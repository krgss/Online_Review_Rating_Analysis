
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import optimizer.LBFGS;
import optimizer.LBFGS.ExceptionWithIflag;
import utilities.*;

public class Laram extends RatingRegression {

    static public boolean SIGMA = false;
    static public double PI = 0.5;

    public static Laram_Model m_model;
    protected double[] m_old_lambda; // in case optimization for alpha failed
    BufferedWriter m_trace;

    // aspect will be determined by the input file for LRR
    public Laram(int lambdaStep, double lambdaTol, int betaStep, double betaTol, double lambda_par) {
        super(lambdaStep, lambdaTol, betaStep, betaTol, lambda_par);

        m_model = null;
        m_old_lambda = null;
    }

    /*public Laram(int lambdastep, double lambdatol,int betastep, double betatol, double lambda_par, String modelfile) {
	  super(lambdastep, lambdatol,betastep, betatol, lambda_par);
	  m_model = null;
	  //Laram_Model(modelfile);
	  m_old_lambda = new double[m_model.m_k];
  }*/
    protected double init(int v, String filename) {
        System.out.println("value of v" + v);
        super.init(v);
        double initV = 1;// likelihood for the first iteration won't matter

        //keep track of the model update trace 
        try {
            m_trace = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("trace.dat"), "UTF-8"));
            for (int i = 0; i < m_k; i++) {
                m_trace.write(String.format("Aspect_%d\t", i));
            }
            m_trace.write("alpha\tbeta\tdata\taux_data\tsigma\n");//column title for the trace file
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (m_model == null) {
            System.out.println("Model Initialise" + v);
            m_model = new Laram_Model(v, filename);
            m_old_lambda = new double[m_model.m_k];

            PI = 2.0;//try to seek a better initialization of beta
            initV = MStep(false);//this is just estimated alpha, no need to update Sigma yet
            PI = 0.5;
        }

        return initV;
    }

    protected double EStep(document d) {
        //step 1: estimate aspect rating
        d.getAspectRating(m_model.m_beta);
        d.getVarAspectRating(m_model.m_beta);
        d.getSigma(m_model.m_delta);

        //step 2: infer aspect weight
        try {
            System.arraycopy(d.m_lambda, 0, m_old_lambda, 0, m_old_lambda.length);
            return infer_lambda(d);
        } catch (ExceptionWithIflag e) {
            System.arraycopy(m_old_lambda, 0, d.m_lambda, 0, m_old_lambda.length);//failed with exceptions
            return -2;
        }
    }

    //we are estimating \hat{alpha}
    protected double getLambdaObjGradient(document d) {
        double expsum = Utilities.expSum(d.m_lambda_hat), orating = -d.rating, s, sum = 0, oRate = 0;

        // initialize the gradient
        Arrays.fill(m_g_lambda, 0);

        for (int i = 0; i < m_model.m_k; i++) {
            d.m_lambda[i] = Math.exp(d.m_lambda_hat[i]) / expsum;//map to aspect weight
            
            orating += d.m_lambda[i] * d.m_pred_bar[i];//estimate the overall rating
            oRate += d.m_lambda[i] * d.m_var_pred[i];
            m_lambda_cache[i] = d.m_lambda_hat[i] - m_model.m_mu[i];//difference with prior

            s = PI * (d.m_pred_bar[i] - d.rating) * (d.m_pred_bar[i] - d.rating);

            if (Math.abs(s) > 1e-10) {//in case we will disable it
                for (int j = 0; j < m_model.m_k; j++) {
                    if (j == i) {
                        m_g_lambda[j] += 0.5 * s * d.m_lambda[i] * (1 - d.m_lambda[i]);
                    } else {
                        m_g_lambda[j] -= 0.5 * s * d.m_lambda[i] * d.m_lambda[j];
                    }
                }
                sum += d.m_lambda[i] * s;
            }
        }

        double diff = orating / m_model.m_delta;
        for (int i = 0; i < m_model.m_k; i++) {
            s = 0;
            for (int j = 0; j < m_model.m_k; j++) {
                // part I of objective function: data likelihood
                if (i == j) {
                    m_g_lambda[j] += (diff * d.m_pred_bar[i] + (oRate * d.m_var_pred[i]) / m_model.m_delta) * d.m_lambda[i] * (1 - d.m_lambda[i]);
                } else {
                    m_g_lambda[j] -= (diff * d.m_pred_bar[i] + oRate) * d.m_lambda[i] * d.m_lambda[j];
                }

                // part II of objective function: prior
                s += m_lambda_cache[j] * m_model.m_sigma_inv[i][j];
            }

            m_g_lambda[i] += s;
            sum += m_lambda_cache[i] * s;
        }

        oRate = 0;
        for (int i = 0; i < m_model.m_k; i++) {
            oRate += d.m_lambda[i] * d.m_var_pred[i] * d.m_lambda[i];
        }

        return 0.5 * ((orating * orating + oRate) / m_model.m_delta + sum);
    }

    private void testLambdaVariance(boolean updateSigma) {
        try {
            int i;
            double v;

            //test the variance of \hat\alpha estimation
            Arrays.fill(m_diag_lambda, 0.0);
            for (document d : LdaEstimate.collection) {
                if (d.m_4train == false) {
                    continue; // do not touch testing cases
                }
                for (i = 0; i < m_k; i++) {
                    v = d.m_lambda_hat[i] - m_model.m_mu[i];
                    m_diag_lambda[i] += v * v; // just for variance
                }
            }

            for (i = 0; i < m_k; i++) {
                m_diag_lambda[i] /= m_trainSize;
                if (i == 0 && updateSigma) {
                    m_trace.write("*");
                }
                m_trace.write(String.format("%.3f:%.3f\t", m_model.m_mu[i], m_diag_lambda[i]));//mean and variance of \hat\alpha
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    protected double infer_lambda(document d) throws ExceptionWithIflag {
        double f = 0;
        int iprint[] = {-1, 0}, iflag[] = {0}, icall = 0, n = m_model.m_k, m = 5;

        //initialize the diagonal matrix
        Arrays.fill(m_diag_lambda, 0);
        double converged = 1;
        double phisum, likelihood = 0;
        double likelihood_old = 0;
        double oldphi[] = new double[n];
        int k, i, var_iter;
        double digamma_gam[] = new double[n];

        // compute posterior dirichlet
        for (k = 0; k < n; k++) {
            d.m_eta[k] = LdaEstimate.INITIAL_ALPHA + (d.docLength / ((double) n));
            digamma_gam[k] = Utils.digamma(d.m_eta[k]);
            for (i = 0; i < d.length; i++) {
                d.phi[i][k] = 1.0 / n;
            }
        }
        var_iter = 0;

        while ((converged > LdaInference.VAR_CONVERGED) && ((var_iter < LdaInference.VAR_MAX_ITER) || (LdaInference.VAR_MAX_ITER == -1))) {
            var_iter++;
            double orating = -d.rating;
            for (k = 0; k < n; k++) {
                orating += d.m_lambda[k] * d.m_pred_bar[k];
            }
            orating = orating * orating;
            orating /= (2 * m_model.m_delta * m_model.m_delta);

            double ORate = 0, calc = 0;;
            for (k = 0; k < n; k++) {
                calc = d.m_lambda[k] * d.m_lambda[k];
                calc += d.m_sigma2[k] * d.m_sigma2[k];
                calc *= d.m_var_pred[k];
                calc += d.m_sigma2[k] * d.m_sigma2[k] * d.m_pred_bar[k] * d.m_pred_bar[k];
                ORate += calc;
            }
            ORate /= (2 * m_model.m_delta * m_model.m_delta);

            for (i = 0; i < d.length; i++) {
                phisum = 0;
                for (k = 0; k < n; k++) {
                    oldphi[k] = d.phi[i][k];
                    d.phi[i][k] = digamma_gam[k] + Math.log(m_model.m_ebsilon[k][d.words[i]]);

                    if (k > 0) {
                        phisum = Utils.log_sum(phisum, d.phi[i][k]);
                    } else {
                        phisum = d.phi[i][k]; // note, phi is in log space
                    }
                }
                for (k = 0; k < n; k++) {
                    d.phi[i][k] = Math.exp(d.phi[i][k] - phisum - (orating + ORate));
                    //d.phi[i][k]-=(orating + ORate);
                    d.m_eta[k] = d.m_eta[k] + d.counts[i] * (d.phi[i][k] - oldphi[k]);
                    digamma_gam[k] = Utils.digamma(d.m_eta[k]);
                }
            }

            d.getAspectRating(m_model.m_beta);
            System.out.println(d.m_lambda_hat[0]);
            d.getVarAspectRating(m_model.m_beta);
            System.out.println("jdsh" + d.m_lambda_hat[0]);
            d.getSigma(m_model.m_delta);
            do {
                f = getLambdaObjGradient(d);//to be minimized
                System.out.println(f + "\t" + m_lambdaTol +"\t"+ iflag) ;
                for(int r=0;r<m_model.m_k;r++ ){
                    System.out.println(d.m_lambda_hat[r]+"  "+ m_g_lambda[r] + "  " + m_diag_lambda[r]);
                }
                LBFGS.lbfgs(n, m, d.m_lambda_hat, f, m_g_lambda, false, m_diag_lambda, iprint, m_lambdaTol, 1e-20, iflag);
            System.out.println(d.m_lambda_hat[0]);
            } while (iflag[0] != 0 && ++icall <= m_lambdaStep);

            likelihood = compute_likelihood(d, d.phi, d.m_eta);
            assert (!new Double(likelihood).isNaN());
            converged = (likelihood_old - likelihood) / likelihood_old;
            likelihood_old = likelihood;
        }

        if (iflag[0] != 0) {
            return -1; // have not converged yet
        } else {
            double expsum = Utilities.expSum(d.m_lambda_hat);
            for (n = 0; n < m_model.m_k; n++) {
                d.m_lambda[n] = Math.exp(d.m_lambda_hat[n]) / expsum;
            }
            return f;
        }

    }

    public static double compute_likelihood(document d, double[][] phi, double[] eita) {
        double likelihood = 0, digsum = 0, var_gamma_sum = 0;
        double dig[] = new double[m_model.m_k];
        int k, n;

        for (k = 0; k < m_model.m_k; k++) {
            dig[k] = Utils.digamma(eita[k]);
            var_gamma_sum += eita[k];
        }
        digsum = Utils.digamma(var_gamma_sum);

        likelihood = Utils.log_gamma(LdaEstimate.INITIAL_ALPHA * m_model.m_k) - m_model.m_k * Utils.log_gamma(LdaEstimate.INITIAL_ALPHA) - (Utils.log_gamma(var_gamma_sum));

        for (k = 0; k < m_model.m_k; k++) {
            likelihood += (LdaEstimate.INITIAL_ALPHA - 1) * (dig[k] - digsum) + Utils.log_gamma(eita[k]) - (eita[k] - 1) * (dig[k] - digsum);

            for (n = 0; n < d.length; n++) {
                if (phi[n][k] > 0) {
                    likelihood += d.counts[n] * (phi[n][k] * ((dig[k] - digsum) - Math.log(phi[n][k]) + m_model.m_ebsilon[k][d.words[n]]));
                }
            }
        }
        return (likelihood);
    }

    public double MStep(boolean updateSigma) {
        updateSigma = false; // shall we update Sigma?
        int i, j, k = m_model.m_k;
        double v;

        //Step 0: initialize the statistics
        Arrays.fill(m_g_lambda, 0.0);

        //Step 1: ML for \mu
        for (document d : LdaEstimate.collection) {
            if (d.m_4train == false) {
                continue; // do not touch testing cases
            }
            for (i = 0; i < k; i++) {
                m_g_lambda[i] += d.m_lambda_hat[i];
            }
        }
        for (i = 0; i < k; i++) {
            m_model.m_mu[i] = m_g_lambda[i] / m_trainSize;
        }
        testLambdaVariance(updateSigma);

        //Step 2: ML for \sigma
        if (updateSigma) {//we may choose to not update \Sigma
            //clear up the cache
            for (i = 0; i < k; i++) {
                Arrays.fill(m_model.m_sigma_inv[i], 0);
            }

            for (document d : LdaEstimate.collection) {
                if (d.m_4train == false) {
                    continue; // do not touch the testing cases
                }
                for (i = 0; i < k; i++) {
                    m_diag_lambda[i] = d.m_lambda_hat[i] - m_model.m_mu[i];
                }
    
                if (SIGMA) {//estimate the whole covariance matrix
                    for (i = 0; i < k; i++) {
                        for (j = 0; j < k; j++) {
                            m_model.m_sigma_inv[i][j] += m_diag_lambda[i] * m_diag_lambda[j] + (d.m_sigma2[i]);
                        }
                    }
                } else {// just for estimate diagonal
                    for (i = 0; i < k; i++) {
                        m_model.m_sigma_inv[i][i] += m_diag_lambda[i] * m_diag_lambda[i] + (d.m_sigma2[i] );
                    };
                }
            }

            for (i = 0; i < k; i++) {
                if (SIGMA) {
                    m_model.m_sigma_inv[i][i] = (1.0 + m_model.m_sigma_inv[i][i]) / (1 + m_trainSize); // prior
                    for (j = 0; j < k; j++) {
                        m_model.m_sigma[i][j] = m_model.m_sigma_inv[i][j];
                    }
                } else {
                    v = (1.0 + m_model.m_sigma_inv[i][i]) / (1 + m_trainSize);
                    m_model.m_sigma[i][i] = v;
                    m_model.m_sigma_inv[i][i] = 1.0 / v;
                }
            }
            m_model.calcSigmaInv(1);
        }

        //calculate the likelihood for the alpha part
        double lambda_likelihood = 0, beta_likelihood = 0;
        for (document d : LdaEstimate.collection) {
            if (d.m_4train == false) {
                continue; // do not touch testing cases
            }
            for (i = 0; i < k; i++) {
                m_diag_lambda[i] = d.m_lambda_hat[i] - m_model.m_mu[i];
                System.out.println(d.m_lambda_hat[i] + "      " + m_model.m_mu[i]);
                System.out.println("lambda  " + m_diag_lambda[i]);
            }
            lambda_likelihood += m_model.calcCovariance(m_diag_lambda);
        }
        lambda_likelihood += Math.log(m_model.calcDet());
        System.out.println("lambda likilihood" + lambda_likelihood);
        //Step 3: ML for \beta
        try {
            ml_beta();
        } catch (ExceptionWithIflag e) {
            e.printStackTrace();
        }

        beta_likelihood = getBetaPriorObj();

        //Step 4: ML for \delta
        double datalikelihood = getDataLikelihood(), auxdata = getAuxDataLikelihood(), oldDelta = m_model.m_delta;
        m_model.m_delta = datalikelihood / m_trainSize;
        datalikelihood /= oldDelta;

        try {
            m_trace.write(String.format("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", lambda_likelihood, beta_likelihood, datalikelihood, auxdata, Math.log(m_model.m_delta)));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return lambda_likelihood + beta_likelihood + datalikelihood + auxdata + Math.log(m_model.m_delta);
    }

    //\beat^T * \beta
    protected double getBetaPriorObj() {
        double likelihood = 0;
        for (int i = 0; i < m_model.m_k; i++) {
            for (int j = 0; j < m_model.m_v + 1; j++) {
                likelihood += m_model.m_beta[i* (m_model.m_v +1) + j] * m_model.m_beta[i* (m_model.m_v +1) + j];
            }
        }
        return m_lambda_par * likelihood;
    }

    protected double getBetaObjGradient() {
        double likelihood = 0, aux_likelihood = 0, orating, diff, oRate;
        int vSize = m_model.m_v + 1;

        // initialize the structure
        Arrays.fill(m_g_beta, 0);

        // part I of objective function: data likelihood
        for (document d : LdaEstimate.collection) {
            if (d.m_4train == false) {
                continue; // do not touch testing cases
            }
            oRate = d.rating;
            orating = -oRate;

            //apply the current model
            d.getAspectRating(m_model.m_beta);
            for (int i = 0; i < m_model.m_k; i++) {
                orating += d.m_lambda[i] * d.m_pred_bar[i];
            }

            likelihood += orating * orating;
           // orating /= m_model.m_delta; // in order to get consistency between aux-likelihood
            for (int i = 0; i < m_model.m_k; i++) {
                m_g_lambda_cache[i] = d.m_lambda[i] * d.m_lambda[i] +  d.m_sigma2[i];
                m_g_lambda_cache[i] *= d.m_var_pred[i];
                m_g_lambda_cache[i] += (d.m_sigma2[i] * d.m_pred_bar[i] * d.m_pred_bar[i]);
                likelihood += m_g_lambda_cache[i];
            }

            double value1, value2 = 0;
            for (int i = 0; i < m_model.m_k; i++) {
                aux_likelihood += d.m_lambda[i] * (d.m_pred_bar[i] - oRate) * (d.m_pred_bar[i] - oRate);
                if (RatingRegression.SCORE_SQUARE) {
                    diff = d.m_lambda[i] * (orating + PI * (d.m_pred_bar[i] - oRate)) * d.m_pred_cache[i];
                } else {
                    diff = d.m_lambda[i] * (orating + PI * (d.m_pred_bar[i] - oRate)) * d.m_pred_bar[i];
                }
                diff += d.m_sigma2[i] * d.m_pred_bar[i];
                value1 = (d.m_lambda[i] * d.m_lambda[i] + d.m_sigma2[i] );
                for (int j = 0; j < d.length; j++) {
                    value2 = m_model.m_beta[i* (m_model.m_v +1) + d.words[j]] * d.counts[j]*(1 - d.phi[j][i]);
                    m_g_beta[m_model.m_v * i + d.words[j]] += ((value1 * value2) + diff)*d.phi[j][i]*d.counts[j];
              
                }

            }
        }

        double reg = 0;
        for (int i = 0; i < m_beta.length; i++) {
            m_g_beta[i] += m_lambda_par * m_beta[i];
            reg += m_beta[i] * m_beta[i];
        }

        return 0.5 * (likelihood / m_model.m_delta + PI * aux_likelihood + m_lambda_par * reg);
    }

    //\sum_d(\sum_i\alpha_{di}\S_{di}-r_d)^2/\sigma^2
    protected double getDataLikelihood() {
        double likelihood = 0, orating;

        // part I of objective function: data likelihood
        for (document d : LdaEstimate.collection) {
            if (d.m_4train == false) {
                continue; // do not touch testing cases
            }
            orating = -d.rating;
            double delta_cache;

            //apply the current model
            d.getAspectRating(m_model.m_beta);
            for (int i = 0; i < d.m_lambda.length; i++) {
                orating += d.m_lambda[i] * d.m_pred_bar[i];
            }

            for (int i = 0; i < m_model.m_k; i++) {
                delta_cache = d.m_lambda[i] * d.m_lambda[i] + d.m_sigma2[i] ;
                delta_cache *= d.m_var_pred[i];
                delta_cache += (d.m_sigma2[i] * d.m_pred_bar[i] * d.m_pred_bar[i]);
                likelihood += delta_cache;
            }

            likelihood += orating * orating;
        }
        return likelihood;
    }

    //\sum_d\pi\sum_i\alpha_{di}(\S_{di}-r_d)^2
    protected double getAuxDataLikelihood() {
        double likelihood = 0.0, orating;

        // part I of objective function: data likelihood
        for (document d : LdaEstimate.collection) {
            if (d.m_4train == false) {
                continue; // do not touch testing cases
            }
            orating = d.rating;
            for (int i = 0; i < d.m_lambda.length; i++) {
                System.out.println("lambda:" + d.m_lambda[i]);
                System.out.println("pred_bar:" + d.m_pred_bar[i]);
                System.out.println("orating:" + orating);
                double x = d.m_lambda[i] * (d.m_pred_bar[i] - orating) * (d.m_pred_bar[i] - orating);
                likelihood += d.m_lambda[i] * (d.m_pred_bar[i] - orating) * (d.m_pred_bar[i] - orating);
            }
        }
        return PI * likelihood;
    }

    protected double ml_beta() throws ExceptionWithIflag {
        double f = 0;
        int iprint[] = {-1, 0}, iflag[] = {0}, icall = 0, n = (1 + m_model.m_v) * m_model.m_k, m = 10;

        for (int i = 0; i < m_model.m_k; i++) {//set up the starting point
            //System.out.println(m_model.m_beta[i].length);
            //System.out.println(m_beta.length);
            //System.out.println()
            //System.out.println(m_model.m_k + "       "+i);
            System.arraycopy(m_model.m_beta,  i * (m_model.m_v + 1), m_beta, i * (m_model.m_v + 1), m_model.m_v + 1);
            //System.out.println(m_model.m_v+1);
            //System.out.println("sjhdg");
        }

        Arrays.fill(m_diag_beta, 0);
        do {
            if (icall % 1000 == 0) {
                System.out.print(".");//keep track of beta update
            }
            f = getBetaObjGradient();//to be minimized
            LBFGS.lbfgs(n, m, m_beta, f, m_g_beta, false, m_diag_beta, iprint, m_betaTol, 1e-20, iflag);
             System.out.println(m_beta[0]);
        } while (iflag[0] != 0 && ++icall <= m_betaStep);

        System.out.print(icall + "\t");
        for (int i = 0; i < m_model.m_k; i++) {
            System.arraycopy(m_beta, i * (m_model.m_v + 1), m_model.m_beta, i * (m_model.m_v + 1) , m_model.m_v + 1);
        }
        return f;
    }

    public void EM_est(int maxIter, double converge, String filename) {
        int iter = 0, lambda_exp = 0, lambda_cov = 0;
        double tag, diff = 10, likelihood = 0, old_likelihood = init(LoadVectors(), filename);
        System.out.println("likelihood  " + old_likelihood);
        System.out.println("[Info]Step\t oMSE\t aMSE\t aCorr\t iCorr\t cov(a)\t exp(a)\t obj\t converge");
        while (iter < Math.min(8, maxIter) || (iter < maxIter && diff > converge)) {
            lambda_exp = 0;
            lambda_cov = 0;

            //E-step
            for (document d : LdaEstimate.collection) {
                if (d.m_4train) {
                    tag = EStep(d);
                    if (tag == -1) // failed to converge
                    {
                        lambda_cov++;
                    } else if (tag == -2) // failed with exceptions
                    {
                        lambda_exp++;
                    }
                }
            }
            System.out.print(iter + "\t");//sign of finishing E-step

            //M-step
            likelihood = MStep(iter % 4 == 3);//updating \Sigma too often will hamper \hat\alpha convergence		

            //evaluateAspect();// evaluating in the testing cases
            diff = (old_likelihood - likelihood) / old_likelihood;
            old_likelihood = likelihood;
            System.out.println(String.format("\t%d\t%d\t%.3f\t%.3f", lambda_cov, lambda_exp, likelihood, diff));
            iter++;
        }

        try {
            m_trace.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void SaveModel(String filename) {
        m_model.Save2File(filename);
    }

}
