
import java.util.*;
import utilities.Utilities;

public class document {

    double rating;
    int[] words;
    int[] counts;
    int length;
    int docLength;
    double[][] phi;
    double[] m_eta; // variational parameter

    double[] m_lambda; // variatiobal parameter of aspect weight
    double[] m_lambda_hat;
    //double[] m_pred_rating;
    double[] m_pred_cache;
    double[] m_var_pred;
    double[] m_pred_bar;
    double[] m_pred_var_cache;
    double m_sigma2[];
    //int[] z_dn;
    //String id;
    boolean m_4train;
    int max_index;
    //int[][] aspectFreq;
    //ArrayList<Integer> index_word;

    public document() {
        rating = 0;
        length = 0;
        docLength = 0;
        m_lambda = new double[LdaEstimate.NTOPICS];
        m_lambda_hat = new double[m_lambda.length];
        m_pred_cache = new double[LdaEstimate.NTOPICS];
        m_var_pred = new double[m_lambda.length];
        m_pred_bar = new double[m_lambda.length];
        m_pred_var_cache = new double[m_lambda.length];
        Utilities.randomize(m_lambda_hat);
        double norm = Utilities.expSum(m_lambda_hat);
        for(int i=0; i<LdaEstimate.NTOPICS; i++)
	      m_lambda[i] = Math.exp(m_lambda_hat[i])/norm;
        //for(int i=0;i<m_lambda_hat.length;i++)
            //System.out.println(m_lambda_hat[i]);
    }
    //apply model onto each aspect

    public void getAspectRating(double[] beta) {
        int vocabSize = Analyzer.m_vocabulary.size();
        for (int i = 0; i < Laram.m_model.m_k; i++) {
            //System.out.println("get Aspect Rating");
            for (int j = 0; j < length; j++) {
                //System.out.println(beta[i][index_word.get(j)]);
                m_pred_cache[i] += beta[i*vocabSize + words[j]] * counts[j]* phi[j][i];
            }

            if (RatingRegression.SCORE_SQUARE)// to avoid negative rating
            {
                m_pred_bar[i] = 0.5 * m_pred_cache[i] * m_pred_cache[i];
            } else {
                m_pred_bar[i] = Math.exp(m_pred_cache[i]);
            }
        }
    }

    public void getVarAspectRating(double[] beta) {
        int vocabSize = Analyzer.m_vocabulary.size();
        for (int i = 0; i < Laram.m_model.m_k; i++) {
            for (int j = 0; j < length; j++) {
                m_pred_var_cache[i] += (beta[i* vocabSize +words[j]] * beta[i*vocabSize+ words[j]] *counts[j]* phi[j][i]) * (1 - phi[j][i]);
            }

            if (RatingRegression.SCORE_SQUARE)// to avoid negative rating
            {
                m_var_pred[i] = 0.5 * m_pred_cache[i] * m_pred_cache[i];
            } else {
                m_var_pred[i] = Math.exp(m_pred_cache[i]);
            }
        }
    }

    public void getSigma(double delta) {
        for (int i = 0; i < Laram.m_model.m_k; i++) {
            double oRate = 0;
            oRate = delta * delta;
            oRate /= (m_var_pred[i] + m_pred_bar[i] * m_pred_bar[i] + delta * delta);
            m_sigma2[i] = oRate;
        }
    }

}
