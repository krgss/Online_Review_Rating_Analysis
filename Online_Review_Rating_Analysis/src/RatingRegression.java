
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Random;
//
//import optimizer.LBFGS;
//import optimizer.LBFGS.ExceptionWithIflag;
//import utilities.Utilities;

public class RatingRegression {
	static final boolean SCORE_SQUARE = false;//rating will be map by s^2 or exp(s)
	static final boolean BY_OVERALL = false;//train aspect rating predictor by overall rating
	
	protected double[] m_diag_beta;// cached diagonal for beta inference
	protected double[] m_g_beta;// cached gradient for beta inference
	protected double[] m_beta;// long vector for the matrix of beta
	
	protected double[] m_diag_sigma2; 
	
	protected double[] m_diag_lambda;// cached diagonal for alpha inference
	protected double[] m_g_lambda;// cached gradient for alpha inference
	private double[] m_lambda; // cached for difference vector
	protected double[] m_lambda_cache; // to map alpha into a simplex by logistic functions
	protected double[] m_g_lambda_cache;
	protected int m_lambdaStep;
	protected double m_lambdaTol;
	
	protected int m_betaStep;
	protected double m_betaTol;
	protected double m_lambda_par;
	protected int m_v, m_k;
	protected int m_trainSize, m_testSize;
	
	protected Random m_rand;
	
	public RatingRegression(int lambdaStep, double lambdaTol, int betaStep, double betaTol, double lambda_par){
		m_lambdaStep =lambdaStep;
		m_lambdaTol =lambdaTol;
		
		m_betaTol = betaTol;
		m_betaStep = betaStep;
		m_lambda_par = lambda_par;
		
		m_rand = new Random(0);//with fixed random seed in order to get the same train/test split
	}
	
	protected double prediction(document d){
		//predict aspect rating
		d.getAspectRating(m_beta);
		double orating = 0;
		for(int i=0; i<m_k; i++)
			orating += m_lambda_cache[i] * d.m_pred_bar[i];
		return orating;
	}
	
	
	
	/*protected void evaluateAspect(){
		double aMSE = 0, oMSE = 0, icorr = 0, acorr = 0, corr, diff;
		int i = -1;
		boolean iError = false, aError = false;
		
		double[][] pred = new double[m_k][m_testSize], ans = new double[m_k][m_testSize];
		for(document d: LdaEstimate.collection){
			if (d.m_4train)
				continue;//only evaluating in testing cases
			i++;
			
			diff = prediction(d) - d.rating;
			oMSE += diff*diff;
			for(int j=0; j<m_k; j++){
				pred[j][i] = d.m_pred_bar[j];
				//ans[j][i] = vct.m_ratings[j+1];
			}
			
			//1. Aspect evaluation: to skip overall rating in ground-truth
			//aMSE += Utilities.MSE(vct.m_pred, vct.m_ratings, 1);
			//corr = Utilities.correlation(vct.m_pred, vct.m_ratings, 1);
			
			/*if (Double.isNaN(corr)==false)
				icorr += corr;
			else
				iError = true;//error occur
		}
		
		//2. entity level evaluation
		for(int j=0; j<m_k; j++){
			corr = Utilities.correlation(pred[j], ans[j], 0);
			if (Double.isNaN(corr)==false)
				acorr += corr;
			else
				aError = true;
		}
		
		//MSE for overall rating, MSE for aspect rating, item level correlation, aspect level correlation
		if (iError)
			System.out.print('x');
		else
			System.out.print('o');
		if (aError)
			System.out.print('x');
		else
			System.out.print('o');
		System.out.print(String.format(" %.3f\t%.3f\t%.3f\t%.3f", Math.sqrt(oMSE/m_testSize), Math.sqrt(aMSE/m_testSize), (icorr/m_testSize), (acorr/m_k)));
		
	}*/
	
	
	protected double init(int v){
		if (LdaEstimate.collection.isEmpty()){
			System.err.println("[Error]Load training data first!");
			return -1;
		}
		System.out.println("Size of v" + v);
		m_v = v;
		m_k = LdaEstimate.NTOPICS;
		
		m_diag_beta = new double[m_k * (m_v+1)];//to include the bias term for each aspect
		m_g_beta = new double[m_diag_beta.length];
		m_beta = new double[m_g_beta.length];
		
		m_diag_lambda = new double[m_k];
		m_g_lambda = new double[m_k];
		m_lambda = new double[m_k];
		m_lambda_cache = new double[m_k];
		m_g_lambda_cache = new double[m_k];
		
		return 0;
	}
	
//	public int LoadVectors(String filename){
//		return LoadVectors(filename, -1);
//	}
	
	public int LoadVectors(){
	
			m_k = LdaEstimate.NTOPICS; 
			m_trainSize = 0;
			m_testSize = 0;
			//double[] aspectSize = null;
		    boolean isTrain;
			for( document d : LdaEstimate.collection){
				if (isTrain = (m_rand.nextDouble()<0.75))//train/test splitting ratio 
					m_trainSize++;
				else
					m_testSize++;
				d.m_4train = isTrain;
			}
			int len = Analyzer.m_vocabulary.size();
				
//				if (aspectSize==null){
//					aspectSize = new double[LdaEstimate.NTOPICS];
//					m_k = aspectSize.length;
//				}
//			
			//System.out.print("[Info]Aspect length proportion:");
		    System.out.println(len);
			
			//System.out.println("[Info]Load " + m_trainSize + "/" + m_testSize + " instances from " + filename + " with feature size " + len);
			return len;
		}
	
	//save all the prediction results
		public void SavePrediction(String filename){
			try {
				BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
				for(document d: LdaEstimate.collection ){
					//writer.write(vct.m_ID);
					
					//all the ground-truth ratings
				     //for(int i=0; i<vct.m_ratings.length; i++)
						//writer.write(String.format("\t%.3f", vct.m_ratings[i]));
					
					//predicted ratings
					d.getAspectRating(m_beta);
					writer.write("\t");
					for(int i=0; i<d.m_pred_bar.length; i++)
						writer.write(String.format("\t%.3f", d.m_pred_bar[i]));
					
					//inferred weights (not meaningful for baseline logistic regression)
					writer.write("\t");
					for(int i=0; i<d.m_lambda.length; i++)
						writer.write(String.format("\t%.3f", d.m_lambda[i]));
					writer.write("\n");
				}
				writer.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		public void SaveModel(String filename){
			try {
				BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
				writer.write(m_k + "\t" + m_v + "\n");
				
				//\mu for \hat\lambda
				for(int i=0; i<m_k; i++)
					writer.write(m_lambda[i] + "\t");
				writer.write("\n");
				
				//\Sigma for \hat\alpha (unknown for logistic regression)
				for(int i=0; i<m_k; i++){
					for(int j=0; j<m_k; j++){
						if (i==j)
							writer.write("1.0\t");
						else
							writer.write("0.0\t");
					}
					writer.write("\n");
				}
				
				//\beta
				for(int i=0; i<m_k; i++){
					for(int j=0; j<=m_v; j++)
						writer.write(m_beta[i*(m_v+1) + j] + "\t");
					writer.write("\n");
				}
				
				//\sigma (unknown for logistic regression)
				writer.write("1.0");
				writer.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
	
}
	
