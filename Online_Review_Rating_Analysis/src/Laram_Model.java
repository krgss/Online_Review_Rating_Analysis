

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Random;

import utilities.MatrixOperations;
import utilities.Utilities;


public class Laram_Model {
	public int m_k = LdaEstimate.NTOPICS;// # of aspects
	public static int m_v; // # of words
	int m_length;
	public static final int OFFSET =0;
	double[] m_mu; // prior for \lambda in each review
	double[][] m_sigma_inv; // precision matrix (NOT covariance!)
	double[][] m_sigma; // only used for calculating inverse(\Sigma)
	double[] m_beta;// word sentiment polarity matrix should have one bias term!
	double [][] m_ebsilon;
	double m_delta; // variance of overall rating prediction (\sigma in the manual)
	
	//Algebra m_alg;
	
	public Laram_Model(int v, String filename){
		m_v = v;
		LoadFromFile(filename);
		init();
	}
	
	
	public Laram_Model(String filename){
		LoadFromFile(filename);
	}
	
	protected void init(){
		create();
		
		Random rand = new Random();
		for(int i=0; i<m_k; i++){
			m_mu[i] = (2.0*rand.nextDouble() - 1.0);
			m_sigma_inv[i][i] = 1.0;
			m_sigma[i][i]=1;
			
		}
                Utilities.randomize(m_beta);
		m_delta = 1.0;
	}

	
	private void create(){
		//System.out.println("aslkdfj" + m_k);
		m_mu = new double[m_k];
		m_sigma = new double[m_k][m_k];
		m_sigma_inv = new double[m_k][m_k];
		m_beta = new double[m_k *(Analyzer.m_vocabulary.size()+1)];
		m_ebsilon = new double[m_k][Analyzer.m_vocabulary.size()+1];
		//m_alg = new Algebra();
	}
	
	public double calcCovariance(double[] vct){
		double sum = 0, s;
		for(int i=0; i<m_k; i++){
			s = 0;
			for(int j=0; j<m_k; j++)
				s += vct[j] * m_sigma_inv[j][i];
			sum += s * vct[i];
		}
		return sum;
	}
	
	public double calcDet(){
		return MatrixOperations.matrixDeterminant(m_sigma);
	}
	
	
   public void calcSigmaInv(double scale)
    {
		double[][] inv = MatrixOperations.invertMatrix(m_sigma);
		for(int i=0; i<m_k; i++){
			for(int j=0; j<m_k; j++)
				m_sigma_inv[i][j] = inv[i][j] * scale;
		}
	}
	
	
   public void Save2File(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			writer.write(m_k + "\t" + m_v + "\n");
			
			//save \mu
			for(int i=0; i<m_k; i++)
				writer.write(m_mu[i] + "\t");
			writer.write("\n");
			
			//save \sigma
			for(int i=0; i<m_k; i++){
				for(int j=0; j<m_k; j++)
					writer.write(m_sigma[i][j]+ "\t");
				writer.write("\n");
			}
			
			//save \beta
			for(int i=0; i<m_k; i++){
				for(int j=0; j<=m_v; j++)
					writer.write(m_beta[i* m_v + j] + "\t");
				writer.write("\n");
			}
			
			//save delta
			writer.write(Double.toString(m_delta));
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void LoadFromFile(String filename){
			create();
			//part 1: aspect size, vocabulary size
			m_length = LdaEstimate.collection.size();
			//m_v = calLength(filename);
			
			m_v = Analyzer.m_vocabulary.size();
			
			//part 2: \mu
			//System.out.println("hjj" + m_mu.length );
			for(int i=0; i<m_k; i++)
				m_mu[i] = 1;
			
			//part 3: \sigma
			for(int i=0; i<m_k; i++){
				for(int j=0; j<m_k; j++){
					if(i==j)
						m_sigma[i][j]=1;
					else
						m_sigma[i][j]=0.0;
					//m_sigma.setQuick(i, j, Double.valueOf(container[j]));
				}	
			}
			calcSigmaInv(1.0);
			
			//part 4: \beta
			for(int i=0;i<m_k;i++) {
					Utilities.randomize(m_beta);
			}
			
			// part 5: ebsilon
			
		try{
			FileReader file = new FileReader(filename);
			BufferedReader buffer = new BufferedReader(file);
			String line;
			int j=0;
			while((line= buffer.readLine())!=null) {
				String[] container;
				container = line.split(" ");
				for(int i=0;i<m_v;i++) {
					m_ebsilon[j][i]= Double.valueOf(container[i]);
				}
			}
			buffer.close();
	       }catch(IOException ex) {
	    	 ex.printStackTrace();   
	       }
			
			//part 6: \delta
			m_delta = 1.0;
	}
	

}

