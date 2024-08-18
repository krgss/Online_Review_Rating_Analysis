import java.io.*;
import java.util.*;
public class LdaEstimate {

	/*
	 * perform inference on a document and update sufficient statistics
	 *
	 */

	public static int LAG = 5;
    public static Analyzer analyzer;
    public static Laram laram;
	public static double EM_CONVERGED;
	public static int EM_MAX_ITER;
	public static int ESTIMATE_ALPHA;
	public static double INITIAL_ALPHA;
	static int NTOPICS;
	public static ArrayList<document> collection = new ArrayList<document>();
	 
	public static double doc_e_step(document doc, double[] gamma,lda_model model, lda_suffstats ss)
	{
	    double likelihood;
	    int n, k;

	    // posterior inference

	    likelihood = LdaInference.lda_inference(doc, model, gamma, doc.phi);

	    // update sufficient statistics

	    double gamma_sum = 0;
	    for (k = 0; k < model.num_topics; k++)
	    {
	        gamma_sum += gamma[k];
	        ss.alpha_suffstats += Utils.digamma(gamma[k]);
	    }
	    ss.alpha_suffstats -= model.num_topics * Utils.digamma(gamma_sum);

	    for (n = 0; n < doc.length; n++)
	    {
	        for (k = 0; k < model.num_topics; k++)
	        {
	            ss.class_word[k][doc.words[n]] += doc.counts[n]*doc.phi[n][k];
	            ss.class_total[k] += doc.counts[n]*doc.phi[n][k];
	        }
	    }

	    ss.num_docs = ss.num_docs + 1;

	    return(likelihood);
	}

	/*
	 * writes the word assignments line for a document to a file
	 *
	 */

	public static void write_word_assignment(BufferedWriter f, document doc, double[][] phi, lda_model model)
	{
	    
		try{
			int n;

	       
	        f.write(doc.length);
	        f.write(" ");
	        for (n = 0; n < doc.length; n++)
	        {
	            
	            f.write(doc.words[n]);
	            f.write(":");
	            f.write(Utils.argmax(phi[n], model.num_topics));
	            f.write("  ");
	         }
	         f.newLine();
	         f.flush();
	        }
		    catch(IOException ex) {
		    	ex.printStackTrace();
		    }
	}


	/*
	 * saves the gamma parameters of the current dataset
	 *
	 */

	public static void save_gamma(String filename, double[][] gamma, int num_docs, int num_topics)
	{
	    int d, k;
	    try{
	       FileWriter writer = new FileWriter(filename);  
	       BufferedWriter buffer = new BufferedWriter(writer); 

	       for (d = 0; d < num_docs; d++)
	       {
	    	   buffer.write(" " +  String.valueOf(gamma[d][0]));
		       for (k = 1; k < num_topics; k++)
		       { 
			       buffer.write(" " + String.valueOf(gamma[d][k]));
		    
		       }
		       buffer.newLine();
	       }
	         buffer.close();
	   }catch(IOException e) {
	       e.printStackTrace();
	   } 
   }


	/*
	 * run_em
	 *
	 */

	public static void run_em(String start, String directory, corpus crps)
	{

	    int d, n;
	    lda_model model;
	    double var_gamma[][];
	    //phi[][];

	    // allocate variational parameters

	    var_gamma = new double[crps.num_docs][NTOPICS];

	    int max_length = lda_data.max_corpus_length(crps);
	   // phi = new double[max_length][NTOPICS];

	    // initialize model

	    String filename;

	    lda_suffstats ss = new lda_suffstats();
	    if (start.equals("seeded"))
	    {
	        model = LdaModel.new_lda_model(crps.num_terms, NTOPICS);
	        ss = LdaModel.new_lda_suffstats(model);
	        LdaModel.corpus_initialize_ss(ss, model, crps);
	        LdaModel.lda_mle(model, ss, 0);
	        model.alpha = INITIAL_ALPHA;
	    }
	    else if (start.equals("random"))
	    {
	        model = LdaModel.new_lda_model(crps.num_terms, NTOPICS);
	        ss = LdaModel.new_lda_suffstats(model);
	        LdaModel.random_initialize_ss(ss, model);
	        LdaModel.lda_mle(model, ss, 0);
	        model.alpha = INITIAL_ALPHA;
	    }
	    else
	    {
	        model = LdaModel.load_lda_model(start);
	        ss = LdaModel.new_lda_suffstats(model);
	    }

	    filename = directory.concat("/000"); 
	    LdaModel.save_lda_model(model, filename);

	    // run expectation maximization
	    
	    filename = directory.concat("/likelihood.dat");
		try {
			int i = 0;
		    double likelihood=0.0, likelihood_old = 0, converged = 1;
		    FileWriter writer = new FileWriter(filename);
	        BufferedWriter buffer = new BufferedWriter(writer); 
	    
	    

	        while (((converged < 0) || (converged > EM_CONVERGED) ||
	        		                  (i <= 2)) && (i <= EM_MAX_ITER))
	        {
	        	i++;  
	        	System.out.println("**** em iteration  ****" + i);
	        	likelihood = 0;
	        	LdaModel.zero_initialize_ss(ss, model);

	        	// e-step

	        	for (d = 0; d < crps.num_docs; d++)
	        	{
	        		if ((d % 1000) == 0) 
	        			System.out.println("document " + d);
	        		likelihood += doc_e_step(crps.docs[d], var_gamma[d], model, ss);
	        	}

	        	// m-step

	        	LdaModel.lda_mle(model, ss, ESTIMATE_ALPHA);
	        	
	        	// check for convergence

	        	converged = (likelihood_old - likelihood) / (likelihood_old);
	        	if (converged < 0) LdaInference.VAR_MAX_ITER = LdaInference.VAR_MAX_ITER * 2;
	        	likelihood_old = likelihood;

	        	// output model and likelihood

	        	buffer.write(String.valueOf(likelihood) + "\t" + String.valueOf(converged));
	        	buffer.newLine();
	        
	        
	        	buffer.flush();
	        	if ((i % LAG) == 0)
	        	{
	            
	        		filename = directory.concat( "/" + String.valueOf(i));
	            
	        		LdaModel.save_lda_model(model, filename);
	            	filename = directory.concat( "/" + String.valueOf(i) + ".gamma");
	            	save_gamma(filename, var_gamma, crps.num_docs, model.num_topics);
	        	}
	        }

	        // output the final model

	        filename = directory.concat("/final");
	        LdaModel.save_lda_model(model, filename);
	        filename = directory.concat("/final.gamma");
	        save_gamma(filename, var_gamma, crps.num_docs, model.num_topics);
	    

	        // output the word assignments (for visualization)

	   
	        filename = directory.concat("/word-assignments.dat");
	        FileWriter filewriter = new FileWriter(filename);  
	        BufferedWriter bufferedWriter = new BufferedWriter(filewriter); 
	        for (d = 0; d < crps.num_docs; d++)
	        {
	        	if ((d % 100) == 0) 
	        		System.out.println("final e step document " + d);
	        	likelihood += LdaInference.lda_inference((crps.docs[d]), model, var_gamma[d], crps.docs[d].phi);
	        	write_word_assignment(bufferedWriter, crps.docs[d], crps.docs[d].phi, model);
	        }
	        bufferedWriter.close();
	        buffer.close();
			} catch (IOException e) {
				e.printStackTrace();
			}  
	}


	/*
	 * read settings.
	 *
	 */

	public static void read_settings(String filename)
	{
	    
		try {
	            FileReader fileReader = new FileReader(filename);
                BufferedReader bufferedReader = new BufferedReader(fileReader);
        
	            String alpha_action="";
	            String line;
	            int count=1;
                while ((line = bufferedReader.readLine()) != null)
        	           
                {
                       String[] container;
                       container = line.split(" "); // container is a string arry of words
                       if(count==1)
                               LdaInference.VAR_MAX_ITER = Integer.valueOf(container[3]);
                       else if(count==2)
        	                   LdaInference.VAR_CONVERGED = Double.valueOf(container[2]);
                       else if(count==3)
	                          EM_MAX_ITER = Integer.valueOf(container[3]);
                       else if(count==4)
	                          EM_CONVERGED = Double.valueOf(container[2]);
                       else 
	                          alpha_action = container[1];
                       count++;
               }
	           if (alpha_action.compareTo("fixed")==0)
	           {
		            ESTIMATE_ALPHA = 0;
	           }
	           else
	           {
	       	       ESTIMATE_ALPHA = 1;
	           }
	           bufferedReader.close();
           }catch(IOException e) {
		       e.printStackTrace();
		   }
	}

	/*
	 * inference only
	 *
	 */

    public static void infer(String model_root, String save, corpus crps)
	{
	    String filename;
	    int  d, n;
	    lda_model model;
	    double var_gamma[][], likelihood;
	    document doc;

	    model = LdaModel.load_lda_model(model_root);
	    var_gamma = new double[crps.num_docs][model.num_topics];
	    
	    filename = save.concat("-lda-lhood.dat");
	    
	    try {
	         FileWriter writer = new FileWriter(filename);  
	         BufferedWriter buffer = new BufferedWriter(writer);  
	         for (d = 0; d < crps.num_docs; d++)
	          {
	        	 
		           if (((d % 100) == 0) && (d>0)) 
		    	        System.out.println("document  "+d);

		           doc = crps.docs[d];
		           //phi = new double[doc.length][model.num_topics];
		           for (n = 0; n < doc.length; n++)
		        	  doc.phi[n] = new double[model.num_topics];
		           likelihood = LdaInference.lda_inference(doc, model, var_gamma[d], doc.phi);
		           buffer.write(String.valueOf(likelihood) + " ");
		           buffer.newLine();
	           }
	    
	          buffer.close();
	    }catch(IOException e) {
	    	e.printStackTrace();
	    }
	    filename = save.concat("-gamma.dat");
	    
	    save_gamma(filename, var_gamma, crps.num_docs, model.num_topics);
	}


	/*
	 * update sufficient statistics
	 *
	 */



	/*
	 * main
	 *
	 */
    
    public LdaEstimate() {
    	analyzer = new Analyzer();
    	analyzer.LoadVocabulary("Data/Seeds/hotel_vocabulary_CHI.dat");
		analyzer.LoadDirectory("Data/Reviews/", ".dat","Data/Vectors/vector_CHI_4000.dat");
		//analyzer.find();
    }

	public static void main(String argv[])
	{
	    // (est / inf) alpha k settings data (random / seed/ model) (directory / out)

	    corpus crps;

	    long t1;
	    
	    //(void) time(&t1);
	    //seedMT(t1);
	    // seedMT(4357U);
	    String directory="";
	    if (argv.length > 1)
	    {
	    	
	        if (argv[1].equals("est"))
	        {
	            INITIAL_ALPHA = Double.parseDouble(argv[2]);
	            NTOPICS = Integer.parseInt(argv[3]);
	            LdaEstimate model = new LdaEstimate();
	   	        collection = analyzer.m_collection;
	            read_settings(argv[4]);
	            crps = lda_data.read_data(argv[5], collection);
	            Utils.make_directory(argv[7]);
	            directory = argv[7];
	            run_em(argv[6], argv[7], crps);
	        }
	        if (argv[1].equals("inf"))
	        {
	            read_settings(argv[2]);
	            crps = lda_data.read_data(argv[4], collection);
	            infer(argv[3], argv[5], crps);
	        }
	    }
	    else
	    {
	        System.out.println("usage : lda est [initial alpha] [k] [settings] [data] [random/seeded/*] [directory]\n");
	        System.out.println("        lda inf [settings] [model] [data] [name]\n");
	    }
	    
	      Laram model2 = new Laram(500,1e-2,5000,1e-2,2.0);
		  //model2.LoadVocublary("filename");
		  model2.EM_est( 10, 1e-4, directory + "/final.beta");	
		  model2.SavePrediction("Data/Results/prediction.dat");
		  model2.SaveModel("Data/Model/model_hotel.dat");
	}

}
