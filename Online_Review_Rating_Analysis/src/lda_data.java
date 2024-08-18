import java.io.*;
import java.util.*;

public class lda_data {
    public static final int OFFSET =0;
    
   
    public static corpus read_data(String data_filename, ArrayList<document>collection)
    {
    	int length, count, word, n, nd, nw;
    	corpus c = null;
    	
    	try {
    		
        	FileReader fileReader = new FileReader(data_filename);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            c = new corpus();
            c.num_terms = 0;
            c.num_docs = 0;
            nd = 0; nw = 0;
            String line;
            System.out.println("reading data from "+ data_filename );
            
            while ((line = bufferedReader.readLine()) != null)
            {
            	nd++;
            }
            bufferedReader.close();
            c.num_docs= nd;
            nd=0;
            c.docs = new document[c.num_docs];
             
            System.out.println(c.num_docs);
            	
             fileReader = new FileReader(data_filename);
             bufferedReader = new BufferedReader(fileReader);
             int max_index = -1;
            while ((line = bufferedReader.readLine()) != null)
            {
              	String[] container;
            	container = line.split(" ");
            	length = Integer.parseInt(container[0]);
               	document d = collection.get(0);
               	collection.remove(0);
               	int d_nw=0;
        		for (n = 0; n < length; n++)
        		{
        			int pos = container[n+1].indexOf(':');
        			word  = Integer.valueOf(container[n+1].substring(0,pos));
        			count = Integer.valueOf(container[n+1].substring(pos+1));
        			word = word - OFFSET;
        			d.words[n] = word;
        			d.counts[n] = count;
        			d.docLength += count;
        			if (word >= nw) { nw = word + 1; }
        			if(word >= d_nw) {
        				d_nw = word+1;
        			}
        			if(d_nw > max_index)
           				max_index = nw;
        		}
        		d.max_index = max_index;
        		d.rating = Double.valueOf(container[n+1]);
        		d.phi = new double[d.docLength][LdaEstimate.NTOPICS];
        		d.m_eta = new double[LdaEstimate.NTOPICS];
        		d.m_sigma2 = new double[LdaEstimate.NTOPICS];
        		//d.aspectFreq = new int[LdaEstimate.NTOPICS][Analyzer.m_vocabulary.size()];
        		c.docs[nd]=d;
        		//d.z_dn= new int[d.docLength];
        		collection.add(d);
        		nd++;
            }
            bufferedReader.close();
            c.num_terms = nw;
            System.out.println("number of docs    : " + nd );
            System.out.println("number of terms   : "+ nw);
            System.out.println("Reading complete");
          }
        catch(IOException ex) {
            System.out.println("Error reading file '" + data_filename + "'");                  
        }
        return(c);
   }

    public static int max_corpus_length(corpus c)
    {
        int n, max = 0;
        document d = new document();
        for (n = 0; n < c.num_docs; n++) {
        	d = c.docs[n];
        	if (d.length > max) 
        		max = d.length;
        }
    	  
        return(max);
    } 
}
