import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.Vector;

//import javax.swing.text.Document;



public class Analyzer {
    public static Hashtable<String, Integer> m_vocabulary;//indexed vocabulary
    
    public static  ArrayList<Double> rating;
    public  static ArrayList<document>m_collection;
	public Analyzer(){
		m_vocabulary = new Hashtable<String, Integer>();
		 rating = new ArrayList<Double>();
		 m_collection = new ArrayList<document>();
		System.out.println("[Info]NLP modules initialized...");
	}
	
	public void LoadDirectory(String path, String suffix, String  filename){
		try{
			FileWriter writer = new FileWriter(filename);
			BufferedWriter buffer = new BufferedWriter(writer);
			buffer.write("");
			buffer.close();
		}catch(IOException ex) {
			ex.printStackTrace();
		}
		
		File dir = new File(path);
		//int size = m_hotelList.size();
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				LoadReviews(f.getAbsolutePath(), filename);
			else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix, filename);
		}
		System.out.println("Loading " +  " hotels from " + path);
	}
	
	public void LoadVocabulary(String filename){
		try {
			m_vocabulary = new Hashtable<String, Integer>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String tmpTxt;
			String[] container;
			while( (tmpTxt=reader.readLine()) != null ){
				container = tmpTxt.split("\t");
				m_vocabulary.put(container[0], m_vocabulary.size());
			}
			reader.close();
			System.out.println("[Info]Load " + m_vocabulary.size() + " control terms...");
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	//the review format is fixed
	public void LoadReviews(String filename, String data_filename) {//load reviews for annotation purpose
		
		      try {
				File f = new File(filename);
				BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(f), "UTF-8"));
				String tmpTxt,content = null;
				int review_size = 0;
				String[] tokens;
				double r=0;
				
				while((tmpTxt=reader.readLine()) != null) {
					if (tmpTxt.startsWith("<Overall>")){//only read those aspects
						try{
					    		r = Double.valueOf(tmpTxt.substring("<Overall>".length()));
					    		//buffer.write(r + " ");
					    		//d.rating = r;
					    		try {
					    			FileWriter file = new FileWriter(data_filename,true);
				    	 	    	BufferedWriter buffer = new BufferedWriter(file);
				    	 	    	buffer.write(Double.toString(r));
				    	 	    	buffer.newLine();
				    	 	    	buffer.close();
					    			
					    		}catch(IOException ex) {
					    			ex.printStackTrace();
					    		}
	                            rating.add(r);
						   }catch (Exception e){
								System.err.println("Error format: ");
								reader.close();
								return;
						}
			    	}
			    	else if (tmpTxt.startsWith("<Content>")){
						
			    		tmpTxt= tmpTxt.toLowerCase();
			    		tmpTxt = tmpTxt.replaceAll( "\\W", " " );     // strip out any non words.
			    		tmpTxt = tmpTxt.replaceAll( "  ", " " );      // strip out any double spaces                                                    //   in the first place!
			    	    tokens = tmpTxt.split( " " );
                        ArrayList< String > items = new ArrayList< String >();
                        ArrayList<String> vector =  new ArrayList<String>();
                        //System.out.println(vector.size());
			    	    items.addAll( Arrays.asList( tokens ) );
			    	    int count = 1;
			    	    int index;
			    	    String word;
			    	    int total=0;
			    	    document d = new document();
			    	    
			    	    /*for(int i=0;i<items.size();i++) {
			    	    	if(m_vocabulary.containsKey(items.get(i)))
			    	    		d.index_word.add(m_vocabulary.get(items.get(i)));
			    	    }*/
			    	    
			    	    for( int i = 0; i < items.size(); i++ )
			    	    {
			    	    	if(m_vocabulary.containsKey(items.get(i)))
			    	    	{
			    	    	    index = m_vocabulary.get(items.get(i));
					    	    
				    	        for( int j = i+1; j < items.size(); j++ )
				    	        {
				    	            if( items.get( i ).equals( items.get( j ) ) )
				    	                count++;
				    	            if( items.get( i ).equals( items.get( j ) ) && count > 1 ) {
				    	                items.remove( j );
				    	                j--;
				    	            }
				    	        }
				    	        word = Integer.toString(index) + ":" + Integer.toString(count);
				    	        vector.add(word);
				    	        count=1; 
			    	    	}
			    	    	else{
			    	    		items.remove(i);
			    	    		i--;
			    	    	}
			    		
					    }
			    	    int length = items.size();
			    	    d.length = length;
			    	    d.words= new int[length];
			    	    d.counts = new int[length];
			    	    d.docLength=0;
			    	    word = Integer.toString(length);
			    	    vector.add(0,word);
			    	    m_collection.add(d);
			    		 try{
			    	 	    	FileWriter file = new FileWriter(data_filename,true);
			    	 	    	BufferedWriter buffer = new BufferedWriter(file);
			    	            for(int i=0;i<vector.size();i++) {
		    	    		
		    	    		          buffer.write(vector.get(i));
		    	    		           buffer.write(" ");
		    	              }
          
                              buffer.close();
			    		 }catch(IOException ex)
			    		 {
			    			 ex.printStackTrace();
			    		 }
			    		
				   }
				}
				reader.close();
				}catch(IOException ex) {
				  ex.printStackTrace();
			}
	}
	
	public void  find() {
		System.out.println(m_collection.size());
		document d = m_collection.get(0);
		System.out.println(d.length);
	}
	
	
}
