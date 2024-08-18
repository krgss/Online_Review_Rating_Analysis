package utilities;

import java.io.File;
public class Utils {

	   public static double log_sum(double log_a, double log_b){
	      // given log(a) and log(b), return log(a+b)
	      
	      double v;
	      if(log_a < log_b){
	          v  = log_b + Math.log(1 + Math.exp(log_a - log_b));
	      }
	      
	      else{
	          v = log_a + Math.log(1 + Math.exp(log_b - log_a));
	      }
	      return v;
	   }       
	   
	   
	   /**
	   * Proc to calculate the value of the trigamma, the   second derivative of the loggamma function.
	   **/
	   
	   public static double trigamma(double x){
	      
	      double p;
	      int i;
	      x = x+6;
	      p = 1/(x*x);
	      p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
	         *p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
	      for (i=0; i<6 ;i++) {
	        x=x-1;
	        p=1/(x*x)+p;
	      }
	      return(p);
	   }
	   
	   // taylor approxiamtion of first derivative of the log gamma function
	   
	   public static double digamma (double x){
	   
	      double p;
	      x=x+6;
	      p=1/(x*x);
	      p=(((0.004166666666667*p-0.003968253986254)*p+ 0.008333333333333)*p-0.083333333333333)*p;
	      p=p+Math.log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
	      return p;
	   }
	   
	  
	    
	   public static double log_gamma(double x){
	     
	     double z=1/(x*x);
	     x=x+6;
	     z=(((-0.000595238095238*z+0.000793650793651)*z - 0.002777777777778)*z+0.083333333333333)/x;
	     z=(x-0.5)*Math.log(x)-x+0.918938533204673+z-Math.log(x-1)-Math.log(x-2)-Math.log(x-3)-Math.log(x-4)-Math.log(x-5)-Math.log(x-6);
	     return z;
	   }
	   
	  /*
	 * make directory
	 *
	  */

	  public static void make_directory(String name){
	     //mkdir(name, S_IRUSR|S_IWUSR|S_IXUSR);
	     new File("./"+ name).mkdirs();
	  }


	  /*
	 * argmax
	 *
	  */

	  public static int argmax(double[] x, int n){
	    int i;
	    double max = x[0];
	    int argmax = 0;
	    for (i = 1; i < n; i++)
	    {
	        if (x[i] > max)
	        {
	            max = x[i];
	            argmax = i;
	        }
	    }
	    return(argmax);
	  }   
}
