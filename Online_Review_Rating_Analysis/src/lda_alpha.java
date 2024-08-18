public class lda_alpha {
   
     public static final int MAX_ALPHA_ITER = 1000;
     public static final double NEWTON_THRESH = 1e-5;
  
    // objective function and its derivatives
 
     public static double alhood(double a, double ss, int D, int K){
    	  return(D * (Utils.log_gamma(K * a) - K * Utils.log_gamma(a)) + (a - 1) * ss); 
     }
     
     
     public static double d_alhood(double a, double ss, int D, int K){
          return(D * (K * Utils.digamma(K * a) - K * Utils.digamma(a)) + ss); 
      }

     public static double d2_alhood(double a, int D, int K){
          return(D * (K * K * Utils.trigamma(K * a) - K * Utils.trigamma(a))); 
     }

    // newtons method

    public static double opt_alpha(double ss, int D, int K){
       double a, log_a, init_a=100;
       double f, df, d2f;
       int iter=0;
    
       log_a = Math.log(init_a);
       do{
           iter++;
           a = Math.exp(log_a);
           if(new Double(a).isNaN()){
                init_a = init_a*10;
                System.out.println("[Info]: alpha is nan; new init =  " +  init_a);
                a=init_a;
                log_a = Math.log(a);
           }
               
           f = alhood(a,ss,D,K);
           df  =d_alhood(a,ss,D,K);
           d2f = d2_alhood(a,D,K);
           log_a = log_a - df/(d2f*a + df);
           System.out.println("Alpha maximization: " + f + "  " + df + "\n");
       }
       while ((Math.abs(df)> NEWTON_THRESH) && (iter < MAX_ALPHA_ITER));
       return Math.exp(log_a);
   } 
 
 }     

