public class LdaInference
{
	
	public static double VAR_CONVERGED;
	public static int VAR_MAX_ITER;

	static double lda_inference(document doc, lda_model model, double[] var_gamma, double[][] phi)
	{
		double converged = 1;
		double phisum = 0, likelihood = 0;
		double likelihood_old = 0;
		double oldphi[] = new double[model.num_topics];
		int k, n, var_iter;
		double digamma_gam[] = new double[model.num_topics];

		// compute posterior dirichlet

		for (k = 0; k < model.num_topics; k++)
		{
			var_gamma[k] = model.alpha + (doc.docLength/((double) model.num_topics));
			digamma_gam[k] = Utils.digamma(var_gamma[k]);
			for (n = 0; n < doc.length; n++)
				phi[n][k] = 1.0/model.num_topics;
		}
		var_iter = 0;

		while ((converged > VAR_CONVERGED) &&((var_iter < VAR_MAX_ITER) || (VAR_MAX_ITER == -1)))
		{
			var_iter++;
			for (n = 0; n < doc.length; n++)
			{
				phisum = 0;
				for (k = 0; k < model.num_topics; k++)
				{
					oldphi[k] = phi[n][k];
					phi[n][k] = digamma_gam[k] + model.log_prob_w[k][doc.words[n]];

					if (k > 0)
						phisum = Utils.log_sum(phisum, phi[n][k]);
					else
						phisum = phi[n][k]; // note, phi is in log space
				}

				for (k = 0; k < model.num_topics; k++)
				{
					phi[n][k] = Math.exp(phi[n][k] - phisum);
					var_gamma[k] = var_gamma[k] + doc.counts[n]*(phi[n][k] - oldphi[k]);
                // !!! a lot of extra digamma's here because of how we're computing it
                // !!! but its more automatically updated too.
					digamma_gam[k] = Utils.digamma(var_gamma[k]);
				}
			}

			likelihood = compute_likelihood(doc, model, phi, var_gamma);
			assert(! new Double(likelihood).isNaN());
			converged = (likelihood_old - likelihood) / likelihood_old;
			likelihood_old = likelihood;
		}
		return(likelihood);
	}


/*
 * compute likelihood bound
 *
 */

	static double compute_likelihood(document doc, lda_model model, double[][] phi, double[] var_gamma)
	{
		double likelihood = 0, digsum = 0, var_gamma_sum = 0;
		double dig[] = new double[model.num_topics];
		int k, n;

		for (k = 0; k < model.num_topics; k++)
		{
		dig[k] = Utils.digamma(var_gamma[k]);
		var_gamma_sum += var_gamma[k];
		}
		digsum = Utils.digamma(var_gamma_sum);

		likelihood = Utils.log_gamma(model.alpha * model.num_topics) - model.num_topics * Utils.log_gamma(model.alpha) - (Utils.log_gamma(var_gamma_sum));

		for (k = 0; k < model.num_topics; k++)
		{
			likelihood += (model.alpha - 1)*(dig[k] - digsum) + Utils.log_gamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum);

			for (n = 0; n < doc.length; n++)
			{
				if (phi[n][k] > 0)
				{
					likelihood += doc.counts[n]*(phi[n][k]*((dig[k] - digsum) - Math.log(phi[n][k]) + model.log_prob_w[k][doc.words[n]]));
				}
			}
		}
		return(likelihood);
	}

}