// (C) Copyright 2009, Jun Zhu (junzhu [at] cs [dot] cmu [dot] edu)

// This file is part of sLDA, implemented based on the basic LDA code
// (LDA-C) by David M. Blei (blei [at] cs [dot] cmu [dot] edu).

// sLDA is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// sLDA is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

//#include "stdafx.h"
#include "slda-inference.h"
//#include <boost/math/special_functions/gamma.hpp>


/*
 * variational inference
 */
double slda_inference(document* doc, slda_model* model, double* var_gamma, double** phi, 
					 double **a, int VAR_MAX_ITER, float VAR_CONVERGED)
{
    double converged = 1;
    double phisum = 0, likelihood = 0;
    double likelihood_old = 0;
	double *oldphi = model->oldphi_;//(double*)malloc(sizeof(double)*model->num_topics);
    int k, n, var_iter;
    double *digamma_gam = model->digamma_gam_;//(double*)malloc(sizeof(double)*model->num_topics);

    // compute posterior dirichlet

	double *phisumarry = model->phisumarry_;//(double*)malloc( sizeof(double) * model->num_topics );
	double *phiNotN = model->phiNotN_;//(double*)malloc( sizeof(double) * model->num_topics );
    for (k = 0; k < model->num_topics; k++)
    {
        var_gamma[k] = model->alpha + (doc->total/((double) model->num_topics));
        digamma_gam[k] = digamma(var_gamma[k]);
		double dTmpVal = 0;
        for (n = 0; n < doc->length; n++) {
            phi[n][k] = 1.0/model->num_topics;
			dTmpVal += phi[n][k] * doc->counts[n];
		}
		phisumarry[k] = dTmpVal;
    }
    var_iter = 0;

	double *phiPtr = NULL;
	int nwrd, ncount;
    while ((converged > VAR_CONVERGED) && ((var_iter < VAR_MAX_ITER) || (VAR_MAX_ITER == -1)))
    {
		var_iter ++;
		for (n = 0; n < doc->length; n++)
		{
			phiPtr = phi[n];
			nwrd = doc->words[n];
			ncount = doc->counts[n];

			/* \eta^\top \phi_{-n} */
			for ( k = 0; k < model->num_topics; k++ ) 
				phiNotN[k] = phisumarry[k] - phiPtr[k]*ncount;
			double dProd = dotprod(model->eta, phiNotN, model->num_topics);
			double dNDeltaSq = model->deltasq * doc->total; /* N \delta^2 */

			phisum = 0; 
			for (k = 0; k < model->num_topics; k++)
			{
				oldphi[k] = phiPtr[k];
				
				/* update the phi: add additional terms here for supervised LDA */
				double dVal = ncount*model->eta[k]*(2*dProd + model->eta[k]*ncount) 
					/ (2*dNDeltaSq*doc->total);

				phiPtr[k] =	digamma_gam[k] + model->log_prob_w[k][nwrd]  // the following two terms for sLDA
					+ (model->eta[k]*doc->responseVar*ncount) / dNDeltaSq
					- dVal;

				if (k > 0) phisum = log_sum(phisum, phiPtr[k]);
				else       phisum = phiPtr[k]; // note, phi is in log space
			}

			// update gamma and normalize phi
			for (k = 0; k < model->num_topics; k++)
			{
				phiPtr[k] = exp(phiPtr[k] - phisum);
				var_gamma[k] = var_gamma[k] + ncount*(phiPtr[k] - oldphi[k]);
				// !!! a lot of extra digamma's here because of how we're computing it
				// !!! but its more automatically updated too.
				digamma_gam[k] = digamma(var_gamma[k]);

				phisumarry[k] = phiNotN[k] + phiPtr[k] * ncount;
			}
		}

		/* compute the E[zz^\top] matrix. */
		amatrix(doc, model, phi, a);

		likelihood = compute_likelihood(doc, model, phi, a, var_gamma);
		//assert(!isnan(likelihood));
		converged = (likelihood_old - likelihood) / likelihood_old;
		likelihood_old = likelihood;

		// printf("[LDA INF] %8.5f %1.3e\n", likelihood, converged);
    }

	//free(oldphi);
 //   free(digamma_gam);
	//free(phisumarry);
	//free(phiNotN);

    return(likelihood);
}


/* 
* Given the model and w, compute the E[Z] for prediction
*/
double slda_inference_prediction(document* doc, slda_model* model, double* var_gamma, double** phi, 
					 double **a, int VAR_MAX_ITER, float VAR_CONVERGED)
{
	int nK_ = model->num_topics;
	int nLength_ = doc->length;
    double converged = 1;
    double phisum = 0, likelihood = 0;
    double likelihood_old = 0;
	double *oldphi = model->oldphi_;//(double*)malloc(sizeof(double)*model->num_topics);
    int k, n, var_iter;
    double *digamma_gam = model->digamma_gam_;//(double*)malloc(sizeof(double)*model->num_topics);

    // compute posterior dirichlet

	//double *phisumarry = (double*)malloc( sizeof(double) * model->num_topics );
	//double *phiNotN = (double*)malloc( sizeof(double) * model->num_topics );
    for (k = 0; k < nK_; k++)
    {
        var_gamma[k] = model->alpha + (doc->total/((double) nK_));
        digamma_gam[k] = digamma(var_gamma[k]);
		//phisumarry[k] = 0;
        for (n = 0; n < nLength_; n++) {
            phi[n][k] = 1.0/nK_;
			//phisumarry[k] += phi[n][k] * doc->counts[n];
		}
    }
    var_iter = 0;

	double *phiPtr = NULL;
	int ncount, nwrd;
    while ((converged > VAR_CONVERGED) && ((var_iter < VAR_MAX_ITER) || (VAR_MAX_ITER == -1)))
    {
		var_iter ++;
		for (n = 0; n < nLength_; n++)
		{
			///* \eta^\top \phi_{-n} */
			//for ( k = 0; k < model->num_topics; k++ ) 
			//	phiNotN[k] = phisumarry[k] - phi[n][k]*doc->counts[n];
			//double dProd = dotprod(model->eta, phiNotN, model->num_topics);
			//double dNDeltaSq = model->deltasq * doc->total; /* N \delta^2 */

			phiPtr = phi[n];
			nwrd = doc->words[n];
			ncount = doc->counts[n];
			phisum = 0; 
			for (k = 0; k < nK_; k++)
			{
				oldphi[k] = phiPtr[k];
				
				///* update the phi: add additional terms here for supervised LDA */
				//double dVal = doc->counts[n]*model->eta[k]*(2*dProd + model->eta[k]*doc->counts[n]) 
				//	/ (2*dNDeltaSq*doc->total);

				phiPtr[k] =	digamma_gam[k] + model->log_prob_w[k][nwrd];
					//+ (model->eta[k]*doc->responseVar*doc->counts[n]) / dNDeltaSq
					//- dVal;

				if (k > 0) phisum = log_sum(phisum, phiPtr[k]);
				else       phisum = phiPtr[k]; // note, phi is in log space
			}

			// update gamma and normalize phi
			for (k = 0; k < nK_; k++)
			{
				phiPtr[k] = exp(phiPtr[k] - phisum);
				var_gamma[k] = var_gamma[k] + ncount*(phiPtr[k] - oldphi[k]);
				// !!! a lot of extra digamma's here because of how we're computing it
				// !!! but its more automatically updated too.
				digamma_gam[k] = digamma(var_gamma[k]);

				//phisumarry[k] = phiNotN[k] + phi[n][k] * doc->counts[n];
			}
		}

		likelihood = compute_likelihood(doc, model, phi, a, var_gamma, false);
		//assert(!isnan(likelihood));
		converged = (likelihood_old - likelihood) / likelihood_old;
		likelihood_old = likelihood;

		// printf("[LDA INF] %8.5f %1.3e\n", likelihood, converged);
    }

	//free(oldphi);
 //   free(digamma_gam);

    return(likelihood);
}

/*
 * compute likelihood bound
 */
double compute_likelihood(document* doc, slda_model* model, double** phi, 
				   double ** a,	double* var_gamma, bool bTrain /*= true*/)
{
	double likelihood = 0, digsum = 0, var_gamma_sum = 0;
	double *dig = model->dig_; //(double*)malloc(sizeof(double)*model->num_topics);
	int k, n;

	for (k = 0; k < model->num_topics; k++)
	{
		dig[k] = digamma(var_gamma[k]);
		var_gamma_sum += var_gamma[k];
	}
	digsum = digamma(var_gamma_sum);

	likelihood = lgamma(model->alpha * model -> num_topics)
		- model -> num_topics * lgamma(model->alpha)
		- (lgamma(var_gamma_sum));

	for (k = 0; k < model->num_topics; k++)
	{
		likelihood += (model->alpha - 1)*(dig[k] - digsum) + lgamma(var_gamma[k])
			- (var_gamma[k] - 1)*(dig[k] - digsum);

		double dVal = 0;
		for (n = 0; n < doc->length; n++)
		{
			if (phi[n][k] > 0) {
				likelihood += doc->counts[n] * (phi[n][k]*((dig[k] - digsum) - log(phi[n][k])
					+ model->log_prob_w[k][doc->words[n]]));
			}
			dVal += phi[n][k] * doc->counts[n] / doc->total;
		}

		/* for the response variables in sLDA */
		if ( bTrain )
			likelihood += (doc->responseVar * model->eta[k] * dVal) / model->deltasq;
	}

	/* for the response variables in sLDA */
	if ( bTrain ) 
	{
		likelihood -= 0.5 * log( model->deltasq * 2 * 3.14159265);
		likelihood -= (doc->responseVar * doc->responseVar) / ( 2 * model->deltasq );

		double *arry = model->arry_; //(double*)malloc(sizeof(double)*model->num_topics);
		matrixprod(model->eta, a, arry, model->num_topics);
		double dVal = dotprod(arry, model->eta, model->num_topics);

		likelihood -= dVal / ( 2 * model->deltasq );

		//free(arry);
	}
	//free(dig);

	return(likelihood);
}

/*
* compute the matrix E[zz^t]
*/
void amatrix(document* doc, slda_model* model, double** phi, double** a)
{
	for ( int k=0; k<model->num_topics; k++ ) {
		for ( int i=0; i<model->num_topics; i++ ) 
			a[k][i] = 0;
	}

	double dnorm = doc->total * doc->total;
	for ( int n=0; n<doc->length; n++ )
	{
		// diag{phi}
		for ( int k=0; k<model->num_topics; k++ ) {
			a[k][k] += (phi[n][k] * doc->counts[n] * doc->counts[n] ) / dnorm;
		}

		for ( int m=n+1; m<doc->length; m++ )
		{
			double dfactor = doc->counts[n] * doc->counts[m] / dnorm;
			addmatrix2(a, phi[n], phi[m], model->num_topics, dfactor);
		}
	}
}