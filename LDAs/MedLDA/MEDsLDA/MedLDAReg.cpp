// (C) Copyright 2009, Jun Zhu (junzhu [at] cs [dot] cmu [dot] edu)

// This file is part of MedLDA for regression.

// MedLDA is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// MedLDA is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "stdafx.h"
#include <boost/math/special_functions/gamma.hpp>
#include "../SVMLight/svm_common.h"
#include "../SVMLight/svm_learn.h"
#include <vector>
#include <string>
#include "MedLDAReg.h"
using namespace std;

MedLDAReg::MedLDAReg(void)
{
}

MedLDAReg::~MedLDAReg(void)
{
}

/*
* perform inference on a document and update sufficient statistics
*/
double MedLDAReg::doc_e_step(document* doc, double* gamma, double** phi,
				  double **a, lda_suffstats* ss)
{
	double likelihood;
	int n, k;

	// posterior inference
	likelihood = inference(doc, mudiff[ss->num_docs], gamma, phi, a);

	// update sufficient statistics
	double *aPtr = NULL;
	double gamma_sum = 0;
	for (k = 0; k < num_topics; k++) {
		gamma_sum += gamma[k];
		ss->alpha_suffstats += digamma(gamma[k]);
	
		// suff-stats for supervised LDA
		aPtr = a[k];
		for ( n=0; n<num_topics; n++ )
			ss->covarmatrix[k][n] += aPtr[n];
	}
	ss->alpha_suffstats -= num_topics * digamma(gamma_sum);

	double *expPtr = ss->exp[ss->num_docs];
	for (k = 0; k < num_topics; k++) 
	{
		double dVal = 0;
		for (n = 0; n < doc->length; n++)
		{
			ss->class_word[k][doc->words[n]] += doc->counts[n]*phi[n][k];
			ss->class_total[k] += doc->counts[n]*phi[n][k];
			dVal += phi[n][k] * doc->counts[n] / doc->total;
		}
	
		// suff-stats for supervised LDA
		ss->ezy[k] += dVal * doc->responseVar;
		expPtr[k] = dVal;
	}
	ss->num_docs = ss->num_docs + 1;

	return(likelihood);
}


/*
* writes the word assignments line for a document to a file
*/

void MedLDAReg::write_word_assignment(FILE* f, document* doc, double** phi)
{
	int n;

	fprintf(f, "%03d", doc->length);
	for (n = 0; n < doc->length; n++)
	{
		fprintf(f, " %04d:%02d", doc->words[n], argmax(phi[n], num_topics));
	}
	fprintf(f, "\n");
	fflush(f);
}


/*
* saves the gamma parameters of the current dataset
*/

void MedLDAReg::save_gamma(char* filename, double** gamma, int num_docs, int num_topics)
{
	FILE* fileptr;
	int d, k;
	fileptr = fopen(filename, "w");

	for (d = 0; d < num_docs; d++)
	{
		fprintf(fileptr, "%5.10f", gamma[d][0]);
		for (k = 1; k < num_topics; k++)
		{
			fprintf(fileptr, " %5.10f", gamma[d][k]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);
}


/*
* save the prediction results and the predictive R^2 value
*/
double MedLDAReg::save_prediction(char *filename, Corpus *corpus, double timeSpend)
{
	double dmean = 0;
	double sumlikelihood = 0;
	int nterms = 0;
	double sumavglikelihood = 0;
	for ( int d=0; d<corpus->num_docs; d++ ) {
		dmean += corpus->docs[d].responseVar / corpus->num_docs;
		sumlikelihood += corpus->docs[d].likelihood;
		nterms += corpus->docs[d].total;
		sumavglikelihood += corpus->docs[d].likelihood / corpus->docs[d].total;
	}
	double perwordlikelihood1 = sumlikelihood / nterms;
	double perwordlikelihood2 = sumavglikelihood / corpus->num_docs;

	double ssd = 0;
	for ( int d=0; d<corpus->num_docs; d++ ) 
		ssd += (corpus->docs[d].responseVar - dmean ) * (corpus->docs[d].responseVar - dmean);

	double press = 0;
	for ( int d=0; d<corpus->num_docs; d++ )
		press += ( corpus->docs[d].responseVar - corpus->docs[d].testresponseVar )
				* ( corpus->docs[d].responseVar - corpus->docs[d].testresponseVar );

	double predictiver2 = 1.0 - press / ssd;

	FILE* fileptr;
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "predictive R^2: %5.10f\n", predictiver2 );
	fprintf(fileptr, "perword likelihood1: %5.10f\n", perwordlikelihood1);
	fprintf(fileptr, "perword likelihood2: %5.10f\n", perwordlikelihood2);

	for (int d=0; d<corpus->num_docs; d++)
	{
		fprintf(fileptr, "%5.10f\t%5.10f\n", corpus->docs[d].testresponseVar, corpus->docs[d].responseVar);
	}
	fclose(fileptr);

	return predictiver2;
}

/*
* run_em
*/
int MedLDAReg::run_em(char* start, char* directory, Corpus* corpus)
{
	int d, n;
	double **var_gamma, **phi;

	// allocate variational parameters

	var_gamma = (double**)malloc(sizeof(double*)*(corpus->num_docs));
	for (d = 0; d < corpus->num_docs; d++)
		var_gamma[d] = (double*)malloc(sizeof(double) * params->NTOPICS);

	int max_length = corpus->max_corpus_length( );
	phi = (double**)malloc(sizeof(double*)*max_length);
	for (n = 0; n < max_length; n++)
		phi[n] = (double*)malloc(sizeof(double) * params->NTOPICS);

	double **a = (double**)malloc(sizeof(double*) * params->NTOPICS);
	for ( int k=0; k<params->NTOPICS; k++ )
		a[k] = (double*)malloc(sizeof(double) * params->NTOPICS);

	// initialize model
	lda_suffstats* ss = NULL;
	if (strcmp(start, "seeded")==0)
	{
		new_model(corpus->num_docs, corpus->num_terms, params->NTOPICS,
								params->INITIAL_C, params->INITIAL_EPSILON);
		ss = new_suffstats( );
		corpus_initialize_ss(ss, corpus);
		mle(ss, 0, 0);
		alpha = params->INITIAL_ALPHA / params->NTOPICS;
	}
	else if (strcmp(start, "random")==0)
	{
		new_model(corpus->num_docs, corpus->num_terms, params->NTOPICS,
								params->INITIAL_C, params->INITIAL_EPSILON);
		ss = new_suffstats( );
		random_initialize_ss(ss, corpus);
		mle(ss, 0, 0);
		alpha = params->INITIAL_ALPHA / params->NTOPICS;
	}
	else
	{
		load_model(start);
		ss = new_suffstats( );

		ss->y = (double*) malloc(sizeof(double) * corpus->num_docs);
		ss->exp = (double**) malloc(sizeof(double*) * corpus->num_docs);
		for (int k=0; k<corpus->num_docs; k++ ) {
			ss->sqresponse += corpus->docs[k].responseVar * corpus->docs[k].responseVar;
			ss->y[k] = corpus->docs[k].responseVar;
			ss->exp[k] = (double*) malloc(sizeof(double)*params->NTOPICS);
		}
	}
	strcpy(ss->dir, directory);


	// set the \delta^2 to be the variance of response variables
	double dmean = 0;
	for ( d=0; d<corpus->num_docs; d++ )
		dmean += corpus->docs[d].responseVar / corpus->num_docs;
	deltasq = 0;
	for ( d=0; d<corpus->num_docs; d++ )
		deltasq += (corpus->docs[d].responseVar - dmean) * (corpus->docs[d].responseVar - dmean)
		/ corpus->num_docs;



	char filename[100];
	//sprintf(filename, "%s\\000",directory);
	//save_lda_model(model, filename);

	// run expectation maximization
	sprintf(filename, "%s/likelihood.dat", directory);
	FILE* likelihood_file = fopen(filename, "w");

	double likelihood, likelihood_old = 0, converged = 1;
	int nIt = 0;
	m_avg_sv = 0;
	double startTime = get_runtime();
	while (((converged < 0) || (converged > params->EM_CONVERGED) 
		|| (nIt <= 2)) && (nIt <= params->EM_MAX_ITER))
	{
		printf("**** em iteration %d ****\n", nIt + 1);
		likelihood = 0;
		zero_initialize_ss(ss);

		// e-step
		for (d = 0; d < corpus->num_docs; d++)
		{
			// initialize to uniform
			for (n = 0; n < max_length; n++)
				for ( int k=0; k<params->NTOPICS; k++ )
					phi[n][k] = 1.0 / (double) params->NTOPICS;

			if ((d % 1000) == 0) printf("document %d\n",d);
			likelihood += doc_e_step( &(corpus->docs[d]), var_gamma[d], phi, a, ss);
		}

		// m-step
		mle(ss, params->ESTIMATE_ALPHA, params->ESTIMATE_DELTASQ, false);
		m_avg_sv += m_num_sv;

		// check for convergence
		for ( int i=0; i<num_topics; i++ )
			likelihood += 0.5 * eta[i] * eta[i];

		converged = (likelihood_old - likelihood) / (likelihood_old);
		if (converged < 0) params->VAR_MAX_ITER = params->VAR_MAX_ITER * 2;
		likelihood_old = likelihood;

		// output model and likelihood

		fprintf(likelihood_file, "%10.10f\t%5.5e\n", likelihood, converged);
		fflush(likelihood_file);
		//if ((nIt % LAG) == 0)
		//{
		//	sprintf(filename,"%s\\%d",directory, nIt + 1);
		//	save_lda_model(model, filename);
		//	sprintf(filename,"%s\\%d.gamma",directory, nIt + 1);
		//	save_gamma(filename, var_gamma, corpus->num_docs, num_topics);
		//}
		nIt ++;
	}
	m_avg_sv /= nIt;
	double endTime = get_runtime();
	m_dTrainTime = (endTime-startTime)/100;
	fprintf(likelihood_file, "train-time: %.3f\n", m_dTrainTime);

	// output the final model
	sprintf(filename,"%s/final",directory);
	save_model(filename);
	sprintf(filename,"%s/final.gamma",directory);
	save_gamma(filename, var_gamma, corpus->num_docs, num_topics);

	// output the word assignments (for visualization)
	sprintf(filename, "%s/word-assignments.dat", directory);
	FILE* w_asgn_file = fopen(filename, "w");
	for (d = 0; d < corpus->num_docs; d++)
	{
		if ((d % 100) == 0) printf("final e step document %d\n",d);
		likelihood += inference(&(corpus->docs[d]), mudiff[d], 
			var_gamma[d], phi, a);
		write_word_assignment(w_asgn_file, &(corpus->docs[d]), phi);
	}
	fclose(w_asgn_file);
	fclose(likelihood_file);


	for (d = 0; d < corpus->num_docs; d++)
		free(var_gamma[d]);
	free(var_gamma);
	for (n = 0; n < max_length; n++)
		free(phi[n]);
	free(phi);
	for ( int k=0; k<num_topics; k++ )
		free(a[k]);
	free(a);

	return nIt;
}

/*
* inference only
*/
void MedLDAReg::infer(char* model_root, char* save, Corpus* corpus)
{
	FILE* fileptr;
	char filename[100];
	int i, d, n;
	double **var_gamma, likelihood;
	document* doc;

	load_model(model_root);
	var_gamma = (double**)malloc(sizeof(double*)*(corpus->num_docs));
	for (i = 0; i < corpus->num_docs; i++)
		var_gamma[i] = (double*)malloc(sizeof(double)*num_topics);
	
	double **a = (double**)malloc(sizeof(double*)*num_topics);
	for ( int k=0; k<num_topics; k++ )
		a[k] = (double*)malloc(sizeof(double)*num_topics);

	double **phi = (double**)malloc(sizeof(double*) * corpus->max_corpus_length());
	for (n = 0; n < corpus->max_corpus_length(); n++) {
		phi[n] = (double*) malloc(sizeof(double) * num_topics);
	}

	sprintf(filename, "%s/evl-lhood.dat", save);
	fileptr = fopen(filename, "w");
	double startTime = get_runtime();
	for (d = 0; d < corpus->num_docs; d++)
	{
		if (((d % 100) == 0) && (d>0)) printf("document %d\n",d);

		doc = &(corpus->docs[d]);
		for (n = 0; n < doc->length; n++) {
			for ( int k=0; k<num_topics; k++ )
				phi[n][k] = 1.0 / (double)num_topics;
		}

		likelihood = inference_prediction(doc, var_gamma[d], phi, a);

		// do prediction
		doc->testresponseVar = 0;
		for ( int k=0; k<num_topics; k++ ) {
			double dVal = 0;
			for ( int n=0; n<doc->length; n++ )
				dVal += phi[n][k] * doc->counts[n] / doc->total;
			doc->testresponseVar += dVal * eta[k];
		}
		doc->likelihood = likelihood;

		fprintf(fileptr, "%5.5f\n", likelihood);
	}
	fclose(fileptr);
	double endTime = get_runtime();
	double timeSpend = (endTime-startTime)/100;

	sprintf(filename, "%s/evl-gamma.dat", save);
	save_gamma(filename, var_gamma, corpus->num_docs, num_topics);

	// save the prediction performance
	sprintf(filename, "%s/evl-performance.dat", save);
	double predictiver2 = save_prediction(filename, corpus, timeSpend);

	fileptr = fopen("overall_res.txt", "a");
	fprintf(fileptr, "K: %d; C: %.3f; predR2: %.3f; test-time: %.3f; train-time: %.3f; final-sv: %d; avg-sv: %.3f\n", 
		num_topics, C, predictiver2, timeSpend, m_dTrainTime,
		m_num_sv, m_avg_sv);
	fclose( fileptr );

	// free memory
	for ( int k=0; k<num_topics; k++ )
		free( a[k] );
	free(a);
	for ( i=0; i<corpus->num_docs; i++ )
		free( var_gamma[i] );
	free( var_gamma );
	for (n = 0; n < corpus->max_corpus_length(); n++) {
		free( phi[n] );
	}
	free( phi );
}

/*
* compute MLE lda model from sufficient statistics
*/

bool MedLDAReg::mle(lda_suffstats* ss, int estimate_alpha, 
			 int estimate_deltasq, bool bInit /*= true*/)
{
	int k; int w;

	// \beta parameters (K x N)
	for (k=0; k<num_topics; k++)
	{
		for (w=0; w<num_terms; w++) {
			if (ss->class_word[k][w] > 0) {
				log_prob_w[k][w] = log(ss->class_word[k][w]) - log(ss->class_total[k]);
			} else {
				log_prob_w[k][w] = -100;
			}
		}
	}

	// \alpha parameters
	if (estimate_alpha == 1) {
		alpha = opt_alpha(ss->alpha_suffstats, ss->num_docs, num_topics);
		printf("new alpha = %5.5f\n", alpha);
	}

	bool bRes = true;
	if ( !bInit ) {
		// inverse to get the \Sigma matrix
		double **A = (double**)malloc(sizeof(double*)*num_topics);
		for ( int k=0; k<num_topics; k++ ) {
			A[k] = (double*)malloc(sizeof(double)*num_topics);
			for ( int j=0; j<num_topics; j++ ) {
				A[k][j] = ss->covarmatrix[k][j];
			}
			A[k][k] += deltasq;
		}

		//printf("\n\n");
		//printmatrix(A, num_topics);
		inverse( A, secmoment, num_topics);

		for ( int k=0; k<num_topics; k++ ) {
			for ( int j=0; j<num_topics; j++ ) {
				secmoment[k][j] *= deltasq;
			}
		}
		for ( int k=0; k<num_topics; k++ ) free(A[k]);
		free(A);


		/* solve the QP dual problem. Up to now, the secmoment is the \Sigma matrix. */
		double *mu = (double*)malloc(sizeof(double)*ss->num_docs*2);
		FILE *fileptr;

		svmLightSolver(ss, secmoment, NULL, mu);
		fileptr = fopen("SVMLight.txt", "a");
		for ( int i=0; i<ss->num_docs; i++ )
			fprintf(fileptr, "%5.10f  %5.10f\n", mu[i], mu[i+ss->num_docs]);
		fprintf(fileptr, "\n\n\n\n");
		fclose(fileptr);

		for ( int i=0; i<ss->num_docs; i++ ) 
			mudiff[i] = mu[i] - mu[i+ss->num_docs];
		free(mu);


		/* \eta for supervised LDA (Blei & McAuliffe, 2007). */
		double *res = (double*)malloc(sizeof(double)*num_topics);
		for ( int k=0; k<num_topics; k++ ) {
			res[k] = ss->ezy[k] / deltasq;
			for ( int i=0; i<ss->num_docs; i++ ) {
				res[k] +=  mudiff[i] * ss->exp[i][k];
			}
		}
		matrixprod(secmoment, res, eta, num_topics);

		// the previous secmoment is the covariance; add the term 
		addmatrix(secmoment, eta, eta, num_topics, 1.0);

		/* \delta^2 for supervised LDA (Blei & McAuliffe, 2007) */
		if (estimate_deltasq == 1) {
			double dVal = 0;
			for ( int k=0; k<num_topics; k++ ) {
				for ( int j=0; j<num_topics; j++ ) {
					dVal += secmoment[k][j] * ss->covarmatrix[k][j];
				}
			}

			//matrixprod(ss->covarmatrix, eta, res, num_topics);
			deltasq = (ss->sqresponse - 2*dotprod(ss->ezy, eta, num_topics)
				+ dVal/*dotprod(eta, res, num_topics)*/) / ss->num_docs;
		}

		printf("\nDeltaSq: %f\n", deltasq);

		free(res);
	} 

	return bRes;
}

void set_init_param(LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm)
{
	strcpy (learn_parm->alphafile, "");
	strcpy (learn_parm->predfile, "trans_predictions");
	//(*verbosity)=1;
	verbosity = 1;

	/* set default */
	learn_parm->biased_hyperplane = 0;
	learn_parm->sharedslack = 0;
	learn_parm->remove_inconsistent = 0;
	learn_parm->skip_final_opt_check = 0;
	learn_parm->svm_maxqpsize = 10;
	learn_parm->svm_newvarsinqp = 0;
	learn_parm->svm_iter_to_shrink = -9999;
	learn_parm->maxiter = 100000;
	learn_parm->kernel_cache_size=40;
	learn_parm->svm_c=0.0;
	learn_parm->eps=0.1;
	learn_parm->transduction_posratio=-1.0;
	learn_parm->svm_costratio=1.0;
	learn_parm->svm_costratio_unlab=1.0;
	learn_parm->svm_unlabbound=1E-5;
	learn_parm->epsilon_crit=0.001;
	learn_parm->epsilon_a=1E-15;
	learn_parm->compute_loo=0;
	learn_parm->rho = 1.0;
	learn_parm->xa_depth = 0;

	kernel_parm->kernel_type = 0;
	kernel_parm->poly_degree = 3;
	kernel_parm->rbf_gamma = 1.0;
	kernel_parm->coef_lin = 1;
	kernel_parm->coef_const = 1;
	strcpy(kernel_parm->custom,"empty");


	if(learn_parm->svm_iter_to_shrink == -9999) {
		if(kernel_parm->kernel_type == LINEAR) 
			learn_parm->svm_iter_to_shrink=2;
		else
			learn_parm->svm_iter_to_shrink=100;
	}

	// by default
	learn_parm->type = REGRESSION;

	if((learn_parm->skip_final_opt_check) 
		&& (kernel_parm->kernel_type == LINEAR)) {
			printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
			learn_parm->skip_final_opt_check=0;
	}  
	if((learn_parm->skip_final_opt_check) && (learn_parm->remove_inconsistent)) {
			printf("\nIt is necessary to do the final optimality check when removing inconsistent \nexamples.\n");
			exit(0);
	}    
	if((learn_parm->svm_maxqpsize<2)) {
		printf("\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n",learn_parm->svm_maxqpsize); 
		exit(0);
	}
	if((learn_parm->svm_maxqpsize<learn_parm->svm_newvarsinqp)) {
		printf("\nMaximum size of QP-subproblems [%ld] must be larger than the number of\n",learn_parm->svm_maxqpsize); 
		printf("new variables [%ld] entering the working set in each iteration.\n",learn_parm->svm_newvarsinqp); 
		exit(0);
	}
	if(learn_parm->svm_iter_to_shrink<1) {
		printf("\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n",learn_parm->svm_iter_to_shrink);
		exit(0);
	}
	if(learn_parm->svm_c<0) {
		printf("\nThe C parameter must be greater than zero!\n\n");
		exit(0);
	}
	if(learn_parm->transduction_posratio>1) {
		printf("\nThe fraction of unlabeled examples to classify as positives must\n");
		printf("be less than 1.0 !!!\n\n");
		exit(0);
	}
	if(learn_parm->svm_costratio<=0) {
		printf("\nThe COSTRATIO parameter must be greater than zero!\n\n");
		exit(0);
	}
	if(learn_parm->epsilon_crit<=0) {
		printf("\nThe epsilon parameter must be greater than zero!\n\n");
		exit(0);
	}
	if(learn_parm->rho<0) {
		printf("\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n");
		printf("be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n");
		printf("Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n");
		exit(0);
	}
	if((learn_parm->xa_depth<0) || (learn_parm->xa_depth>100)) {
		printf("\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n");
		printf("for switching to the conventional xa/estimates described in T. Joachims,\n");
		printf("Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n");
		exit(0);
	}
}

// need transformation: cannot use the standard form
void MedLDAReg::svmLightSolver(lda_suffstats* ss, 
					double **sigma, double **sigmaInverse, double *res)
{
	KERNEL_CACHE *kernel_cache = NULL;
	LEARN_PARM learn_parm;
	KERNEL_PARM kernel_parm;
	MODEL *svm = (MODEL *)my_malloc(sizeof(MODEL));
	double *alpha_in=NULL;

	DOC **docs;
	double *target;

	// do transformation
	vector<double> alpha(num_topics, 0);
	for ( int k=0; k<num_topics; k++ ) {
		alpha[k] = 0;
		for ( int d=0; d<ss->num_docs; d++ ) {
			alpha[k] += ss->y[d] * ss->exp[d][k];
		}
		alpha[k] /= deltasq;
	}

	vector<double> beta(num_topics, 0);
	for ( int k=0; k<num_topics; k++ ) {
		beta[k] = 0;
		for ( int i=0; i<num_topics; i++ )
			beta[k] += sigma[k][i] * alpha[i];
	}
	
	// Cholesky-decomposition of sigmaInverse
	double **U = (double **)malloc(sizeof(double*) * num_topics);
	//double **Uinverse = (double **)malloc(sizeof(double*) * num_topics);
	for ( int k=0; k<num_topics; k++ ) {
		U[k] = (double*)malloc(sizeof(double) * num_topics);
		//Uinverse[k] = (double*)malloc(sizeof(double) * num_topics);
	}
	choleskydec(sigma, U, num_topics, true);

	//inverse(U, Uinverse, num_topics);

	// output the tranformed data into an intermediate file
	char buff[512];
	sprintf(buff, "%s/Feature.txt", ss->dir);
	FILE *fileptr = fopen(buff, "w");
	for ( int d=0; d<ss->num_docs; d++ )
	{
		/* response value. */
		double response = ss->y[d];
		for ( int k=0; k<num_topics; k++ )
			response -= beta[k] * ss->exp[d][k];
		fprintf(fileptr, "%d %.10f", num_topics, response);

		/* expected topic proportion. */
		for ( int k=0; k<num_topics; k++ ) {
			double feature = 0;
			for ( int i=0; i<num_topics; i++ )
				feature += U[k][i] * ss->exp[d][i];

			fprintf(fileptr, " %d:%.10f", k, feature/*ss->exp[d][k]*/);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);
	
	/* learn SVM regression based on the transformed data. */
	long totwords, totdocs;
	read_documents(buff, &docs, &target, &totwords, &totdocs);

	set_init_param(&learn_parm, &kernel_parm);

	learn_parm.svm_c = C;
	learn_parm.svm_c_factor = 1.0;
	learn_parm.eps = epsilon;

	svm_learn_regression(docs, target, ss->num_docs, totwords, &learn_parm,
		&kernel_parm, &kernel_cache, svm);

	/* get the dual solution. no inverse transformation is needed
	   because the dual problem doesn't change in the reformulation. */
	for ( int k=0; k<ss->num_docs*2; k++ ) res[k] = 0;
	for ( int k=1; k<svm->sv_num; k++ ) {
		int docnum = svm->supvec[k]->docnum;
		if ( svm->primal[k] == 1) res[docnum] = svm->alpha[k];
		else res[docnum + ss->num_docs] = 0 - svm->alpha[k];
	}
	//b = svm->b;
	m_num_sv = svm->sv_num - 1;

	free_model(svm, 0);
	for(int i=0; i<ss->num_docs; i++) free_example(docs[i], 1);
	free(docs);
	free(target);
	for ( int k=0; k<num_topics; k++ ) {
		free(U[k]);
	}
	free(U);
}

/*
* allocate sufficient statistics
*/
lda_suffstats* MedLDAReg::new_suffstats( )
{
	int i,j;

	lda_suffstats* ss = (lda_suffstats*)malloc(sizeof(lda_suffstats));
	ss->class_total = (double*)malloc(sizeof(double)*num_topics);
	ss->class_word = (double**)malloc(sizeof(double*)*num_topics);
	for (i = 0; i < num_topics; i++)
	{
		ss->class_total[i] = 0;
		ss->class_word[i] = (double*)malloc(sizeof(double)*num_terms);
		for (j = 0; j < num_terms; j++) {
			ss->class_word[i][j] = 0;
		}
	}

	// for sLDA only
	ss->covarmatrix = (double**)malloc(sizeof(double*)*num_topics);
	ss->ezy = (double*)malloc(sizeof(double) * num_topics);
	for ( int k=0; k<num_topics; k++ ) {
		ss->ezy[k] = 0;
		ss->covarmatrix[k] = (double*)malloc(sizeof(double)*num_topics);
		for ( i=0; i<num_topics; i++ )
			ss->covarmatrix[k][i] = 0;
	}
	ss->sqresponse = 0;

	return(ss);
}


/*
* various intializations for the sufficient statistics
*/
void MedLDAReg::zero_initialize_ss(lda_suffstats* ss)
{
	int k, w;
	for (k=0; k<num_topics; k++) {
		ss->class_total[k] = 0;
		for (w=0; w<num_terms; w++) {
			ss->class_word[k][w] = 0;
		}
	}
	ss->num_docs = 0;
	ss->alpha_suffstats = 0;

	// for sLDA only
	for ( int k=0; k<num_topics; k++ ) {
		ss->ezy[k] = 0;
		for ( int i=0; i<num_topics; i++ )
			ss->covarmatrix[k][i] = 0;
	}
}


void MedLDAReg::random_initialize_ss(lda_suffstats* ss, Corpus* c)
{
	int k, n;
	for (k = 0; k < num_topics; k++)
	{
		for (n = 0; n < num_terms; n++)
		{
			ss->class_word[k][n] += /*1.0/num_terms*/1 + myrand(); // uniform!!
			ss->class_total[k] += ss->class_word[k][n];
		}
	}

	// for sLDA only
	for ( int k=0; k<num_topics; k++ ) {
		ss->ezy[k] = -1 + 2 * k / num_topics;
		for ( int i=0; i<num_topics; i++ )
			ss->covarmatrix[k][i] = 0;
	}

	ss->y = (double*) malloc(sizeof(double) * c->num_docs);
	ss->exp = (double**) malloc(sizeof(double*) * c->num_docs);
	for ( k=0; k<c->num_docs; k++ )
	{
		ss->sqresponse += c->docs[k].responseVar * c->docs[k].responseVar;
		ss->y[k] = c->docs[k].responseVar;
		ss->exp[k] = (double*) malloc(sizeof(double)*num_topics);
	}
}


void MedLDAReg::corpus_initialize_ss(lda_suffstats* ss, Corpus* c)
{
	int i, k, d, n;
	document* doc;

	for (k = 0; k < num_topics; k++)
	{
		for (i = 0; i < NUM_INIT; i++)
		{
			d = floor(myrand() * c->num_docs);
			printf("initialized with document %d\n", d);
			doc = &(c->docs[d]);
			for (n = 0; n < doc->length; n++) {
				ss->class_word[k][doc->words[n]] += doc->counts[n];
			}
		}
		for (n=0; n<num_terms; n++) {
			ss->class_word[k][n] += 1.0;
			ss->class_total[k] = ss->class_total[k] + ss->class_word[k][n];
		}
	}

	// for sLDA only
	ss->y = (double*) malloc(sizeof(double) * c->num_docs);
	ss->exp = (double**) malloc(sizeof(double*) * c->num_docs);
	for ( d=0; d<c->num_docs; d++ ) {
		ss->sqresponse += c->docs[d].responseVar * c->docs[d].responseVar;
		ss->y[d] = c->docs[d].responseVar;
		ss->exp[d] = (double*) malloc(sizeof(double)*num_topics);
	}

	for ( int k=0; k<num_topics; k++ ) {
		ss->ezy[k] = -1 + 2 * k / num_topics;
		for ( int i=0; i<num_topics; i++ )
			ss->covarmatrix[k][i] = 0;
	}
}

/*
* allocate new model
*/
void MedLDAReg::new_model(int num_docs, int terms, int topics,
						 double dC, double dEpsilon)
{
	int i,j;
	num_topics = topics;
	num_terms = terms;
	alpha = 1.0 / num_topics;
	log_prob_w = (double**)malloc(sizeof(double*)*num_topics);
	eta = (double*)malloc(sizeof(double)*num_topics);
	secmoment = (double**)malloc(sizeof(double*)*num_topics);
	for (i = 0; i < num_topics; i++) {
		secmoment[i] = (double*)malloc(sizeof(double)*num_topics);
		log_prob_w[i] = (double*)malloc(sizeof(double)*num_terms);
		eta[i] = 0;
		for (j = 0; j < num_terms; j++)
			log_prob_w[i][j] = 0;

		// initialize the covariance matrix to be identity
		for (j=0; j<num_topics; j++ ) {
			secmoment[i][j] = 0;
			if ( i == j ) secmoment[i][i] = 1;
		}
	}

	deltasq = 1;

	dim = num_docs;
	mudiff = (double*) malloc(sizeof(double)*dim);
	for ( int i=0; i<dim; i++ ) mudiff[i] = 0;
	epsilon = dEpsilon;
	C = dC;


	oldphi = (double*)malloc(sizeof(double)*num_topics);
    digamma_gam = (double*)malloc(sizeof(double)*num_topics);
	phisumarry = (double*)malloc( sizeof(double) * num_topics );
	phiNotN = (double*)malloc( sizeof(double) * num_topics );
	arry = (double*)malloc(sizeof(double)*num_topics);
	dig = (double*)malloc(sizeof(double)*num_topics);
}

/*
* deallocate new medlda model
*/
void MedLDAReg::free_medlda()
{
	int i;

	for (i=0; i<num_topics; i++) {
		free(log_prob_w[i]);
		free(secmoment[i]);
	}
	free(log_prob_w);
	free(secmoment);
	free(eta);

	free( oldphi );
    free( digamma_gam );
	free( phisumarry );
	free( phiNotN );
	free( arry );
	free( dig );
}

/*
* save an lda model
*/
void MedLDAReg::save_model(char* model_root)
{
	char filename[100];
	FILE* fileptr;
	int i, j;

	sprintf(filename, "%s.beta", model_root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "%5.10f\n", deltasq);

	// for covariance matrix
	for (i = 0; i<num_topics; i++) {
		for (j = 0; j<num_topics; j++)
			fprintf(fileptr, "%5.10f ", secmoment[i][j]);
		fprintf(fileptr, "\n");
	}

	for (i = 0; i<num_topics; i++) {
		// the first element is eta[k]
		fprintf(fileptr, "%5.10f", eta[i]);

		for (j=0; j<num_terms; j++) {
			fprintf(fileptr, " %5.10f", log_prob_w[i][j]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	sprintf(filename, "%s.other", model_root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "num_topics %d\n", num_topics);
	fprintf(fileptr, "num_terms %d\n", num_terms);
	fprintf(fileptr, "num_docs %d\n", dim);
	fprintf(fileptr, "alpha %5.10f\n", alpha);
	fprintf(fileptr, "C %5.10f\n", C);
	fprintf(fileptr, "epsilon %5.10f\n", epsilon);
	fprintf(fileptr, "final-svs %d\n", m_num_sv);
	fprintf(fileptr, "avg-svs %.3f\n", m_avg_sv);
	fprintf(fileptr, "train-time %.3f\n", m_dTrainTime);
	fclose(fileptr);
}

void MedLDAReg::load_model(char* model_root)
{
	char filename[100];
	FILE* fileptr;
	int i, j, num_docs;
	float x;

	sprintf(filename, "%s.other", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "num_topics %d\n", &num_topics);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "num_docs %d\n", &num_docs);
	fscanf(fileptr, "alpha %lf\n", &alpha);
	fscanf(fileptr, "C %lf\n", &C);
	fscanf(fileptr, "epsilon %lf\n", &epsilon);
	fscanf(fileptr, "final-svs %d\n", &m_num_sv);
	fscanf(fileptr, "avg-svs %lf\n", &m_avg_sv);
	fscanf(fileptr, "train-time %lf\n", &m_dTrainTime);
	fclose(fileptr);

	new_model(num_docs, num_terms, num_topics, C, epsilon);

	sprintf(filename, "%s.beta", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "%f", &x);
	deltasq = x;

	// for covariance matrix
	for (i = 0; i < num_topics; i++) {
		for (j = 0; j < num_topics; j++) {
			fscanf(fileptr, "%f", &x);
			secmoment[i][j] = x;
		}
	}

	for (i = 0; i < num_topics; i++) {
		fscanf(fileptr, "%f", &x);
		eta[i] = x;
		for (j = 0; j < num_terms; j++) {
			fscanf(fileptr, "%f", &x);
			log_prob_w[i][j] = x;
		}
	}
	fclose(fileptr);
}



/*
 * variational inference
 */
double MedLDAReg::inference(document* doc, double mudiff,
					 double* var_gamma, double** phi, double **a)
{
    double converged = 1;
    double phisum = 0, likelihood = 0;
    double likelihood_old = 0;
    int k, n, var_iter;

	// compute posterior dirichlet
    for (k = 0; k < num_topics; k++) {
        var_gamma[k] = alpha + (doc->total/((double) num_topics));
        digamma_gam[k] = digamma(var_gamma[k]);
		phisumarry[k] = 0;
        for (n = 0; n < doc->length; n++) {
            phi[n][k] = 1.0/num_topics;
			phisumarry[k] += phi[n][k] * doc->counts[n];
		}
    }
    var_iter = 0;


	double *phiPtr = NULL;
	int ncount = 0;
    while ((converged > params->VAR_CONVERGED) && ((var_iter < params->VAR_MAX_ITER) 
		|| (params->VAR_MAX_ITER == -1)))
    {
		var_iter ++;
		for (n = 0; n < doc->length; n++)
		{
			phiPtr = phi[n];
			ncount = doc->counts[n];

			/* \eta^\top \phi_{-n} */
			for ( k = 0; k < num_topics; k++ ) 
				phiNotN[k] = phisumarry[k] - phiPtr[k]*ncount;
			//double dProd = dotprod(eta, phiNotN, num_topics);
			double dNDeltaSq = deltasq * doc->total; /* N \delta^2 */

			phisum = 0; 
			for (k = 0; k < num_topics; k++)
			{
				oldphi[k] = phiPtr[k];
				
				/* update the phi: add additional terms here for supervised LDA */
				double dVal = 0;
				for ( int i=0; i<num_topics; i++ )
					dVal += phiNotN[i] * secmoment[i][k];
				dVal = dVal * 2 * ncount;

				dVal += ncount * ncount * secmoment[k][k];
				dVal = dVal / (2 * dNDeltaSq * doc->total);

				// update the phi_{d,n,k}
				phiPtr[k] =	digamma_gam[k] + log_prob_w[k][doc->words[n]]  // the following two terms for sLDA
					+ (eta[k]*doc->responseVar*ncount) / dNDeltaSq
					- dVal
					+ (eta[k]*ncount*mudiff) / doc->total;

				if (k > 0) phisum = log_sum(phisum, phiPtr[k]);
				else       phisum = phiPtr[k]; // note, phi is in log space
			}

			// update gamma and normalize phi
			for (k = 0; k < num_topics; k++)
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
		amatrix(doc, phi, a);

		likelihood = compute_likelihood(doc, phi, a, var_gamma);
		//assert(!isnan(likelihood));
		converged = (likelihood_old - likelihood) / likelihood_old;
		likelihood_old = likelihood;

		// printf("[LDA INF] %8.5f %1.3e\n", likelihood, converged);
    }

    return(likelihood);
}


/* 
* Given the model and w, compute the E[Z] for prediction
*/
double MedLDAReg::inference_prediction(document* doc, double* var_gamma, double** phi, double **a)
{
    double converged = 1;
    double phisum = 0, likelihood = 0;
    double likelihood_old = 0;
    int k, n, var_iter;

    // compute posterior dirichlet
    for (k = 0; k < num_topics; k++)
    {
        var_gamma[k] = alpha + (doc->total/((double)num_topics));
        digamma_gam[k] = digamma(var_gamma[k]);
        for (n = 0; n < doc->length; n++) {
            phi[n][k] = 1.0/num_topics;
		}
    }

	var_iter = 0;
    while ((converged > params->VAR_CONVERGED) && ((var_iter < params->VAR_MAX_ITER) 
		|| (params->VAR_MAX_ITER == -1)))
    {
		var_iter ++;
		for (n = 0; n < doc->length; n++)
		{
			phisum = 0; 
			for (k = 0; k < num_topics; k++)
			{
				oldphi[k] = phi[n][k];
				phi[n][k] =	digamma_gam[k] + log_prob_w[k][doc->words[n]];

				if (k > 0) phisum = log_sum(phisum, phi[n][k]);
				else       phisum = phi[n][k]; // note, phi is in log space
			}

			// update gamma and normalize phi
			for (k = 0; k < num_topics; k++) {
				phi[n][k] = exp(phi[n][k] - phisum);
				var_gamma[k] = var_gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]);
				// !!! a lot of extra digamma's here because of how we're computing it
				// !!! but its more automatically updated too.
				digamma_gam[k] = digamma(var_gamma[k]);
			}
		}

		likelihood = compute_likelihood(doc, phi, a, var_gamma, false);
		converged = (likelihood_old - likelihood) / likelihood_old;
		likelihood_old = likelihood;
	}

    return(likelihood);
}

/*
 * compute likelihood bound
 */
double MedLDAReg::compute_likelihood(document* doc, double** phi, 
				   double ** a,	double* var_gamma, bool bTrain /*= true*/)
{
	double likelihood = 0, digsum = 0, var_gamma_sum = 0;
	int k, n;

	for (k = 0; k<num_topics; k++) {
		dig[k] = digamma(var_gamma[k]);
		var_gamma_sum += var_gamma[k];
	}
	digsum = digamma(var_gamma_sum);

	likelihood = boost::math::lgamma(alpha * num_topics)
		- num_topics * boost::math::lgamma(alpha)
		- (boost::math::lgamma(var_gamma_sum));

	for (k=0; k<num_topics; k++)
	{
		likelihood += (alpha - 1)*(dig[k] - digsum) + boost::math::lgamma(var_gamma[k])
			- (var_gamma[k] - 1)*(dig[k] - digsum);

		double dVal = 0;
		for (n = 0; n < doc->length; n++)
		{
			if (phi[n][k] > 0) {
				likelihood += doc->counts[n] * (phi[n][k]*((dig[k] - digsum) - log(phi[n][k])
					+ log_prob_w[k][doc->words[n]]));
			}
			dVal += phi[n][k] * doc->counts[n] / doc->total;
		}

		/* for the response variables in sLDA */
		if ( bTrain )
			likelihood += (doc->responseVar * eta[k] * dVal) / deltasq;
	}

	/* for the response variables in sLDA */
	if ( bTrain ) 
	{
		likelihood -= 0.5 * log( deltasq * 2 * 3.14159265);
		likelihood -= (doc->responseVar * doc->responseVar) / ( 2 * deltasq );

		matrixprod(eta, a, arry, num_topics);
		double dVal = dotprod(arry, eta, num_topics);

		likelihood -= dVal / ( 2 * deltasq );
	}

	return(likelihood);
}

/*
* compute the matrix E[zz^t]
*/
void MedLDAReg::amatrix(document* doc, double** phi, double** a)
{
	for ( int k=0; k<num_topics; k++ ) {
		for ( int i=0; i<num_topics; i++ ) 
			a[k][i] = 0;
	}

	double *phiPtr = NULL;
	double dnorm = doc->total * doc->total;
	for ( int n=0; n<doc->length; n++ )
	{
		phiPtr = phi[n];
		int ncount = doc->counts[n];
		// diag{phi}
		for ( int k=0; k<num_topics; k++ ) {
			a[k][k] += (phiPtr[k] * ncount * ncount ) / dnorm;
		}

		for ( int m=n+1; m<doc->length; m++ ) {
			double dfactor = ncount * doc->counts[m] / dnorm;
			addmatrix2(a, phi[n], phi[m], num_topics, dfactor);
		}
	}
}