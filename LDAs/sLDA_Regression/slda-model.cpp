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
#include "slda-model.h"

/*
* compute MLE lda model from sufficient statistics
*/

bool slda_mle(slda_model* model, slda_suffstats* ss, int estimate_alpha, 
			 int estimate_deltasq, bool bInit /*= true*/)
{
	int k; int w;

	// \beta parameters (K x N)
	for (k = 0; k < model->num_topics; k++)
	{
		for (w = 0; w < model->num_terms; w++)
		{
			if (ss->class_word[k][w] > 0) {
				model->log_prob_w[k][w] = log(ss->class_word[k][w]) - log(ss->class_total[k]);
			} else {
				model->log_prob_w[k][w] = -100;
			}
		}
	}

	// \alpha parameters
	if (estimate_alpha == 1) {
		model->alpha = opt_alpha(ss->alpha_suffstats, ss->num_docs, model->num_topics);
		printf("new alpha = %5.5f\n", model->alpha);
	}

	bool bRes = true;
	if ( !bInit ) {
		// \eta for supervised LDA (Blei & McAuliffe, 2007)
		double **inversmatrix = (double**)malloc(sizeof(double*)*model->num_topics);
		for ( int i=0; i<model->num_topics; i++ )
			inversmatrix[i] = (double*)malloc(sizeof(double)*model->num_topics);

		bRes = inverse( ss->covarmatrix, inversmatrix, model->num_topics );



		matrixprod(inversmatrix, ss->ezy, model->eta, model->num_topics);

		// \delta^2 for supervised LDA (Blei & McAuliffe, 2007)
		if (estimate_deltasq == 1) {
			model->deltasq = (ss->sqresponse - dotprod(ss->ezy, model->eta, model->num_topics)) 
				/ ss->num_docs;
		}

		printf("\nDeltaSq: %f\n", model->deltasq);

		for (int i=0; i<model->num_topics; i++ )
			free(inversmatrix[i]);	
		free(inversmatrix);
	} 

	return bRes;
}

/*
* allocate sufficient statistics
*
*/

slda_suffstats* new_slda_suffstats(slda_model* model)
{
	int num_topics = model->num_topics;
	int num_terms = model->num_terms;
	int i,j;

	slda_suffstats* ss = (slda_suffstats*)malloc(sizeof(slda_suffstats));
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
	ss->covarmatrix = (double**)malloc(sizeof(double*)*model->num_topics);
	ss->ezy = (double*)malloc(sizeof(double) * model->num_topics);
	for ( int k=0; k<model->num_topics; k++ ) {
		ss->ezy[k] = 0;
		ss->covarmatrix[k] = (double*)malloc(sizeof(double)*model->num_topics);
		for ( i=0; i<model->num_topics; i++ )
			ss->covarmatrix[k][i] = 0;
	}
	ss->sqresponse = 0;

	return(ss);
}


/*
* various intializations for the sufficient statistics
*/

void zero_initialize_ss(slda_suffstats* ss, slda_model* model)
{
	int k, w;
	for (k = 0; k < model->num_topics; k++)
	{
		ss->class_total[k] = 0;
		for (w = 0; w < model->num_terms; w++) {
			ss->class_word[k][w] = 0;
		}
	}
	ss->num_docs = 0;
	ss->alpha_suffstats = 0;

	// for sLDA only
	for ( int k=0; k<model->num_topics; k++ ) {
		ss->ezy[k] = 0;
		for ( int i=0; i<model->num_topics; i++ )
			ss->covarmatrix[k][i] = 0;
	}
}


void random_initialize_ss(slda_suffstats* ss, slda_model* model, corpus* c)
{
	int num_topics = model->num_topics;
	int num_terms = model->num_terms;
	int k, n;
	for (k = 0; k < num_topics; k++)
	{
		for (n = 0; n < num_terms; n++)
		{
			ss->class_word[k][n] += 1.0/num_terms + myrand();
			ss->class_total[k] += ss->class_word[k][n];
		}
	}

	// for sLDA only
	for ( int k=0; k<model->num_topics; k++ ) {
		ss->ezy[k] = -1 + 2 * k / model->num_topics;
		for ( int i=0; i<model->num_topics; i++ )
			ss->covarmatrix[k][i] = 0;
	}
	for ( k=0; k<c->num_docs; k++ )
		ss->sqresponse += c->docs[k].responseVar * c->docs[k].responseVar;
}


void corpus_initialize_ss(slda_suffstats* ss, slda_model* model, corpus* c)
{
	int num_topics = model->num_topics;
	int i, k, d, n;
	document* doc;

	for (k = 0; k < num_topics; k++)
	{
		for (i = 0; i < NUM_INIT; i++)
		{
			d = floor(myrand() * c->num_docs);
			printf("initialized with document %d\n", d);
			doc = &(c->docs[d]);
			for (n = 0; n < doc->length; n++)
			{
				ss->class_word[k][doc->words[n]] += doc->counts[n];
			}
		}
		for (n = 0; n < model->num_terms; n++)
		{
			ss->class_word[k][n] += 1.0;
			ss->class_total[k] = ss->class_total[k] + ss->class_word[k][n];
		}
	}

	// for sLDA only
	for ( d=0; d<c->num_docs; d++ ) {
		ss->sqresponse += c->docs[d].responseVar * c->docs[d].responseVar;
	}

	for ( int k=0; k<model->num_topics; k++ ) {
		ss->ezy[k] = -1 + 2 * k / model->num_topics;
		for ( int i=0; i<model->num_topics; i++ )
			ss->covarmatrix[k][i] = 0;
	}
}

/*
* allocate new lda model
*/

slda_model* new_slda_model(int num_terms, int num_topics)
{
	int i,j;
	slda_model* model;

	model = (slda_model*)malloc(sizeof(slda_model));
	model->num_topics = num_topics;
	model->num_terms = num_terms;
	model->alpha = 1.0 / num_topics;
	model->log_prob_w = (double**)malloc(sizeof(double*)*num_topics);
	model->eta = (double*)malloc(sizeof(double)*num_topics);
	for (i = 0; i < num_topics; i++)
	{
		model->log_prob_w[i] = (double*)malloc(sizeof(double)*num_terms);
		model->eta[i] = 0;
		for (j = 0; j < num_terms; j++)
			model->log_prob_w[i][j] = 0;
	}

	model->oldphi_ = (double*)malloc(sizeof(double)*num_topics);
    model->digamma_gam_ = (double*)malloc(sizeof(double)*num_topics);
	model->phisumarry_ = (double*)malloc( sizeof(double) * num_topics );
	model->phiNotN_ = (double*)malloc( sizeof(double) * num_topics );
	model->dig_ = (double*)malloc( sizeof(double) * num_topics );
	model->arry_ = (double*)malloc( sizeof(double) * num_topics );

	model->deltasq = 1;

	return(model);
}


/*
* deallocate new lda model
*/

void free_slda_model(slda_model* model)
{
	int i;

	for (i = 0; i < model->num_topics; i++)
	{
		free(model->log_prob_w[i]);
	}
	free(model->log_prob_w);
	free(model->eta);
	free(model->oldphi_);
	free(model->phiNotN_);
	free(model->phisumarry_);
	free(model->digamma_gam_);
	free(model->dig_);
	free(model->arry_);
}


/*
* save an slda model
*/
void save_slda_model(slda_model* model, char* model_root)
{
	char filename[100];
	FILE* fileptr;
	int i, j;

	sprintf(filename, "%s.beta", model_root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "%5.10f\n", model->deltasq);

	for (i = 0; i < model->num_topics; i++)
	{
		// the first element is eta[k]
		fprintf(fileptr, "%5.10f", model->eta[i]);

		for (j = 0; j < model->num_terms; j++)
		{
			fprintf(fileptr, " %5.10f", model->log_prob_w[i][j]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	sprintf(filename, "%s.other", model_root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "num_topics %d\n", model->num_topics);
	fprintf(fileptr, "num_terms %d\n", model->num_terms);
	fprintf(fileptr, "alpha %5.10f\n", model->alpha);
	fprintf(fileptr, "train_time: %.4f\n", model->m_dTrainTime);
	fclose(fileptr);
}


slda_model* load_lda_model(char* model_root)
{
	char filename[100];
	FILE* fileptr;
	int i, j, num_terms, num_topics;
	float x, alpha;
	double dTrainTime;

	sprintf(filename, "%s.other", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "num_topics %d\n", &num_topics);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "alpha %f\n", &alpha);
	fscanf(fileptr, "train_time: %lf\n", &dTrainTime);
	fclose(fileptr);

	slda_model* model = new_slda_model(num_terms, num_topics);
	model->alpha = alpha;
	model->m_dTrainTime = dTrainTime;

	sprintf(filename, "%s.beta", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "%f", &x);
	model->deltasq = x;
	for (i = 0; i < num_topics; i++)
	{
		fscanf(fileptr, "%f", &x);
		model->eta[i] = x;
		for (j = 0; j < num_terms; j++)
		{
			fscanf(fileptr, "%f", &x);
			model->log_prob_w[i][j] = x;
		}
	}
	fclose(fileptr);
	return(model);
}
