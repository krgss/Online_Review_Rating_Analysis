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
#pragma once
#include <math.h>
#include <float.h>
#include <assert.h>
#include "utils.h"
#include "Corpus.h"
#include <stdlib.h>
#include <stdio.h>
#include "opt-alpha.h"
#include "cokus.h"
#include "params.h"

#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
#define NUM_INIT 1

typedef struct
{
    double** class_word;
    double* class_total;
    double alpha_suffstats;
    int num_docs;

	// for supervised LDA, by Jun Zhu
	double **covarmatrix; // E[zz^\top]
	double *ezy;		  // y*E[z]
	double sqresponse;	  // \sum_{d=1}^D y_d^2, the sum of square response variables
	double **exp;		  // the array of E[z_d]
	double *y;			  // the vector of response value
	char dir[512];
} lda_suffstats;


class MedLDAReg
{
public:
	MedLDAReg(void);
	~MedLDAReg(void);

	double doc_e_step(document* doc, double* gamma, double** phi, double **a, lda_suffstats* ss);

	void save_gamma(char* filename, double** gamma, int num_docs, int num_topics);
	double save_prediction(char *filename, Corpus *corpus, double timeSpend);

	int run_em(char* start, char* directory, Corpus* corpus);

	void read_settings(char* filename);

	void infer(char* model_root, char* save, Corpus* corpus);


	// create & initialize model.
	void free_medlda( );
	void save_model(char*);
	void new_model(int, int, int, double, double);
	lda_suffstats* new_suffstats( );
	void corpus_initialize_ss( lda_suffstats* ss, Corpus* c);
	void random_initialize_ss( lda_suffstats* ss, Corpus* c);
	void zero_initialize_ss( lda_suffstats* ss );
	bool mle( lda_suffstats* ss, int estimate_alpha, int estimate_deltasq, bool bInit = true);
	void load_model(char* model_root);

	void svmLightSolver(lda_suffstats* ss, double **sigma, double **sigmaInverse, double *res);


	// inference.
	double inference(document*, double, double*, double**, double **);
	double inference_prediction(document*, double*, double**, double **);
	double compute_likelihood(document*, double**, double **, double*, bool bTrain = true);
	void amatrix(document* doc, double** phi, double** a);

	void write_word_assignment(FILE* f, document* doc, double** phi);

public:
    double alpha;
    double** log_prob_w;
    int num_topics;
    int num_terms;
	double *eta;		  // column vector E[\eta]
	double ** secmoment;  // E[\eta \eta']
	double deltasq;		  // \delta^2
	double *mudiff;		  // the difference \mu - \mu^\star
	int dim;			  // the dimension of mudiff
	double C;			  // the regularization constant
	double epsilon;		  // for epsilon-intensive regression
	int m_num_sv;
	double m_avg_sv;
	double m_dTrainTime;

	MedLDA_Params *params;

private:
	double *oldphi;
    double *digamma_gam;
	double *phisumarry;
	double *phiNotN;
	double *arry;
	double *dig;
};
