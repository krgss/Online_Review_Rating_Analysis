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

#ifndef SLDA_H
#define SLDA_H

typedef struct
{
	double responseVar;	// the response variable value
	double testresponseVar;	// the response variable value
	double likelihood;
    int* words;
    int* counts;
    int length;
    int total;
} document;


typedef struct
{
    document* docs;
    int num_terms;
    int num_docs;
	int max_length;
} corpus;


typedef struct
{
    double alpha;
    double** log_prob_w;
    int num_topics;
    int num_terms;
	double *eta;		// \eta
	double deltasq;		// \delta^2
	double m_dTrainTime;

	// for fast computing.
	double *oldphi_;
	double *digamma_gam_;
	double *phisumarry_;
	double *phiNotN_;
	double *dig_;
	double *arry_;
} slda_model;


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

} slda_suffstats;

#endif
