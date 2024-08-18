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
#ifndef SLDA_ESTIMATE_H
#define SLDA_ESTIMATE_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#include "slda.h"
#include "slda-data.h"
#include "slda-inference.h"
#include "slda-model.h"
#include "slda-alpha.h"
#include "utils.h"
#include "cokus.h"

int LAG = 5;

float EM_CONVERGED;
int EM_MAX_ITER;
int ESTIMATE_ALPHA;
int ESTIMATE_DELTASQ;
double INITIAL_ALPHA;
int NTOPICS;
int NFOLDS;
int FOLDIX;
char DATA_FILENAME[512];

double doc_e_step(document* doc,
                  double* gamma, double** phi, double **a,
				  slda_model* model, slda_suffstats* ss);

void save_gamma(char* filename,
                double** gamma,
                int num_docs,
                int num_topics);
double save_prediction(char *filename, corpus *corpus);


int run_em(char* start,
            char* directory,
            corpus* corpus);

void read_settings(char* filename);

void infer(char* model_root,
           char* save,
           corpus* corpus);
void partitionData(corpus *corpus, double** gamma, int ntopic);
void outputData(corpus *corpus, double** gamma, int ntopic);

#endif