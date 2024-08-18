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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "params.h"

void MedLDA_Params::read_params(char* filename)
{
    FILE* fileptr;
	char alpha_action[100];
	char delta_action[100];

    fileptr = fopen(filename, "r");
	fscanf(fileptr, "var max iter %d\n", &VAR_MAX_ITER);
	fscanf(fileptr, "var convergence %f\n", &VAR_CONVERGED);
	fscanf(fileptr, "em max iter %d\n", &EM_MAX_ITER);
	fscanf(fileptr, "em convergence %f\n", &EM_CONVERGED);
	fscanf(fileptr, "model C %f\n", &INITIAL_C);
	fscanf(fileptr, "model epsilon %f\n", &INITIAL_EPSILON);
	fscanf(fileptr, "alpha %s\n", alpha_action);
	fscanf(fileptr, "deltasq %s\n", delta_action);
	fscanf(fileptr, "lag %d\n", &LAG );
	if (strcmp(alpha_action, "fixed")==0) {
		ESTIMATE_ALPHA = 0;
	} else {
		ESTIMATE_ALPHA = 1;
	}

	if (strcmp(delta_action, "fixed")==0) {
		ESTIMATE_DELTASQ = 0;
	} else {
		ESTIMATE_DELTASQ = 1;
	}

	fscanf(fileptr, "train-data: %s\n", train_filename);
	fscanf(fileptr, "test-data: %s\n", test_filename);
}


void MedLDA_Params::print_params()
{
    printf("em max iter %d\n", EM_MAX_ITER);
    printf("var max iter %d\n", VAR_MAX_ITER);
    printf("em convergence %lf\n", EM_CONVERGED);
    printf("var convergence %lf\n", VAR_CONVERGED);
	printf("model C %f\n", INITIAL_C);
	printf("model epsilon %f\n", INITIAL_EPSILON);
	printf("alpha est? %d\n", ESTIMATE_ALPHA);
	printf("deltasq est? %d\n", ESTIMATE_DELTASQ);
	printf("lag %d\n", LAG);

	printf("train-data: %s\n", train_filename);
	printf("test-data: %s\n", test_filename);
}


void MedLDA_Params::default_params()
{
    EM_MAX_ITER = 1000;
    VAR_MAX_ITER = 500;
    EM_CONVERGED = 1e-3;
    VAR_CONVERGED = 1e-5;
    LAG = 5;
	INITIAL_C = 1;
	INITIAL_EPSILON = 0.01;
	ESTIMATE_ALPHA = 1;
	ESTIMATE_DELTASQ = 1;
}
