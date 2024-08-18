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

#ifndef MEDLDA_PARAMS_H
#define MEDLDA_PARAMS_H

#define MLE 0
#define SHRINK 1

class MedLDA_Params
{
public:
	MedLDA_Params() {
		train_filename = new char[1024];
		test_filename  = new char[1024];
	}
	~MedLDA_Params() {
		delete[] train_filename;
		delete[] test_filename;
	}

	void read_params(char*);
	void print_params();
	void default_params();

public:
	int VAR_MAX_ITER;
	float VAR_CONVERGED;
	int LAG;

	float EM_CONVERGED;
	int EM_MAX_ITER;
	int ESTIMATE_ALPHA;
	int ESTIMATE_DELTASQ;
	double INITIAL_ALPHA;
	float INITIAL_C;
	float INITIAL_EPSILON;
	int NTOPICS;
	int NFOLDS;
	int FOLDIX;

	char *train_filename;
	char *test_filename;
};

#endif
