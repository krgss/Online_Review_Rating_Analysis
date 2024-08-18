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

#ifndef SLDA_MODEL_H
#define SLDA_MODEL

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "slda.h"
#include "slda-alpha.h"
#include "cokus.h"

#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
#define NUM_INIT 1

void free_slda_model(slda_model*);
void save_slda_model(slda_model*, char*);
slda_model* new_slda_model(int, int);
slda_suffstats* new_slda_suffstats(slda_model* model);
void corpus_initialize_ss(slda_suffstats* ss, slda_model* model, corpus* c);
void random_initialize_ss(slda_suffstats* ss, slda_model* model, corpus* c);
void zero_initialize_ss(slda_suffstats* ss, slda_model* model);
bool slda_mle(slda_model* model, slda_suffstats* ss, int estimate_alpha, 
			 int estimate_deltasq, bool bInit = true);
slda_model* load_lda_model(char* model_root);

#endif
