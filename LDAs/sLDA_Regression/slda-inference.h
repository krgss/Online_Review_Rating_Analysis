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
#ifndef SLDA_INFERENCE_H
#define SLDA_INFERENCE_H

#include <math.h>
#include <float.h>
#include <assert.h>
#include "slda.h"
#include "utils.h"

double slda_inference(document*, slda_model*, double*, 
					 double**, double **, int VAR_MAX_ITER, float VAR_CONVERGED);
double slda_inference_prediction(document*, slda_model*, double*, 
					 double**, double **, int VAR_MAX_ITER, float VAR_CONVERGED);
double compute_likelihood(document*, slda_model*, double**, double **, double*, bool bTrain = true);
void amatrix(document* doc, slda_model* model, double** phi, double** a);
#endif
