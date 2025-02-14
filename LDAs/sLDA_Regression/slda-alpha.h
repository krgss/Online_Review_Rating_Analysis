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

#ifndef SLDA_ALPHA_H
#define SLDA_ALPHA_H

#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "slda.h"
#include "utils.h"

#define NEWTON_THRESH 1e-5
#define MAX_ALPHA_ITER 1000

double alhood(double a, double ss, int D, int K);
double d_alhood(double a, double ss, int D, int K);
double d2_alhood(double a, int D, int K);
double opt_alpha(double ss, int D, int K);
void maximize_alpha(double** gamma, slda_model* model, int num_docs);

#endif
