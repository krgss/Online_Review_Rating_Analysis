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

#ifndef SLDA_DATA_H
#define SLDA_DATA_H

#include <stdio.h>
#include <stdlib.h>

#include "slda.h"

#define OFFSET 0;                  // offset for reading data

corpus* read_data(char* data_filename);
corpus* get_traindata(corpus* c, const int&nfold, const int &foldix);
corpus* get_testdata(corpus* c, const int&nfold, const int &foldix);
void equal_partion_data(corpus *c, corpus* tr_c, corpus *tst_c);
void reorder(corpus* c, char *filename);
void save_data(corpus *c, char *fileroot);

int max_corpus_length(corpus* c);

#endif
