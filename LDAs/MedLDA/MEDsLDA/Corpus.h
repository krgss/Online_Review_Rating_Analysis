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

#define OFFSET 0;                  // offset for reading data

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

class Corpus
{
public:
	Corpus(void);
	~Corpus(void);

	void read_data(char* data_filename);
	void read_train_data(char* data_filename);
	void read_test_data(char* data_filename);
	Corpus *get_traindata(const int&nfold, const int &foldix);
	Corpus *get_testdata(const int&nfold, const int &foldix);
	void reorder(char *filename);

	int max_corpus_length();
public:
	document* docs;
    int num_terms;
    int num_docs;
};
