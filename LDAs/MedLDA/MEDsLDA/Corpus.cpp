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
#include "StdAfx.h"
#include "Corpus.h"
#include <stdlib.h>
#include <stdio.h>

Corpus::Corpus(void)
{
}

Corpus::~Corpus(void)
{
}
Corpus* Corpus::get_traindata(const int&nfold, const int &foldix)
{
	int nunit = num_docs / nfold;

	Corpus *subc = (Corpus*)malloc(sizeof(Corpus));
	subc->docs = 0;
	subc->num_docs = 0;
	subc->num_terms = 0;
	int nd = 0, nw = 0;
	for ( int i=0; i<num_docs; i++ )
	{
		if ( foldix < nfold ) {
			if ( (i >= (foldix-1)*nunit) && ( i < foldix*nunit ) ) continue;
		} else {
			if ( i >= (foldix-1) * nunit ) continue;
		}

		subc->docs = (document*) realloc(subc->docs, sizeof(document)*(nd+1));
		subc->docs[nd].length = docs[i].length;
		subc->docs[nd].total = docs[i].total;
		subc->docs[nd].words = (int*)malloc(sizeof(int)*docs[i].length);
		subc->docs[nd].counts = (int*)malloc(sizeof(int)*docs[i].length);
		
		// read the response variable
		subc->docs[nd].responseVar = docs[i].responseVar;

		for (int n = 0; n < docs[i].length; n++)
		{
			subc->docs[nd].words[n] = docs[i].words[n];
			subc->docs[nd].counts[n] = docs[i].counts[n];
			if (docs[i].words[n] >= nw) { nw = docs[i].words[n] + 1; }
		}
		nd++;
	}
	subc->num_docs = nd;
	subc->num_terms = nw;
	return subc;
}

Corpus* Corpus::get_testdata(const int&nfold, const int &foldix)
{
	int nunit = num_docs / nfold;

	Corpus *subc = (Corpus*)malloc(sizeof(Corpus));
	subc->docs = 0;
	subc->num_docs = 0;
	subc->num_terms = 0;
	int nd = 0, nw = 0;
	for ( int i=0; i<num_docs; i++ )
	{
		if ( foldix < nfold ) {
			if ( i < ((foldix-1)*nunit) || i >= foldix*nunit ) continue;
		} else {
			if ( i < (foldix-1) * nunit ) continue;
		}

		subc->docs = (document*) realloc(subc->docs, sizeof(document)*(nd+1));
		subc->docs[nd].length = docs[i].length;
		subc->docs[nd].total = docs[i].total;
		subc->docs[nd].words = (int*)malloc(sizeof(int)*docs[i].length);
		subc->docs[nd].counts = (int*)malloc(sizeof(int)*docs[i].length);
		
		// read the response variable
		subc->docs[nd].responseVar = docs[i].responseVar;

		for (int n = 0; n < docs[i].length; n++)
		{
			subc->docs[nd].words[n] = docs[i].words[n];
			subc->docs[nd].counts[n] = docs[i].counts[n];
			if (docs[i].words[n] >= nw) { nw = docs[i].words[n] + 1; }
		}
		nd++;
	}
	subc->num_docs = nd;
	subc->num_terms = nw;
	return subc;
}

void Corpus::reorder(char *filename)
{
	int num, ix=0;
	int *order = (int*)malloc(sizeof(int)*num_docs);
	FILE *fileptr = fopen(filename, "r");
	while ( (fscanf(fileptr, "%10d", &num) != EOF ) )
	{
		order[ix] = num;
		ix ++;
	}
	
	document *docs_ = (document*)malloc(sizeof(document) * num_docs);
	for ( int i=0; i<num_docs; i++ )
		docs_[i] = docs[i];
	for ( int i=0; i<num_docs; i++ )
		docs[i] = docs_[order[i]];
	free(docs_);
	free(order);
}

void Corpus::read_data(char* data_filename)
{
	FILE *fileptr;
	int length, count, word, n, nd, nw;

	printf("reading data from %s\n", data_filename);
	docs = 0;
	num_terms = 0;
	num_docs = 0;
	fileptr = fopen(data_filename, "r");
	nd = 0; nw = 0;
	while ((fscanf(fileptr, "%10d", &length) != EOF))
	{
		docs = (document*) realloc(docs, sizeof(document)*(nd+1));
		docs[nd].length = length;
		docs[nd].total = 0;
		docs[nd].words = (int*)malloc(sizeof(int)*length);
		docs[nd].counts = (int*)malloc(sizeof(int)*length);
		
		// read the response variable
		float responsVal;
		fscanf(fileptr, "%f", &responsVal);
		docs[nd].responseVar = responsVal;

		for (n = 0; n < length; n++)
		{
			fscanf(fileptr, "%10d:%10d", &word, &count);
			word = word - OFFSET;
			docs[nd].words[n] = word;
			docs[nd].counts[n] = count;
			docs[nd].total += count;
			if (word >= nw) { nw = word + 1; }
		}
		nd++;
	}
	fclose(fileptr);
	num_docs = nd;
	num_terms = nw;
	printf("number of docs    : %d\n", nd);
	printf("number of terms   : %d\n", nw);
}

void Corpus::read_train_data(char* data_filename)
{
	// read all the data
	Corpus* c = new Corpus();
	c->read_data( data_filename );

	docs = 0;
	num_docs = 0;
	num_terms = 0;
	int nd = 0;
	// keep a half for training
	int nunit = c->num_docs / 5;
	for ( int k=0; k<5; k++ ) {
		for ( int i=0; i<nunit/2; i++ ) {
			int nix = k * nunit + i;

			docs = (document*) realloc(docs, sizeof(document)*(nd+1));
			docs[nd].length = c->docs[nix].length;
			docs[nd].total = c->docs[nix].total;
			docs[nd].words = (int*)malloc(sizeof(int)*c->docs[nix].length);
			docs[nd].counts = (int*)malloc(sizeof(int)*c->docs[nix].length);

			// read the response variable
			docs[nd].responseVar = c->docs[nix].responseVar;

			for (int n = 0; n < c->docs[nix].length; n++)
			{
				docs[nd].words[n] = c->docs[nix].words[n];
				docs[nd].counts[n] = c->docs[nix].counts[n];
			}
			nd++;
		}
	}
	num_docs = nd;
	num_terms = c->num_terms;
	free(c->docs);
	delete c;
}

void Corpus::read_test_data(char* data_filename)
{
	// read all the data
	Corpus* c = new Corpus();
	c->read_data( data_filename );

	docs = 0;
	num_docs = 0;
	num_terms = 0;
	int nd = 0;
	// keep a half for training
	int nunit = c->num_docs / 5;
	for ( int k=0; k<5; k++ ) {
		for ( int i=nunit/2; i<nunit; i++ ) {
			int nix = k * nunit + i;

			docs = (document*) realloc(docs, sizeof(document)*(nd+1));
			docs[nd].length = c->docs[nix].length;
			docs[nd].total = c->docs[nix].total;
			docs[nd].words = (int*)malloc(sizeof(int)*c->docs[nix].length);
			docs[nd].counts = (int*)malloc(sizeof(int)*c->docs[nix].length);

			// read the response variable
			docs[nd].responseVar = c->docs[nix].responseVar;

			for (int n = 0; n < c->docs[nix].length; n++)
			{
				docs[nd].words[n] = c->docs[nix].words[n];
				docs[nd].counts[n] = c->docs[nix].counts[n];
			}
			nd++;
		}
	}
	num_docs = nd;
	num_terms = c->num_terms;
	free(c->docs);
	delete c;
}

int Corpus::max_corpus_length( )
{
	int n, max = 0;
	for (n = 0; n <num_docs; n++)
		if (docs[n].length > max) max = docs[n].length;

	return(max);
}
