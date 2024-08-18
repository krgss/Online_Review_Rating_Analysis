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

//#include "stdafx.h"

#include "slda-data.h"

void save_data(corpus *c, char *fileroot)
{
	char buff[512];
	sprintf(buff, "%s.dat", fileroot);
	FILE *fptr = fopen(buff, "w");
	sprintf(buff, "%s_normalized.dat", fileroot);
	FILE *fptr_norm = fopen(buff, "w");

	int num_docs = c->num_docs;
	for ( int d=0; d<num_docs; d++ ) {
		document doc = c->docs[d];

		fprintf(fptr, "%d %.5f", doc.length, doc.responseVar);
		fprintf(fptr_norm, "%d %.5f", doc.length, doc.responseVar);
		for ( int n=0; n<doc.length; n++ ) {
			fprintf(fptr, " %d:%d", doc.words[n], doc.counts[n]);
			fprintf(fptr_norm, " %d:%.5f", doc.words[n], (double)doc.counts[n]/(double)doc.total);
		}
		fprintf(fptr, "\n");
		fprintf(fptr_norm, "\n");
	}

	fclose(fptr);
	fclose(fptr_norm);
}

// equally divide the data into training set and testing set.
void equal_partion_data(corpus *c, corpus* tr_c, corpus *tst_c)
{
	int nunit = c->num_docs / 5; // 5 is the # ratings

	tr_c->docs = 0;
	tr_c->num_docs = 0;
	tr_c->num_terms = c->num_terms;

	tst_c->docs = 0;
	tst_c->num_docs = 0;
	tst_c->num_terms = c->num_terms;

	int tr_nd = 0, tst_nd = 0, nw = 0;
	for ( int k=0; k<5; k++ ) //
	{
		for ( int i=0; i<nunit; i++ ) 
		{
			document doc = c->docs[k*nunit + i];

			if ( i < nunit / 2 ) { // training
				tr_c->docs = (document*) realloc(tr_c->docs, sizeof(document)*(tr_nd+1));
				tr_c->docs[tr_nd].length = doc.length;
				tr_c->docs[tr_nd].total = doc.total;
				tr_c->docs[tr_nd].words = (int*)malloc(sizeof(int)*doc.length);
				tr_c->docs[tr_nd].counts = (int*)malloc(sizeof(int)*doc.length);

				// read the response variable
				tr_c->docs[tr_nd].responseVar = doc.responseVar;

				for (int n = 0; n < doc.length; n++) {
					tr_c->docs[tr_nd].words[n] = doc.words[n];
					tr_c->docs[tr_nd].counts[n] = doc.counts[n];
					if (doc.words[n] >= nw) { nw = c->docs[i].words[n] + 1; }
				}
				tr_nd++;
			} else { // testing
				tst_c->docs = (document*) realloc(tst_c->docs, sizeof(document)*(tst_nd+1));
				tst_c->docs[tst_nd].length = doc.length;
				tst_c->docs[tst_nd].total = doc.total;
				tst_c->docs[tst_nd].words = (int*)malloc(sizeof(int)*doc.length);
				tst_c->docs[tst_nd].counts = (int*)malloc(sizeof(int)*doc.length);

				// read the response variable
				tst_c->docs[tst_nd].responseVar = doc.responseVar;

				for (int n = 0; n < doc.length; n++) {
					tst_c->docs[tst_nd].words[n] = doc.words[n];
					tst_c->docs[tst_nd].counts[n] = doc.counts[n];
					if (doc.words[n] >= nw) { nw = doc.words[n] + 1; }
				}
				tst_nd++;
			}
		}
	}

	tr_c->num_docs  = tr_nd;
	tst_c->num_docs = tst_nd;
}

corpus* get_traindata(corpus* c, const int&nfold, const int &foldix)
{
	int nunit = c->num_docs / nfold;

	corpus *subc = (corpus*)malloc(sizeof(corpus));
	subc->docs = 0;
	subc->num_docs = 0;
	subc->num_terms = 0;
	int nd = 0, nw = 0;
	for ( int i=0; i<c->num_docs; i++ )
	{
		if ( foldix < nfold ) {
			if ( (i >= (foldix-1)*nunit) && ( i < foldix*nunit ) ) continue;
		} else {
			if ( i >= (foldix-1) * nunit ) continue;
		}

		subc->docs = (document*) realloc(subc->docs, sizeof(document)*(nd+1));
		subc->docs[nd].length = c->docs[i].length;
		subc->docs[nd].total = c->docs[i].total;
		subc->docs[nd].words = (int*)malloc(sizeof(int)*c->docs[i].length);
		subc->docs[nd].counts = (int*)malloc(sizeof(int)*c->docs[i].length);
		
		// read the response variable
		subc->docs[nd].responseVar = c->docs[i].responseVar;

		for (int n = 0; n < c->docs[i].length; n++)
		{
			subc->docs[nd].words[n] = c->docs[i].words[n];
			subc->docs[nd].counts[n] = c->docs[i].counts[n];
			if (c->docs[i].words[n] >= nw) { nw = c->docs[i].words[n] + 1; }
		}
		nd++;
	}
	subc->num_docs = nd;
	subc->num_terms = nw;
	return subc;
}

corpus* get_testdata(corpus* c, const int&nfold, const int &foldix)
{
	int nunit = c->num_docs / nfold;

	corpus *subc = (corpus*)malloc(sizeof(corpus));
	subc->docs = 0;
	subc->num_docs = 0;
	subc->num_terms = 0;
	int nd = 0, nw = 0;
	for ( int i=0; i<c->num_docs; i++ )
	{
		if ( foldix < nfold ) {
			if ( i < ((foldix-1)*nunit) || i >= foldix*nunit ) continue;
		} else {
			if ( i < (foldix-1) * nunit ) continue;
		}

		subc->docs = (document*) realloc(subc->docs, sizeof(document)*(nd+1));
		subc->docs[nd].length = c->docs[i].length;
		subc->docs[nd].total = c->docs[i].total;
		subc->docs[nd].words = (int*)malloc(sizeof(int)*c->docs[i].length);
		subc->docs[nd].counts = (int*)malloc(sizeof(int)*c->docs[i].length);
		
		// read the response variable
		subc->docs[nd].responseVar = c->docs[i].responseVar;

		for (int n = 0; n < c->docs[i].length; n++)
		{
			subc->docs[nd].words[n] = c->docs[i].words[n];
			subc->docs[nd].counts[n] = c->docs[i].counts[n];
			if (c->docs[i].words[n] >= nw) { nw = c->docs[i].words[n] + 1; }
		}
		nd++;
	}
	subc->num_docs = nd;
	subc->num_terms = nw;
	return subc;
}

void reorder(corpus* c, char *filename)
{
	int num, ix=0;
	int *order = (int*)malloc(sizeof(int)*c->num_docs);
	FILE *fileptr = fopen(filename, "r");
	while ( (fscanf(fileptr, "%10d", &num) != EOF ) )
	{
		order[ix] = num;
		ix ++;
	}
	
	document *docs = (document*)malloc(sizeof(document) * c->num_docs);
	for ( int i=0; i<c->num_docs; i++ )
		docs[i] = c->docs[i];
	for ( int i=0; i<c->num_docs; i++ )
		c->docs[i] = docs[order[i]];
	free(docs);
	free(order);
}

corpus* read_data(char* data_filename)
{
	FILE *fileptr;
	int length, count, word, n, nd, nw;
	corpus* c;

	printf("reading data from %s\n", data_filename);
	c = (corpus*)malloc(sizeof(corpus));
	c->docs = 0;
	c->num_terms = 0;
	c->num_docs = 0;
	c->max_length = 0;
	fileptr = fopen(data_filename, "r");
	nd = 0; nw = 0;
	while ((fscanf(fileptr, "%10d", &length) != EOF))
	{
		c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
		c->docs[nd].length = length;
		c->docs[nd].total = 0;
		c->docs[nd].words = (int*)malloc(sizeof(int)*length);
		c->docs[nd].counts = (int*)malloc(sizeof(int)*length);
		
		// read the response variable
		float responsVal;
		fscanf(fileptr, "%f", &responsVal);
		c->docs[nd].responseVar = responsVal;

		for (n = 0; n < length; n++)
		{
			fscanf(fileptr, "%10d:%10d", &word, &count);
			word = word - OFFSET;
			c->docs[nd].words[n] = word;
			c->docs[nd].counts[n] = count;
			c->docs[nd].total += count;
			if (word >= nw) { nw = word + 1; }
		}
		if ( c->max_length < length ) {
			c->max_length = length;
		}

		nd++;
	}
	fclose(fileptr);
	c->num_docs = nd;
	c->num_terms = nw;
	printf("number of docs    : %d\n", nd);
	printf("number of terms   : %d\n", nw);
	return(c);
}

int max_corpus_length(corpus* c)
{
	int n, max = 0;
	for (n = 0; n < c->num_docs; n++)
		if (c->docs[n].length > max) max = c->docs[n].length;
	return(max);
}
