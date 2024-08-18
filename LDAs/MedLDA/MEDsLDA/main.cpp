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
#include "windows.h"
#include "io.h"
#include <time.h>
#include "MedLDAReg.h"
#include <vector>
#include <string>
using namespace std;

int main(int argc, char* argv[])
{

	//long t1;
	//(void) time(&t1);
	seedMT( time(NULL) );
	// seedMT(4357U);

	if (argc > 1)
	{
		Corpus* c = new Corpus();
		MedLDA_Params *params = new MedLDA_Params();
		if (strcmp(argv[1], "estinf")==0)
		{
			params->INITIAL_ALPHA = atof(argv[2]);
			params->NTOPICS = atoi(argv[3]);
			int nfold = atoi(argv[4]);
			params->read_params( argv[5] );
			if ( argc > 6 ) params->INITIAL_C = atof(argv[6]);

			Corpus *tr_c = new Corpus();
			tr_c->read_train_data( params->train_filename );
			Corpus *tst_c = new Corpus();
			tst_c->read_test_data( params->train_filename );

			//make_directory(argv[7]);
			char dir[512];
			sprintf(dir, "res%d_%d", params->NTOPICS, nfold);
			::CreateDirectoryA( dir, NULL );
			MedLDAReg med;
			med.params = params;
			med.run_em("random", dir, tr_c);

			char model_root[128];
			sprintf(model_root, "%s/final", dir);
			MedLDAReg evlMed;
			evlMed.params = params;
			evlMed.infer(model_root, dir, tst_c);

			free(tr_c->docs);
			free(tst_c->docs);
		}
		if (strcmp(argv[1], "est")==0)
		{
			params->read_params( "settings.txt" );
			params->INITIAL_ALPHA = atof(argv[2]);
			params->NTOPICS = atoi(argv[3]);
			c->read_data( params->train_filename );

			//make_directory(argv[7]);
			::CreateDirectoryA( argv[5], NULL );
			MedLDAReg med;
			med.params = params;
			med.run_em(argv[4], argv[5], c);
			free(c->docs);
		}
		if (strcmp(argv[1], "inf")==0)
		{
			params->read_params( "settings.txt" );
			//read_settings(argv[2]);
			c->read_test_data( params->train_filename );
			MedLDAReg med;
			med.params = params;
			med.infer(argv[2], argv[3], c);
			free(c->docs);
		}

		if ( strcmp(argv[1], "cv") == 0 ) // cross-validation (run one fold at a time)
		{
			params->NFOLDS = atoi(argv[2]);
			params->FOLDIX = atoi(argv[3]);

			params->INITIAL_ALPHA = atof(argv[4]);
			params->NTOPICS = atoi(argv[5]);
			params->read_params( "settings.txt" );
			c->read_data(argv[6]);
			c->reorder("randomorder.txt");

			Corpus *traincorpus = c->get_traindata(params->NFOLDS, params->FOLDIX);
			Corpus *testcorpus = c->get_testdata(params->NFOLDS, params->FOLDIX);

			// estimate on training data (return the number of iterations)
			char resDir[512];
			sprintf(resDir, "%dtopic_cv%d(%d)", params->NTOPICS, params->NFOLDS, params->FOLDIX);
			::CreateDirectoryA(resDir, NULL);
			if ( GetLastError() == ERROR_ALREADY_EXISTS ) {
				::RemoveDirectoryA(resDir);
			}
			::CreateDirectoryA(resDir, NULL);
			
			MedLDAReg med;
			med.params = params;
			int nIt = med.run_em(argv[8], resDir, traincorpus);

			// predict on test corpus
			char chrRes[512];
			sprintf(chrRes, "%s/evlRes%d(%d)_it%d.txt", resDir, params->NFOLDS, params->FOLDIX, nIt);
			char chrModel[512];
			sprintf(chrModel, "%s/final", resDir);
			MedLDAReg evlMed;
			evlMed.params = params;
			evlMed.infer(chrModel, chrRes, testcorpus);

			free(traincorpus->docs);
			free(testcorpus->docs);
			free(c->docs);
		}

		if ( strcmp(argv[1], "vis") == 0 ) // visulization of the top words in each topic
		{
			MedLDAReg med;
			med.params = params;
			med.load_model(argv[2]);
			FILE *fileptr = fopen("E:\\OnlineReviewExtract\\data\\rev2500_6000_voc12000_alpha.dat", "r");
			char buff[512];
			vector<string> vecStr;
			int i = -1;
			while ( (fscanf(fileptr, "%s", buff) != EOF) ) {
				//if ( i == -1 ) {
				//	i = 0;
					fscanf(fileptr, "%s", buff); /*fscanf(fileptr, "%s", buff);
					fscanf(fileptr, "%s", buff); fscanf(fileptr, "%s", buff);*/
				//} else {
					vecStr.push_back( string( buff ) );
				//	fscanf(fileptr, "%s", buff);
				//	fscanf(fileptr, "%s", buff);
				//	fscanf(fileptr, "%s", buff);
				//	fscanf(fileptr, "%s", buff);
				//}
			}

			// find top 50 words
			fileptr = fopen("TopWords.txt", "w");
			vector<vector<string> > vec_topwrd( med.num_topics );
			for ( i = 0; i<med.num_topics; i++ )
			{
				for ( int k=0; k<150; k++ ) {
					int maxix = -1;
					double maxval = -100;
					for ( int j=0; j<med.num_terms; j++ )
					{
						if ( med.log_prob_w[i][j] > maxval ) {
							maxix = j;
							maxval = med.log_prob_w[i][j];
						}
					}
					vec_topwrd[i].push_back( vecStr[maxix] );
					med.log_prob_w[i][maxix] = -100;
				}
			}

			for ( int k=0; k<50; k++ ) {
				for ( int i=0; i<med.num_topics; i++ ) {
					fprintf(fileptr, "\& %s ", vec_topwrd[i][k].c_str() );
				}
				fprintf(fileptr, "\n");
			}
			fclose(fileptr);
		}
		delete c;
	}
	else
	{
		printf("usage : medlda est [initial alpha] [k] [random/seeded/*] [directory]\n");
		printf("        medlda cv [foldnum] [foldix] [initial alpha] [k] [settings] [data] [random/seeded/*] [directory]\n");
		printf("        medlda inf [model] [name]\n");
	}
	return(0);
}