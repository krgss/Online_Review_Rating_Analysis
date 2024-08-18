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

#include "slda-estimate.h"
//#include "io.h"
#include <fstream>
#include <vector>
#include <stdio.h>
using namespace std;

int VAR_MAX_ITER;
float VAR_CONVERGED;
//char* DATA_FILENAME;
/*
* perform inference on a document and update sufficient statistics
*
*/

double doc_e_step(document* doc, double* gamma, double** phi,
				  double **a, slda_model* model, slda_suffstats* ss)
{
	double likelihood;
	int n, k;

	// posterior inference

	likelihood = slda_inference(doc, model, gamma, phi, a, VAR_MAX_ITER, VAR_CONVERGED);

	// update sufficient statistics

	double gamma_sum = 0, *covPtr=NULL;
	for (k = 0; k < model->num_topics; k++) {
		gamma_sum += gamma[k];
		ss->alpha_suffstats += digamma(gamma[k]);
	
		// suff-stats for supervised LDA
		covPtr = ss->covarmatrix[k];
		for ( n=0; n<model->num_topics; n++ )
			covPtr[n] += a[k][n];
	}
	ss->alpha_suffstats -= model->num_topics * digamma(gamma_sum);

	for (k = 0; k < model->num_topics; k++) 
	{
		double dVal = 0;
		double *wrdPtr = ss->class_word[k];
		for (n = 0; n < doc->length; n++) {
			wrdPtr[doc->words[n]] += doc->counts[n]*phi[n][k];
			ss->class_total[k] += doc->counts[n]*phi[n][k];
			dVal += phi[n][k] * doc->counts[n] / doc->total;
		}
	
		// suff-stats for supervised LDA
		ss->ezy[k] += dVal * doc->responseVar;
	}
	ss->num_docs = ss->num_docs + 1;

	return(likelihood);
}


/*
* writes the word assignments line for a document to a file
*
*/

void write_word_assignment(FILE* f, document* doc, double** phi, slda_model* model)
{
	int n;

	fprintf(f, "%03d", doc->length);
	for (n = 0; n < doc->length; n++)
	{
		fprintf(f, " %04d:%02d", doc->words[n], argmax(phi[n], model->num_topics));
	}
	fprintf(f, "\n");
	fflush(f);
}


/*
* saves the gamma parameters of the current dataset
*
*/

void save_gamma(char* filename, double** gamma, int num_docs, int num_topics)
{
	FILE* fileptr;
	int d, k;
	fileptr = fopen(filename, "w");

	for (d = 0; d < num_docs; d++)
	{
		fprintf(fileptr, "%5.10f", gamma[d][0]);
		for (k = 1; k < num_topics; k++)
		{
			fprintf(fileptr, " %5.10f", gamma[d][k]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);
}


/*
* save the prediction results and the predictive R^2 value
*/
double save_prediction(char *filename, corpus *corpus)
{
	double dmean = 0;
	double sumlikelihood = 0;
	int nterms = 0;
	double sumavglikelihood = 0;
	for ( int d=0; d<corpus->num_docs; d++ ) {
		dmean += corpus->docs[d].responseVar / corpus->num_docs;
		sumlikelihood += corpus->docs[d].likelihood;
		nterms += corpus->docs[d].total;
		sumavglikelihood += corpus->docs[d].likelihood / corpus->docs[d].total;
	}
	double perwordlikelihood1 = sumlikelihood / nterms;
	double perwordlikelihood2 = sumavglikelihood / corpus->num_docs;

	double ssd = 0;
	for ( int d=0; d<corpus->num_docs; d++ ) 
		ssd += (corpus->docs[d].responseVar - dmean ) * (corpus->docs[d].responseVar - dmean);

	double press = 0;
	for ( int d=0; d<corpus->num_docs; d++ )
		press += ( corpus->docs[d].responseVar - corpus->docs[d].testresponseVar )
				* ( corpus->docs[d].responseVar - corpus->docs[d].testresponseVar );

	double predictiver2 = 1.0 - press / ssd;

	FILE* fileptr;
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "predictive R^2: %5.10f\n", predictiver2 );
	fprintf(fileptr, "perword likelihood1: %5.10f\n", perwordlikelihood1);
	fprintf(fileptr, "perword likelihood2: %5.10f\n", perwordlikelihood2);

	for (int d=0; d<corpus->num_docs; d++)
	{
		fprintf(fileptr, "%5.10f\t%5.10f\n", corpus->docs[d].testresponseVar, corpus->docs[d].responseVar);
	}
	fclose(fileptr);

	return predictiver2;
}

/*
* run_em
*
*/

int run_em(char* start, char* directory, corpus* corpus)
{

	int d, n;
	slda_model *model = NULL;
	double **var_gamma, **phi;
	bool mycovergence = false;

	// allocate variational parameters
	var_gamma = (double**)malloc(sizeof(double*)*(corpus->num_docs));
	double **exp = (double**)malloc(sizeof(double*)*(corpus->num_docs));
	for (d = 0; d < corpus->num_docs; d++) {
		var_gamma[d] = (double*)malloc(sizeof(double) * NTOPICS);
		exp[d] = (double*)malloc(sizeof(double) * NTOPICS);
	}

	int max_length = max_corpus_length(corpus);
	phi = (double**)malloc(sizeof(double*)*max_length);
	for (n = 0; n < max_length; n++)
		phi[n] = (double*)malloc(sizeof(double) * NTOPICS);

	double **a = (double**)malloc(sizeof(double*) * NTOPICS);
	for ( int k=0; k<NTOPICS; k++ )
		a[k] = (double*)malloc(sizeof(double) * NTOPICS);

	// initialize model
	slda_suffstats* ss = NULL;
	if (strcmp(start, "seeded")==0)
	{
		model = new_slda_model(corpus->num_terms, NTOPICS);
		ss = new_slda_suffstats(model);
		corpus_initialize_ss(ss, model, corpus);
		slda_mle(model, ss, 0, 0);
		model->alpha = INITIAL_ALPHA / NTOPICS;
	}
	else if (strcmp(start, "random")==0)
	{
		model = new_slda_model(corpus->num_terms, NTOPICS);
		ss = new_slda_suffstats(model);
		random_initialize_ss(ss, model, corpus);
		slda_mle(model, ss, 0, 0);
		model->alpha = INITIAL_ALPHA / NTOPICS;
	}
	else
	{
		model = load_lda_model(start);
		ss = new_slda_suffstats(model);
		for (int k=0; k<corpus->num_docs; k++ )
			ss->sqresponse += corpus->docs[k].responseVar * corpus->docs[k].responseVar;
	}

	// set the \delta^2 to be the variance of response variables
	double dmean = 0;
	for ( d=0; d<corpus->num_docs; d++ )
		dmean += corpus->docs[d].responseVar / corpus->num_docs;
	model->deltasq = 0;
	for ( d=0; d<corpus->num_docs; d++ )
		model->deltasq += (corpus->docs[d].responseVar - dmean) * (corpus->docs[d].responseVar - dmean)
		/ corpus->num_docs;



	char filename[100];
	//sprintf(filename, "%s\\000",directory);
	//save_slda_model(model, filename);

	// run expectation maximization
	sprintf(filename, "%s\\likelihood.dat", directory);
	FILE* likelihood_file = fopen(filename, "w");

	long runtime_start = get_runtime();
	int i = 0;
	double likelihood, likelihood_old = 0, converged = 1;
	int nIt = 0;
	while (((converged < 0) || (converged > EM_CONVERGED) || (i <= 2)) && (i <= EM_MAX_ITER))
	{
		printf("**** em iteration %d ****\n", i + 1);
		likelihood = 0;
		zero_initialize_ss(ss, model);

		// e-step

		for (d = 0; d < corpus->num_docs; d++)
		{
			// initialize to uniform
			for (n = 0; n < max_length; n++)
				for ( int k=0; k<NTOPICS; k++ )
					phi[n][k] = 1.0 / (double) NTOPICS;

			printf("document %d\n",d);
			likelihood += doc_e_step( &(corpus->docs[d]),
										var_gamma[d],
										phi,
										a,
										model,
										ss);
		}

		// m-step

		if ( slda_mle(model, ss, ESTIMATE_ALPHA, ESTIMATE_DELTASQ, false) ) {
			nIt = i + 1;
		} else {
			break;
		}

		// check for convergence

		converged = (likelihood_old - likelihood) / (likelihood_old);
		if (converged < 0) VAR_MAX_ITER = VAR_MAX_ITER * 2;
		likelihood_old = likelihood;

		// output model and likelihood

		fprintf(likelihood_file, "%10.10f\t%5.5e\n", likelihood, converged);
		fflush(likelihood_file);
		//if ((i % LAG) == 0)
		//{
		//	sprintf(filename,"%s\\%d",directory, i + 1);
		//	save_slda_model(model, filename);
		//	sprintf(filename,"%s\\%d.gamma",directory, i + 1);
		//	save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);
		//}
		i ++;
	}
	long runtime_end = get_runtime();
	model->m_dTrainTime = ((double)runtime_end-(double)runtime_start)/100.0;
	printf("Training time in (cpu-seconds): %.3f\n", model->m_dTrainTime);

	if(converged >= 0 || converged <= EM_CONVERGED){
		myconvergence = true;
		for(d=0; d<corpus->num_docs; d++){
			write_probcounts(corpus->docs[d], d, myconvergence, model);
		}

	}


	// output the final model
	sprintf(filename,"%s/final",directory);
	save_slda_model(model, filename);
	sprintf(filename,"%s/final.gamma",directory);
	save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);

	// output the word assignments (for visualization)

	sprintf(filename, "%s/word-assignments.dat", directory);
	FILE* w_asgn_file = fopen(filename, "w");
	for (d = 0; d < corpus->num_docs; d++)
	{
		printf("final e step document %d\n",d);
		likelihood += slda_inference(&(corpus->docs[d]), model, var_gamma[d], 
			phi, a, VAR_MAX_ITER, VAR_CONVERGED);
		write_word_assignment(w_asgn_file, &(corpus->docs[d]), phi, model);

		for ( int k=0; k<model->num_topics; k++ ) {
			double dVal = 0;
			for ( int n=0; n<corpus->docs[d].length; n++ )
				dVal += phi[n][k] * corpus->docs[d].counts[n] / corpus->docs[d].total;
			exp[d][k] = dVal;
		}
	}
	fclose(w_asgn_file);
	fclose(likelihood_file);

	//// output the low-dimensional representation of data
	//if ( NFOLDS == 5 ) partitionData(corpus, exp/*var_gamma*/, model->num_topics);
	//else outputData(corpus, exp/*var_gamma*/, model->num_topics);

	//FILE *fptr = fopen("overall-res.txt", "a");
	//fprintf(fptr, "K: %d; train-time: %.3f; ", model->num_topics, dTrainTime);
	//fclose(fptr);
	for (d = 0; d < corpus->num_docs; d++) {
		free(var_gamma[d]); free(exp[d]);
	}
	free(var_gamma); free(exp);
	for (n = 0; n < max_length; n++)
		free(phi[n]);
	free(phi);
	for ( int k=0; k<model->num_topics; k++ )
		free(a[k]);
	free(a);

	return nIt;
}

// save the low-dimentional data for 5 fold cv
void partitionData(corpus *corpus, double** gamma, int ntopic)
{
	ifstream ifs("randomorder.txt", ios_base::in);
	char buff[512];
	vector<double> order;
	while ( !ifs.eof() ) {
		ifs.getline(buff, 512);
		order.push_back( atof(buff) );
	}
	ifs.close();

	// partition into 5 parts for 5 fold cv
	int nunit = corpus->num_docs / 5;
	for ( int k=1; k<=5; k++ ) {
		sprintf(buff, "train_tfidf5000(%dtopic)_cv5(%d).txt", ntopic, k);
		ofstream ofs(buff, ios_base::out | ios_base::trunc);
		sprintf(buff, "test_tfidf5000(%dtopic)_cv5(%d).txt", ntopic, k);
		ofstream ofs2(buff, ios_base::out | ios_base::trunc);
		for ( int i=0; i<corpus->num_docs; i++ ) {
			int ndocIx = order[i];
			double dlog = corpus->docs[ndocIx].responseVar;

			bool btrain = true;
			if ( k < 5 && (i>=(k-1)*nunit) && (i<k*nunit) ) btrain = false;
			else if ( k == 5 && (i >= (k-1) * nunit) ) btrain = false;

			double dNorm = 0;
			for ( int k=0; k<ntopic; k++ ) dNorm += gamma[ndocIx][k];

			if ( btrain ) {
				ofs << ntopic << " " << dlog;
				for ( int k=0; k<ntopic; k++ ) ofs << " " << k << ":" << gamma[ndocIx][k] / dNorm;
				ofs << endl;
			} else {
				ofs2 << ntopic << " " << dlog;
				for ( int k=0; k<ntopic; k++ ) ofs2 << " " << k << ":" << gamma[ndocIx][k] / dNorm;
				ofs2 << endl;
			}write_probcounts
		}
		ofs2.close();
		ofs.close();
	}
}
// save the low-dimentional data
void outputData(corpus *corpus, double** gamma, int ntopic)
{
	char buff[512];

	sprintf(buff, "sLDA_(%dtopic)_train.txt", ntopic);
	ofstream ofs(buff, ios_base::out | ios_base::trunc);
	sprintf(buff, "sLDA_(%dtopic)_test.txt", ntopic);
	ofstream ofs2(buff, ios_base::out | ios_base::trunc);

	for ( int i=0; i<corpus->num_docs; i++ ) {
		double dlog = corpus->docs[i].responseVar;

		double dNorm = 0;
		for ( int k=0; k<ntopic; k++ ) dNorm += gamma[i][k];

		/* 856 is for binary dataset, 11269 for multi-class dataset. */
		if ( i < 856/*11269*/ ) {
			ofs << ntopic << " " << dlog;
			for ( int k=0; k<ntopic; k++ ) ofs << " " << k << ":" << gamma[i][k] / dNorm;
			ofs << endl;
		} else {
			ofs2 << ntopic << " " << dlog;
			for ( int k=0; k<ntopic; k++ ) ofs2 << " " << k << ":" << gamma[i][k] / dNorm;
			ofs2 << endl;			
		}
	}
	ofs.close();
	ofs2.close();
}
/*
* read settings.
*
*/

void read_settings(char* filename)
{
	FILE* fileptr;
	char alpha_action[100];
	char delta_action[100];
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "var max iter %d\n", &VAR_MAX_ITER);
	fscanf(fileptr, "var convergence %f\n", &VAR_CONVERGED);
	fscanf(fileptr, "em max iter %d\n", &EM_MAX_ITER);
	fscanf(fileptr, "em convergence %f\n", &EM_CONVERGED);
	fscanf(fileptr, "alpha %s\n", alpha_action);
	fscanf(fileptr, "deltasq %s\n", delta_action);
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
	fscanf(fileptr, "data-file: %s", DATA_FILENAME);

	fclose(fileptr);
}


/*
* inference only
*
*/

void infer(char* model_root, char* save, corpus* corpus)
{
	FILE* fileptr;
	char filename[100];
	int i, d, n;
	slda_model *model;
	double **var_gamma, likelihood, **phi;
	document* doc;

	model = load_lda_model(model_root);
	var_gamma = (double**)malloc(sizeof(double*)*(corpus->num_docs));
	for (i = 0; i < corpus->num_docs; i++)
		var_gamma[i] = (double*)malloc(sizeof(double)*model->num_topics);
	
	double **a = (double**)malloc(sizeof(double*)*model->num_topics);
	for ( int k=0; k<model->num_topics; k++ )
		a[k] = (double*)malloc(sizeof(double) * model->num_topics);
	int nMaxLength = max_corpus_length(corpus);
	phi = (double**) malloc(sizeof(double*) * nMaxLength );
	for (n = 0; n < nMaxLength; n++) {
		phi[n] = (double*) malloc(sizeof(double) * model->num_topics);
	}

	long time_start = get_runtime();
	int nAcc = 0;
	double dUniformPhiVal = 1.0 / (double) model->num_topics, *phiPtr=NULL;
	sprintf(filename, "%s/evl-lda-lhood.dat", save);
	fileptr = fopen(filename, "w");
	for (d = 0; d < corpus->num_docs; d++)
	{
		if (((d % 100) == 0) && (d>0)) printf("document %d\n",d);

		doc = &(corpus->docs[d]);
		// initialize to uniform distrubtion
		for (n = 0; n < doc->length; n++) {
			phiPtr = phi[n];
			for ( int k=0; k<model->num_topics; k++ )
				phiPtr[k] = dUniformPhiVal;
		}

		likelihood = slda_inference_prediction(doc, model, var_gamma[d], phi, a, 
			VAR_MAX_ITER, VAR_CONVERGED);

		// do prediction
		doc->testresponseVar = 0;
		for ( int k=0; k<model->num_topics; k++ ) {
			double dVal = 0;
			for ( int n=0; n<doc->length; n++ )
				dVal += phi[n][k] * doc->counts[n] / doc->total;
			doc->testresponseVar += dVal * model->eta[k];
		}
		doc->likelihood = likelihood;

		if ( doc->testresponseVar > 0.5 && doc->responseVar > 0.5 ) nAcc ++;
		if ( doc->testresponseVar <= 0.5 && doc->responseVar < 0.5 ) nAcc ++;

		fprintf(fileptr, "%5.5f\n", likelihood);
	}
	fclose(fileptr);
	long time_end = get_runtime();
	double dTestTime = ((double)time_end - (double)time_start) / 100.0;
	sprintf(filename, "%s/evl-gamma.dat", save);
	save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);

	// save the prediction performance
	sprintf(filename, "%s/evl-performance.dat", save);
	double dPredR2 = save_prediction(filename, corpus);

	printf("\nPredictive R2: %.3f; Binary Classification Accuracy: %.4f\n", dPredR2, (nAcc * 100.0) / corpus->num_docs);

	FILE *fptr = fopen("overall-res.txt", "a");
	fprintf(fptr, "K: %d; alpha: %.3f; predR2: %.3f; test-time: %.3f; train-time: %.3f\n", 
		model->num_topics, model->alpha, dPredR2, dTestTime, model->m_dTrainTime);
	fclose(fptr);
	// free memory
	for ( int k=0; k<model->num_topics; k++ )
		free( a[k] );
	free(a);
	for (n = 0; n < corpus->max_length; n++) {
		free( phi[n] );
	}
	free(phi);
	for ( i=0; i<corpus->num_docs; i++ )
		free( var_gamma[i] );
	free( var_gamma );
}

void write_probcounts(docuent* doc, int d, boolean b, slda_model* model)
{
	int n, k;
	int hotelids[corpus->max_length];
	double ratings[corpus->max_length][8];
	FILE *fp = fopen("hotelratings.txt", "w");
	for (int i = 0; i < corpus->max_length; ++i)
	{
		fscanf(fp, "%d", hotelids[i]);
		for(int j=0; j<8; j++)
			fscanf(fp, "%lf", ratings[i][j]);
	}
	fscanf(fp, );

	if(myconvergence == true){
		FILE *fptr = fopen("sLDAprobcounts.txt", "a");
		if()
	}
}
/*
* main
*/
int main(int argc, char* argv[])
{
	// (est / inf) alpha k settings data (random / seed/ modelf) (directory / out)

	corpus* c;

	//long t1;
	//(void) time(&t1);
	seedMT( time(NULL) );
	// seedMT(4357U);

	if (argc > 1)
	{
		NFOLDS = 0;
		if (strcmp(argv[1], "estinf") == 0 ) // for the learning and evaluation
		{
			NTOPICS = atoi(argv[2]);
			NFOLDS = atoi(argv[3]);
			INITIAL_ALPHA = atof(argv[4]);
			read_settings(argv[5]); // "settings.txt";
			c = read_data(DATA_FILENAME);
			char dir[512];
			sprintf(dir, "rev%d_%d", NTOPICS, NFOLDS);
			mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);
			corpus *tr_c, *tst_c;
			tr_c  = (corpus*)malloc(sizeof(corpus));
			tst_c = (corpus*)malloc(sizeof(corpus));

			equal_partion_data(c, tr_c, tst_c);

			//save_data(tr_c, "rev2500_6000_nlp_lda_voc12000_5000_train");
			//save_data(tst_c, "rev2500_6000_nlp_lda_voc12000_5000_test");

			run_em("random", dir, tr_c);

			char model_root[512];
			sprintf(model_root, "%s/final", dir);
			infer(model_root, dir, tst_c);
		}

		if (strcmp(argv[1], "est")==0)
		{
			NTOPICS = atoi(argv[2]);
			NFOLDS = atoi(argv[3]);
			INITIAL_ALPHA = atof(argv[4]);
			read_settings(argv[5]); //"settings.txt";
			c = read_data(DATA_FILENAME);
			char dir[512];
			sprintf(dir, "rev%d_%d", NTOPICS, NFOLDS);
			mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);
			run_em("random", dir, c);
		}
		if (strcmp(argv[1], "inf")==0)
		{
			read_settings(argv[2]); //"settings.txt";
			c = read_data(DATA_FILENAME);
			corpus *tr_c, *tst_c;
			tr_c  = (corpus*)malloc(sizeof(corpus));
			tst_c = (corpus*)malloc(sizeof(corpus));

			equal_partion_data(c, tr_c, tst_c);

			infer(argv[3], argv[4], tst_c);
		}

		if ( strcmp(argv[1], "cv") == 0 ) // cross-validation (run one fold at a time)
		{
			NFOLDS = atoi(argv[2]);
			FOLDIX = atoi(argv[3]);

			INITIAL_ALPHA = atof(argv[4]);
			NTOPICS = atoi(argv[5]);
			read_settings("settings.txt");
			c = read_data(argv[6]);

			reorder(c, "randomorder.txt");

			corpus *traincorpus = get_traindata(c, NFOLDS, FOLDIX);
			corpus *testcorpus = get_testdata(c, NFOLDS, FOLDIX);

			// estimate on training data (return the number of iterations)
			char resDir[512];
			sprintf(resDir, "%dtopic_cv%d(%d)", NTOPICS, NFOLDS, FOLDIX);
			mkdir(resDir, S_IRUSR|S_IWUSR|S_IXUSR);
			/*if ( GetLastError() == ERROR_ALREADY_EXISTS ) {
				int x = rmdir(resDir);
			}*/
			//mkdir(resDir, S_IRUSR|S_IWUSR|S_IXUSR);

			int nIt = run_em(argv[7], resDir, traincorpus);

			// predict on test corpus
			char chrRes[512];
			sprintf(chrRes, "%s/evlRes%d(%d)_it%d.txt", resDir, NFOLDS, FOLDIX, nIt);
			char chrModel[512];
			sprintf(chrModel, "%s/final", resDir);
			infer(chrModel, chrRes, testcorpus);

			free(traincorpus->docs);
			free(testcorpus->docs);
		}
		free(c->docs);
	}
	else
	{
		printf("usage : sLDAr estinf [k] [fold] [initial alpha] [settings file]\n");
		printf("        sLDAr est [k] [fold] [initial alpha] [settings file]\n");
		printf("        sLDAr inf [settings file] [model root] [dir]\n");
		printf("        sLDAr cv [foldnum] [foldix] [initial alpha] [k] [settings] [data] [random/seeded/*]\n");
	}
	return(0);
}
