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
#include "utils.h"

#include "ap.h"

#include "cholesky.h"
#include "spdinverse.h"

/*
* given log(a) and log(b), return log(a + b)
*
*/

double log_sum(double log_a, double log_b)
{
	double v;

	if (log_a < log_b)
	{
		v = log_b+log(1 + exp(log_a-log_b));
	}
	else
	{
		v = log_a+log(1 + exp(log_b-log_a));
	}
	return(v);
}

/**
* Proc to calculate the value of the trigamma, the second
* derivative of the loggamma function. Accepts positive matrices.
* From Abromowitz and Stegun.  Uses formulas 6.4.11 and 6.4.12 with
* recurrence formula 6.4.6.  Each requires workspace at least 5
* times the size of X.
*
**/

double trigamma(double x)
{
	double p;
	int i;

	x=x+6;
	p=1/(x*x);
	p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
		*p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
	for (i=0; i<6 ;i++)
	{
		x=x-1;
		p=1/(x*x)+p;
	}
	return(p);
}


/*
* taylor approximation of first derivative of the log gamma function
*
*/

double digamma(double x)
{
	double p;
	x=x+6;
	p=1/(x*x);
	p=(((0.004166666666667*p-0.003968253986254)*p+
		0.008333333333333)*p-0.083333333333333)*p;
	p=p+log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
	return p;
}


double log_gamma(double x)
{
	double z=1/(x*x);

	x=x+6;
	z=(((-0.000595238095238*z+0.000793650793651)
		*z-0.002777777777778)*z+0.083333333333333)/x;
	z=(x-0.5)*log(x)-x+0.918938533204673+z-log(x-1)-
		log(x-2)-log(x-3)-log(x-4)-log(x-5)-log(x-6);
	return z;
}



/*
* make directory
*
*/

//void make_directory(char* name)
//{
//	//mkdir(name, S_IRUSR|S_IWUSR|S_IXUSR);
//	make_directory(name);
//}


/*
* argmax
*
*/

int argmax(double* x, int n)
{
	int i;
	double max = x[0];
	int argmax = 0;
	for (i = 1; i < n; i++)
	{
		if (x[i] > max)
		{
			max = x[i];
			argmax = i;
		}
	}
	return(argmax);
}

double dotprod(double *a, double *b, const int&n)
{
	double res = 0;
	for ( int i=0; i<n; i++ ) {
		res += a[i] * b[i];
	}
	return res;
}
/* a vector times a (n x n) square matrix  */
void matrixprod(double *a, double **A, double *res, const int&n)
{
	for ( int i=0; i<n; i++ ) {
		res[i] = 0;
		for ( int j=0; j<n; j++ ) {
			res[i] += a[j] * A[j][i];
		}
	}
}
/* a (n x n) square matrix times a vector. */
void matrixprod(double **A, double *a, double *res, const int&n)
{
	for ( int i=0; i<n; i++ ) {
		res[i] = 0;
		for ( int j=0; j<n; j++ ) {
			res[i] += a[j] * A[i][j];
		}
	}
}

/* A + ab^\top*/
void addmatrix(double **A, double *a, double *b, const int &n, double factor)
{
	for (int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			A[i][j] += a[i] * b[j] * factor;
		}
	}
}

/* A + ab^\top + ba^\top*/
void addmatrix2(double **A, double *a, double *b, const int &n, double factor)
{
	for (int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			A[i][j] += (a[i] * b[j] + b[i] * a[j]) * factor;
		}
	}
}

/* the inverse of a matrix. */
bool inverse(double **A, double **res, const int &n)
{
    ap::real_2d_array a;
    a.setbounds(0, n-1, 0, n-1);

	// upper-triangle matrix
	for ( int i=0; i<n; i++ ) 
	{
		for ( int j=0; j<n; j++ )
		{
			if ( j < i ) a(i, j) = 0;
			else a(i, j) = A[i][j];
		}
	}
	//printf("\n\n");
	//printmatrix(A, n);

	bool bRes = true;
	//if ( !spdmatrixcholesky(a, n, true) ) {
	//	printf("matrix is not positive-definite\n");
	//	bRes = false;
	//}

	//if ( spdmatrixcholeskyinverse(a, n, true) ) {
	//} else {
	//	bRes = false;

		if( spdmatrixinverse(a, n, true) )
		{
		} else {
			printf("Inverse matrix error!");
			bRes = false;
		//exit(0);
		}

	for ( int i=0; i<n; i++ ) 
	{
		for ( int j=0; j<n; j++ )
		{
			if ( j < i ) res[i][j] = a(j, i);
			else res[i][j] = a(i, j);
		}
	}
	//printf("\n\n");
	//printmatrix(res, n);

	//printf("\n");
	/*double dsqerr = 0;
	for ( int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			double dval = 0;
			for ( int k=0; k<n; k++ )
				dval += A[i][k] * res[k][j];
			printf("\t%f", dval);
			if ( i == j ) dsqerr += (dval - 1) * (dval - 1);
			else dsqerr += dval * dval;
		}
		printf("\n");
	}
	printf("Sq Err: %f\n", dsqerr);*/


	return bRes;
}

void printmatrix(double **A, double n)
{
	for ( int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			printf("\t%f", A[i][j]);
		}
		printf("\n");
	}
}
long get_runtime(void)
{
	long start;
	start = clock();
	return((long)((double)start*100.0/(double)CLOCKS_PER_SEC));
}