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
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

double log_sum(double log_a, double log_b);
double trigamma(double x);
double digamma(double x);
double log_gamma(double x);
//void make_directory(char* name);
int argmax(double* x, int n);
double dotprod(double *a, double *b, const int&n);
void matrixprod(double *a, double **A, double *res, const int&n);
void matrixprod(double **A, double *a, double *res, const int&n);
void addmatrix(double **A, double *a, double *b, const int &n, double factor);
void addmatrix2(double **A, double *a, double *b, const int &n, double factor);
bool inverse(double **A, double **res, const int &n);
bool choleskydec(double **A, double **res, const int &n, bool isupper);
void printmatrix(double **A, double n);

#endif
