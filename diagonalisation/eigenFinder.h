#ifndef EIGENFINDER_H
#define EIGENFINDER_H

#include <iostream>
#include <complex>
#include "mkl.h"

void findEigenReal(float *A, float *w, float *z, int N);
void findEigenReal(double *A, double *w, double *z, int N);

#endif // EIGENFINDER_H
