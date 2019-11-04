#ifndef KERNEL_FUNCTIONS_CUH
#define KERNEL_FUNCTIONS_CUH

#include <cuda_runtime.h>
#include <cusolverDn.h>

double diagonalise_kernel(float  *d_A, float  *d_W, int m, int batchSize);
double diagonalise_kernel(double *d_A, double *d_W, int m, int batchSize);

#endif // KERNEL_FUNCTIONS_CUH


