#ifndef DSYEVJ_BATCH_FUNCTIONS_CUH
#define DSYEVJ_BATCH_FUNCTIONS_CUH

#include <cuda_runtime.h>
#include <cusolverDn.h>

double diagonalise_batch_syevj(float  *d_A, float  *d_W, int m, int batchSize);
double diagonalise_batch_syevj(double *d_A, double *d_W, int m, int batchSize);

#endif // DSYEVJ_BATCH_FUNCTIONS_CUH

