#ifndef DSYEVD_STREAM_FUNCTIONS_CUH
#define DSYEVD_STREAM_FUNCTIONS_CUH

#include <cuda_runtime.h>
#include <cusolverDn.h>

double diagonalise_stream_syevd(float  *d_A, float  *d_W, int m, int batchSize);
double diagonalise_stream_syevd(double *d_A, double *d_W, int m, int batchSize);

#endif // DSYEVD_STREAM_FUNCTIONS_CUH

