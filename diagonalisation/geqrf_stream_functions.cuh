#ifndef GEQRF_STREAM_FUNCTIONS_CUH
#define GEQRF_STREAM_FUNCTIONS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

double diagonalise_stream_geqrf(float  *d_A, float  *d_W, int m, int batchSize);
double diagonalise_stream_geqrf(double *d_A, double *d_W, int m, int batchSize);

#endif // GEQRF_STREAM_FUNCTIONS_CUH

