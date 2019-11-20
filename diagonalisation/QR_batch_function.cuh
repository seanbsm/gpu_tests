#ifndef QR_BATCH_FUNCTION_CUH
#define QR_BATCH_FUNCTION_CUH

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <iomanip>
#include <iostream>
#include <vector>
#include <cusolverDn.h>

double diagonalise_batch_QR(float  *d_A, float  *d_W, int m, int batchSize);
double diagonalise_batch_QR(double *d_A, double *d_W, int m, int batchSize);

#endif // QR_BATCH_FUNCTION_CUH

