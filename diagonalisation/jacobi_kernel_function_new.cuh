#ifndef JACOBI_KERNEL_FUNCTION_NEW_CUH
#define JACOBI_KERNEL_FUNCTION_NEW_CUH

#include <cuda_runtime.h>

/* Type definitions */
#include "typeDefs.h"

#include <iomanip>
#include <iostream>
#include <vector>

double jacobi_kernels_parallel(floatType *d_A, floatType *d_W, int m, int batchSize);

#endif // JACOBI_KERNEL_FUNCTION_NEW_CUH


