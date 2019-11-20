#ifndef MAGMA_DSYEV_BATCH_FUNCTIONS_CUH
#define MAGMA_DSYEV_BATCH_FUNCTIONS_CUH

//~ #include "magma_v2.h"
//~ #include "magma_operators.h"
//~ #include "magma_lapack.h"


double diagonalise_batch_syev(float  *d_A, float  *d_W, int m, int batchSize);
double diagonalise_batch_syev(double *d_A, double *d_W, int m, int batchSize);

#endif // MAGMA_DSYEV_BATCH_FUNCTIONS_CUH


