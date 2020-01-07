
#include "eigenFinder.h"

/* Finds the eigenvalues and eigenvectors
 * of a real, symmetric matrix A.
 * For simplicity here, A must be
 * stored as an upper triangle. w will
 * be filled with the eigenvalues in
 * ascending order, and z will be a matrix
 * with the corresponding eigenvectors
 * (column by column). Lastly, we work
 * with row major matrices, as is usual
 * with C & C++ */
 
void findEigenReal(float *A, float *w, float *z, int N){
	char jobz = 'V';
	char uplo = 'U';
	
	LAPACKE_sspevd(LAPACK_ROW_MAJOR, jobz, uplo, N, A, w, z, N);
}

void findEigenReal(double *A, double *w, double *z, int N){
	char jobz = 'V';
	char uplo = 'U';
	
	LAPACKE_dspevd(LAPACK_ROW_MAJOR, jobz, uplo, N, A, w, z, N);
}
