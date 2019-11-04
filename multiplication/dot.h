#ifndef DOT_H
#define DOT_H

#include <iostream>
#include <complex>
//#include <cblas.h>
#include "mkl.h"

//extern "C" void zgemm_( __complex__ double* a,  __complex__ double* b, int* dim1, int* dim2, __complex__ double* a, int* lda, int* ipiv, int* info);

/* Product functions for SQUARE matrices.
 * M: matrix
 * V: vector
 * S: scalar
 * N: size along an axis (so a matrix has N^2 elements)
 */
 
void dot_VS(float  *A, float  *B, float  *C, int N);	// single Vector-Scalar dot product (row-major)
void dot_VS(double *A, double *B, double *C, int N);	// double Vector-Scalar dot product (row-major)
double dot_VV(float  *x, float  *y, int N);			// single Vector-Vector dot product (row-major)
double dot_VV(double *x, double *y, int N);			// double Vector-Vector dot product (row-major)
void dot_MS(float  *A, float  *B, float  *C, int N);	// single Matrix-Scalar dot product (row-major)
void dot_MS(double *A, double *B, double *C, int N);	// double Matrix-Scalar dot product (row-major)
void dot_MV(float  *A, float  *B, float  *C, int N);	// single Matrix-Vector dot product (row-major)
void dot_MV(double *A, double *B, double *C, int N);	// double Matrix-Vector dot product (row-major)
void dot_MM(float  *A, float  *B, float  *C, int N, int K, int M);	// single Matrix-Matrix dot product (row-major)
void dot_MM(double *A, double *B, double *C, int N, int K, int M);	// double Matrix-Matrix dot product (row-major)

void cdot_VS_colMaj(std::complex<float>  *A, std::complex<float>  *B, int N);										// std::complex(single) Vector-Scalar dot product (row-major)
void cdot_VS_colMaj(std::complex<double> *A, std::complex<double> *B, int N);										// std::complex(double) Vector-Scalar dot product (row-major)
void cdot_VV(std::complex<float>  A[], std::complex<float>  B[], std::complex<float>  C[], int N);					// std::complex(single) Vector-Vector dot product (row-major)
void cdot_VV(std::complex<double> A[], std::complex<double> B[], std::complex<double> C[], int N);					// std::complex(double) Vector-Vector dot product (row-major)
void cdot_MS(std::complex<float>  A[], std::complex<float>  B[], std::complex<float>  C[], int N);					// std::complex(single) Matrix-Scalar dot product (row-major)
void cdot_MS(std::complex<double> A[], std::complex<double> B[], std::complex<double> C[], int N);					// std::complex(double) Matrix-Scalar dot product (row-major)
void cdot_MV(std::complex<float>  *A,  std::complex<float>  *x,  std::complex<float>  *y, int N, int stepLength);	// std::complex(single) Matrix-Vector dot product (row-major)
void cdot_MV(std::complex<double> *A,  std::complex<double> *x,  std::complex<double> *y, int N, int stepLength);	// std::complex(double) Matrix-Vector dot product (row-major)
void cdot_MM(std::complex<float>  *A,  std::complex<float>  *B,  std::complex<float>  *C, int N, int M, int K);		// std::complex(single) Matrix-Matrix dot product (row-major)
void cdot_MM(std::complex<double> *A,  std::complex<double> *B,  std::complex<double> *C, int N, int M, int K);		// std::complex(double) Matrix-Matrix dot product (row-major)


void cdot_MM_M(std::complex<float> A[], std::complex<float> B[], std::complex<float> C[], __complex__ float alpha, __complex__ float beta, int N);		// std::complex(single) Matrix-Matrix dot product Plus Matrix (row-major)
void cdot_MM_M(std::complex<double> A[], std::complex<double> B[], std::complex<double> C[], __complex__ double alpha, __complex__ double beta, int N);	// std::complex(double) Matrix-Matrix dot product Plus Matrix (row-major)


#endif // DOT_H
