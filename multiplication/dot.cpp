
#include "dot.h"

void dot_VS(float *A, float *B, int N){
	cblas_sscal(N, *B, A, 1);
}
void dot_VS(double *A, double *B, int N){
	cblas_dscal(N, *B, A, 1);
}

void dot_MS(float *A, float *B, float *C, int N){
}
void dot_MS(double *A, double *B, double *C, int N){
}

void dot_MV(float *A, float *B, float *C, int N){
}
void dot_MV(double *A, double *B, double *C, int N){
}

void dot_MM(float *A, float *B, float *C, int N, int K, int M){
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, K, 1.0, A, K, B, M, 0.0, C, M);
}
void dot_MM(double *A, double *B, double *C, int N, int K, int M){
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, K, 1.0, A, K, B, M, 0.0, C, M);
}

/* A -> A*B, where B is a scalar and A a vector */
void cdot_VS_colMaj(std::complex<float> *A, std::complex<float> *B, int N){
	cblas_cscal(N, B, A, N);
}
void cdot_VS_colMaj(std::complex<double> *A, std::complex<double> *B, int N){
	cblas_zscal(N, B, A, N);
}

double ddot_VV(float *x, float *y, int N){
	return cblas_sdot(N, x, 1, y, 1);
}
double ddot_VV(double *x, double *y, int N){
	return cblas_ddot(N, x, 1, y, 1);
}

/* y := A*x. The stepLength is used if you want to
 * multiply a column of one matrix with another matrix. */
void cdot_MV(std::complex<float> *A, std::complex<float> *x, std::complex<float> *y, int N, int stepLength){
	float beta = 0;
	float alpha = 1;
	cblas_cgemv(CblasRowMajor, CblasNoTrans, N, N, &alpha, A, N, x, stepLength, &beta, y, 1);
}
void cdot_MV(std::complex<double> *A, std::complex<double> *x, std::complex<double> *y, int N, int stepLength){
	double beta = 0;
	double alpha = 1;
	cblas_zgemv(CblasRowMajor, CblasNoTrans, N, N, &alpha, A, N, x, stepLength, &beta, y, 1);
}

/* Dot product for C := alpha*A*B + beta*C for dimensions C(N,M), A(N,K), and B(K,M) (row-major) */
void cdot_MM(std::complex<float> *A, std::complex<float> *B, std::complex<float> *C, int N, int K, int M){
	float beta = 0;
	float alpha = 1;
	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, K, &alpha, A, K, B, M, &beta, C, M);
}
void cdot_MM(std::complex<double> *A, std::complex<double> *B, std::complex<double> *C, int N, int K, int M){
	double beta = 0;
	double alpha = 1;
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, K, &alpha, A, K, B, M, &beta, C, M);
}

void cdot_MM_M(std::complex<float> A[], std::complex<float> B[], std::complex<float> C[], __complex__ float alpha, __complex__ float beta, int N){
	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, &alpha, reinterpret_cast <__complex__ float*>(&*A), N, reinterpret_cast <__complex__ float*>(&*B), N, &beta, reinterpret_cast <__complex__ float*>(&*C), N);
}
void cdot_MM_M(std::complex<double> A[], std::complex<double> B[], std::complex<double> C[], __complex__ double alpha, __complex__ double beta, int N){
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, &alpha, reinterpret_cast <__complex__ double*>(&*A), N, reinterpret_cast <__complex__ double*>(&*B), N, &beta, reinterpret_cast <__complex__ double*>(&*C), N);
}
