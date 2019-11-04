
/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include syevd_example.cpp 
 *   g++ -o a.out syevd_example.o -L/usr/local/cuda/lib64 -lcudart -lcusolver
 *
 */
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda_runtime.h>

//~ #include <cuda.h>
#include "cublas_v2.h" 

/* Time-keeping modules */
#include <chrono>
#include <ctime>

#include "dot.h"

void printMatrix(int m, int n, const double*A, const char* name){
	for(int row = 0 ; row < m ; row++){
		for(int col = 0 ; col < n ; col++){
			double Areg = A[row + col*m];
			printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
		}
	}
}

/* Meant for mxm matrices */
void fillMatrix(double *A, int m){
	for (int i=0; i<m; i++){
		for (int j=0; j<m; j++){
			A[i + j*m] = (i + j*j) * sqrt((double)(i + j));
		}
	}
}

void fillMatrices(double *h_A, int m, int batchSize){
	for (int b=0; b<batchSize; b++){
		fillMatrix(&h_A[b*m*m], m);
	}
}

/* This function multiplies by calling Dgemm in parallel using batch routines */
double multiplication_batch(double *d_A, double *d_B, double *d_C, int m, int batchSize){
	
	const double alpha = 1.;
	const double beta  = 0.;
	
	cublasHandle_t cublasH;
	cublasCreate(&cublasH);
	
	//~ auto start = std::chrono::system_clock::now();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	/* Synchronise before running multiplication scheme */
	//~ cudaDeviceSynchronize();
	cublasDgemmStridedBatched(cublasH,
							  CUBLAS_OP_N,
							  CUBLAS_OP_N,
							  m, m, m,
							  &alpha,
							  d_B, m,
							  m*m,
							  d_A, m,
							  m*m,
							  &beta,
							  d_C, m,
							  m*m, 
							  batchSize);
	/* Synchronise after running mulitplication scheme */
	//~ cudaDeviceSynchronize();
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	
	//~ auto end = std::chrono::system_clock::now();
	//~ std::chrono::duration<double> time = end-start;
	
	//~ return time.count();
	return (double) time*1e-3;
}

/* This function multiplies by calling Dgemm in parallel using streaming */
double multiplication_stream(double *d_A, double *d_B, double *d_C, int m, int batchSize){
	
	int NBSTREAM = batchSize;
	
	const double alpha = 1.f;
	const double beta  = 0.f;
	
	cublasHandle_t cublasH;
	cublasCreate(&cublasH);
	
	cudaError_t err;
	cublasStatus_t stat;
	
	//~ cudaStream_t stream [NBSTREAM];
	cudaStream_t *stream = (cudaStream_t *)malloc(NBSTREAM*sizeof(cudaStream_t));
	for (int i=0; i<NBSTREAM; i++){
		//~ cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
		err = cudaStreamCreate(&(stream[i]));
		
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	}
	
	//~ auto start = std::chrono::system_clock::now();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	/* Synchronise before running multiplication scheme */
	//~ cudaDeviceSynchronize();
	for(int i=0;i<NBSTREAM;i++){
		
		cublasSetStream(cublasH, stream[i]);
		stat = cublasDgemm(cublasH,
						   CUBLAS_OP_N,
						   CUBLAS_OP_N,
						   m, m, m,
						   &alpha,
						   &d_A[i*m*m], m,
						   &d_B[i*m*m], m,
						   &beta,
						   &d_C[i*m*m], m);
					
		if(stat!=CUBLAS_STATUS_SUCCESS){printf("error code %d, line(%d)\n", stat, __LINE__);exit(EXIT_FAILURE);}
	}
	/* Synchronise after running multiplication scheme */
	//~ cudaDeviceSynchronize();
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	//~ auto end = std::chrono::system_clock::now();
	//~ std::chrono::duration<double> time = end-start;
	
	//~ return time.count();
	return (double) time*1e-3;
}

int main(int argc, char*argv[]){
	
	const int m = 4*32;
	const int batchSize = 2*32;

	/* Declare host arrays */
	double *A = new double [m*m*batchSize];
	double *B = new double [m*m*batchSize];
	double *C = new double [m*m*batchSize];
	
	/* Declare device arrays */
	double *d_A  = NULL; /* m-by-m-by-batchSize */
	double *d_B  = NULL; /* m-by-m-by-batchSize */
	double *d_C  = NULL; /* m-by-m-by-batchSize */
	
	/* Fill up array A and B with matrix elements */
	fillMatrices(A, m, batchSize);
	fillMatrices(B, m, batchSize);
	
	/* Allocate A, B, and C on device */
	cudaMalloc ((void**)&d_A, sizeof(double)*m*m*batchSize);
	cudaMalloc ((void**)&d_B, sizeof(double)*m*m*batchSize);
	cudaMalloc ((void**)&d_C, sizeof(double)*m*m*batchSize);
	
	/* Copy A to device */
	cudaMemcpy(d_A, A, sizeof(double)*m*m*batchSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeof(double)*m*m*batchSize, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	 
	/* Call multiplication routine of choice */
	//~ double time_gpu = multiplication_stream(d_A, d_B, d_C, m, batchSize);
	double time_gpu = multiplication_batch(d_A, d_B, d_C, m, batchSize);
	
	std::cout<<"Time gpu: "<< time_gpu << " s" << std::endl;
	
	/* Code C from device */
	cudaMemcpy(C, d_C, sizeof(double) * m * m * batchSize, cudaMemcpyDeviceToHost);
	
	/* CPU BENCHMARKING */
	
	/* Declare CPU arrays */
	double *A_CPU = new double [m*m*batchSize];
	double *B_CPU = new double [m*m*batchSize];
	double *C_CPU = new double [m*m*batchSize];
	
	/* Fill up array A and B with matrix elements */
	fillMatrices(A_CPU, m, batchSize);
	fillMatrices(B_CPU, m, batchSize);
	
	auto start = std::chrono::system_clock::now();
	
	/* Multipli A and B using MKL LAPACK */
	/* Arg 1 is A */
	/* Arg 1 is B */
	/* Arg 1 is C */
	/* Arg 4, 5, and 6 are the dimensions (all m) */
	for (int i=0; i<batchSize; i++){
		dot_MM(&A_CPU[i*m*m], &B_CPU[i*m*m], &C_CPU[i*m*m], m, m, m);
	}
	
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> time_raw = end-start;
	double time_cpu = time_raw.count();
	
	std::cout<<"Time cpu: "<< time_cpu << " s" << std::endl;
	
	//~ for (int M=0; M<batchSize; M++){
		//~ for (int i=0; i<m; i++){
			//~ for (int j=0; j<m; j++){
				//~ std::cout << V_CPU[M*m*m + i*m+j] << std::endl;
			//~ }
		//~ }
	//~ }
	
	/* Print any numerically big differences in eigenvalues between GPU and CPU*/
	double maxDiff = 0;
	for (int M=0; M<batchSize; M++){
		for (int i=0; i<m*m; i++){
			double diff = abs(C_CPU[M*m*m + i] - C[M*m*m + i]);
			if (diff > maxDiff){
				maxDiff = diff;
				//~ std::cout << M << " " << i << " " << diff << std::endl;
			}
		}
	}
	
	std::cout << std::endl;
	std::cout << "Max diff: " << maxDiff << std::endl;

	/* free resources */
	if (d_A) cudaFree(d_A);
	if (d_B) cudaFree(d_B);
	if (d_C) cudaFree(d_C);

	delete [] A;
	delete [] B;
	delete [] C;

	delete [] A_CPU;
	delete [] B_CPU;
	delete [] C_CPU;

	cudaDeviceReset();

	return 0;
}

