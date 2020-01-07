
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
#include <algorithm>

#include <cuda_runtime.h>
#include <cusolverDn.h>

/* Time-keeping modules */
#include <chrono>
#include <ctime>

/* Type definitions */
#include "typeDefs.h"

/* CUDA functions */
#include "dsyevd_stream_functions.cuh"
#include "dsyevj_stream_functions.cuh"
#include "dsyevj_batch_functions.cuh"

/* Self-written Jacobi algorithm */
#include "jacobi_kernel_function_new.cuh"

/* Additional functions */
#include "fill_matrices.cuh"

void printMatrix(int m, int n, const floatType*A, const char* name){
	for(int row = 0 ; row < m ; row++){
		for(int col = 0 ; col < n ; col++){
			floatType Areg = A[row + col*m];
			printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
		}
	}
}

int main(int argc, char*argv[]){
	
	if (argc != 3){
		std::cout << "you need to input matrix size and number, respectively" << std::endl;
	}
	
	const int N = atoi(argv[1]);
	const int m = N;
	const int batchSize = atoi(argv[2]);

	/* Declare host arrays */
	floatType *A = new floatType [m*m*batchSize];
	floatType *V = new floatType [m*m*batchSize];
	floatType *W = new floatType [m*batchSize];
	
	/* Declare device arrays */
	floatType *d_A  = NULL; /* m-by-m-by-batchSize */
	floatType *d_W  = NULL; /* m-by-batchSizee */
	
	/* Fill up array A with matrix elements */
	fillSymmetricMatrices_full(A, m, batchSize);
	
	/* Allocate A on device */
	cudaMalloc ((void**)&d_A   , sizeof(floatType) * m * m * batchSize);
	cudaMalloc ((void**)&d_W   , sizeof(floatType) * m * batchSize);
	
	/* Copy A to device */
	cudaMemcpy(d_A, A, sizeof(floatType) * m * m * batchSize, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	 
	/* Call diagonalisation routine of choice */
	//~ double time_gpu = diagonalise_kernel(d_A, d_W, m, batchSize);
	//~ double time_gpu = diagonalise_stream_syevd(d_A, d_W, m, batchSize);
	//~ double time_gpu = diagonalise_stream_syevj(d_A, d_W, m, batchSize);
	//~ double time_gpu = jacobi_kernels(d_A, d_W, m, batchSize);
	//~ double time_gpu = jacobi_kernels_parallel(d_A, d_W, m, batchSize);
	//~ double time_gpu = diagonalise_batch_QR(d_A, d_W, m, batchSize);
	double time_gpu = diagonalise_batch_syevj(d_A, d_W, m, batchSize);
	
	std::cout<<"Time gpu: "<< time_gpu << " s" << std::endl;
	
	/* Code A and W from device */
	cudaMemcpy(V, d_A, sizeof(floatType) * m * m * batchSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(W, d_W, sizeof(floatType) * m * batchSize      , cudaMemcpyDeviceToHost);
	
	//~ if (N <=6){
		for (int i=0; i<m; i++){
			std::cout << std::endl;
			std::cout << "E: " << W[i] << std::endl;
			//~ for (int j=0; j<m; j++){
				//~ std::cout << V[i*m+j] << std::endl;
			//~ }
		}
	//~ }
	
	/* free resources */
	if (d_A) cudaFree(d_A);
	if (d_W) cudaFree(d_W);

	delete [] A;
	delete [] V;
	delete [] W;

	cudaDeviceReset();

	return 0;
}
