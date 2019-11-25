
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
//~ #include <cusolverDn.h>

/* Time-keeping modules */
#include <chrono>
#include <ctime>

/* Type definitions */
#include "typeDefs.h"

/* CUDA functions */
#include "eigenFinder.h"

/* MAGMA functions */
//~ #include "magma_dsyev_batch_functions.cuh"

/* Self-improvised QR algorithm */
//~ #include "QR_batch_function.cuh"

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
	double time_gpu = jacobi_kernels_parallel(d_A, d_W, m, batchSize);
	//~ double time_gpu = diagonalise_batch_QR(d_A, d_W, m, batchSize);
	//~ double time_gpu = diagonalise_batch_syevj(d_A, d_W, m, batchSize);
	
	std::cout<<"Time gpu: "<< time_gpu << " s" << std::endl;
	
	/* Code A and W from device */
	cudaMemcpy(V, d_A, sizeof(floatType) * m * m * batchSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(W, d_W, sizeof(floatType) * m * batchSize      , cudaMemcpyDeviceToHost);
	
	/* CPU BENCHMARKING */
	/* Symmetric matrix size */
	int matSize = m*(m+1)/2;
	
	/* Declare CPU arrays */
	double *A_CPU = new double [matSize*batchSize];
	double *V_CPU = new double [m*m*batchSize];
	double *W_CPU = new double [m*batchSize];
	
	/* Fill up array A with matrix elements */
	fillSymmetricMatrices_symm(A_CPU, m, batchSize);
	
	auto start = std::chrono::system_clock::now();
	
	/* Diagonalise A using MKL LAPACK */
	/* Arg 1 is A and must be upper-triangle */
	/* Arg 2 are the eigenvalues */
	/* Arg 3 are the eigenvectors */
	/* Arg 4 is the dimension */
	for (int i=0; i<batchSize; i++){
		findEigenReal(&A_CPU[i*matSize], &W_CPU[m*i], &V_CPU[i*m*m], m);
	}
	
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> time_raw = end-start;
	double time_cpu = time_raw.count();
	
	std::cout<<"Time cpu: "<< time_cpu << " s" << std::endl;
	
	//~ if (N <=6){
		//~ for (int i=0; i<m; i++){
			//~ std::cout << std::endl;
			//~ std::cout << "E: " << W[i] << std::endl;
			//~ for (int j=0; j<m; j++){
				//~ std::cout << V[i*m+j] << std::endl;
			//~ }
		//~ }
	//~ }
	
	/* Print any numerically big differences in eigenvalues between GPU and CPU*/
	floatType maxDiff = 0;
	int max_h, max_i;
	for (int M=0; M<batchSize; M++){
		
		std::vector<floatType> eigenVals_gpu (m);
		for (int i=0; i<m; i++){
			eigenVals_gpu[i] = W[M*m + i];
		}
		std::sort(eigenVals_gpu.begin(), eigenVals_gpu.end());
		
		for (int i=0; i<m; i++){
		
			//~ std::cout << M << " " << i << " " << eigenVals_gpu[i] << std::endl;
			//~ std::cout << M << " " << i << " " << W_CPU[M*m+i] << "\n" << std::endl;
			
			floatType diff = abs(W_CPU[M*m + i] - eigenVals_gpu[i]);
			if (diff > maxDiff){
				maxDiff = diff;
				max_h = M;
				max_i = i;
				//~ std::cout << M << " " << i << " " << diff << std::endl;
			}
		}
	}
	
	std::cout << std::endl;
	std::cout << "Max diff: " << maxDiff << std::endl;
	std::cout << "h: " << max_h << " i: " << max_i << std::endl; 

	/* free resources */
	if (d_A) cudaFree(d_A);
	if (d_W) cudaFree(d_W);

	delete [] A;
	delete [] V;
	delete [] W;

	delete [] A_CPU;
	delete [] V_CPU;
	delete [] W_CPU;

	cudaDeviceReset();

	return 0;
}
