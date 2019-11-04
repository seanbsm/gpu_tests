
#include "kernel_functions.cuh"

__global__
void EigKernel_single(float  *d_A, float  *d_W, int m, int batchSize, cusolverDnHandle_t handle, int lWork, float  *dWork, int *dInfo){
	
	unsigned matIdx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(matIdx < batchSize) {
		float  *tMatrix = d_A + m*m*matIdx;
    	float  *tResults = d_W + (m+1)*m*matIdx;  // room for eigenvalues and vectors
		
		//~ cusolverDnSsyevd(handle,
						 //~ CUSOLVER_EIG_MODE_VECTOR,
						 //~ CUBLAS_FILL_MODE_UPPER,
						 //~ m,
						 //~ tMatrix,
						 //~ m,
						 //~ tResults,
						 //~ dWork,
						 //~ lWork,
						 //~ dInfo);
	}
}

__global__
void EigKernel_double(double *d_A, double *d_W, int m, int batchSize, cusolverDnHandle_t handle, int lWork, double *dWork, int *dInfo){
	
	unsigned matIdx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(matIdx < batchSize) {
		double *tMatrix = d_A + m*m*matIdx;
    	double *tResults = d_W + (m+1)*m*matIdx;  // room for eigenvalues and vectors
		
		//~ cusolverDnDsyevd(handle,
						 //~ CUSOLVER_EIG_MODE_VECTOR,
						 //~ CUBLAS_FILL_MODE_UPPER,
						 //~ m,
						 //~ tMatrix,
						 //~ m,
						 //~ tResults,
						 //~ dWork,
						 //~ lWork,
						 //~ dInfo);
	}
}

/* This function diagonalises by calling dsysev in parallel using kernels */
double diagonalise_kernel(float  *d_A, float  *d_W, int m, int batchSize){
	
	cusolverDnHandle_t cusolverH;
	cusolverDnCreate(&cusolverH);
	
	int 	l_work = 0;
	float  *d_work = NULL;
	int    *d_info = NULL;
	
	cudaMalloc((void **)d_info, sizeof(int));
	
	cusolverDnSsyevd_bufferSize(cusolverH,
									CUSOLVER_EIG_MODE_VECTOR,
									CUBLAS_FILL_MODE_UPPER,
									m,
									d_A,
									m,
									d_W,
									&l_work);
	cudaMalloc((void **)&d_work, sizeof(float)*l_work);
	
	int threads_per_block = 32;
	int block_count = (batchSize + threads_per_block-1) / threads_per_block;
	
	EigKernel_single<<<block_count, threads_per_block>>>(d_A, d_W, m, batchSize, cusolverH, l_work, d_work, d_info);
												  
	return 0;
}

/* This function diagonalises by calling dsysev in parallel using kernels */
double diagonalise_kernel(double *d_A, double *d_W, int m, int batchSize){
	
	cusolverDnHandle_t cusolverH;
	cusolverDnCreate(&cusolverH);
	
	int 	l_work = 0;
	double *d_work = NULL;
	int    *d_info = NULL;
	
	cudaMalloc((void **)d_info, sizeof(int));
	
	cusolverDnDsyevd_bufferSize(cusolverH,
									CUSOLVER_EIG_MODE_VECTOR,
									CUBLAS_FILL_MODE_UPPER,
									m,
									d_A,
									m,
									d_W,
									&l_work);
	cudaMalloc((void **)&d_work, sizeof(double)*l_work);
	
	int threads_per_block = 32;
	int block_count = (batchSize + threads_per_block-1) / threads_per_block;
	
	EigKernel_double<<<block_count, threads_per_block>>>(d_A, d_W, m, batchSize, cusolverH, l_work, d_work, d_info);
												  
	return 0;
}
