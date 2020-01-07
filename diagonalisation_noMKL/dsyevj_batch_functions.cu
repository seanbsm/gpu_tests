
#include "dsyevj_batch_functions.cuh"

/* Uses cuda batch routines to diagonalise using Jacobi method.
 * NOTE: DUE TO SHARED MEMORY, THIS FUNCTION IS LIMITIED TO MAX 32X32
 * MATRICES (NO LIMIT ON BATCHSIZE) */
double diagonalise_batch_syevj(float  *d_A, float  *d_W, int m, int batchSize){
	
	cusolverDnHandle_t cusolverH = NULL;
	cudaStream_t stream			 = NULL;
	syevjInfo_t syevj_params	 = NULL;
	
	int* d_info    = NULL; 		/* batchSize */
	int lwork 	   = 0;  		/* size of workspace */
	float *d_work = NULL; 		/* device workspace for syevjBatched */
	
	const float tol = 1.e-7;
	const int max_sweeps = 20;
	const int sort_eig  = 1;   /* don't sort eigenvalues */
	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute eigenvectors */
	const cublasFillMode_t  uplo = CUBLAS_FILL_MODE_UPPER;
	
	/* step 1: create cusolver handle, bind a stream  */
	cusolverDnCreate(&cusolverH);
	
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cusolverDnSetStream(cusolverH, stream);
	
	/* step 2: configuration of syevj */
	cusolverDnCreateSyevjInfo(&syevj_params);
	
	/* default value of tolerance is machine zero */
	cusolverDnXsyevjSetTolerance(syevj_params, tol);

	/* default value of max. sweeps is 100 */
	cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
	
	/* disable sorting */
	cusolverDnXsyevjSetSortEig( syevj_params, sort_eig);
	
	cudaMalloc ((void**)&d_info, sizeof(int) * batchSize);
	
	//~ auto start = std::chrono::system_clock::now();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	//~ cudaDeviceSynchronize();
	
	/* step 4: query working space of syevjBatched */
	cusolverDnSsyevjBatched_bufferSize(cusolverH,
									   jobz,
									   uplo,
									   m,
									   d_A,
									   m,
									   d_W,
									   &lwork,
									   syevj_params,
									   batchSize);
	
	cudaMalloc((void**)&d_work, sizeof(float)*lwork);
	
	/* step 5: compute spectrum of A0 and A1 */
	cusolverDnSsyevjBatched(cusolverH,
							jobz,
							uplo,
							m,
							d_A,
							m,
							d_W,
							d_work,
							lwork,
							d_info,
							syevj_params,
							batchSize);
	/* Without this synchronisation I get crazy eigenvalues in my print test */
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	
	//~ if (d_info ) cudaFree(d_info);
	//~ if (d_work ) cudaFree(d_work);

	//~ if (cusolverH) cusolverDnDestroy(cusolverH);
	//~ if (stream      ) cudaStreamDestroy(stream);
	//~ if (syevj_params) cusolverDnDestroySyevjInfo(syevj_params);

	return (double) time*1e-3;
}

/* Uses cuda batch routines to diagonalise using Jacobi method.
 * NOTE: DUE TO SHARED MEMORY, THIS FUNCTION IS LIMITIED TO MAX 32X32
 * MATRICES (NO LIMIT ON BATCHSIZE) */
double diagonalise_batch_syevj(double *d_A, double *d_W, int m, int batchSize){
	
	cusolverDnHandle_t cusolverH = NULL;
	cudaStream_t stream			 = NULL;
	syevjInfo_t syevj_params	 = NULL;
	
	int* d_info    = NULL; 		/* batchSize */
	int lwork 	   = 0;  		/* size of workspace */
	double *d_work = NULL; 		/* device workspace for syevjBatched */
	
	const double tol = 1.e-15;
	const int max_sweeps = 20;
	const int sort_eig  = 1;   /* don't sort eigenvalues */
	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute eigenvectors */
	const cublasFillMode_t  uplo = CUBLAS_FILL_MODE_UPPER;
	
	/* step 1: create cusolver handle, bind a stream  */
	cusolverDnCreate(&cusolverH);
	
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cusolverDnSetStream(cusolverH, stream);
	
	/* step 2: configuration of syevj */
	cusolverDnCreateSyevjInfo(&syevj_params);
	
	/* default value of tolerance is machine zero */
	cusolverDnXsyevjSetTolerance(syevj_params, tol);

	/* default value of max. sweeps is 100 */
	cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
	
	/* disable sorting */
	cusolverDnXsyevjSetSortEig( syevj_params, sort_eig);
	
	cudaMalloc ((void**)&d_info, sizeof(int) * batchSize);
	
	//~ auto start = std::chrono::system_clock::now();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	//~ cudaDeviceSynchronize();
	
	/* step 4: query working space of syevjBatched */
	cusolverDnDsyevjBatched_bufferSize(cusolverH,
									   jobz,
									   uplo,
									   m,
									   d_A,
									   m,
									   d_W,
									   &lwork,
									   syevj_params,
									   batchSize);
	
	cudaMalloc((void**)&d_work, sizeof(double)*lwork);
	
	/* step 5: compute spectrum of A0 and A1 */
	cusolverDnDsyevjBatched(cusolverH,
							jobz,
							uplo,
							m,
							d_A,
							m,
							d_W,
							d_work,
							lwork,
							d_info,
							syevj_params,
							batchSize);
	/* Without this synchronisation I get crazy eigenvalues in my print test */
	cudaDeviceSynchronize();
	
	//~ auto end = std::chrono::system_clock::now();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	
	//~ if (d_info ) cudaFree(d_info);
	//~ if (d_work ) cudaFree(d_work);

	//~ if (cusolverH) cusolverDnDestroy(cusolverH);
	//~ if (stream      ) cudaStreamDestroy(stream);
	//~ if (syevj_params) cusolverDnDestroySyevjInfo(syevj_params);
	
	return (double) time*1e-3;
}
