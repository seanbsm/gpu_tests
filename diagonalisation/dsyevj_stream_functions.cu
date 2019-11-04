
#include "dsyevj_stream_functions.cuh"

/* This function diagonalises by calling dsyevj in parallel using streaming */

/* This function diagonalises by calling dsyevj in parallel using streaming */
double diagonalise_stream_syevj(float *d_A, float *d_W, int m, int batchSize){
	
	
	syevjInfo_t syevj_params = NULL;
	/* configuration of syevj  */
	const float tol = 1.e-7;
	const int max_sweeps = 20;

	/* numerical results of syevj  */
	float residual = 0;
	int executed_sweeps = 0;
	/* step 2: configuration of syevj */
	cusolverDnCreateSyevjInfo(&syevj_params);

	/* default value of tolerance is machine zero */
	cusolverDnXsyevjSetTolerance(syevj_params, tol);
	
	/* default value of max. sweeps is 100 */
	cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
	
	
	int NBSTREAM = batchSize;
	
	cusolverDnHandle_t cusolverH;
	//~ cudaStream_t 	   stream	 [NBSTREAM];
	cudaStream_t *stream = (cudaStream_t *)malloc(NBSTREAM*sizeof(cudaStream_t));
	
	int 	l_work [NBSTREAM];
	float  *d_work [NBSTREAM];
	int    *d_info [NBSTREAM];
	
	
	for (int i=0; i<NBSTREAM; i++){
		cudaMalloc((void **)&d_info[i], sizeof(int));
	}
	
	for (int i=0; i<NBSTREAM; i++){
		//~ cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
		cudaStreamCreate(&stream[i]);
	}
	
	cusolverDnCreate(&cusolverH);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	for(int i=0;i<NBSTREAM;i++){
		
		cusolverDnSsyevj_bufferSize(cusolverH,
									CUSOLVER_EIG_MODE_VECTOR,
									CUBLAS_FILL_MODE_UPPER,
									m,
									&d_A[i*m*m],
									m,
									&d_W[i*m],
									&l_work[i],
									syevj_params);
	
		cudaMalloc((void **)&d_work[i], sizeof(float)*l_work[i]);
	}
	
	for(int i=0;i<NBSTREAM;i++){
		cusolverDnSetStream(cusolverH, stream[i]);
						 
		cusolverDnSsyevj(cusolverH,
						 CUSOLVER_EIG_MODE_VECTOR,
						 CUBLAS_FILL_MODE_UPPER,
						 m,
						 &d_A[i*m*m],
						 m,
						 &d_W[i*m],
						 d_work[i],
						 l_work[i],
						 d_info[i],
						 syevj_params);
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	
	return (double) time*1e-3;
}

double diagonalise_stream_syevj(double *d_A, double *d_W, int m, int batchSize){
	
	
	syevjInfo_t syevj_params = NULL;
	/* configuration of syevj  */
	const double tol = 1.e-15;
	const int max_sweeps = 20;

	/* numerical results of syevj  */
	double residual = 0;
	int executed_sweeps = 0;
	/* step 2: configuration of syevj */
	cusolverDnCreateSyevjInfo(&syevj_params);

	/* default value of tolerance is machine zero */
	cusolverDnXsyevjSetTolerance(syevj_params, tol);
	
	/* default value of max. sweeps is 100 */
	cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
	
	
	int NBSTREAM = batchSize;
	
	cusolverDnHandle_t cusolverH;
	//~ cudaStream_t 	   stream	 [NBSTREAM];
	cudaStream_t *stream = (cudaStream_t *)malloc(NBSTREAM*sizeof(cudaStream_t));
	
	int 	l_work [NBSTREAM];
	double *d_work [NBSTREAM];
	int    *d_info [NBSTREAM];
	
	
	for (int i=0; i<NBSTREAM; i++){
		cudaMalloc((void **)&d_info[i], sizeof(int));
	}
	
	for (int i=0; i<NBSTREAM; i++){
		//~ cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
		cudaStreamCreate(&stream[i]);
	}
	
	cusolverDnCreate(&cusolverH);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	for(int i=0;i<NBSTREAM;i++){
		
		cusolverDnDsyevj_bufferSize(cusolverH,
									CUSOLVER_EIG_MODE_VECTOR,
									CUBLAS_FILL_MODE_UPPER,
									m,
									&d_A[i*m*m],
									m,
									&d_W[i*m],
									&l_work[i],
									syevj_params);
	
		cudaMalloc((void **)&d_work[i], sizeof(double)*l_work[i]);
	}
	
	for(int i=0;i<NBSTREAM;i++){
		cusolverDnSetStream(cusolverH, stream[i]);
						 
		cusolverDnDsyevj(cusolverH,
						 CUSOLVER_EIG_MODE_VECTOR,
						 CUBLAS_FILL_MODE_UPPER,
						 m,
						 &d_A[i*m*m],
						 m,
						 &d_W[i*m],
						 d_work[i],
						 l_work[i],
						 d_info[i],
						 syevj_params);
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	
	return (double) time*1e-3;
}
