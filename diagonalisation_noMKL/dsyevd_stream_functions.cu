
#include "dsyevd_stream_functions.cuh"

/* This function diagonalises by calling dsyevd in parallel using streaming */
double diagonalise_stream_syevd(float  *d_A, float  *d_W, int m, int batchSize){
	
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
		cusolverDnSsyevd_bufferSize(cusolverH,
									CUSOLVER_EIG_MODE_VECTOR,
									CUBLAS_FILL_MODE_UPPER,
									m,
									&d_A[i*m*m],
									m,
									&d_W[i*m],
									&l_work[i]);
	
		cudaMalloc((void **)&d_work[i], sizeof(float)*l_work[i]);
	}
	
	for(int i=0;i<NBSTREAM;i++){
		cusolverDnSetStream(cusolverH, stream[i]);
		
		cusolverDnSsyevd(cusolverH,
						 CUSOLVER_EIG_MODE_VECTOR,
						 CUBLAS_FILL_MODE_UPPER,
						 m,
						 &d_A[i*m*m],
						 m,
						 &d_W[i*m],
						 d_work[i],
						 l_work[i],
						 d_info[i]);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	
	return (double) time*1e-3;
}

/* This function diagonalises by calling dsyevd in parallel using streaming */
double diagonalise_stream_syevd(double *d_A, double *d_W, int m, int batchSize){
	
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
		cusolverDnDsyevd_bufferSize(cusolverH,
									CUSOLVER_EIG_MODE_VECTOR,
									CUBLAS_FILL_MODE_UPPER,
									m,
									&d_A[i*m*m],
									m,
									&d_W[i*m],
									&l_work[i]);
	
		cudaMalloc((void **)&d_work[i], sizeof(double)*l_work[i]);
	}
	
	for(int i=0;i<NBSTREAM;i++){
		cusolverDnSetStream(cusolverH, stream[i]);
		
		cusolverDnDsyevd(cusolverH,
						 CUSOLVER_EIG_MODE_VECTOR,
						 CUBLAS_FILL_MODE_UPPER,
						 m,
						 &d_A[i*m*m],
						 m,
						 &d_W[i*m],
						 d_work[i],
						 l_work[i],
						 d_info[i]);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	
	return (double) time*1e-3;
}
