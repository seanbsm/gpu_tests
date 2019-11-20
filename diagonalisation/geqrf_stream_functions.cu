
#include "geqrf_stream_functions.cuh"

double diagonalise_stream_geqrf(float  *d_A, float  *d_W, int m, int batchSize){
	
	int NBSTREAM = batchSize;
	
	cusolverDnHandle_t cusolverH;
	cublasHandle_t 	   cublasH;
	
	cusolverDnCreate(&cusolverH);
	cublasCreate(&cublasH);
	
	const double one = 1;
	
	//~ cudaStream_t 	   stream	 [NBSTREAM];
	cudaStream_t *stream = (cudaStream_t *)malloc(NBSTREAM*sizeof(cudaStream_t));
	
	int 	l_work [NBSTREAM];
	float  *d_work [NBSTREAM];
	int    *d_info [NBSTREAM];
	float  *d_tau  [NBSTREAM];
	
	for (int i=0; i<NBSTREAM; i++){
		cudaMalloc((void **)&d_info[i], sizeof(int));
		cudaMalloc ((void**)&d_tau[i],  sizeof(double)*m);
	}
	
	for (int i=0; i<NBSTREAM; i++){
		//~ cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
		cudaStreamCreate(&stream[i]);
	}
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	//~ for(int i=0;i<NBSTREAM;i++){
		//~ cusolverDnSgeqrf_bufferSize(cusolverH, 
									//~ m, 
									//~ m, 
									//~ &d_A[i*m*m], 
									//~ m, 
									//~ &l_work[i]);
	
		//~ cudaMalloc((void **)&d_work[i], sizeof(double)*l_work[i]);
	//~ }
	
	//~ for(int i=0;i<NBSTREAM;i++){
		//~ cusolverDnSetStream(cusolverH, stream[i]);
		
		//~ cusolverDnSgeqrf(cusolverH, 
						 //~ m, 
						 //~ m, 
						 //~ &d_A[i*m*m], 
						 //~ m, 
						 //~ d_tau[i*m], 
						 //~ d_work[i], 
						 //~ l_work[i], 
						 //~ d_info[i]);
	//~ }
	
	//~ for(int i=0;i<NBSTREAM;i++){
		//~ cusolverDnSetStream(cusolverH, stream[i]);
		
		//~ cusolverDnSormqr(cusolverH, 
						 //~ CUBLAS_SIDE_LEFT, 
						 //~ CUBLAS_OP_T,
						 //~ m, 
						 //~ m, 
						 //~ m, 
						 //~ &d_A[i*m*m], 
						 //~ m,
						 //~ d_tau[i*m],
						 //~ &d_B[i*m*m],
						 //~ m,
						 //~ d_work[i],
						 //~ l_work[i],
						 //~ d_info[i]);
	//~ }
	
	//~ for(int i=0;i<NBSTREAM;i++){
		//~ cusolverDnSetStream(cusolverH, stream[i]);
		
		//~ cublasStrsm(cublasH,
					//~ CUBLAS_SIDE_LEFT,
					//~ CUBLAS_FILL_MODE_UPPER,
					//~ CUBLAS_OP_N, 
					//~ CUBLAS_DIAG_NON_UNIT,
					//~ m,
					//~ m,
					//~ &one,
					//~ &d_A[i*m*m],
					//~ m,
					//~ &d_B[i*m*m],
					//~ m);
	//~ }
	
	//~ cudaEventRecord(stop, 0);
	//~ cudaEventSynchronize(stop);
	//~ float time;
	//~ cudaEventElapsedTime(&time, start, stop);
	
	//~ return (double) time*1e-3;
}

double diagonalise_stream_geqrf(double *d_A, double *d_W, int m, int batchSize){
}
