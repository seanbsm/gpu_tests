
#include "jacobi_kernel_function_new.cuh"

/* Symmetric 2-by-2 Schur Decomposition */
__device__
void sym_Schur2(floatType *A, floatType *c, floatType *s, int p, int q, int n){
	
	floatType tau = 0;
	floatType t   = 0;
	
	if ( A[n*p + q]!=0. ){
		tau = (A[n*q + q] - A[n*p + p]) / (2.*A[n*p + q]);
		
		if (tau>=0){
			t =  1. / (tau + sqrt(1+tau*tau));
		}
		else{
			t = -1. / (-tau + sqrt(1+tau*tau));
		}
		
		*c = 1 / sqrt(1+t*t);
		*s = t*(*c);
	}
	else{
		*c = 1;
		*s = 0;
	}
}

/* Symmetric 2-by-2 Schur Decomposition */
__global__
void sym_Schur2_all(floatType *d_A, floatType *d_c, floatType *d_s, int *d_top, int *d_bot, int n){
	
	int h = blockIdx.x;
	int k = threadIdx.x;
	
	floatType *A = &d_A[h*n*n];
	floatType *c = &d_c[k + h*n/2];
	floatType *s = &d_s[k + h*n/2];
	
	int tk = d_top[k];
	int bk = d_bot[k];
	int p = (tk<bk)*tk + (tk>bk)*bk;
	int q = (tk>bk)*tk + (tk<bk)*bk;
	
	sym_Schur2(A, c, s, p, q, n);
}

__global__
void Jacobi_parallel_row_rot(floatType *d_A, floatType *d_V, floatType *d_c, floatType *d_s, int *d_top, int *d_bot, int n){
	
	int h = blockIdx.x;
	int K = threadIdx.x;
	
	int k = K / (n-1);
	int i = K % (n-1);
	
	/* Only usage of h and k */
	floatType *A = &d_A[h*n*n];
	int tk 		 = d_top[k];
	int bk 		 = d_bot[k];
	
	floatType *c_set  = &d_c[h*n/2];
	floatType *s_set  = &d_s[h*n/2];
	floatType c = c_set[k];
	floatType s = s_set[k];
	
	/* Set p to the smallest of tk and bk */
	int p = (tk<bk)*tk + (tk>bk)*bk;
	
	/* Set q to the largest of tk and bk */
	int q = (tk>bk)*tk + (tk<bk)*bk;
	
	floatType Api, Aqi, App, Apq, Aqp, Aqq;
	
	if (i>=p){
		i += 1;
		
		if (i>=q){
			i += 1;
		}
	}
	
	/* TEMP SOLUTION */
	if (i==n){
		App = A[n*p + p];
		Apq = A[n*p + q];
		Aqp = A[n*q + p];
		Aqq = A[n*q + q];
		
		A[n*p + p] = c*c*App - c*s*(Apq + Aqp) + s*s*Aqq;
		A[n*q + q] = s*s*App + c*s*(Apq + Aqp) + c*c*Aqq;
		
		A[n*p + q] = 0;
		A[n*q + p] = 0;
	}
	else{
		Api = A[n*p + i];
		Aqi = A[n*q + i];
		
		A[n*p + i] = c*Api - s*Aqi;
		A[n*q + i] = c*Aqi + s*Api;
	}
	
	//~ /* Update A
	 //~ * 3 loops are actually slower than the if-test for many small matrices */
	//~ for (int i=0; i<n; i++){
		//~ if (i!=p && i!=q){
			//~ Api = A[n*p + i];
			//~ Aqi = A[n*q + i];
			
			//~ A[n*p + i] = c*Api - s*Aqi;
			//~ A[n*q + i] = c*Aqi + s*Api;
		//~ }
	//~ }
	
	//~ App = A[n*p + p];
	//~ Apq = A[n*p + q];
	//~ Aqp = A[n*q + p];
	//~ Aqq = A[n*q + q];
	
	//~ A[n*p + p] = c*c*App - c*s*(Apq + Aqp) + s*s*Aqq;
	//~ A[n*q + q] = s*s*App + c*s*(Apq + Aqp) + c*c*Aqq;
	
	//~ A[n*p + q] = 0;
	//~ A[n*q + p] = 0;
}

__global__
void Jacobi_parallel_col_rot(floatType *d_A, floatType *d_V, floatType *d_c, floatType *d_s, int *d_top, int *d_bot, int n){
	int h = blockIdx.x;
	int K = threadIdx.x;
	
	int k = K / (n/2 - 1);
	int i = K % (n/2 - 1);
	
	if (i>=k){
		i += 1;
	}
	
	/* Only usage of h and k */
	floatType *A 	   = &d_A[h*n*n];
	int tk 		   = d_top[k];
	int bk 		   = d_bot[k];
	
	floatType *c_set  = &d_c[h*n/2];
	floatType *s_set  = &d_s[h*n/2];
	
	/* Set p to the smallest of tk and bk */
	int p = (tk<bk)*tk + (tk>bk)*bk;
	/* Set q to the largest of tk and bk */
	int q = (tk>bk)*tk + (tk<bk)*bk;
	
	int ti = d_top[i];
	int bi = d_bot[i];
	
	int p_i = (ti<bi)*ti + (ti>bi)*bi;
	int q_i = (ti>bi)*ti + (ti<bi)*bi;
	
	floatType c = c_set[i];
	floatType s = s_set[i];
	
	floatType Ap_pi = A[n*p + p_i];
	floatType Aq_pi = A[n*q + p_i];
	
	floatType Ap_qi = A[n*p + q_i];
	floatType Aq_qi = A[n*q + q_i];
	
	A[n*p + p_i] = c*Ap_pi - s*Ap_qi;
	A[n*q + p_i] = c*Aq_pi - s*Aq_qi;
	
	A[n*p + q_i] = c*Ap_qi + s*Ap_pi;
	A[n*q + q_i] = c*Aq_qi + s*Aq_pi;
}

__global__
void update_A_prev(floatType *d_A, floatType *d_A_prev){
	int idx = threadIdx.x;
	
	d_A_prev[idx] = d_A[idx];
}

__global__
void rotational_sets_copy(int *top_new, int *bot_new, int *top, int *bot){
	int k = threadIdx.x;
	
	top_new[k] = top[k];
	bot_new[k] = bot[k];
}

__global__
void rotational_sets(int *top_new, int *bot_new, int *top, int *bot, int N){
	int k = threadIdx.x;
	int m = N/2;
	
	if (k==0){
		top_new[0] = 0;
	}
	else if (k==1){
		top_new[k] = bot[0];
	}
	else{
		top_new[k] = top[k-1];
	}
	
	if (k==m-1){
		bot_new[k] = top[k];
	}
	else{
		bot_new[k] = bot[k+1];
	}
}

double jacobi_kernels_parallel(floatType *d_A, floatType *d_W, int m, int batchSize){
	
	int nBlocks = batchSize;
	int nThreads = m/2;
	
	//~ const floatType eps = 1e-10;
	int maxIter = m+1;
	int m_half = m/2;
	
	int top [m_half];
	int bot [m_half];
	/* Initialise top and bottom indices */
	for (int i=0; i<m_half; i++){
		top[i] = 2*i;
		bot[i] = 2*i+1;
	}
	
	/* Declare rotation set device arrays */
	int *d_top = NULL; int *d_top_temp = NULL;
	int *d_bot = NULL; int *d_bot_temp = NULL;
	
	/* Allocate rotation set device arrays */
	cudaMalloc( (void**)&d_top, m_half*sizeof(int) );
	cudaMalloc( (void**)&d_bot, m_half*sizeof(int) );
	cudaMalloc( (void**)&d_top_temp, m_half*sizeof(int) );
	cudaMalloc( (void**)&d_bot_temp, m_half*sizeof(int) );
	
	/* Copy rotation set host arrays to corresponding device arrays */
	cudaMemcpy(d_top, top, sizeof(int) * m_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bot, bot, sizeof(int) * m_half, cudaMemcpyHostToDevice);
	
	/* Declare rotation angle-set device arrays */
	floatType *d_c = NULL;
	floatType *d_s = NULL;
	cudaMalloc( (void**)&d_c,  batchSize*m_half*sizeof(floatType) );
	cudaMalloc( (void**)&d_s,  batchSize*m_half*sizeof(floatType) );
	
	floatType *d_V = NULL;
	cudaMalloc( (void**)&d_V,  batchSize*m*m*sizeof(floatType) );
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	/* Loop over number of rotations */
	for (int iter=0; iter<maxIter; iter++){
		
		/* Loop over rotation sets */
		for (int j=0; j<m-1; j++){
			/* Calculate all rotation angles before we start rotating */
			sym_Schur2_all<<<nBlocks, nThreads>>>(d_A, d_c, d_s, d_top, d_bot, m);
			
			//~ for (int k=0; k<nThreads; k++){
				//~ Jacobi_parallel_algo<<<nBlocks, 1>>>(d_A, d_V, d_c, d_s, &d_top[k], &d_bot[k], m);
			//~ }
			
			Jacobi_parallel_row_rot<<<nBlocks, nThreads * (m-1)>>>(d_A, d_V, d_c, d_s, d_top, d_bot, m);
			Jacobi_parallel_col_rot<<<nBlocks, nThreads*(nThreads-1)>>>(d_A, d_V, d_c, d_s, d_top, d_bot, m);
			
			
			rotational_sets_copy<<<1, nThreads>>>(d_top_temp, d_bot_temp, d_top, d_bot);
			rotational_sets<<<1, nThreads>>>(d_top, d_bot, d_top_temp, d_bot_temp, m);
		}
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	
	floatType A [m*m*batchSize];
	floatType V [m*m*batchSize];
	floatType W [m*batchSize];
	
	cudaMemcpy(A, d_A, sizeof(floatType) * m*m*batchSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(V, d_V, sizeof(floatType) * m*m*batchSize, cudaMemcpyDeviceToHost);
	
	for (int h=0; h<batchSize; h++){
		for (int i=0; i<m; i++){
			W[m*h + i] = A[m*m*h + m*i + i];
		}
		for (int i=0; i<m*m; i++){
			A[m*m*h + i] = V[m*m*h + i];
		}
	}
	
	cudaMemcpy(d_A, A, sizeof(floatType) * m*m*batchSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_W, W, sizeof(floatType) * m*batchSize, cudaMemcpyHostToDevice);
	
	return (double) time*1e-3;
}
