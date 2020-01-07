
#include "jacobi_kernel_function_new.cuh"

/* Symmetric 2-by-2 Schur decomposition */
__global__
void sym_Schur2_all(floatType *d_A, floatType *d_c, floatType *d_s, int *d_P, int *d_Q, int n){
	
	int h = blockIdx.x;
	int k = threadIdx.x;
	
	floatType *A = &d_A[h*n*n];
	floatType *C = &d_c[h*n/2];
	floatType *S = &d_s[h*n/2];
	
	int p = d_P[k];
	int q = d_Q[k];
	
	floatType tau, t, c, s, Apq, App, Aqq;
	
	Apq = A[n*p + q];
	App = A[n*p + p];
	Aqq = A[n*q + q];
	
	if ( Apq!=0 ){
		tau = (Aqq - App) / (2.*Apq);
		
		if (tau>=0){
			t =  1. / (tau + sqrt(1+tau*tau));
		}
		else{
			t = -1. / (-tau + sqrt(1+tau*tau));
		}
		
		c = 1. / sqrt(1+t*t);
		s = t*c;
	}
	else{
		c = 1;
		s = 0;
	}
	
	C[k] = c;
	S[k] = s;
}

__global__
void Jacobi_parallel_row_rot(floatType *d_A, floatType *d_c, floatType *d_s, int *d_P, int *d_Q, int n){
	int H = blockIdx.x;
	int i = threadIdx.x;
	int h = H / (n/2);
	int k = H % (n/2);
	
	/* Only usage of h and k */
	floatType *A = &d_A[h*n*n];
	
	floatType *c_set  = &d_c[h*n/2];
	floatType *s_set  = &d_s[h*n/2];
	floatType c = c_set[k];
	floatType s = s_set[k];
	
	int p = d_P[k];
	int q = d_Q[k];
	
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
}

__global__
void Jacobi_parallel_col_rot(floatType *d_A, floatType *d_c, floatType *d_s, int *d_P, int *d_Q, int n){
	
	int H = blockIdx.x;
	int i = threadIdx.x;
	int h = H / (n/2);
	int k = H % (n/2);
	
	if (i>=k){
		i += 1;
	}
	
	/* Only usage of h and k */
	floatType *A 	   = &d_A[h*n*n];
	
	floatType *c_set  = &d_c[h*n/2];
	floatType *s_set  = &d_s[h*n/2];
	
	int p = d_P[k];
	int q = d_Q[k];
	
	int p_i = d_P[i];
	int q_i = d_Q[i];
	
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
void Jacobi_parallel_rot_shared(floatType *d_A, floatType *d_c, floatType *d_s, int *d_P, int *d_Q, int n){
	extern __shared__ floatType shared_array[];
	
	int H = blockIdx.x;
	int i = threadIdx.x;
	int h = H / (n/2);
	int k = H % (n/2);
	
	floatType *Ap = (floatType*)shared_array;
	floatType *Aq = (floatType*)&Ap[n];
	
	floatType *c_set  = &d_c[h*n/2];
	floatType *s_set  = &d_s[h*n/2];
	floatType c = c_set[k];
	floatType s = s_set[k];
	
	int p = d_P[k];
	int q = d_Q[k];
	
	floatType *Ap_glob = &d_A[h*n*n + n*p];
	floatType *Aq_glob = &d_A[h*n*n + n*q];
	
	Ap[i] = Ap_glob[i];
	__syncthreads();
	Aq[i] = Aq_glob[i];
	__syncthreads();
	
	/* ROW ROTATIONS */
	floatType Api, Aqi, App, Apq, Aqp, Aqq;
	
	if (i!=q && i!=p){
		Api = Ap[i];
		Aqi = Aq[i];
	}
	else{
		App = Ap[p];
		Apq = Ap[q];
		Aqp = Aq[p];
		Aqq = Aq[q];
	}
	
	__syncthreads();
	
	if (i!=q && i!=p){
		Ap[i] = c*Api - s*Aqi;
		Aq[i] = c*Aqi + s*Api;
	}
	else if (i==p){
		Ap[i] = c*c*App - c*s*(Apq + Aqp) + s*s*Aqq;
		Aq[i] = 0;
	}
	else{	
		Aq[i] = s*s*App + c*s*(Apq + Aqp) + c*c*Aqq;
		Ap[i] = 0;
	}
	/* END OF ROW ROTATIONS */
	
	__syncthreads();
	
	/* COLUMNS ROTATIONS */
	int p_i = d_P[i/2];
	int q_i = d_Q[i/2];
		
	c = c_set[i/2];
	s = s_set[i/2];
		
	floatType Ap_pi = Ap[p_i];
	floatType Aq_pi = Aq[p_i];
		
	floatType Ap_qi = Ap[q_i];
	floatType Aq_qi = Aq[q_i];
	
	__syncthreads();
	
	if (i/2 != k){
		if (i%2 == 0){
			Ap[p_i] = c*Ap_pi - s*Ap_qi;
			Aq[p_i] = c*Aq_pi - s*Aq_qi;
		}
		else{
			Ap[q_i] = c*Ap_qi + s*Ap_pi;
			Aq[q_i] = c*Aq_qi + s*Aq_pi;
		}
	}
	/* END OF COLUMN ROTATIONS */
	
	__syncthreads();
	Ap_glob[i] = Ap[i];
	__syncthreads();
	Aq_glob[i] = Aq[i];
}

__global__
void Jacobi_parallel_vec_rot(floatType *d_V, floatType *d_c, floatType *d_s, int *d_P, int *d_Q, int n){
	int H = blockIdx.x;
	int i = threadIdx.x;
	int h = H / (n/2);
	int k = H % (n/2);
	
	/* Only usage of h and k */
	floatType *V = &d_V[h*n*n];
	
	floatType *c_set  = &d_c[h*n/2];
	floatType *s_set  = &d_s[h*n/2];
	floatType c = c_set[k];
	floatType s = s_set[k];
	
	int p = d_P[k];
	int q = d_Q[k];
	
	floatType Viq, Vip;
	
	Vip = V[i*n + p];
	Viq = V[i*n + q];
	
	V[i*n + p] = c*Vip - s*Viq;
	V[i*n + q] = c*Viq + s*Vip;
}

__global__
void Jacobi_parallel_vec_rot_shared(floatType *d_V, floatType *d_c, floatType *d_s, int *d_P, int *d_Q, int n){
	extern __shared__ floatType shared_array[];
	
	int H = blockIdx.x;
	int i = threadIdx.x;
	int h = H / (n/2);
	int k = H % (n/2);
	
	floatType *V_p = (floatType*)shared_array;
	floatType *V_q = (floatType*)&V_p[n];
	
	/* Only usage of h and k */
	floatType *V_glob = &d_V[h*n*n];
	
	floatType *c_set  = &d_c[h*n/2];
	floatType *s_set  = &d_s[h*n/2];
	floatType c = c_set[k];
	floatType s = s_set[k];
	
	int p = d_P[k];
	int q = d_Q[k];
	
	V_p[i] = V_glob[i*n + p];
	V_q[i] = V_glob[i*n + q];
	
	floatType Viq, Vip;
	
	Vip = V_p[i];
	Viq = V_q[i];
	
	V_p[i] = c*Vip - s*Viq;
	V_q[i] = c*Viq + s*Vip;
	
	V_glob[i*n + p] = V_p[i];
	V_glob[i*n + q] = V_q[i];
}

__global__
void rotational_sets_shared_mem(int *top, int *bot, int *d_P, int *d_Q, int N){
	extern __shared__ floatType shared_array[];
	
	floatType *top_temp = (floatType*)shared_array;
	floatType *bot_temp = (floatType*)&top_temp[N/2];

	int k = threadIdx.x;
	
	/* Copy global arrays into shared memory, and sync threads */
	top_temp[k] = top[k];
	bot_temp[k] = bot[k];
	__syncthreads();
	
	/* Create new rotational set */
	if (k==0){
		top[0] = 0;
	}
	else if (k==1){
		top[k] = bot_temp[0];
	}
	else{
		top[k] = top_temp[k-1];
	}
	
	if (k==N/2-1){
		bot[k] = top_temp[k];
	}
	else{
		bot[k] = bot_temp[k+1];
	}
	
	int tk = top[k];
	int bk = bot[k];
	
	/* Set p to the smallest of tk and bk */
	d_P[k] = (tk<bk)*tk + (tk>bk)*bk;
	/* Set q to the largest of tk and bk */
	d_Q[k] = (tk>bk)*tk + (tk<bk)*bk;
}

__global__
void setEigenValues(floatType *d_A, floatType *d_W, int n){
	int h = blockIdx.x;
	int i = threadIdx.x;
	
	d_W[i + h*n] = d_A[h*n*n + i*n + i];
}

__global__
void setEigenVectors(floatType *d_A, floatType *d_V, int n){
	int H = blockIdx.x;
	int h = H / n;
	int i = H % n;
	int j = threadIdx.x;
	
	d_A[h*n*n + i*n + j] = d_V[h*n*n + i*n + j];
}

double jacobi_kernels_parallel(floatType *d_A, floatType *d_W, int m, int batchSize){
	
	int nBlocks = batchSize;
	int nThreads = m/2;
	
	//~ const floatType eps = 1e-10;
	int maxIter = 10;//m/8;//m+1;
	int m_half = m/2;
	
	int top [m_half]; int P [m_half];
	int bot [m_half]; int Q [m_half];
	/* Initialise top and bottom indices */
	for (int i=0; i<m_half; i++){
		top[i] = 2*i;
		bot[i] = 2*i+1;
	}
	for (int i=0; i<m_half; i++){
		P[i] = 2*i;
		Q[i] = 2*i+1;
	}
	
	
	/* Declare rotation set device arrays */
	int *d_top = NULL; int *d_P = NULL;
	int *d_bot = NULL; int *d_Q = NULL;
	
	cudaMalloc( (void**)&d_P, m_half*sizeof(int) );
	cudaMalloc( (void**)&d_Q, m_half*sizeof(int) );
	cudaMemcpy(d_P, P, sizeof(int) * m_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Q, Q, sizeof(int) * m_half, cudaMemcpyHostToDevice);
	
	/* Allocate rotation set device arrays */
	cudaMalloc( (void**)&d_top, m_half*sizeof(int) );
	cudaMalloc( (void**)&d_bot, m_half*sizeof(int) );
	cudaMemcpy(d_top, top, sizeof(int) * m_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bot, bot, sizeof(int) * m_half, cudaMemcpyHostToDevice);
	
	/* Declare rotation angle-set device arrays */
	floatType *d_c = NULL;
	floatType *d_s = NULL;
	cudaMalloc( (void**)&d_c,  batchSize*m_half*sizeof(floatType) );
	cudaMalloc( (void**)&d_s,  batchSize*m_half*sizeof(floatType) );
	
	
	floatType *d_V = NULL;
	cudaMalloc( (void**)&d_V,  batchSize*m*m*sizeof(floatType) );
	
	floatType *V = new floatType [m*m*batchSize];
	for (int h=0; h<batchSize; h++){
		for (int i=0; i<m; i++){
			V[h*m*m + i*m + i] = 1;
		}
	}
	cudaMemcpy(d_V, V, sizeof(floatType) * m*m*batchSize, cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	int blocks = batchSize*m/2;
	
	int threads_row = m-1;
	int threads_col = m/2 - 1;
	int threads_vec = m;
	
	std::cout << "Blocks:      " << blocks << std::endl;
	std::cout << "Row threads: " << threads_row << std::endl;
	std::cout << "Col threads: " << threads_col << "\n" << std::endl;
	
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	
	/* Loop over number of rotations */
	for (int iter=0; iter<maxIter; iter++){
		/* Loop over rotation sets */
		for (int j=0; j<m-1; j++){
			sym_Schur2_all<<<nBlocks, nThreads>>>(d_A, d_c, d_s, d_P, d_Q, m);
			
			//~ Jacobi_parallel_row_rot<<<blocks, threads_row>>>(d_A, d_c, d_s, d_P, d_Q, m);
			//~ Jacobi_parallel_col_rot<<<blocks, threads_col>>>(d_A, d_c, d_s, d_P, d_Q, m);
			Jacobi_parallel_rot_shared<<<blocks, m, 2*m*sizeof(floatType)>>>(d_A, d_c, d_s, d_P, d_Q, m);
			
			//~ Jacobi_parallel_vec_rot<<<blocks, threads_vec>>>(d_V, d_c, d_s, d_P, d_Q, m);
			Jacobi_parallel_vec_rot_shared<<<blocks, m, 2*m*sizeof(floatType)>>>(d_V, d_c, d_s, d_P, d_Q, m);
			
			rotational_sets_shared_mem<<<1, nThreads, m*sizeof(floatType)>>>(d_top, d_bot, d_P, d_Q, m);
		}
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	
	setEigenValues<<<batchSize, m>>>(d_A, d_W, m);
	setEigenVectors<<<batchSize*m, m>>>(d_A, d_V, m);
	
	delete [] V;
	
	return (double) time*1e-3;
}
