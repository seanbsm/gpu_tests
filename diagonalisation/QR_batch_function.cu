
#include "QR_batch_function.cuh"

double diagonalise_batch_QR(float  *d_A, float  *d_W, int m, int batchSize){
}

__global__
void fill_v(double *vj, double** d_d_T_ptrs, double** d_d_A_ptrs, int m, int j, int h){
	
	int k = blockIdx.x;	// channel index
	
	if (j>k){
		vj[k] = 0;
	}
	else if (j<k){
		vj[k] = -d_d_T_ptrs[h][0] * d_d_A_ptrs[h][j*m + k];
	}
	else{
		vj[k] = -d_d_T_ptrs[h][0];
	}
}

__global__
void copy_h(double *H_copy_from, double *H_copy_to){
	
	int idx = blockIdx.x;
	
	H_copy_to[idx] = H_copy_from[idx];
}

__global__
void set_H(double *H, int m){
	
	int i = blockIdx.x;
	int j = blockIdx.y;
	
	if (i==j){
		H[m*i + j] = 1;
	}
	else{
		H[m*i + j] = 0;
	}
}

__global__
void copy_diag(double *mat, double *eig, int m){
	
	int idx = blockIdx.x;
	int chn = blockIdx.y;
	
	
	eig[m*chn + idx] = mat[m*m*chn + idx*m + idx];
}

void printMatrix(int m, int n, const double *A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

double diagonalise_batch_QR(double *d_A, double *d_W, int m, int batchSize){
	
	const double alpha = 1.;
	const double beta = 0.;
	
	const int blockSize 	= 1;//ceil(N/32);
	const dim3 block(blockSize, 1, 1); 
	const dim3 grid_make_v(m, 1, 1);
	const dim3 grid_make_H(m, m, 1);
	const dim3 grid_copy_v(m, 1, 1);
	const dim3 grid_copy_H(m*m, 1, 1);
	const dim3 grid_copy_A(m*m*batchSize, 1, 1);
	const dim3 grid_copy_diag(m, batchSize, 1);
	
	/* Copy d_A into d_A_copy */
	double *d_A_copy = NULL;
	cudaMalloc( (void**)&d_A_copy,  batchSize*m*m*sizeof(double) );
	
	cublasHandle_t cublasH;
	cublasCreate(&cublasH);
	
	cusolverDnHandle_t cusolverH;
	cusolverDnCreate(&cusolverH);
	
	int info;
	
	double *h_d_A_ptrs[batchSize];
	double *h_d_T_ptrs[batchSize];
	
	/* Create a host pointer which will point to device memory */
	for (int i=0; i<batchSize; i++){
		cudaMalloc((void**)&h_d_A_ptrs[i], sizeof(double*));
		cudaMalloc((void**)&h_d_T_ptrs[i], sizeof(double*));
		
		h_d_A_ptrs[i] = &d_A[m*m*i];
	}
	
	/* Create a device pointer which will point to device memory */
	double** d_d_A_ptrs = NULL;
	double** d_d_T_ptrs = NULL;
	/* Allocate space for pointers on device at above location */
	cudaMalloc((void**)&d_d_A_ptrs, sizeof(double*) * batchSize);
	cudaMalloc((void**)&d_d_T_ptrs, sizeof(double*) * batchSize);
	/* Copy the pointers from the host memory to the device array */
	cudaMemcpy(d_d_A_ptrs, h_d_A_ptrs, sizeof(double*) * batchSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_d_T_ptrs, h_d_T_ptrs, sizeof(double*) * batchSize, cudaMemcpyHostToDevice);
	
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	/* QR algorithm loop */
	for (int i=0; i<1; i++){
		
		copy_h<<<grid_copy_A, block>>>(d_A, d_A_copy);
		
		cublasDgeqrfBatched(cublasH, 
							m, 
							m,
							d_d_A_ptrs,  
							m, 
							d_d_T_ptrs,                                                         
							&info,
							batchSize);
		
		/* Housholder reflections */
		/* H     = H_1*H_2*H_3*...*H_m
		 * H_j   = I - tau_j*v_j*v_j^T
		 * v_j,i = 0     for i>j
		 * v_j,i = 1     for i=j
		 * v_j,i = A_i,j for i<j */
		 
		double *H   = NULL;
		double *Ht  = NULL;
		double *Hj  = NULL;
		double *vj  = NULL;
		double *vjt = NULL;
		
		cudaMalloc( (void**)&H,   m*m*sizeof(double) );
		cudaMalloc( (void**)&Ht,  m*m*sizeof(double) );
		cudaMalloc( (void**)&Hj,  m*m*sizeof(double) );
		cudaMalloc( (void**)&vj,    m*sizeof(double) );
		cudaMalloc( (void**)&vjt,   m*sizeof(double) );
		
		/* Loop through all Hj matrices */
		for (int h=0; h<batchSize; h++){
			
			/* Loop through all elements of vj */
			//~ for (int j=0; j<m; j++){
				
				//~ fill_v<<<grid_make_v, block>>>(vj, d_d_T_ptrs, d_d_A_ptrs, m, j, h);
				//~ set_H<<<grid_make_H, block>>>(Hj, m);
				
				//~ copy_h<<<grid_copy_v, block>>>(vj, vjt);
				
				//~ /* Create Hj */
				//~ cublasDgemm(cublasH,
                           //~ CUBLAS_OP_N,
                           //~ CUBLAS_OP_T,
                           //~ m, m, 1,
                           //~ &alpha,
                           //~ vj,  m,
                           //~ vjt, m,
                           //~ &alpha,
                           //~ Hj, m);
				
				//~ if (j==0){
					//~ copy_h<<<grid_copy_H, block>>>(Hj, H);
				//~ }
				//~ else{
					//~ cublasDgemm(cublasH,
								//~ CUBLAS_OP_N,
								//~ CUBLAS_OP_N,
								//~ m, m, m,
								//~ &alpha,
								//~ Hj, m,
								//~ Ht, m,
								//~ &beta,
								//~ H, m);
				//~ }
				
				//~ copy_h<<<grid_copy_H, block>>>(H, Ht);
			//~ }
			
			int *devInfo = NULL;
			int lwork = 0;
			double *d_work = NULL;
			cudaMalloc ((void**)&devInfo, sizeof(int));
			
			//~ cusolverDnDgeqrf_bufferSize(cusolverH, 
										//~ m, 
										//~ m, 
										//~ d_A, 
										//~ m, 
										//~ &lwork);
			
			//~ cudaMalloc((void**)&d_work, sizeof(double)*lwork);
			
			cusolverDnDorgqr_bufferSize(cusolverH,
										m,
										m,
										m,
										&d_A[h*m*m],
										m,
										h_d_T_ptrs[h],
										&lwork);
			cudaMalloc((void**)&d_work, sizeof(double)*lwork);
			
			cusolverDnDormqr(cusolverH,
							 CUBLAS_SIDE_RIGHT,
							 CUBLAS_OP_N,
							 m,
							 m,
							 m,
							 &d_A[h*m*m],
							 m,
							 h_d_T_ptrs[h],
							 H,
							 m,
							 d_work,
							 lwork,
							 devInfo);
			
			//~ cusolverDnDorgqr(cusolverH,
							 //~ m,
							 //~ m,
							 //~ m,
							 //~ &d_A[h*m*m],
							 //~ m,
							 //~ h_d_T_ptrs[h],
							 //~ d_work,
							 //~ lwork,
							 //~ devInfo);
			
			std::vector<double> cpu_array (m*m);
			cudaMemcpy(&(cpu_array[0]), H, sizeof(double)*m*m, cudaMemcpyDeviceToHost);	
			printMatrix(m,m, &(cpu_array[0]), m, "Q");
			
			//~ cublasDgemm(cublasH,
						//~ CUBLAS_OP_T,
						//~ CUBLAS_OP_N,
						//~ m, m, m,
						//~ &alpha,
						//~ Ht, m,
						//~ &d_A_copy[h*m*m], m,
						//~ &beta,
						//~ Hj, m);
						
			//~ cublasDgemm(cublasH,
						//~ CUBLAS_OP_N,
						//~ CUBLAS_OP_N,
						//~ m, m, m,
						//~ &alpha,
						//~ Hj, m,
						//~ H, m,
						//~ &beta,
						//~ &d_A[h*m*m], m);
		}
		
		//~ cublasDgemmBatched(cublasH,
						   //~ CUBLAS_OP_N, 
						   //~ CUBLAS_OP_N,
						   //~ m, m, m,
						   //~ &alpha,
						   //~ d_d_A_ptrs, m,
						   //~ d_d_B_ptrs, m,
						   //~ &beta,
						   //~ d_d_C_ptrs, m,, 
						   //~ batchSize);
	}
	
	copy_diag<<<grid_copy_diag, block>>>(d_A, d_W, m);
	
	std::cout << "finished function" << std::endl;
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	
	return (double) time*1e-3;
}
