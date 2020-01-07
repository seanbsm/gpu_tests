
#include "fill_matrices.cuh"

/* ################### FULL MATRICES ################### */ 
/* Meant for full symmetric mxm matrices */
void fillSymmetricMatrix_full(float *A, int m){
	for (int i=0; i<m; i++){
		for (int j=0; j<m; j++){
			//~ A[i + j*m] = (i + j*j) * sqrt((float)(i + j));
			A[i + j*m]= (i + j) * sqrt(i + j) / 1e3;
		}
	}
}
void fillSymmetricMatrices_full(float *h_A, int m, int batchSize){
	for (int b=0; b<batchSize; b++){
		fillSymmetricMatrix_full(&h_A[b*m*m], m);
	}
}

/* Meant for full symmetric mxm matrices */
void fillSymmetricMatrix_full(double *A, int m){
	for (int i=0; i<m; i++){
		for (int j=0; j<m; j++){
			//~ A[i + j*m] = (i + j*j) * sqrt((double)(i + j));
			A[i + j*m] = (i + j) * sqrt(i + j) / 1e3;
		}
	}
}
void fillSymmetricMatrices_full(double *h_A, int m, int batchSize){
	for (int b=0; b<batchSize; b++){
		fillSymmetricMatrix_full(&h_A[b*m*m], m);
	}
}

/* ################### SYMM MATRICES ################### */ 

/* Meant for symmetric mxm matrices stored as upper triangular*/
void fillSymmetricMatrix_symm(float  *A, int m){
	for (int i=0; i<m; i++){
		for (int j=i; j<m; j++){
			
			int idx_ij = m*(m-1)/2 - (m-i)*(m-i-1)/2 + j;
			
			//~ A[idx_ij] = (i + j*j) * sqrt((float)(i + j));
			A[idx_ij]= (i + j) * sqrt(i + j) / 1e3;
		}
	}
}
void fillSymmetricMatrices_symm(float  *h_A, int m, int batchSize){
	int matSize = m*(m+1)/2;
	for (int b=0; b<batchSize; b++){
		fillSymmetricMatrix_symm(&h_A[b*matSize], m);
	}
}

/* Meant for symmetric mxm matrices stored as upper triangular*/
void fillSymmetricMatrix_symm(double *A, int m){
	for (int i=0; i<m; i++){
		for (int j=i; j<m; j++){
			
			int idx_ij = m*(m-1)/2 - (m-i)*(m-i-1)/2 + j;
			
			//~ A[idx_ij] = (i + j*j) * sqrt((double)(i + j));
			A[idx_ij]= (i + j) * sqrt(i + j) / 1e3;
		}
	}
}
void fillSymmetricMatrices_symm(double *h_A, int m, int batchSize){
	int matSize = m*(m+1)/2;
	for (int b=0; b<batchSize; b++){
		fillSymmetricMatrix_symm(&h_A[b*matSize], m);
	}
}
