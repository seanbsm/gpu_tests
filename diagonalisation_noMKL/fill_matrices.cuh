#ifndef FILL_MATRICES_CUH
#define FILL_MATRICES_CUH

void fillSymmetricMatrix_full(float *A, int m);
void fillSymmetricMatrices_full(float *h_A, int m, int batchSize);

void fillSymmetricMatrix_full(double *A, int m);
void fillSymmetricMatrices_full(double *h_A, int m, int batchSize);

void fillSymmetricMatrix_symm(float  *A, int m);
void fillSymmetricMatrices_symm(float  *h_A, int m, int batchSize);

void fillSymmetricMatrix_symm(double *A, int m);
void fillSymmetricMatrices_symm(double *h_A, int m, int batchSize);

#endif // FILL_MATRICES_CUH


