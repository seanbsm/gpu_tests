
#include "device.cuh"

__global__
void fill_array(double *d_A){
	for (int i=0; i<1000; i++){
		d_A[i] = i;
	}
}

__global__
void fill_c_array(thrust::complex<double> *d_A){
	for (int i=0; i<1000; i++){
		d_A[i] = i;
	}
}


thrust::device_vector<thrust::complex<double>> d_vec_A;


void get_cuda_array_ptr(double **array_ptr){
	
	double *d_A  = NULL;
	
	cudaMalloc ((void**)&d_A, sizeof(double) * 1000);
	
	fill_array<<<1,1>>>(d_A);
	
	*array_ptr = d_A;
}

void get_cuda_c_array_ptr(std::complex<double> **array_ptr){
	
	thrust::complex<double> *d_A  = NULL;
	
	cudaMalloc ((void**)&d_A, sizeof(thrust::complex<double>) * 1000);
	
	fill_c_array<<<1,1>>>(d_A);
	
	*array_ptr = reinterpret_cast<std::complex<double>*>(d_A);
}

void get_cuda_thrust_vector_ptr(std::complex<double> **array_ptr){
	
	d_vec_A.resize(1000);
	
	fill_c_array<<<1,1>>>(thrust::raw_pointer_cast(&d_vec_A[0]));
	
	*array_ptr = reinterpret_cast<std::complex<double>*>( thrust::raw_pointer_cast(&d_vec_A[0]) );
}

void use_cuda_array_and_check(double *array_ptr){
	
	double *h_A = new double [1000];
	
	
	cudaMemcpy(h_A, array_ptr, sizeof(double) * 1000, cudaMemcpyDeviceToHost);
	
	for (int i=0; i<10; i++){
		std::cout << h_A[i] << std::endl;
	} 
}

void use_cuda_c_array_and_check(std::complex<double> *array_ptr){
	
	std::complex<double> *h_A = new std::complex<double> [1000];
	
	cudaMemcpy(h_A, array_ptr, sizeof(std::complex<double>) * 1000, cudaMemcpyDeviceToHost);
	
	for (int i=0; i<10; i++){
		std::cout << h_A[i] << std::endl;
	} 
}
