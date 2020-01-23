
#include <iostream>
#include <complex>

#include "device.cuh"

int main(int argc, char*argv[]){
	
	
	double *array_ptr;
	std::complex<double>* c_array_ptr;
	std::complex<double>* c_vector_ptr;
	
	get_cuda_array_ptr(&array_ptr);
	use_cuda_array_and_check(array_ptr);
	
	std::cout << std::endl;
	
	get_cuda_c_array_ptr(&c_array_ptr);
	use_cuda_c_array_and_check(c_array_ptr);
	
	std::cout << std::endl;
	
	get_cuda_thrust_vector_ptr(&c_vector_ptr);
	use_cuda_c_array_and_check(c_vector_ptr);
	
	return 0;
}
