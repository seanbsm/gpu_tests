#ifndef DEVICE_CUH
#define DEVICE_CUH

#include <iostream>
#include <complex>

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

void get_cuda_array_ptr(double **array_ptr);
void get_cuda_c_array_ptr(std::complex<double> **array_ptr);
void get_cuda_thrust_vector_ptr(std::complex<double> **array_ptr);

void use_cuda_array_and_check(double *array_ptr);
void use_cuda_c_array_and_check(std::complex<double> *array_ptr);

#endif // DEVICE_CUH
