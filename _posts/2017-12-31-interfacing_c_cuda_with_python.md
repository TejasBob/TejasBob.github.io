---
layout: post
title: Hello World!
---


This article is about reusing existing c/c++ CUDA implementation in Python.

The rationale behind doing this is, doing fast prototyping in Python while CUDA does most of the heavy lifting in C/C++

We are going to use shared objects to do so.

To explain the process I'm going to perform RGB to Gray conversion using CUDA.

RGB to Gray conversion is a standard operation in Image-Processing.

Mathematically, 

gray = 0.299 * r + 0.587 * g + 0.114 * b

let us write a kernal for the rgb to gray conversion.

	#include <cuda.h>
	#include <cuda_runtime_api.h>
	#include<stdio.h>

	__global__ void cuda_gray_kernel(unsigned char *b, unsigned char *g, unsigned char *r, unsigned char *gray, size_t size)
	{
	    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	    if (idx >= size) {
	        return;
	    }
	    
	    gray[idx] = (unsigned char)(0.114f*b[idx] + 0.587f*g[idx] + 0.299f*r[idx] + 0.5);

	}


In the above code,
b,g,r are one dimensional arrays of unsigned chars holding B, G, R color values of the given image.


first we calculate the thread index and write the computed grayscale value at that index.


Now let's see the memory allocation part,


	extern "C" {
	void cuda_gray(unsigned char *b, unsigned char *g, unsigned char *r, unsigned char *gray, size_t size)
	{

	    cudaEvent_t start, stop;
	    cudaEventCreate(&start);
	    cudaEventCreate(&stop);


	    unsigned char *d_b, *d_g, *d_r, *d_gray;

	    cudaMalloc((void **)&d_b, size * sizeof(char));
	    cudaMalloc((void **)&d_g, size * sizeof(char));
	    cudaMalloc((void **)&d_r, size * sizeof(char));
	    cudaMalloc((void **)&d_gray, size * sizeof(char));

	    cudaMemcpy(d_b, b, size * sizeof(char), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_g, g, size * sizeof(char), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_r, r, size * sizeof(char), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_gray, gray, size * sizeof(char), cudaMemcpyHostToDevice);


	    cudaEventRecord(start);
	    cuda_gray_kernel <<< ceil(size / 1024.0), 1024 >>> (d_a, d_b, d_c, d_d, size);
	    cudaEventRecord(stop);
	    cudaEventSynchronize(stop);
	    float milliseconds = 0;
	    cudaEventElapsedTime(&milliseconds, start, stop);
	    printf("Time on GPU : %f msec\n", milliseconds);

	    cudaMemcpy(gray, d_gray, size * sizeof(char), cudaMemcpyDeviceToHost);

	    cudaFree(d_b);
	    cudaFree(d_g);
	    cudaFree(d_r);
	    cudaFree(d_gray);
	}


	}

In this code we allocate memory for r,g,b channel arrays on gpu. THe variables d_a, d_b, d_c are the pointers to memory on GPU (d_ in the name refers to device variable, GPU is called as Device and CPU machines are Host)

Once the file is ready, run the following command to generate the .so file.

	path_to_nvcc -Xcompiler -fPIC -shared -o cuda_gray.so cuda_gray.cu

Now let's see the python part.

	import numpy as np
	import ctypes
	import cv2
	from ctypes import *


	def get_cuda_gray():
	    dll = ctypes.CDLL('./cuda_lib.so', mode=ctypes.RTLD_GLOBAL)
	    func = dll.cuda_gray
	    func.argtypes = [POINTER(c_ubyte), POINTER(c_ubyte), POINTER(c_ubyte), POINTER(c_ubyte), c_size_t]
	    return func


	__cuda_gray = get_cuda_gray()


	def cuda_gray(b, g, r, gray, size):
	    b_p = b.ctypes.data_as(POINTER(c_ubyte))
	    g_p = g.ctypes.data_as(POINTER(c_ubyte))
	    r_p = r.ctypes.data_as(POINTER(c_ubyte))
	    gray_p = gray.ctypes.data_as(POINTER(c_ubyte))
	    __cuda_gray(b_p, g_p, r_p, gray_p, size)

Aboe code reads cuda_gray function from shared object file.

Let's use run the code

	if __name__ == '__main__':

	    img = cv2.imread('river.jpeg')
	    rows, columns = img.shape[:2]
	    b, g, r = cv2.split(img)
	    gray = np.zeros((rows * columns, 1))

	    b = np.copy(b.reshape((rows * columns, 1))).astype('uint8')
	    g = np.copy(g.reshape((rows * columns, 1))).astype('uint8')
	    r = np.copy(r.reshape((rows * columns, 1))).astype('uint8')
	    gray = np.copy(gray).astype('uint8')

	    cuda_gray(b, g, r, gray, rows * columns)
	    gray = gray.reshape(rows, columns,)
	    cv2.imwrite('gray.jpg', gray)

run as:
	python cuda_test.py