//
// Assignment 1: ParallelSine
// CSCI 415: Networking and Parallel Computation
// Spring 2017
// Name(s): Kelan Riley
// Sine implementation derived from slides here: http://15418.courses.cs.cmu.edu/spring2016/lecture/basicarch


// standard imports
#include <stdio.h>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/time.h>

// problem size (vector length) N
// remember that a vector is just a series of values that we'd like to refer to
// as one thing, so we can refer to the whole series by just saying the word
// vector
static const int N = 134215680;

// Number of terms to use when approximating sine
static const int TERMS = 6;

// need a better understanding of this algorithm for computing sine
// kernel function (CPU - Do not modify)
void sine_serial(float *input, float *output)
{
  // loop counter
  int i;

  // iterate as many times as there are numbers to work on
  for (i=0; i<N; i++) {
      // fetch ith number in the input array
      float value = input[i]; 
      // multiply the number by 3 initially
      float numer = input[i] * input[i] * input[i]; 
      int denom = 6; // 3! 
      int sign = -1; 
      // this loops TERMS number of times
      for (int j=1; j<=TERMS;j++) 
      { 
         value += sign * numer / denom; 
         numer *= input[i] * input[i]; 
         denom *= (2*j+2) * (2*j+3); 
         sign *= -1; 
      } 
      output[i] = value; 
    }
}


// kernel function (CUDA device)
// TODO: Implement your graphics kernel here. See assignment instructions for method information
// need to tell cuda that this is a kernel to run... need special syntax here...
//__global__ is the syntax for doing that, the below code will run on threads executing in the GPU
__global__ void sine_parallel(float *input, float *output) {
  // the thread id of the current thread that is running this kernel
  // threadIdx is a dim3 structure with x, y, and z fields (up to three dimensions)
  // make sure to compute the idx as the block number offset by the thread id in the block!
  int idx = blockIdx.x * 1024 + threadIdx.x;
  // fetch ith number in the input array
  float value = input[idx]; 
  // multiply the number by 3 initially
  float numer = input[idx] * input[idx] * input[idx]; 
  int denom = 6; // 3! 
  int sign = -1; 
  // this loops TERMS number of times
  for (int j=1; j<=TERMS;j++) 
  { 
    value += sign * numer / denom; 
    numer *= input[idx] * input[idx]; 
    denom *= (2*j+2) * (2*j+3); 
    sign *= -1; 
  }
  // write out the result into the output array 
  output[idx] = value;   
}

// BEGIN: timing and error checking routines (do not modify)

// Returns the current time in microseconds
long long start_timer() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}


// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, std::string name) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
        std::cout << std::setprecision(5);	
	std::cout << name << ": " << ((float) (end_time - start_time)) / (1000 * 1000) << " sec\n";
	return end_time - start_time;
}

void checkErrors(const char label[])
{
  // we need to synchronise first to catch errors due to
  // asynchroneous operations that would otherwise
  // potentially go unnoticed

  cudaError_t err;

  err = cudaThreadSynchronize();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }
}

// END: timing and error checking routines (do not modify)



int main (int argc, char **argv)
{
  // first I'm going to save the total number of bytes this array takes up in a variable
  int total_array_bytes = N * sizeof(float);
  int half_size = N / 2;
  int half_array_bytes = half_size * sizeof(float);

  //BEGIN: CPU implementation (do not modify)
  float *h_cpu_result = (float*)malloc(N*sizeof(float));
  float *h_input = (float*)malloc(N*sizeof(float));
  //Initialize data on CPU
  int i;
  for (i=0; i<N; i++)
  {
    h_input[i] = 0.1f * i;
  }

  //Execute and time the CPU version
  long long CPU_start_time = start_timer();
  sine_serial(h_input, h_cpu_result);
  long long CPU_time = stop_timer(CPU_start_time, "\nCPU Run Time");
  //END: CPU implementation (do not modify)


  //TODO: Prepare and run your kernel, make sure to copy your results back into h_gpu_result and display your timing results
  // allocating the results array on the host (cpu)
  float *h_gpu_result = (float*)malloc(total_array_bytes);

  // declare 4 pointers (because 2 devices) to memory on the GPU
  float *d_in_1;
  float *d_out_1;
  float *d_out_2;
  float *d_in_2;
 
  // explicitly set which device is being used 
  cudaSetDevice(0);
  // insert some timing code now
  long long GPU_memory_allocation_start_time = start_timer(); 
  // now actually allocate GPU memory for input and output
  cudaMalloc((void **) &d_in_1, half_array_bytes); 
   
  cudaMalloc((void **) &d_out_1, half_array_bytes);

  // using the second device because why not
  cudaSetDevice(1);
  cudaMalloc((void **) &d_in_2, half_array_bytes);
  cudaMalloc((void **) &d_out_2, half_array_bytes);
 
  long long GPU_memory_allocation_time = stop_timer(GPU_memory_allocation_start_time, "\nGPU Memory Allocation"); 

  // time the memory copy to devices
  long long host_to_device_start_time = start_timer(); 
  // the second thing to do would be to copy the input array over into the gpu 2's memory
  cudaMemcpy(d_in_2, h_input, half_array_bytes, cudaMemcpyHostToDevice); 
 
  cudaSetDevice(0); 
  //copy the other half of the input array over onto the first device
  cudaMemcpy(d_in_1, &h_input[half_size], half_array_bytes, cudaMemcpyHostToDevice);
  long long host_to_device_time = stop_timer(host_to_device_start_time, "GPU Memory Copy to Device");

  // time how long it takes for the kernel to run
  long long kernel_start_time = start_timer(); 
  // now I think I'm ready to launch the kernel on the GPU
  // my original call was faulty since I can't run more than 1024 threads per block!
  
  sine_parallel<<<65535, 1024>>>(d_in_1, d_out_1);
  // also run the kernel on the second device
  cudaSetDevice(1);
  sine_parallel<<<65535, 1024>>>(d_in_2, d_out_2);

  // checking to see that there were no errors with the kernel parameters when it got launched
  long long kernel_time = stop_timer(kernel_start_time, "GPU Kernel Run Time");

  // time how long it takes to copy the results on the GPU back onto the CPU
  long long device_to_host_start_time = start_timer(); 
  // now copy the results on GPU 2's memory to CPU memory
  cudaMemcpy(h_gpu_result, d_out_2, half_array_bytes, cudaMemcpyDeviceToHost); 
  // now copy the results on GPU 1's memory to CPU memory
  cudaMemcpy(&h_gpu_result[half_size], d_out_1, half_array_bytes, cudaMemcpyDeviceToHost);
  long long device_to_host_time = stop_timer(device_to_host_start_time, "GPU Memory Copy to Host");
  
  // get the total time on the GPU
  long long total_time = stop_timer(GPU_memory_allocation_start_time, "Total GPU Run Time");
  std::cout << "\n";
  // Checking to make sure the CPU and GPU results match - Do not modify
  int errorCount = 0;
  for (i=0; i<N; i++)
  {
    if (abs(h_cpu_result[i]-h_gpu_result[i]) > 1e-6)
      errorCount = errorCount + 1;
  }
  if (errorCount > 0)
    printf("Result comparison failed.\n");
  else
    printf("Result comparison passed.\n");

  // Cleaning up memory
  free(h_input);
  free(h_cpu_result);
  free(h_gpu_result);

  // make sure to free the memory on the GPU too!
  cudaFree(d_in_1);
  cudaFree(d_out_1);
  cudaFree(d_in_2);
  cudaFree(d_out_2);

  return 0;
}






