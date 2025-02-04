/* Host code that implements a  separable convolution filter of a 
 * 2D signal with a gaussian kernel.
 * 
 * Author: Naga Kandasamy
 * Date modified: May 26, 2020
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

extern "C" void compute_gold(float *, float *, int, int, int);
extern "C" float *create_kernel(float, int);
void print_kernel(float *, int);
void print_matrix(float *, int, int);

/* Width of convolution kernel */
#define HALF_WIDTH 8
#define COEFF 10

/* Size of thread blocks */
#define THREAD_BLOCK_SIZE 32

__constant__ float kernel_c[HALF_WIDTH * 2 + 1]; /* Allocation for the kernel in GPU constant memory */

/* Uncomment line below to spit out debug information */
// #define DEBUG

/* Include device code */
#include "separable_convolution_kernel.cu"

/* FIXME: Edit this function to compute the convolution on the device.*/
void compute_on_device(float *gpu_result,float *gpu_result_opt, float *matrix_c,\
                   float *kernel, int num_cols,\
                   int num_rows, int half_width)
{
    struct timeval start, stop;
    int width = 2 * half_width + 1;
    int num_elements = num_rows * num_cols;

    float *rDevice = NULL; /* result on device */
    float *mDevice = NULL; /* matrix on device */
    float *mDevice_opt = NULL; /* matrix on device */
    float *kDevice = NULL; /* kernel on device */

    /* Memory allocation for GPU */
    cudaMalloc((void **)&rDevice, num_elements * sizeof(float));
    cudaMalloc((void **)&mDevice, num_elements * sizeof(float));
    cudaMalloc((void **)&mDevice_opt, num_elements * sizeof(float));
    cudaMalloc((void **)&kDevice, width * sizeof(float));

    /* Copy over matrices */
    cudaMemcpy(mDevice, matrix_c, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mDevice_opt, matrix_c, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kDevice, kernel, width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    dim3 grid(num_cols / THREAD_BLOCK_SIZE, num_rows / THREAD_BLOCK_SIZE);

    /* Naive implementation */
    gettimeofday (&start, NULL);
    convolve_rows_kernel_naive<<< grid, threads >>>(rDevice, mDevice, kDevice, num_cols, num_rows, half_width);
    cudaDeviceSynchronize();
    convolve_columns_kernel_naive<<< grid, threads >>>(mDevice, rDevice, kDevice, num_cols, num_rows, half_width);
    gettimeofday (&stop, NULL);
    printf ("CUDA Naive Execution Time = %fs\n", (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float) 1000000));

    cudaMemcpy(gpu_result, mDevice, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    /* Optimized implementation */
    gettimeofday (&start, NULL);
    /* We copy the mask to GPU constant memory to improve performance */
    cudaMemcpyToSymbol(kernel_c, kernel, width * sizeof(float));
    convolve_rows_kernel_optimized<<< grid, threads >>>(rDevice, mDevice_opt, num_cols, num_rows, half_width);
    cudaDeviceSynchronize();
    convolve_columns_kernel_optimized<<< grid, threads >>>(mDevice_opt, rDevice, num_cols, num_rows, half_width);
    gettimeofday (&stop, NULL);
    printf ("CUDA Optimized Execution Time = %fs\n", (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float) 1000000));

    cudaMemcpy(gpu_result_opt, mDevice, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(rDevice);
    cudaFree(mDevice);
    cudaFree(mDevice_opt);
    cudaFree(kDevice);

    return;
}


int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s num-rows num-columns\n", argv[0]);
        printf("num-rows: height of the matrix\n");
        printf("num-columns: width of the matrix\n");
        exit(EXIT_FAILURE);
    }

    int num_rows = atoi(argv[1]);
    int num_cols = atoi(argv[2]);

    struct timeval start, stop;

    /* Create input matrix */
    int num_elements = num_rows * num_cols;
    printf("Creating input matrix of %d x %d\n", num_rows, num_cols);
    float *matrix_a = (float *)malloc(sizeof(float) * num_elements);
    float *matrix_c = (float *)malloc(sizeof(float) * num_elements);
	
    srand(time(NULL));
    int i;
    for (i = 0; i < num_elements; i++) {
        matrix_a[i] = rand()/(float)RAND_MAX;			 
        matrix_c[i] = matrix_a[i]; /* Copy contents of matrix_a into matrix_c */
    }
	 
	/* Create Gaussian kernel */	  
    float *gaussian_kernel = create_kernel((float)COEFF, HALF_WIDTH);	
#ifdef DEBUG
    print_kernel(gaussian_kernel, HALF_WIDTH); 
#endif
	  
    /* Convolve matrix along rows and columns. 
       The result is stored in matrix_a, thereby overwriting the 
       original contents of matrix_a.		
     */
    printf("\nConvolving the matrix on the CPU\n");
    gettimeofday (&start, NULL);
    compute_gold(matrix_a, gaussian_kernel, num_cols,\
                  num_rows, HALF_WIDTH);
    gettimeofday (&stop, NULL);
    printf ("CPU Execution Time = %fs\n", (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float) 1000000));
#ifdef DEBUG	 
    print_matrix(matrix_a, num_cols, num_rows);
#endif
  
    float *gpu_result = (float *)malloc(sizeof(float) * num_elements);
    float *gpu_result_opt = (float *)malloc(sizeof(float) * num_elements);
    
    /* FIXME: Edit this function to complete the functionality on the GPU.
       The input matrix is matrix_c and the result must be stored in 
       gpu_result.
     */
    printf("\nConvolving matrix on the GPU\n");
    compute_on_device(gpu_result, gpu_result_opt, matrix_c, gaussian_kernel, num_cols,\
                       num_rows, HALF_WIDTH);
    
    #ifdef DEBUG	 
        print_matrix(gpu_result, num_cols, num_rows);
        printf("\n");
        print_matrix(gpu_result_opt, num_cols, num_rows);
    #endif
    printf("\nComparing CPU and GPU results (Naive)\n");
    float sum_delta = 0, sum_ref = 0;
    for (i = 0; i < num_elements; i++) {
        sum_delta += fabsf(matrix_a[i] - gpu_result[i]);
        sum_ref   += fabsf(matrix_a[i]);
    }
        
    float L1norm = sum_delta / sum_ref;
    float eps = 1e-6;
    printf("L1 norm: %E\n", L1norm);
    printf((L1norm < eps) ? "TEST PASSED\n" : "TEST FAILED\n");

    printf("\nComparing CPU and GPU results (Optimized)\n");
    sum_delta = 0;
    sum_ref = 0;
    for (i = 0; i < num_elements; i++) {
        sum_delta += fabsf(matrix_a[i] - gpu_result_opt[i]);
        sum_ref   += fabsf(matrix_a[i]);
    }
        
    L1norm = sum_delta / sum_ref;
    printf("L1 norm: %E\n", L1norm);
    printf((L1norm < eps) ? "TEST PASSED\n" : "TEST FAILED\n");

    free(matrix_a);
    free(matrix_c);
    free(gpu_result);
    free(gpu_result_opt);
    free(gaussian_kernel);

    exit(EXIT_SUCCESS);
}


/* Check for errors reported by the CUDA run time */
void check_for_error(char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s)\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    return;
} 

/* Print convolution kernel */
void print_kernel(float *kernel, int half_width)
{
    int i, j = 0;
    for (i = -half_width; i <= half_width; i++) {
        printf("%0.2f ", kernel[j]);
        j++;
    }

    printf("\n");
    return;
}

/* Print matrix */
void print_matrix(float *matrix, int num_cols, int num_rows)
{
    int i,  j;
    float element;
    for (i = 0; i < num_rows; i++) {
        for (j = 0; j < num_cols; j++){
            element = matrix[i * num_cols + j];
            printf("%0.2f ", element);
        }
        printf("\n");
    }

    return;
}

