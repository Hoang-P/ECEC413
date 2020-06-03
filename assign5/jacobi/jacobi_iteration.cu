/* Host code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Build as follws: make clean && make

 * Author: Naga Kandasamy
 * Date modified: May 21, 2020
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code */
#include "jacobi_iteration_kernel.cu"

/* Uncomment the line below if you want the code to spit out debug information. */ 
/* #define DEBUG */

int main(int argc, char **argv) 
{
	if (argc > 1) {
		printf("This program accepts no arguments\n");
		exit(EXIT_FAILURE);
	}

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
	matrix_t gpu_naive_solution_x;  /* Solution computed by naive kernel */
    matrix_t gpu_opt_solution_x;    /* Solution computed by optimized kernel */

	/* Initialize the random number generator */
    srand(time(NULL));
    
    struct timeval start, stop;

	/* Generate diagonally dominant matrix */ 
    printf("\nGenerating %d x %d system\n", MATRIX_SIZE, MATRIX_SIZE);
	A = create_diagonally_dominant_matrix(MATRIX_SIZE, MATRIX_SIZE);
	if (A.elements == NULL) {
        printf("Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create the other vectors */
    B = allocate_matrix_on_host(MATRIX_SIZE, 1, 1);
	reference_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
	gpu_naive_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
    gpu_opt_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    /* Compute Jacobi solution on CPU */
    printf("\nPerforming Jacobi iteration on the CPU\n");
    gettimeofday (&start, NULL);
    compute_gold(A, reference_x, B);
    gettimeofday (&stop, NULL);
    printf ("Reference Execution Time = %fs\n", (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float) 1000000));
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	
	/* Compute Jacobi solution on device. Solutions are returned 
       in gpu_naive_solution_x and gpu_opt_solution_x. */
    printf("\nPerforming Jacobi iteration on device\n");
	compute_on_device(A, gpu_naive_solution_x, gpu_opt_solution_x, B);
    display_jacobi_solution(A, gpu_naive_solution_x, B); /* Display statistics */
    display_jacobi_solution(A, gpu_opt_solution_x, B); 
    
    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(gpu_naive_solution_x.elements);
    free(gpu_opt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}


/* FIXME: Complete this function to perform Jacobi calculation on device */
void compute_on_device(const matrix_t A, matrix_t gpu_naive_sol_x, 
                       matrix_t gpu_opt_sol_x, const matrix_t B)
{
    struct timeval start, stop;
    int size = A.num_rows * A.num_columns;
    unsigned int num_rows = A.num_rows;
    unsigned int num_cols = A.num_columns;
    unsigned int done = 0;
    int num_iter_naive = 0;
    double ssd, mse, *ssd_dev;
    int pingpong = 1;

    /* Naive implementation */
    float *A_device = NULL;
    float *B_device = NULL;
    float *naive = NULL;
    float *opt = NULL;
    float *new_A = (float *)malloc(size * sizeof(float));
    int *mutex_on_device = NULL;

    /* Allocate n x 1 matrix to hold iteration values */
    float *new_x = NULL;

    /* Create grid and thread block sizes */
    dim3 threads(THREAD_BLOCK_SIZE, 1);
    dim3 grid(MATRIX_SIZE / THREAD_BLOCK_SIZE, 1);

    gettimeofday (&start, NULL);
    /* Malloc CUDA kernel arguments */
    cudaMalloc((void **)&A_device, size * sizeof(float));
    cudaMalloc((void **)&B_device, MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&naive, MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&opt, MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&new_x, MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&ssd_dev, sizeof(double));

    /* Copy from host to device */
    cudaMemcpy(A_device, A.elements, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B.elements, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(naive, B.elements, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(opt, B.elements, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    /* Allocate space for the lock on GPU and initialize it */
	cudaMalloc((void **)&mutex_on_device, sizeof(int));
	cudaMemset(mutex_on_device, 0, sizeof(int));

    while(!done)
    {
        (pingpong) ? (jacobi_iteration_kernel_naive<<< grid, threads >>>(A_device, naive, B_device, new_x, ssd_dev, mutex_on_device)) : \
                        (jacobi_iteration_kernel_naive<<< grid, threads >>>(A_device, new_x, B_device, naive, ssd_dev, mutex_on_device));
        cudaDeviceSynchronize();
        
        num_iter_naive++;
        pingpong = !pingpong;
        cudaMemcpy(&ssd, ssd_dev, sizeof(double), cudaMemcpyDeviceToHost);
        mse = sqrt (ssd); /* Mean squared error. */
        // printf("Iteration: %d. MSE = %f\n", num_iter_naive, mse);
        if (mse <= THRESHOLD)
            done = 1;
    }

    (!pingpong) ? (cudaMemcpy(gpu_naive_sol_x.elements, naive, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost)) : \
                        (cudaMemcpy(gpu_naive_sol_x.elements, new_x, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    gettimeofday (&stop, NULL);
    printf("\nConvergence achieved after %d iterations \n", num_iter_naive);
    printf ("CUDA (Naive) Execution Time = %fs\n\n", (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float) 1000000));

    /* Optimized implementation */
    done = 0;
    pingpong = 1;
    num_iter_naive = 0;

    gettimeofday (&start, NULL);
    
    /* Convert A matrix from row major format to column major format */
    for (int i = 0; i < num_cols; ++i)
        for (int j = 0; j < num_rows; ++j)
            new_A[ i * num_rows + j ] = A.elements[ j * num_cols + i ];

    /* Malloc CUDA kernel arguments */
    cudaMalloc((void **)&A_device, size * sizeof(float));
    cudaMalloc((void **)&B_device, MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&naive, MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&opt, MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&new_x, MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&ssd_dev, sizeof(double));

    /* Allocate space for the lock on GPU and initialize it */
    cudaMemcpy(A_device, new_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B.elements, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(naive, B.elements, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(opt, B.elements, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    /* Allocate space for the lock on GPU and initialize it */
	cudaMalloc((void **)&mutex_on_device, sizeof(int));
	cudaMemset(mutex_on_device, 0, sizeof(int));
    
    while(!done)
    {
        (pingpong) ? (jacobi_iteration_kernel_optimized<<< grid, threads >>>(A_device, opt, B_device, new_x, ssd_dev, mutex_on_device)) : \
                        (jacobi_iteration_kernel_optimized<<< grid, threads >>>(A_device, new_x, B_device, opt, ssd_dev, mutex_on_device));
        cudaDeviceSynchronize();
        
        num_iter_naive++;
        pingpong = !pingpong;
        cudaMemcpy(&ssd, ssd_dev, sizeof(double), cudaMemcpyDeviceToHost);
        mse = sqrt (ssd); /* Mean squared error. */
        // printf("Iteration: %d. MSE = %f\n", num_iter_naive, mse);
        if (mse <= THRESHOLD)
            done = 1;
    }

    (!pingpong) ? (cudaMemcpy(gpu_opt_sol_x.elements, opt, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost)) : \
                        (cudaMemcpy(gpu_opt_sol_x.elements, new_x, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    gettimeofday (&stop, NULL);
    printf("\nConvergence achieved after %d iterations \n", num_iter_naive);
    printf ("CUDA (Optimized) Execution Time = %fs\n\n", (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float) 1000000));
    
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(naive);
    cudaFree(opt);
    cudaFree(new_x);
    free(new_A);

    return;
}

/* Allocate matrix on the device of same size as M */
matrix_t allocate_matrix_on_device(const matrix_t M)
{
    matrix_t Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void **)&Mdevice.elements, size);
    return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix_on_host(int num_rows, int num_columns, int init)
{	
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *)malloc(size * sizeof(float));
	for (unsigned int i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Copy matrix to device */
void copy_matrix_to_device(matrix_t Mdevice, const matrix_t Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device to host */
void copy_matrix_from_device(matrix_t Mhost, const matrix_t Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

/* Prints the matrix out to screen */
void print_matrix(const matrix_t M)
{
	for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++) {
			printf("%f ", M.elements[i * M.num_columns + j]);
        }
		
        printf("\n");
	} 
	
    printf("\n");
    return;
}

/* Returns a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check for errors in kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
    
    return;    
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(unsigned int num_rows, unsigned int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));
    if (M.elements == NULL)
        return M;

	/* Create a matrix with random numbers between [-.5 and .5] */
    unsigned int i, j;
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
	for (i = 0; i < num_rows; i++) {
		float row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    return M;
}

