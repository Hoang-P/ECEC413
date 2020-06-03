#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */

/* Use compare and swap to acquire mutex */
__device__ void lock(int *mutex) 
{	  
    while (atomicCAS(mutex, 0, 1) != 0);
    return;
}

/* Use atomic exchange operation to release mutex */
__device__ void unlock(int *mutex) 
{
    atomicExch(mutex, 0);
    return;
}

__global__ void jacobi_iteration_kernel_naive(float *A, float *naive, float *B, float *new_x, double *ssd, int *mutex)
{
    __shared__ double ssd_array[THREAD_BLOCK_SIZE];

    /* Find our place in thread block/grid. */
    unsigned int threadID = threadIdx.x;
    unsigned int dataID = blockIdx.x * blockDim.x + threadIdx.x;

    /* Reset ssd to 0 */
    if (dataID == 0)
        *ssd = 0.0;

    /* Perform jacobi */
    double sum = -A[dataID * MATRIX_SIZE + dataID] * naive[dataID];
    for (int j = 0; j < MATRIX_SIZE; j++) {
        sum += A[dataID * MATRIX_SIZE + j] * naive[j];
    }

    new_x[dataID] = (B[dataID] - sum)/A[dataID * MATRIX_SIZE + dataID];

    /* Copy data to shared memory from global memory. */ 
	if (dataID < MATRIX_SIZE)
        ssd_array[threadID] = (new_x[dataID] - naive[dataID]) * (new_x[dataID] - naive[dataID]);
    else
        ssd_array[threadID] = 0.0;

    __syncthreads();

    /* Parallel reduction */
	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride = stride >> 1) {
		if(threadID < stride)
            ssd_array[threadID] += ssd_array[threadID + stride];
		__syncthreads();
    }

	/* Store result to global ssd. */
    if (threadID == 0) {
		lock(mutex);
		*ssd += ssd_array[0];
		unlock(mutex);
	}

    return;
}

__global__ void jacobi_iteration_kernel_optimized(float *A, float *naive, float *B, float *new_x, double *ssd, int *mutex)
{
    __shared__ double ssd_array[THREAD_BLOCK_SIZE];

    /* Find our place in thread block/grid. */
    unsigned int threadID = threadIdx.x;
    unsigned int dataID = blockIdx.x * blockDim.x + threadIdx.x;

    /* Reset ssd to 0 */
    if (dataID == 0)
        *ssd = 0.0;

    /* Perform jacobi */
    double sum = -A[dataID * MATRIX_SIZE + dataID] * naive[dataID];
    for (int j = 0; j < MATRIX_SIZE; j++) {
        sum += A[dataID + MATRIX_SIZE * j] * naive[j];
    }

    new_x[dataID] = (B[dataID] - sum)/A[dataID * MATRIX_SIZE + dataID];

    /* Copy data to shared memory from global memory. */ 
	if (dataID < MATRIX_SIZE)
        ssd_array[threadID] = (new_x[dataID] - naive[dataID]) * (new_x[dataID] - naive[dataID]);
    else
        ssd_array[threadID] = 0.0;

    __syncthreads();

    /* Parallel reduction */
	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride = stride >> 1) {
		if(threadID < stride)
            ssd_array[threadID] += ssd_array[threadID + stride];
		__syncthreads();
    }

	/* Store result to global ssd. */
    if (threadID == 0) {
		lock(mutex);
		*ssd += ssd_array[0];
		unlock(mutex);
	}

    return;
}

