/* FIXME: Edit this file to complete the functionality of 2D separable 
 * convolution on the GPU. You may add additional kernel functions 
 * as necessary. 
 */

__device__ void convolve_rows_kernel_naive(float *gpu_result, float *matrix_c, float *kernel,\
    int num_cols, int num_rows, int half_width)
{

    

    int x = blockDim.x * blockIdx.x + threadIdx.x; 
    int y = blockDim.y * blockIdx.y + threadIdx.y; /* (x,y) is the point in the matrix */
    // int kernel_width = 2 * half_width + 1;

    int i, i1; 
    int j, j1, j2;

    j1 = x - half_width; /* min row element */ 
    j2 = x + half_width; /* max row element */

    if (j1 < 0) {
        j1 = 0;
    } /* don't let j go prior to element 0; */ 
        
    if (j2 >= num_cols) {
        j2 = num_cols - 1;
    } /* don't let j go past number of columns */
        
    i1 = j1 - x; /* relative index of our range [0 - kernel_width], start at 0 */ 

    j1 = j1 - x + half_width; 
    j2 = j2 - x + half_width;


    float sum = 0.0;

    for (i = i1, j = j1; j <= j2; j++, i++) {
        sum += kernel[j] * matrix_c[y * num_cols + x + i];
    }

    gpu_result[y * num_cols + x] = sum;

    return;
}


__device__ void convolve_columns_kernel_naive(float *gpu_result, float *matrix_c, float *kernel,\
    int num_cols, int num_rows, int half_width)
{

    int x = blockDim.x * blockIdx.x + threadIdx.x; 
    int y = blockDim.y * blockIdx.y + threadIdx.y; /* (x,y) is the point in the matrix */
    // int kernel_width = 2 * half_width + 1;

    int i, i1; 
    int j, j1, j2;

    j1 = y - half_width; /* min col element */ 
    j2 = y + half_width; /* max col element */

    if (j1 < 0) {
        j1 = 0;
    } /* don't let j go prior to element 0; */ 
        
    if (j2 >= num_rows) {
        j2 = num_rows - 1;
    } /* don't let j go past number of rows */

    i1 = j1 - y; /* relative index of our range [0 - kernel_width], start at 0 */ 

    j1 = j1 - y + half_width; 
    j2 = j2 - y + half_width;

    float sum = 0.0;

    for (i = i1, j = j1; j <= j2; j++, i++) {
        sum += kernel[j] * matrix_c[y * num_cols + x + (i * num_cols) ];
    }

    gpu_result[y * num_cols + x] = sum;

    return;


}

__global__ void convolve_kernel_naive(float *gpu_result, float *matrix_c, float *kernel,\
    int num_cols, int num_rows, int half_width)
{
    convolve_rows_kernel_naive(gpu_result, matrix_c, kernel, num_cols, num_rows, half_width);

    __syncthreads();

    convolve_columns_kernel_naive(matrix_c, gpu_result, kernel, num_cols, num_rows, half_width);

    __syncthreads();

    return;
}

__global__ void convolve_rows_kernel_optimized()
{
    return;
}

__global__ void convolve_columns_kernel_optimized()
{
    return;
}




