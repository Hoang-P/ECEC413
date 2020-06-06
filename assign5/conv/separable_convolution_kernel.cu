/* FIXME: Edit this file to complete the functionality of 2D separable 
 * convolution on the GPU. You may add additional kernel functions 
 * as necessary. 
 */

__global__ void convolve_rows_kernel_naive(float *result, float *input, float *kernel, int num_cols, int num_rows, int half_width)
{
    int i, i1;
    int j, j1, j2;
    int x, y;

    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    j1 = x - half_width;
    j2 = x + half_width;

    /* Clamp at the edges of the matrix */
    if (j1 < 0) 
        j1 = 0;
    if (j2 >= num_cols) 
        j2 = num_cols - 1;

    /* Obtain relative position of starting element from element being convolved */
    i1 = j1 - x;

    j1 = j1 - x + half_width; /* Obtain operating width of the kernel */
    j2 = j2 - x + half_width;

    /* Convolve along row */
    result[y * num_cols + x] = 0.0f;
    for(i = i1, j = j1; j <= j2; j++, i++)
        result[y * num_cols + x] += kernel[j] * input[y * num_cols + x + i];

    return;
}

__global__ void convolve_columns_kernel_naive(float *result, float *input, float *kernel, int num_cols, int num_rows, int half_width)
{
    int i, i1;
    int j, j1, j2;
    int x, y;

    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    j1 = y - half_width;
    j2 = y + half_width;

    /* Clamp at the edges of the matrix */
    if (j1 < 0) 
        j1 = 0;
    if (j2 >= num_rows) 
        j2 = num_rows - 1;

    /* Obtain relative position of starting element from element being convolved */
    i1 = j1 - y;

    j1 = j1 - y + half_width; /* Obtain the operating width of the kernel.*/
    j2 = j2 - y + half_width;

    /* Convolve along column */
    result[y * num_cols + x] = 0.0f;
    for (i = i1, j = j1; j <= j2; j++, i++)
        result[y * num_cols + x] += kernel[j] * input[y * num_cols + x + (i * num_cols)];

    return;
}

__global__ void convolve_rows_kernel_optimized(float *result, float *input, int num_cols, int num_rows, int half_width)
{
    __shared__ float input_ts[(THREAD_BLOCK_SIZE + HALF_WIDTH * 2) * THREAD_BLOCK_SIZE];
    int i, i1;
    int j, j1, j2;
    int x, y;

    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    /* Load the left halo elements from the previous tile. 
     * The number of halo elements will be half_width.
     */
    int left_halo_index = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x >= (blockDim.x - half_width)) {
        if (left_halo_index < 0)
            input_ts[(threadIdx.x - (blockDim.x - half_width)) + threadIdx.y * (THREAD_BLOCK_SIZE + HALF_WIDTH * 2)] = 0.0;
        else
            input_ts[(threadIdx.x - (blockDim.x - half_width)) + threadIdx.y * (THREAD_BLOCK_SIZE + HALF_WIDTH * 2)] = input[left_halo_index + y * num_rows];
    }

    /* Load the center elements for the tile */
    if (x < num_cols)
        input_ts[half_width + threadIdx.x + threadIdx.y * (THREAD_BLOCK_SIZE + HALF_WIDTH * 2)] = input[y * num_rows + x];
    else
        input_ts[half_width + threadIdx.x + threadIdx.y * (THREAD_BLOCK_SIZE + HALF_WIDTH * 2)] = 0.0;

    /* Load the right halo elements from the next tile.
     * The number of halo elements will again be half_width.
     */
    int right_halo_index = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x < half_width) {
        if (right_halo_index >= num_cols)
            input_ts[threadIdx.x + (blockDim.x + half_width) + threadIdx.y * (THREAD_BLOCK_SIZE + HALF_WIDTH * 2)] = 0.0;
        else
            input_ts[threadIdx.x + (blockDim.x + half_width) + threadIdx.y * (THREAD_BLOCK_SIZE + HALF_WIDTH * 2)] = input[right_halo_index + y * num_rows];
    }

    __syncthreads();

    j1 = x - half_width;
    j2 = x + half_width;

    /* Clamp at the edges of the matrix */
    if (j1 < 0)
        j1 = 0;
    if (j2 >= num_cols)
        j2 = num_cols - 1;

    /* Obtain relative position of starting element from element being convolved */
    i1 = j1 - x;

    j1 = j1 - x + half_width; /* Obtain operating width of the kernel */
    j2 = j2 - x + half_width;

    /* Convolve along row */
    result[y * num_cols + x] = 0.0f;
    for(i = i1, j = j1; j <= j2; j++, i++)
        // result[y * num_cols + x] += kernel_c[j] * input[y * num_cols + x + i];
        result[y * num_cols + x] += kernel_c[j] * input_ts[threadIdx.y * blockDim.y + threadIdx.x + i + half_width];
    
    return;
}

__global__ void convolve_columns_kernel_optimized(float *result, float *input, int num_cols, int num_rows, int half_width)
{
    int i, i1;
    int j, j1, j2;
    int x, y;

    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;

    j1 = y - half_width;
    j2 = y + half_width;
    /* Clamp at the edges of the matrix */
    if (j1 < 0) 
        j1 = 0;
    if (j2 >= num_rows) 
        j2 = num_rows - 1;

    /* Obtain relative position of starting element from element being convolved */
    i1 = j1 - y; 
    
    j1 = j1 - y + half_width; /* Obtain the operating width of the kernel.*/
    j2 = j2 - y + half_width;

    /* Convolve along column */            
    result[y * num_cols + x] = 0.0f;
    for (i = i1, j = j1; j <= j2; j++, i++)
        result[y * num_cols + x] += kernel_c[j] * input[y * num_cols + x + (i * num_cols)];
    
    return;
}
 