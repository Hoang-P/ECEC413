/* Blur filter. Device code. */

#ifndef _BLUR_FILTER_KERNEL_H_
#define _BLUR_FILTER_KERNEL_H_

#include "blur_filter.h"

__global__ void 
blur_filter_kernel (const float *in, float *out, int size)
{
    /* Obtain thread location within the block */
    int tx, ty;
    tx = threadIdx.x;
    ty = threadIdx.y;

    int bx, by;
    bx = blockIdx.x;
    by = blockIdx.y;

    int bdimx, bdimy;
    bdimx = blockDim.x;
    bdimy = blockDim.y;
    
    // int imgSize = size;
    int row = bdimy * by + ty;
    int col = bdimx * bx + tx;

    int i, j;
    int curr_row, curr_col;
    float blur_value;
    int num_neighbors;

    /* Apply blur filter to current pixel */
    blur_value = 0.0;
    num_neighbors = 0;
    for (i = -BLUR_SIZE; i < (BLUR_SIZE + 1); i++) {
        for (j = -BLUR_SIZE; j < (BLUR_SIZE + 1); j++) {
            /* Accumulate values of neighbors while checking for 
             * boundary conditions */
            curr_row = row + i;
            curr_col = col + j;
            if ((curr_row > -1) && (curr_row < size) &&\
                    (curr_col > -1) && (curr_col < size)) {
                blur_value += in[curr_row * size + curr_col];
                num_neighbors += 1;
            }
        }
    }
    out[row * size + col] = blur_value/num_neighbors;

    return;
}

#endif /* _BLUR_FILTER_KERNEL_H_ */
