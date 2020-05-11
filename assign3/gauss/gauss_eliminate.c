/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date of last update: April 29, 2020
 *
 * Student names(s): FIXME
 * Date: FIXME
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -fopenmp -std=c99 -Wall -O3 -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_omp(Matrix, int);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);


int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);

     /* Things to run threaded programs */
    int threads[] = {4, 8, 16, 32};
    float pthread_times[5];

    Matrix A;			                                            /* Input matrix */
    Matrix U_reference;		                                        /* Upper triangular matrix computed by reference code */
    Matrix U_mt;			                                        /* Upper triangular matrix computed by pthreads */

    fprintf(stderr, "Generating input matrices\n");
    srand(time (NULL));                                             /* Seed random number generator */
    A = allocate_matrix(matrix_size, matrix_size, 1);               /* Allocate and populate random square matrix */
    U_reference = allocate_matrix (matrix_size, matrix_size, 0);    /* Allocate space for reference result */
    U_mt = allocate_matrix (matrix_size, matrix_size, 0);           /* Allocate space for multi-threaded result */

    /* Copy contents A matrix into U matrices */
    int i, j;
    for (i = 0; i < A.num_rows; i++) {
        for (j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    
    int status = compute_gold(U_reference.elements, A.num_rows);
  
    gettimeofday(&stop, NULL);
    pthread_times[5] = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000);
    // fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec
    //             + (stop.tv_usec - start.tv_usec) / (float)1000000));

    if (status < 0) {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }
  
    status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */ 
    if (status < 0) {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");
  
    /* FIXME: Perform Gaussian elimination using OpenMP. 
     * The resulting upper triangular matrix should be returned in U_mt */
    fprintf(stderr, "\nPerforming gaussian elimination using omp\n");
    Matrix U_mt_thread[4];
    for (int i = 0; i < sizeof(threads)/sizeof(threads[0]); i++)
    {
        U_mt_thread[i] = allocate_matrix (matrix_size, matrix_size, 0);
        memcpy(U_mt_thread[i].elements, U_mt.elements, sizeof(float)*matrix_size*matrix_size);
    }
    for (int i = 0; i < sizeof(threads)/sizeof(threads[0]); i++)
    {
        // *U_mt_thread.elements = *U_mt.elements;
        printf("Performing with %i threads\n", threads[i]);
        gettimeofday(&start, NULL);
        gauss_eliminate_using_omp(U_mt_thread[i], threads[i]);
        gettimeofday(&stop, NULL);
        pthread_times[i] = (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float) 1000000);

        status = perform_simple_check(U_mt_thread[i]);	/* Check that principal diagonal elements are 1 */
        if (status < 0) {
            fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
            exit(EXIT_FAILURE);
        }
        fprintf(stderr, "Multi-threaded Gaussian elimination was successful.\n");

        /* Check if pthread result matches reference solution within specified tolerance */
        fprintf(stderr, "Checking results\n");
        int size = matrix_size * matrix_size;
        int res = check_results(U_reference.elements, U_mt_thread[i].elements, size, 1e-6);
        fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED\n" : "FAILED\n");
        pthread_times[i] = (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float) 1000000);
    }

    printf ("Ref Execution Time = %fs\n", pthread_times[5]);
    for (int i = 0; i < sizeof(threads)/sizeof(threads[0]); i++)
    {
        printf("%i-Threaded Execution Time = %fs\n", threads[i], pthread_times[i]);
    }

    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}


/* FIXME: Write code to perform gaussian elimination using omp */
void gauss_eliminate_using_omp(Matrix U, int thread_count)
{
    // int thread_count = 4;
    int num_elements = U.num_rows;


#pragma omp parallel num_threads(thread_count) shared(U, num_elements) 
    {
    int tid = omp_get_thread_num();
    
    
    for (int k = 0; k < num_elements; k++) {
        if ( (k % thread_count) == tid) {
            for (int j = (k + 1); j < num_elements; j++) {
                if (U.elements[num_elements * k + k] == 0) {
                    fprintf(stderr, "Numerical instability. The principal diagonal element is zero.\n");
                }

                U.elements[num_elements * k + j] = (float)(U.elements[num_elements * k + j] / U.elements[num_elements * k + k]); 	/* Division step */
        
            }

            U.elements[num_elements * k + k] = 1; /* Set the principal diagonal entry in U to 1 */
        
        }

        #pragma omp barrier

        for (int i = k + 1; i < num_elements; i++) {
            if ( (i % thread_count) == tid) {
                for (int j = (k + 1); j < num_elements; j++) {
                    U.elements[num_elements * i + j] -= (U.elements[num_elements * i + k] * U.elements[num_elements * k + j]);
                }

                U.elements[num_elements * i + k] = 0;
            }
        }

        #pragma omp barrier

    }

    if (tid == (thread_count - 1) ) {
        U.elements[num_elements * (num_elements - 1) + num_elements - 1] = 1;
    }

    }

    // Print out matrix
    // for (int i = 0; i < num_elements; i++) {
    //     for (int j = 0; j < num_elements; j++) {
    //         fprintf(stderr, "%.2f ", U.elements[num_elements * i + j]);
    //     }
    // fprintf(stderr, "\n");     
    // }

}


/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if(fabsf(A[i] - B[i]) > tolerance)
            return -1;
    return 0;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    Matrix M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));
  
    for (i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }
  
    return M;
}

/* Return a random floating-point number between [min, max] */ 
float get_random_number(int min, int max)
{
    return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
    int i;
    for (i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
            return -1;
  
    return 0;
}
