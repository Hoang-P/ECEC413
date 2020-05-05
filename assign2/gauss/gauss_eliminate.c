/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date of last update: April 22, 2020
 *
 * Student names(s): FIXME
 * Date: FIXME
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -lpthread -lm -std=c99
 */

#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

/* Shared data structure used by the threads */
typedef struct args_for_thread_t {
    int tid;                          /* The thread ID */
    int num_threads;                  /* Number of worker threads */
    // int num_elements;                 /* Number of elements in the vectors */
    int rows;
    int columns;
    float *elements;
    pthread_barrier_t *barrier;
} ARGS_FOR_THREAD;

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix, int);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);
void *gauss (void *args);


int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        // fprintf(stderr, "num-threads: Number of threads\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);
    // int num_threads = atoi(argv[2]);

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
  
    /* FIXME: Perform Gaussian elimination using pthreads. 
     * The resulting upper triangular matrix should be returned in U_mt */
    fprintf(stderr, "\nPerforming gaussian elimination using pthreads\n");
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
        gauss_eliminate_using_pthreads(U_mt_thread[i], threads[i]);
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


/* FIXME: Write code to perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix U, int num_threads)
{
    /* FIXME: Complete this function */
    pthread_t *tid = (pthread_t *) malloc (sizeof (pthread_t) * num_threads); /* Data structure to store the thread IDs */
    if (tid == NULL) {
        perror ("malloc");
        exit (EXIT_FAILURE);
    }

    pthread_attr_t attributes;                  /* Thread attributes */
    pthread_attr_init (&attributes);            /* Initialize the thread attributes to the default values */
    
    int i;
    ARGS_FOR_THREAD **args_for_thread;
    args_for_thread = malloc (sizeof (ARGS_FOR_THREAD) * num_threads);

    pthread_barrier_t *barrier = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t *));
    pthread_barrier_init(barrier,NULL,num_threads);

    for (i = 0; i < num_threads; i++){
        args_for_thread[i] = (ARGS_FOR_THREAD *) malloc (sizeof (ARGS_FOR_THREAD));
        args_for_thread[i]->tid = i;
        args_for_thread[i]->num_threads = num_threads;
        args_for_thread[i]->rows = U.num_rows;
        args_for_thread[i]->columns = U.num_columns;
        args_for_thread[i]->elements = U.elements;
        args_for_thread[i]->barrier = barrier;
        pthread_create (&tid[i], &attributes, gauss, (void *) args_for_thread[i]);
    }
					 
    /* Wait for the workers to finish */
    for(i = 0; i < num_threads; i++)
        pthread_join (tid[i], NULL);
		
    /* Free data structures */
    for(i = 0; i < num_threads; i++)
        free ((void *) args_for_thread[i]);
}

void *gauss (void *args)
{
    ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *) args; /* Typecast the argument to a pointer the the ARGS_FOR_THREAD structure */

    int num_elements = args_for_me->rows;
    for (int k = 0; k < num_elements; k++) {
        if((k % args_for_me->num_threads) == args_for_me->tid) {
            for (int j = (k + 1); j < num_elements; j++)
            {
                if (args_for_me->elements[num_elements * k + k] == 0) {
                    fprintf(stderr, "Numerical instability. The principal diagonal element is zero.\n");
                    pthread_exit ((void *)0);
                }
                args_for_me->elements[num_elements * k + j] = (float)(args_for_me->elements[num_elements * k + j] / args_for_me->elements[num_elements * k + k]);	/* Division step */
            }
            args_for_me->elements[num_elements * k + k] = 1;	/* Set the principal diagonal entry in U to 1 */
        }
        pthread_barrier_wait(args_for_me->barrier);
        

        for (int i = k + 1; i < num_elements; i++)
        {
            if((i % args_for_me->num_threads) == args_for_me->tid) {
                for (int j = (k + 1); j < num_elements; j++)
                    args_for_me->elements[num_elements * i + j] -= (args_for_me->elements[num_elements * i + k] * args_for_me->elements[num_elements * k + j]);	/* Elimination step */
            
                args_for_me->elements[num_elements * i + k] = 0;
            }
        }
        pthread_barrier_wait(args_for_me->barrier);
    }

    if (args_for_me->tid == (args_for_me->num_threads - 1))
        args_for_me->elements[num_elements * (num_elements - 1) + num_elements - 1] = 1;

    pthread_exit ((void *)0);
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
