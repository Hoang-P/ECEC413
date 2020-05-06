/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: April 22, 2020
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm -std=c99
*/

#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>
#include "jacobi_solver.h"

/* Shared data structure used by the threads */
typedef struct args_for_thread_t {
    int tid;                          /* The thread ID */
    int num_threads;                  /* Number of worker threads */
    int rows;                         /* Number of rows in the vectors */
    int columns;                      /* Number of columns in the vectors */
    matrix_t A;                       /* Matrix A */
    matrix_t B;                       /* Matrix B */
    matrix_t x1;                      /* Matrix x (ping) */
    matrix_t x2;                      /* Matrix x (pong) */
    int num_iter;                     /* Iteration counter */
    pthread_barrier_t *barrier;       /* Barrier */
    double *diff;                     /* Partial diff array (ping) */
    double *diff2;                    /* Partial diff array (pong) */
} ARGS_FOR_THREAD;

/* Uncomment the line below to spit out debug information */ 
/* #define DEBUG */

int main(int argc, char **argv) 
{
	if (argc != 2) {
		fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
		exit(EXIT_FAILURE);
	}

    int matrix_size = atoi(argv[1]);

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
    matrix_t mt_solution_x;         /* Solution computed by pthread code */

    /* Things to run threaded programs */
    int threads[] = {4, 8, 16, 32};
    float pthread_times[4];

	/* Generate diagonally dominant matrix */
    fprintf(stderr, "\nCreating input matrices\n");
	srand(time(NULL));
	A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
	if (A.elements == NULL) {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
	reference_x = allocate_matrix(matrix_size, 1, 0);
	mt_solution_x = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    struct timeval start, stop, start1, stop1; /* Create time structures */

    /* Compute Jacobi solution using reference code */
	fprintf(stderr, "Generating solution using reference code\n");
    int max_iter = 100000; /* Maximum number of iterations to run */
    gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B, max_iter);
    gettimeofday(&stop, NULL);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	
	/* Compute the Jacobi solution using pthreads. 
     * Solutions are returned in mt_solution_x.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads\n\n");
    for (int i = 0; i < sizeof(threads)/sizeof(threads[0]); i++)
    {
        printf("Performing with %i threads\n", threads[i]);
        gettimeofday(&start1, NULL);
        compute_using_pthreads(A, mt_solution_x, B, threads[i]);
        gettimeofday(&stop1, NULL);
        display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
        printf("\n");
        pthread_times[i] = (float) (stop1.tv_sec - start1.tv_sec + (stop1.tv_usec - start1.tv_usec)/(float) 1000000);
    }

    printf ("\n");
    printf ("Ref Execution Time = %fs\n", (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float) 1000000));
    for (int i = 0; i < sizeof(threads)/sizeof(threads[0]); i++)
    {
        printf("%i-Threaded Execution Time = %fs\n", threads[i], pthread_times[i]);
    }
    // printf ("Thread Execution time = %fs\n", (float) (stop1.tv_sec - start1.tv_sec + (stop1.tv_usec - start1.tv_usec)/(float) 1000000));
    printf ("\n");
    
    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(mt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads. 
 * Result must be placed in mt_sol_x. */
void compute_using_pthreads (const matrix_t A, matrix_t mt_sol_x, const matrix_t B, int num_threads)
{
    pthread_t *tid = (pthread_t *) malloc (sizeof (pthread_t) * num_threads); /* Data structure to store the thread IDs */
    if (tid == NULL) {
        perror ("malloc");
        exit (EXIT_FAILURE);
    }

    pthread_attr_t attributes;                  /* Thread attributes */    
    pthread_attr_init (&attributes);            /* Initialize the thread attributes to the default values */

    ARGS_FOR_THREAD **args_for_thread;
    args_for_thread = malloc (sizeof (ARGS_FOR_THREAD) * num_threads);

    int i, j;
    pthread_barrier_t *barrier = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t *));
    pthread_barrier_init(barrier,NULL,num_threads);

    /* Create partial diff arrays for threads */
    double *diff = (double *) malloc (num_threads * sizeof (double));
    if (diff == NULL) {
        perror ("Malloc");
        return;
    }
    double *diff2 = (double *) malloc (num_threads * sizeof (double));
    if (diff2 == NULL) {
        perror ("Malloc");
        return;
    }
    
    /* Initialize current jacobi solution. */
    for (i = 0; i < A.num_rows; i++)
        mt_sol_x.elements[i] = B.elements[i];

    matrix_t new_x = allocate_matrix(A.num_rows, 1, 0);

    for (i = 0; i < num_threads; i++) {
        args_for_thread[i] = (ARGS_FOR_THREAD *) malloc (sizeof (ARGS_FOR_THREAD));
        args_for_thread[i]->tid = i;
        args_for_thread[i]->num_threads = num_threads;
        args_for_thread[i]->rows = A.num_rows;
        args_for_thread[i]->columns = A.num_columns;
        args_for_thread[i]->A = A;
        args_for_thread[i]->B = B;
        args_for_thread[i]->x1 = mt_sol_x;
        args_for_thread[i]->x2 = new_x;
        args_for_thread[i]->num_iter = 0;
        args_for_thread[i]->barrier = barrier;
        args_for_thread[i]->diff = diff;
        args_for_thread[i]->diff2 = diff2;
        pthread_create (&tid[i], &attributes, jacobi, (void *) args_for_thread[i]);
    }

    for (i = 0; i < num_threads; i++)
        pthread_join (tid[i], NULL);

    fprintf(stderr, "Convergence achieved after %d iterations\n", args_for_thread[0]->num_iter);

    /* Free data structures */
    for(j = 0; j < num_threads; j++)
        free ((void *) args_for_thread[j]);
}

void * jacobi (void *args) 
{
    ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *) args; /* Typecast the argument to a pointer the the ARGS_FOR_THREAD structure */

    /* Perform Jacobi iteration. */
    int done = 0;
    double mse, sum, total;

    /* Pingpong buffers (pointers) */
    float * x1 = args_for_me->x1.elements;
    float * x2 = args_for_me->x2.elements;

    double * diff = args_for_me->diff;
    double * diff2 = args_for_me->diff2;

    while (!done)
    {
        diff[args_for_me->tid] = 0.0;
        total = 0.0;
        int i, j;
        for (i = args_for_me->tid; i < args_for_me->rows; i += args_for_me->num_threads)
        {
            sum = -args_for_me->A.elements[i * args_for_me->columns + i] * x1[i];
            
            for (j = 0; j < args_for_me->columns; j++) 
                sum += args_for_me->A.elements[i * args_for_me->columns + j] * x1[j];

            x2[i] = (args_for_me->B.elements[i] - sum)/args_for_me->A.elements[i * args_for_me->columns + i];
            diff[args_for_me->tid] += (x2[i] - x1[i]) * (x2[i] - x1[i]);
        }
        args_for_me->num_iter++;

        /* Swap buffers */
        float * temp = x1;
        x1 = x2;
        x2 = temp;
        
        double * temp2 = diff;
        diff = diff2;
        diff2 = temp2;

        pthread_barrier_wait(args_for_me->barrier);

        for(int i = 0; i < args_for_me->num_threads; i++) /* Accumulate total difference for mse */
            total += diff2[i];
        
        mse = sqrt(total); /* Mean squared error. */

        if (mse <= THRESHOLD)
            done = 1;
    }

    pthread_exit ((void *)0);
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;    
    matrix_t M;
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

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
	for (i = 0; i < M.num_rows; i++) {
        for (j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_columns + j]);
        }
		
        fprintf(stderr, "\n");
	} 
	
    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand ()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
	float diag_element;
	float sum;
	for (i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		
        if (diag_element <= sum)
			return -1;
	}

	return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix (int num_rows, int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
	fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
	for (i = 0; i < num_rows; i++) {
		row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    /* Check if matrix is diagonal dominant */
	if (check_if_diagonal_dominant(M) < 0) {
		free(M.elements);
		M.elements = NULL;
	}
	
    return M;
}



