/* Particle swarm optimizer.
 *
 * Note: This is an implementation of the original algorithm proposed in: 
 *
 * Yuhui Shi, "Particle Swarm Optimization," IEEE Neural Networks Society, pp. 8-13, February, 2004.
 *
 * Compile using provided Makefile: make 
 * If executable exists or if you have made changes to the .h file but not to the .c files, delete the executable and rebuild 
 * as follows: make clean && make
 *
 * Author: Naga Kandasamy
 * Date created: April 30, 2020
 * Date modified: May 4, 2020 
 *
 * Student/team: Hoang Pham (hdp38) and Nicholas Syrylo (njs76)
 * Date: 5/15/20
 */  
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "pso.h"

int main(int argc, char **argv)
{
    if (argc < 7) {
        fprintf(stderr, "Usage: %s function-name dimension swarm-size xmin xmax max-iter num-threads\n", argv[0]);
        fprintf(stderr, "function-name: name of function to optimize\n");
        fprintf(stderr, "dimension: dimensionality of search space\n");
        fprintf(stderr, "swarm-size: number of particles in swarm\n");
        fprintf(stderr, "xmin, xmax: lower and upper bounds on search domain\n");
        fprintf(stderr, "max-iter: number of iterations to run the optimizer\n");
        // fprintf(stderr, "num-threads: number of threads to create\n");
        exit(EXIT_FAILURE);
    }

    char *function = argv[1];
    int dim = atoi(argv[2]);
    int swarm_size = atoi(argv[3]);
    float xmin = atof(argv[4]);
    float xmax = atof(argv[5]);
    int max_iter = atoi(argv[6]);
    // int num_threads = atoi(argv[7]);

    struct timeval start, stop, start1, stop1; /* Create time structures */

    /* Optimize using reference version */
    int status;
    gettimeofday(&start, NULL);
    status = optimize_gold(function, dim, swarm_size, xmin, xmax, max_iter);
    gettimeofday(&stop, NULL);
    if (status < 0) {
        fprintf(stderr, "Error optimizing function using reference code\n");
        exit (EXIT_FAILURE);
    }

    /* FIXME: Complete this function to perform PSO using OpenMP. 
     * Return -1 on error, 0 on success. Print best-performing 
     * particle within the function prior to returning. 
     */
    /* Things to run threaded programs */
    int threads[] = {4, 8, 16};
    float pthread_times[3];

    for (int i = 0; i < sizeof(threads)/sizeof(threads[0]); i++)
    {
        printf("\nPerforming with %i threads\n", threads[i]);
        gettimeofday(&start1, NULL);
        status = optimize_using_omp(function, dim, swarm_size, xmin, xmax, max_iter, threads[i]);
        gettimeofday(&stop1, NULL);
        if (status < 0) {
            fprintf(stderr, "Error optimizing function using OpenMP\n");
            exit (EXIT_FAILURE);
        }
        pthread_times[i] = (float) (stop1.tv_sec - start1.tv_sec + (stop1.tv_usec - start1.tv_usec)/(float) 1000000);
    }

    printf ("\nRef Execution Time = %fs\n", (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float) 1000000));
    for (int i = 0; i < sizeof(threads)/sizeof(threads[0]); i++)
    {
        printf("%i-Threaded Execution Time = %fs\n", threads[i], pthread_times[i]);
    }
    
    exit(EXIT_SUCCESS);
}

/* Print command-line arguments */
void print_args(char *function, int dim, int swarm_size, float xmin, float xmax)
{
    fprintf(stderr, "Function to optimize: %s\n", function);
    fprintf(stderr, "Dimensionality of search space: %d\n", dim);
    fprintf(stderr, "Number of particles: %d\n", swarm_size);
    fprintf(stderr, "xmin: %f\n", xmin);
    fprintf(stderr, "xmax: %f\n", xmax);
    return;
}

