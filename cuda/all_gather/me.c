#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <mpi.h>
int main( int argc, char** argv )
{
    MPI_Init (&argc, &argv);
    int direct;
    int rank, size, chunk;
    int *h_buff = NULL;
    int *h_buff2 = NULL;
    int *h_buff3 = NULL;
    int *d_buff = NULL;
    int *d_buff2 = NULL;
    int i;
    MPI_Request request;
    // Ensure that RDMA ENABLED CUDA is set correctly
    // direct = getenv("MPICH_RDMA_ENABLED_CUDA")==NULL?0:atoi(getenv ("MPICH_RDMA_ENABLED_CUDA"));
    // if(direct != 1){
    //     printf ("MPICH_RDMA_ENABLED_CUDA not enabled!\n");
    //     exit (EXIT_FAILURE);
    // }
    srand (time(NULL));
    chunk = 1000;
    h_buff = (int*)malloc(sizeof(int)*chunk);
    h_buff2 = (int*)malloc(sizeof(int)*chunk);
    h_buff3 = (int*)malloc(sizeof(int)*chunk);
    cudaMalloc(&d_buff, sizeof(int)*chunk);
    cudaMalloc(&d_buff2, sizeof(int)*chunk);
    for(i=0; i<chunk; i++){
        h_buff[i] = rand();
    }
    cudaMemcpy(d_buff, h_buff, sizeof(int)*chunk, cudaMemcpyHostToDevice);
    MPI_Allreduce(h_buff, h_buff2, chunk, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Iallreduce(d_buff, d_buff2, chunk, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    // Check that the GPU buffer is correct
    cudaMemcpy(h_buff3, d_buff2, sizeof(int)*chunk, cudaMemcpyDeviceToHost);
    for(i=0; i<chunk; i++){
        if(h_buff2[i] != h_buff3[i]) {
            printf ("Iallreduce Failed!\n");
            exit (EXIT_FAILURE);
        }
    }
    printf("Success!\n");
    // Clean up
    free(h_buff3);
    free(h_buff2);
    free(h_buff);
    cudaFree(d_buff2);
    cudaFree(d_buff);
    MPI_Finalize();
    return 0;
}
