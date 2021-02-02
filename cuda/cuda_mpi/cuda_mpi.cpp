#include <mpi.h>
#include <vector>
#include <numeric>
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);


    int N = 100;
    int size = N * sizeof(float);
    std::vector<float> host(N);

    float* device;
    cudaMalloc(&device, size);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
	std::fill(host.begin(), host.end(), 1);

        cudaMemcpy(device, host.data(), size, cudaMemcpyHostToDevice);

	MPI_Send(device, N, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);

    } else {
        MPI_Status status;
        MPI_Recv(device, N, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);

        cudaMemcpy(host.data(), device, size, cudaMemcpyDeviceToHost);

        float sum = std::accumulate(host.begin(), host.end(), 0);
	std::cout << "Exact: " << N << ", Computed: " << sum << std::endl;
    }


    cudaFree(device);

    MPI_Finalize();
    return 0;
}
