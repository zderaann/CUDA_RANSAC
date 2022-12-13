
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<fstream>
#include<iostream>
#include<string>
#include<limits>

#define NUM_OF_ITERATIONS 1000
#define NUM_OF_SAMPLES 100
#define NUM_FIT 10
#define THRESHOLD 0.05
#define THREADS_PER_BLOCK 32

class results {
public:
    float scale;
    float translation[3];
    float rotation[9];
    float error;

    results() {
        scale = 1;
        for (int i = 0; i < 3; i++) {
            translation[i] = 0;
            for (int j = 0; j < 3; j++) {
                rotation[i * 3 + j] = j == i;
            }
        }
        error = std::numeric_limits<float>::max();
    }
};

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
results best_result;
int* pc_count1;
int* pc_count2;
results* devOut;
float* devInPC1;
float* devInPC2;
float* devOutPC;
int* devInCount1;
int* devInCount2;
float* result_pc;
float* matrix_GPU;
int* matrix_threads;

__device__ int mutex = 0;


void CPU_non_paralel(float** pc1, float** pc2) {
    // python
}

float* readPLY(const char* path, int* cnt) {
    std::fstream pc_file;
    pc_file.open(path, std::ios::in);
    std::string line;
    std::string header_end = "end_header";
    std::string element_vertex = "element vertex";
    while (std::getline(pc_file, line)) {
        if (line.compare(0, element_vertex.size(), element_vertex) == 0) {
            size_t index = line.find("vertex ") + 7;
            std::string s_num = line.substr(index, line.size() - index);
            *cnt = std::stoi(s_num);
            std::cout << *cnt << std::endl;
        }
        if (line.compare(header_end) == 0) {
            break;
        }
    }
    
    float* pc = new float[*cnt * 3];
    for (int i = 0; i < *cnt; i++) {
        std::getline(pc_file, line);
        for (int j = 0; j < 3; j++) {
            size_t end = line.find(" ");
            pc[i * 3 + j] = stof(line.substr(0, line.size() - end));
            line = line.substr(end + 1, line.size() - end - 1);
        }
    }


    pc_file.close();

    return pc;
}

void writePLY(float* aligned_pc, int count, const char* path) {
    std::fstream output_file;
    output_file.open(path, std::ios::out);
    output_file << "ply\nformat ascii 1.0\nelement vertex " << count << "\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n";
    for (int i = 0; i < count; i++) {
        output_file << aligned_pc[i * 3] << " " << aligned_pc[i * 3 + 1] << " " << aligned_pc[i * 3 + 2] << "\n";
    }
    output_file.close();
}
/*
__global__ void iteration(float* pc1, float* pc2, int* pc1_count, int* pc2_count, results* devOut)
{
    int i = threadIdx.x;
    //pro SAMPLES vypocitat transformaci
    //alignout
    //vypocitat chybu
    atomicCAS(&mutex, 0, 1); //mozna ve whileu
    if (tmp->error < devOut->error) {
        devOut->error = tmp->error;
        devOut->scale = tmp->scale;
        for (int i = 0; i < 3; i++) {
            devOut->translation[i] = tmp->translation[i];
            for (int j = 0; j < 3; j++) {
                devOut->rotation[i * 3 + j] = tmp->rotation[i * 3 + j];
            }
        }
        
    }
    atomicExch(&mutex, 0);
    //atomicke operace?
    //mozna paralelni redukce?
}*/

__global__ void symmetricFactorization(float* devIn, float* devOut, int m, int n) {
    //vznikne matice mxm
    int id = threadIdx.x;
    devOut[id] = 0;
    if ((id % (m + 1)) == 0) {
        for (int i = id % m; i < m * m; i+= m) {
            
            devOut[id] += devIn[i] * devIn[i];
        }
    }
    else {
        for (int i = 0; i < m; i++) {
            printf("id: %d, index: %d, %d\n", id, id / m + i * m, id % m + i * m);
            devOut[id] += devIn[id % m + i * m] * devIn[id / m + i * m];
        }
    }

}

__global__ void countThreads(int* n, int* count) {
    int id = threadIdx.x;
    atomicAdd(count, *n - id);
}


void SVD(float* matrix) {
    int m = 3;
    int num_of_threads = m * m;
    int* devM;
    float* matrixOut;
    float* factorized_matrix = new float[m*m];
    cudaMalloc((void**)&devM, sizeof(int));
    cudaMalloc((void**)&matrixOut, m * m * sizeof(float));
    cudaMemcpy(devM, &m, sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridRes(1, 1, 1);
    dim3 blockRes(num_of_threads, 1, 1);
    //countThreads << < gridRes, blockRes >> > (devM, matrix_threads);
    //cudaDeviceSynchronize();
    //cudaMemcpy(&num_of_threads, matrix_threads, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrix_threads, &num_of_threads, sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "Threads to be run: " << num_of_threads << std::endl;

    blockRes.x = num_of_threads;
    symmetricFactorization << < gridRes, blockRes >> > (matrix_GPU, matrixOut, m, m);
    cudaDeviceSynchronize();
    cudaMemcpy(factorized_matrix, matrixOut, m * m * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << matrix[i * m + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "--------------" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << factorized_matrix[i * m + j] << " ";
        }
        std::cout << std::endl;
    }

    // TODO: EIGEN VALUES OF FACTORIZED MATRIX
    //       S = diag(sqrt(e1), sqrt(e2), sqrt(e3))
    //       ...

    cudaFree(devM);
    cudaFree(matrixOut);

}

__global__ void alignment(float* pc, results* transform, float* devOut)
{
    int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    devOut[i * 3] = transform->scale * (pc[i * 3] * transform->rotation[0] + pc[i * 3 + 1] * transform->rotation[1] + pc[i * 3 + 2] * transform->rotation[2]) + transform->translation[0];
    devOut[i * 3 + 1] = transform->scale * (pc[i * 3] * transform->rotation[3] + pc[i * 3 + 1] * transform->rotation[4] + pc[i * 3 + 2] * transform->rotation[5]) + transform->translation[1];
    devOut[i * 3 + 2] = transform->scale * (pc[i * 3] * transform->rotation[6] + pc[i * 3 + 1] * transform->rotation[7] + pc[i * 3 + 2] * transform->rotation[8]) + transform->translation[2];

}

void initializeCUDA(float * pc1, float* pc2) {
    cudaMalloc((void**)&devInPC1, *pc_count1 * 3 * sizeof(float));
    cudaMalloc((void**)&devInPC2, *pc_count2 * 3 * sizeof(float));
    cudaMalloc((void**)&devOutPC, *pc_count2 * 3 * sizeof(float));
    cudaMalloc((void**)&devOut, sizeof(results));
    cudaMalloc((void**)&matrix_threads, sizeof(int));


    //CHECK_ERROR();
    cudaMemcpy(devInPC1, pc1, *pc_count1 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devInPC2, pc2, *pc_count2 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devOut, &best_result, sizeof(results), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&(devInCount1), sizeof(int));
    cudaMemset(devInCount1, *pc_count1, sizeof(int));
    cudaMalloc((void**)&(devInCount2), sizeof(int));
    cudaMemset(devInCount2, *pc_count2, sizeof(int));
    cudaMemset(matrix_threads, 0, sizeof(int));




    // Bohuzel CHECK_ERROR ani eventy mi v mem prostredi zpusobovaly kompilacni errory 
    //CHECK_ERROR(cudaEventCreate(&start));
    //CHECK_ERROR(cudaEventCreate(&stop));
}


void finalizeCUDA() {

    cudaFree(devInCount1);
    cudaFree(devOut);
    cudaFree(devInCount2);
    cudaFree(devInPC1);
    cudaFree(devInPC2);
    cudaFree(devOutPC);

}

void ransacGPU() {
    int numBlocks = (int)ceil((float)NUM_OF_ITERATIONS / (float)(THREADS_PER_BLOCK));
    dim3 gridRes(numBlocks, 1, 1);
    dim3 blockRes(THREADS_PER_BLOCK, 1, 1);
    /*for (int i = 0; i < NUM_OF_ITERATIONS; i++) {
        iteration<<< gridRes, blockRes >>> (devInPC1, devInPC2, devInCount1, devInCount2, devOut);
    }*/

    cudaDeviceSynchronize();
    //cudaMemcpy(best_result, devOut, sizeof(results), cudaMemcpyDeviceToHost);
    /*float* m = new float[20];
    for (int i = 0; i < 20; i++) {
        m[i] = 0;
    }
    m[0] = 1;
    m[4] = 2;
    m[7] = 3;
    m[16] = 2;*/
    float* m = new float[9];
    for (int i = 0; i < 9; i++) {
        m[i] = 1;
    }
    m[0] = 3;
    m[2] = 0;
    m[4] = 2;
    m[5] = 2;
    m[6] = 0;

    cudaMalloc((void**)&matrix_GPU, 20 * sizeof(float));
    cudaMemcpy(matrix_GPU, m, 20 * sizeof(float), cudaMemcpyHostToDevice);

    SVD(m);
    //implementovat vlastni SVD nebo pouzit CUDA funkci?
    numBlocks = (int)ceil((float)*pc_count2 / (float)(THREADS_PER_BLOCK));
    dim3 gridResAlign(numBlocks, 1, 1);
    alignment <<< gridResAlign, blockRes >>> (devInPC2, devOut, devOutPC);
    cudaDeviceSynchronize();

    
    cudaMemcpy(result_pc, devOutPC, *pc_count2 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&best_result, devOut, sizeof(results), cudaMemcpyDeviceToHost);

    //todo spustit pro kazdou iteraci vlakno
    //todo v kazde iteraci vlakno pro urcitou podmnozinu bodu
    //zarovnani
    //todo nasobeni matic GPU?
}

int main()
{
    float* pc1;
    pc_count1 = new int;
    pc1 = readPLY("source.ply", pc_count1);
    printf("Point-cloud 1 loaded, %d vertices\n", *pc_count1);
    //writePLY(pc1, *pc_count1, "out.ply");
    
    float* pc2;
    pc_count2 = new int;
    pc2 = readPLY("target.ply", pc_count2);
    printf("Point-cloud 2 loaded, %d vertices\n", *pc_count2);
    result_pc = new float[*pc_count2 * 3];

    initializeCUDA(pc1, pc2);
    ransacGPU();
    writePLY(result_pc, *pc_count2, "out.ply");

    finalizeCUDA();

    return 0;
}


