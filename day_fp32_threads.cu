#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>


constexpr int TILE = 32;
constexpr int ROWS = 8;


// C[MxN] = A[MxK] * B[KxN]
__global__ void gemm0(int M, int N, int K, const float *A, const float *B, float *C)
{
    int x = TILE * blockIdx.y + threadIdx.y;
    int y = TILE * blockIdx.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
    {
        sum += A[x * K + k] * B[k * N + y];
    }
    if (x < M && y < N) { C[x * N + y] = sum; }
}


// C[MxN] = A[MxK] * B[KxN], add simple shared memory smem tiles
__global__ void gemm1(int M, int N, int K, const float *A, const float *B, float *C)
{
    int x = TILE * blockIdx.y + threadIdx.y;
    int y = TILE * blockIdx.x + threadIdx.x;

    __shared__ float tile_a[TILE][TILE], tile_b[TILE][TILE];

    float sum = 0.0f;
    for (int k = 0; k < K; k += TILE)
    {
        tile_a[threadIdx.y][threadIdx.x] = (x < M && k + threadIdx.x < K) ? A[x * K + k + threadIdx.x] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (k + threadIdx.y < K && y < N) ? B[(k + threadIdx.y) * N + y] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; kk++)
        {
            sum += tile_a[threadIdx.y][kk] * tile_b[kk][threadIdx.x];
        }
        __syncthreads();
    }
    if (x < M && y < N) { C[x * N + y] = sum; }
}


// C[MxN] = A[MxK] * B[KxN], ILP instruction level parallelism
template <int ACC>
__global__ void gemm2(int M, int N, int K, const float *A, const float *B, float *C)
{
    int x = TILE * blockIdx.y + threadIdx.y;
    int y = TILE * blockIdx.x + threadIdx.x;

    static_assert(TILE == ROWS * ACC);

    __shared__ float tile_a[TILE][TILE], tile_b[TILE][TILE];

    float sum[ACC] = {0.0f};
    for (int k = 0; k < K; k += TILE)
    {
        #pragma unroll
        for (int j = 0; j < ACC; j++)
        {
            tile_a[threadIdx.y + j * ROWS][threadIdx.x] = (x + j * ROWS < M && k + threadIdx.x < K) ? A[(x + j * ROWS) * K + k + threadIdx.x] : 0.0f;
            tile_b[threadIdx.y + j * ROWS][threadIdx.x] = (k + j * ROWS + threadIdx.y < K && y < N) ? B[(k + j * ROWS + threadIdx.y) * N + y] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; kk++)
        {
            float b = tile_b[kk][threadIdx.x];
            #pragma unroll
            for (int j = 0; j < ACC; j++)
            {
                sum[j] += tile_a[threadIdx.y + j * ROWS][kk] * b;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < ACC; j++)
    {
        if (x + j * ROWS < M && y < N) { C[(x + j * ROWS) * N + y] = sum[j]; }
    }
}


// C[MxN] = A[MxK] * B[KxN], Thread Coarsening
template <int BM, int BN, int BK, int TM, int TN, int TY, int TX, int NT>
__global__ void gemm3(int M, int N, int K, const float *A, const float *B, float *C)
{
    __shared__ float As[BM][BK], Bs[BK][BN];

    static_assert(BM == TM * TY && BN == TN * TX && NT == TY * TX);

    int tid = threadIdx.y * TX + threadIdx.x;
    int row0 = blockIdx.y * BM;
    int col0 = blockIdx.x * BN;

    float sum[TM][TN] = {0.0f};
    float a[TM], b[TN];

    for (int k = 0; k < K; k += BK)
    {
        #pragma unroll
        for (int i = 0; i < BM * BK / NT; i++)
        {
            int idx = tid + i * NT;
            int r = idx / BK, c = idx % BK;
            As[r][c] = (row0 + r < M && k + c < K) ? A[(row0 + r) * K + k + c] : 0.0f;
        }

        #pragma unroll
        for (int i = 0; i < BK * BN / NT; i++)
        {
            int idx = tid + i * NT;
            int r = idx / BN, c = idx % BN;
            Bs[r][c] = (k + r < K && col0 + c < N) ? B[(k + r) * N + col0 + c] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++)
        {
            #pragma unroll
            for (int ai = 0; ai < TM; ai++)
            {
                a[ai] = As[threadIdx.y + ai * TY][kk];
            }
            #pragma unroll
            for (int bi = 0; bi < TN; bi++)
            {
                b[bi] = Bs[kk][threadIdx.x + bi * TX];
            }

            #pragma unroll
            for (int ai = 0; ai < TM; ai++)
            {
                #pragma unroll
                for (int bi = 0; bi < TN; bi++)
                {
                    sum[ai][bi] += a[ai] * b[bi];
                }
            }

        }
        __syncthreads();
    }
    #pragma unroll
    for (int ai = 0; ai < TM; ai++)
    {
        #pragma unroll
        for (int bi = 0; bi < TN; bi++)
        {
            if (row0 + ai * TY + threadIdx.y < M && col0 + bi * TX + threadIdx.x < N)
            {
                C[(row0 + ai * TY + threadIdx.y) * N + col0 + bi * TX + threadIdx.x] = sum[ai][bi];
            }
        }
    }
}


// C[MxN] = A[MxK] * B[KxN], vectorized uploading
template <int BM, int BN, int BK, int TM, int TN, int TY, int TX, int NT>
__global__ void gemm4(int M, int N, int K, const float *A, const float *B, float *C)
{
    __shared__ __align__(16) float As[BK][BM], Bs[BK][BN];

    static_assert(BM == TM * TY && BN == TN * TX && NT == TY * TX);

    int tid = threadIdx.y * TX + threadIdx.x;
    int row0 = blockIdx.y * BM;
    int col0 = blockIdx.x * BN;

    float sum[TM][TN] = {0.0f};
    float a[TM], b[TN];

    for (int k = 0; k < K; k += BK)
    {
        #pragma unroll
        for (int i = 0; i < BM * BK / NT / 4; i++)
        {
            int idx = tid + i * NT;
            int r = idx / (BK / 4), c = 4 * (idx % (BK / 4));
            float4 v = reinterpret_cast<const float4*>(&A[(row0 + r) * K + k + c])[0];
            As[c + 0][r] = v.x;
            As[c + 1][r] = v.y;
            As[c + 2][r] = v.z;
            As[c + 3][r] = v.w;
        }

        #pragma unroll
        for (int i = 0; i < BN * BK / NT / 4; i++)
        {
            int idx = tid + i * NT;
            int r = idx / (BN / 4), c = 4 * (idx % (BN / 4));
            float4 v = reinterpret_cast<const float4*>(&B[(k + r) * N + col0 + c])[0];
            Bs[r][c + 0] = v.x;
            Bs[r][c + 1] = v.y;
            Bs[r][c + 2] = v.z;
            Bs[r][c + 3] = v.w;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++)
        {
            #pragma unroll
            for (int ai = 0; ai < TM / 4; ai++)
            {
                float4 v = reinterpret_cast<const float4*>(&As[kk][threadIdx.y * TM + 4 * ai])[0];
                a[4 * ai + 0] = v.x;
                a[4 * ai + 1] = v.y;
                a[4 * ai + 2] = v.z;
                a[4 * ai + 3] = v.w;
            }
            #pragma unroll
            for (int bi = 0; bi < TN / 4; bi++)
            {
                float4 v = reinterpret_cast<const float4*>(&Bs[kk][threadIdx.x * TN + 4 * bi])[0];
                b[4 * bi + 0] = v.x;
                b[4 * bi + 1] = v.y;
                b[4 * bi + 2] = v.z;
                b[4 * bi + 3] = v.w;
            }

            #pragma unroll
            for (int ai = 0; ai < TM; ai++)
            {
                #pragma unroll
                for (int bi = 0; bi < TN; bi++)
                {
                    sum[ai][bi] += a[ai] * b[bi];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int ai = 0; ai < TM; ai++)
    {
        #pragma unroll
        for (int bi = 0; bi < TN / 4; bi++)
        {
            float4 v = make_float4(sum[ai][4 * bi + 0], sum[ai][4 * bi + 1],
                                sum[ai][4 * bi + 2], sum[ai][4 * bi + 3]);
            reinterpret_cast<float4*>(
                &C[(row0 + threadIdx.y * TM + ai) * N + col0 + threadIdx.x * TN + 4 * bi]
            )[0] = v;
        }
    }
}


template <typename F>
float bench(F launch, int R = 20)
{
    for (int i = 0; i < 3; i++) { launch(); }

    std::vector<cudaEvent_t> s(R), e(R);
    for (int i = 0; i < R; i++) { cudaEventCreate(&s[i]); cudaEventCreate(&e[i]); }
    for (int i = 0; i < R; i++)
    {
        cudaEventRecord(s[i]);
        launch();
        cudaEventRecord(e[i]);
    }
    cudaDeviceSynchronize();

    std::vector<float> t(R);
    for (int i = 0; i < R; i++)
    {
        cudaEventElapsedTime(&t[i], s[i], e[i]);
        cudaEventDestroy(s[i]);
        cudaEventDestroy(e[i]);
    }

    std::sort(t.begin(), t.end());
    return t[R / 2];
}



int main() 
{
    int M = 4096, N = 4096, K = 4096;
    std::vector<float> hA(size_t(M) * K), hB(size_t(K) * N), hC(size_t(M) * N), hRef(size_t(M) * N);
    for (auto& v : hA) { v = rand() / float(RAND_MAX) * 2.0f - 1.0f; }
    for (auto& v : hB) { v = rand() / float(RAND_MAX) * 2.0f - 1.0f; }
    float *A, *B, *C, *Ref;
    cudaMalloc(&A, sizeof(float) * hA.size());
    cudaMalloc(&B, sizeof(float) * hB.size());
    cudaMalloc(&C, sizeof(float) * hC.size());
    cudaMalloc(&Ref, sizeof(float) * hRef.size());
    
    cudaMemcpy(A, hA.data(), sizeof(float) * hA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB.data(), sizeof(float) * hB.size(), cudaMemcpyHostToDevice);
    
    // cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    float alpha = 1.0f, beta = 0.0f;
    float ms = bench([&] {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, B, N, A, K, &beta, Ref, N);
    } );
    cudaMemcpy(hRef.data(), Ref, sizeof(float) * hRef.size(), cudaMemcpyDeviceToHost);
    printf("\ncublas %.5f\n", ms);

    // gemm
    auto report = [&] (const char *name, float ms)
    {
        cudaMemcpy(hC.data(), C, sizeof(float) * hC.size(), cudaMemcpyDeviceToHost);
        float maxerr = 0.0f;
        for (size_t i = 0; i < hC.size(); i++) { maxerr = std::max(maxerr, std::abs(hC[i] - hRef[i])); }
        printf("\n%s %.5f ms | %.5f err\n", name, ms, maxerr);
        cudaMemset(C, 0, sizeof(float) * hC.size());
    };
    
    dim3 grid0((N + TILE - 1) / TILE, (M + TILE - 1) / TILE), block0(TILE, TILE);
    dim3 block1(TILE, ROWS);
    dim3 grid2((N + 63) / 64, (M + 63) / 64), block2(16, 16);
    dim3 grid3((N + 127) / 128, (M + 127) / 128), block3(16, 16);
    dim3 grid4((N + 63) / 64, (M + 127) / 128), block4(16, 16);
    
    report("gemm0", bench([&]{ gemm0<<<grid0, block0>>>(M, N, K, A, B, C); }));
    report("gemm1", bench([&]{ gemm1<<<grid0, block0>>>(M, N, K, A, B, C); }));
    report("gemm2", bench([&]{ gemm2<TILE/ROWS><<<grid0, block1>>>(M, N, K, A, B, C); }));
    report("gemm3 0", bench([&]{
        gemm3<64, 64, 16, 4, 4, 16, 16, 256><<<grid2, block2>>>(M, N, K, A, B, C);
    }));
    report("gemm3 1", bench([&]{
        gemm3<128, 128, 16, 8, 8, 16, 16, 256><<<grid3, block3>>>(M, N, K, A, B, C);
    }));
    report("gemm3 2", bench([&]{  
        gemm3<128, 64, 16, 8, 4, 16, 16, 256><<<grid4, block4>>>(M, N, K, A, B, C);
    }));   
    report("gemm4 0", bench([&]{
        gemm4<64, 64, 16, 4, 4, 16, 16, 256><<<grid2, block2>>>(M, N, K, A, B, C);
    }));
    report("gemm4 1", bench([&]{
        gemm4<128, 128, 16, 8, 8, 16, 16, 256><<<grid3, block3>>>(M, N, K, A, B, C);
    }));
    report("gemm4 2", bench([&]{
        gemm4<128, 64, 16, 8, 4, 16, 16, 256><<<grid4, block4>>>(M, N, K, A, B, C);
    }));

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(Ref);
    cublasDestroy(handle);
    
    return 0;
}   







































