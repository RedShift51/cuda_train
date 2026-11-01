#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>


// C[MxN] = A[MxK] * B[KxN]
template <int WM, int WN, int WK>
__global__ void wmma0(int M, int N, int K, const __half *A, const __half *B, float *C)
{
    int tid = threadIdx.x;
    int warp = tid / 32;
    int wy = warp / 2, wx = warp % 2;
    int row0 = (blockIdx.y * 2 + wy) * WM, col0 = (blockIdx.x * 2 + wx) * WN;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WM, WN, WK, __half, nvcuda::wmma::row_major> a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WM, WN, WK, __half, nvcuda::wmma::row_major> b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, float> c;
    nvcuda::wmma::fill_fragment(c, 0.0f);

    for (int k = 0; k < K; k += WK)
    {
        nvcuda::wmma::load_matrix_sync(a, A + row0 * K + k, K);
        nvcuda::wmma::load_matrix_sync(b, B + k * N + col0, N);
        nvcuda::wmma::mma_sync(c, a, b, c);
    }
    nvcuda::wmma::store_matrix_sync(
        C + row0 * N + col0,
        c,
        N,
        nvcuda::wmma::mem_row_major
    );
}


// C[MxN] = A[MxK] * B[KxN], smem + tensor cores 
template <int WM, int WN, int WK, int BM, int BN, int BK, int WARPS_Y, int WARPS_X, int NT>
__global__ void wmma1(int M, int N, int K, const __half *A, const __half *B, float *C)
{
    int tid = threadIdx.x;
    int warp = tid / 32;
    int wy = warp / WARPS_X, wx = warp % WARPS_X;
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

    __shared__ __align__(32) __half As[BM][BK], Bs[BK][BN];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WM, WN, WK, __half, nvcuda::wmma::row_major> a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WM, WN, WK, __half, nvcuda::wmma::row_major> b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, float> c;
    nvcuda::wmma::fill_fragment(c, 0.0f);

    for (int k = 0; k < K; k += BK)
    {
        #pragma unroll
        for (int i = 0; i < BM * BK / NT; i++)
        {
            int idx = tid + i * NT;
            int rr = idx / BK, cc = idx % BK;
            As[rr][cc] = A[(row0 + rr) * K + k + cc];
        }
        #pragma unroll
        for (int i = 0; i < BK * BN / NT; i++)
        {
            int idx = tid + i * NT;
            int rr = idx / BN, cc = idx % BN;
            Bs[rr][cc] = B[(k + rr) * N + col0 + cc];
        }
        __syncthreads();

        for (int kk = 0; kk < BK; kk += WK)
        {
            nvcuda::wmma::load_matrix_sync(a, &As[wy * WM][kk], BK);
            nvcuda::wmma::load_matrix_sync(b, &Bs[kk][wx * WN], BN);
            nvcuda::wmma::mma_sync(c, a, b, c);
        }
        __syncthreads();
    }
    nvcuda::wmma::store_matrix_sync(
        C + (row0 + wy * WM) * N + col0 + wx * WN,
        c,
        N,
        nvcuda::wmma::mem_row_major
    );
}


// C[MxN] = A[MxK] * B[KxN], tensor cores+bank SKEW + vector upload 
template <int WM, int WN, int WK, int FM, int FN, int BM, int BN, int BK, int WARPS_Y, int WARPS_X, int NT, int SKEW>
__global__ void wmma2(int M, int N, int K, const __half *A, const __half *B, float *C)
{
    int tid = threadIdx.x;
    int warp = tid / 32;
    int wy = warp / WARPS_X, wx = warp % WARPS_X;
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

    __shared__ __align__(32) __half As[BM][BK + SKEW], Bs[BK][BN + SKEW];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WM, WN, WK, __half, nvcuda::wmma::row_major> a[FM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WM, WN, WK, __half, nvcuda::wmma::row_major> b[FN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, float> c[FM][FN];
    #pragma unroll
    for (int ai = 0; ai < FM; ai++)
    {
        #pragma unroll
        for (int bi = 0; bi < FN; bi++)
        {
            nvcuda::wmma::fill_fragment(c[ai][bi], 0.0f);
        }
    }

    for (int k = 0; k < K; k += BK)
    {
        #pragma unroll
        for (int i = 0; i < BM * BK / NT / 8; i++)
        {
            int idx = tid + i * NT;
            int rr = idx / (BK / 8), cc = 8 * (idx % (BK / 8));
            uint4 v = reinterpret_cast<const uint4 *>(&A[(row0 + rr) * K + k + cc])[0];
            reinterpret_cast<uint4*>(&As[rr][cc])[0] = v;
        }
        #pragma unroll
        for (int i = 0; i < BK * BN / NT / 8; i++)
        {
            int idx = tid + i * NT;
            int rr = idx / (BN / 8), cc = 8 * (idx % (BN / 8));
            uint4 v = reinterpret_cast<const uint4*>(&B[(k + rr) * N + col0 + cc])[0];
            reinterpret_cast<uint4*>(&Bs[rr][cc])[0] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WK)
        {
            #pragma unroll
            for (int ai = 0; ai < FM; ai++)
            {
                nvcuda::wmma::load_matrix_sync(a[ai], &As[(wy * FM + ai) * WM][kk], BK + SKEW);
            }
            #pragma unroll
            for (int bi = 0; bi < FN; bi++)
            {
                nvcuda::wmma::load_matrix_sync(b[bi], &Bs[kk][(wx * FN + bi) * WN], BN + SKEW);
            }

            #pragma unroll
            for (int ai = 0; ai < FM; ai++)
            {
                #pragma unroll
                for (int bi = 0; bi < FN; bi++)
                {
                    nvcuda::wmma::mma_sync(c[ai][bi], a[ai], b[bi], c[ai][bi]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ai = 0; ai < FM; ai++)
    {
        #pragma unroll
        for (int bi = 0; bi < FN; bi++)
        {
            nvcuda::wmma::store_matrix_sync(
                C + (row0 + (wy * FM + ai) * WM) * N + col0 + (wx * FN + bi) * WN,
                c[ai][bi],
                N,
                nvcuda::wmma::mem_row_major
            );
        }
    }
}


// C[MxN] = A[MxK] * B[KxN], tensor cores + async upload + smem
template <int WM, int WN, int WK, int FM, int FN, int BM, int BN, int BK, int WARPS_Y, int WARPS_X, int NT, int SKEW>
__global__ void wmma3(int M, int N, int K, const __half *A, const __half *B, float *C)
{
    int tid = threadIdx.x;
    int warp = tid / 32;
    int wy = warp / WARPS_X, wx = warp % WARPS_X;
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

    __shared__ __align__(32) __half As[2][BM][BK + SKEW], Bs[2][BK][BN + SKEW];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WM, WN, WK, __half, nvcuda::wmma::row_major> a[FM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WM, WN, WK, __half, nvcuda::wmma::row_major> b[FN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, float> c[FM][FN];
    #pragma unroll
    for (int ai = 0; ai < FM; ai++)
    {
        #pragma unroll
        for (int bi = 0; bi < FN; bi++)
        {
            nvcuda::wmma::fill_fragment(c[ai][bi], 0.0f);
        }
    }

    auto load_tile = [&] (int buf, int k)
    {
        #pragma unroll
        for (int i = 0; i < BM * BK / NT / 8; i++)
        {
            int idx = tid + i * NT;
            int rr = idx / (BK / 8), cc = 8 * (idx % (BK / 8));
            __pipeline_memcpy_async(
                &As[buf][rr][cc],
                &A[(row0 + rr) * K + k + cc],
                16
            );
        }
        #pragma unroll
        for (int i = 0; i < BK * BN / NT / 8; i++)
        {
            int idx = tid + i * NT;
            int rr = idx / (BN / 8), cc = 8 * (idx % (BN / 8));
            __pipeline_memcpy_async(
                &Bs[buf][rr][cc],
                &B[(k + rr) * N + col0 + cc],
                16
            );
        }
    };

    int nt = K / BK;
    load_tile(0, 0);
    __pipeline_commit();

    for (int t = 0; t < nt; t++)
    {
        int cur = t % 2;
        __pipeline_wait_prior(0);
        __syncthreads();
        if (t + 1 < nt)
        {
            load_tile(1 - cur, (t + 1) * BK);
            __pipeline_commit();
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WK)
        {
            #pragma unroll
            for (int ai = 0; ai < FM; ai++)
            {
                nvcuda::wmma::load_matrix_sync(a[ai], &As[cur][(wy * FM + ai) * WM][kk], BK + SKEW);
            }
            #pragma unroll
            for (int bi = 0; bi < FN; bi++)
            {
                nvcuda::wmma::load_matrix_sync(b[bi], &Bs[cur][kk][(wx * FN + bi) * WN], BN + SKEW);
            }

            #pragma unroll
            for (int ai = 0; ai < FM; ai++)
            {
                #pragma unroll
                for (int bi = 0; bi < FN; bi++)
                {
                    nvcuda::wmma::mma_sync(c[ai][bi], a[ai], b[bi], c[ai][bi]);
                }
            }
        }
    }

    #pragma unroll
    for (int ai = 0; ai < FM; ai++)
    {
        #pragma unroll
        for (int bi = 0; bi < FN; bi++)
        {
            nvcuda::wmma::store_matrix_sync(
                C + (row0 + (wy * FM + ai) * WM) * N + col0 + (wx * FN + bi) * WN,
                c[ai][bi],
                N,
                nvcuda::wmma::mem_row_major
            );
        }
    }
}

template <typename F>
float bench(F launch, int R = 30)
{
    for (int i = 0; i < 3; i++) { launch(); }
    std::vector<cudaEvent_t> s(R), e(R);
    for (int i = 0; i < R; i++) { cudaEventCreate(&s[i]); cudaEventCreate(&e[i]); }
    cudaDeviceSynchronize();
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
    std::vector<__half> hA(size_t(M) * K), hB(size_t(N) * K);
    std::vector<float> hC(size_t(M) * N), hRef(size_t(M) * N);
    for (size_t i = 0; i < hA.size(); i++) { hA[i] = rand() / float(RAND_MAX) * 2.0f - 1.0f; }
    for (size_t i = 0; i < hB.size(); i++) { hB[i] = rand() / float(RAND_MAX) * 2.0f - 1.0f; }
    
    __half *A, *B;
    float *C, *Ref;
    cudaMalloc(&A, sizeof(__half) * hA.size());
    cudaMalloc(&B, sizeof(__half) * hB.size());
    cudaMalloc(&C, sizeof(float) * hC.size());
    cudaMalloc(&Ref, sizeof(float) * hRef.size());

    cudaMemcpy(A, hA.data(), sizeof(__half) * hA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB.data(), sizeof(__half) * hB.size(), cudaMemcpyHostToDevice);

    // cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    float ms = bench([&]
    {
        cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, CUDA_R_16F, N,
            A, CUDA_R_16F, K,
            &beta,
            Ref, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );
    });
    printf("\ncublas %.5f\n", ms);
    cudaMemcpy(hRef.data(), Ref, sizeof(float) * hRef.size(), cudaMemcpyDeviceToHost);

    auto report = [&] (const char *name, float ms)
    {
        cudaMemcpy(hC.data(), C, sizeof(float) * hC.size(), cudaMemcpyDeviceToHost);
        float maxerr = 0.0f;
        for (size_t i = 0; i < hC.size(); i++) { maxerr = std::max(maxerr, std::abs(hC[i] - hRef[i])); }
        printf("\n%s %.5f ms | maxerr %.5f\n", name, ms, maxerr);
        cudaMemset(C, 0.0f, sizeof(float) * hC.size());
    };

    dim3 grid0(N / 32, M / 32), block0(128);
    dim3 grid1(N / 64, M / 64), block1(512);
    dim3 grid2(N / 128, M / 128), block2(512);

    report("wmma0", bench([&] { wmma0<16, 16, 16><<<grid0, block0>>>(M, N, K, A, B, C); }));
    report("wmma1", bench([&] { wmma1<16, 16, 16, 64, 64, 32, 4, 4, 512><<<grid1, block1>>>(M, N, K, A, B, C); }));
    report("wmma2", bench([&] { wmma2<16, 16, 16, 2, 2, 128, 128, 32, 4, 4, 512, 8><<<grid2, block2>>>(M, N, K, A, B, C); }));
    report("wmma3", bench([&] { wmma3<16, 16, 16, 2, 2, 128, 128, 32, 4, 4, 512, 8><<<grid2, block2>>>(M, N, K, A, B, C); }));
    // another registers pack
    report("wmma3 256", bench([&] { wmma3<16, 16, 16, 4, 2, 128, 128, 32, 2, 4, 256, 8><<<grid2, dim3(256)>>>(M, N, K, A, B, C); }));

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(Ref);
    cublasDestroy(handle);
    return 0;
}
















































