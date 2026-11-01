# cuda_train: hand-written GEMM, FP32 → Tensor Cores

Two single-file CUDA programs that build up a 4096×4096×4096 GEMM step by step and compare every step with cuBLAS on the same card, in the same run.

| File | What it is |
|---|---|
| `day_fp32_threads.cu` | FP32 SGEMM: `gemm0` coalesced → `gemm1` shared memory → `gemm2` 1D blocktiling → `gemm3` 2D blocktiling (3 tile configs) → `gemm4` float4 loads + transposed `As` (3 tile configs) |
| `day_tensor_cores.cu` | FP16 in / FP32 accumulate on Tensor Cores (WMMA): `wmma0` fragments from global → `wmma1` smem tile → `wmma2` 2×2 fragments per warp, `uint4` loads, row skew → `wmma3` `cp.async` double buffering (16 warps, then an 8-warp config) |

## Build

Needs the CUDA toolkit (`nvcc`) and cuBLAS. Pick `-arch` for your card: A100 `sm_80`, RTX 3090 `sm_86`, L4 / RTX 40xx `sm_89`, H100 `sm_90`.

```bash
nvcc -O3 -arch=sm_80 day_fp32_threads.cu  -lcublas -o fp32
nvcc -O3 -arch=sm_80 day_tensor_cores.cu  -lcublas -o tc     # needs sm_80+ (cp.async)

./fp32
./tc
```

Registers and shared memory per kernel (this is how the occupancy numbers were checked):

```bash
nvcc -O3 -arch=sm_80 -Xptxas -v day_tensor_cores.cu -lcublas -o tc 2>&1 | grep -E "registers|smem"
cuobjdump -sass tc | grep -c LDS.128      # are the shared-memory reads vectorised?
```

## What you will see

Each line is `name  time (ms)  | max abs error vs cuBLAS`. The first line is cuBLAS itself.

```
cublas 0.63500
wmma0 7.76300 | maxerr ...
wmma1 6.83700 | maxerr ...
wmma2 4.35100 | maxerr ...
wmma3 1.30900 | maxerr ...
wmma3 256 0.82600 | maxerr ...
```

- Time is the **median** of 20 runs (FP32 file) / 30 runs (Tensor Core file), 3 warm-up launches, measured with `cudaEvent`.
- Error is against cuBLAS on the same inputs; summation order differs, so a small non-zero error is normal (not a bug).
- TFLOP/s = `2·M·N·K / time`. Compare steps with each other and with the cuBLAS line, not with datasheet peaks.

## Results from my runs (4096³)

FP32 (cuBLAS without TF32), percent of cuBLAS:

| Card | best hand-written kernel | % of cuBLAS |
|---|---|---|
| A100 SXM | `gemm4`, 4×4 / 8×4 tile | 96% |
| L4 | `gemm4`, 8×8 tile | 78% |
| H100 SXM | `gemm4`, 8×8 tile | 75% |

FP16 Tensor Cores, WMMA (FP16 in, FP32 accumulate):

| Step | A100 TFLOP/s | % of cuBLAS |
|---|---|---|
| `wmma0` fragments from global | 17.7 | 8% |
| `wmma3` + `uint4` + row skew | 80.7 | 37% |
| `wmma3` + `cp.async` | 105 | 48% |
| `wmma3`, 8 warps (2 blocks/SM) | 166 | 77% |

On H100 the same kernel reaches 267 TFLOP/s (36% of cuBLAS).

## Take-aways

- **Row padding / skew in shared memory:** ×2.3 on the Tensor Core GEMM (bank conflicts), but it *hurts* the FP32 GEMM (33-float rows can't be read as `LDS.128`).
- **Occupancy is a step function:** ≤128 registers per thread with 256 threads means two blocks per SM. 168 → 151 registers changed nothing, 122 gave the win.
- **Fewer, fatter warps** (8 instead of 16, 4×2 fragments each) gave ×1.6: more reuse per warp and a second block per SM.
- **`int` indices instead of `size_t`:** +11% on H100 (fewer integer instructions).
- **The best tile depends on the card:** the same code is 96% on A100 and 75–78% on H100/L4.
- **H100 gap:** `mma.sync` (WMMA) is Ampere-style; cuBLAS on Hopper uses `wgmma` + TMA. Part of the gap is also my simpler pipeline (no `ldmatrix`, 2 stages).

## Caveats

- Sizes are fixed in `main` (4096³). The Tensor Core kernels have **no edge handling**: M, N, K must be multiples of 128.
- Run-to-run noise: L4 is power-limited (cuBLAS varies by up to ~30%); measure the cuBLAS baseline in the same run.
- Don't trust one number: re-run, check `maxerr`, and look at registers (`-Xptxas -v`) before explaining a speed-up.
- Profiler counters (`ncu`) are blocked on many rented GPUs (`ERR_NVGPUCTRPERM`); the occupancy claims here come from register/smem counts, not from `ncu`.

Progression follows the well-known siboehm SGEMM write-up; all numbers are my own measurements.
