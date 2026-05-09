#include <cuda_runtime.h>
#include <math.h>

// Tile size for GEMM
#define TILE 32

// Fused Linear + optional GELU kernel
// Assumes W is in PyTorch format: [N_out, K]
template <bool UseGeLU>
__global__ void mlp_linear_kernel(
    const float* __restrict__ X,      // [M, K]
    const float* __restrict__ W,      // [N_out, K]
    const float* __restrict__ B_bias, // [N_out]
    float* __restrict__ Y,            // [M, N_out]
    int M, int K, int N_out
) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float s_X[TILE][TILE];
    __shared__ float s_W[TILE][TILE + 1]; // Pad to avoid bank conflicts

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Load X tile: [M, K]
        if (row < M && t * TILE + threadIdx.x < K) {
            s_X[threadIdx.y][threadIdx.x] = X[row * K + t * TILE + threadIdx.x];
        } else {
            s_X[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load W tile: W is [N_out, K]
        // We want s_W[row_in_block][col_in_block] to be W^T
        // To coalesce global reads, threadIdx.x maps to K dimension
        int w_row = blockIdx.x * TILE + threadIdx.y; // N_out dim
        int w_col = t * TILE + threadIdx.x;          // K dim
        if (w_row < N_out && w_col < K) {
            s_W[threadIdx.x][threadIdx.y] = W[w_row * K + w_col];
        } else {
            s_W[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc = __fmaf_rn(s_X[threadIdx.y][i], s_W[i][threadIdx.x], acc);
        }
        __syncthreads();
    }

    if (row < M && col < N_out) {
        if (B_bias != nullptr) {
            acc += B_bias[col];
        }

        if (UseGeLU) {
            const float SQRT_2_OVER_PI = 0.7978845608f;
            const float COEF = 0.044715f;
            float z = SQRT_2_OVER_PI * acc * (1.0f + COEF * acc * acc);
            
            float tanh_z;
            if (z > 20.0f) {
                tanh_z = 1.0f;
            } else if (z < -20.0f) {
                tanh_z = -1.0f;
            } else {
                float e2z = __expf(2.0f * z);
                tanh_z = __fdividef(e2z - 1.0f, e2z + 1.0f);
            }
            acc = 0.5f * acc * (1.0f + tanh_z);
        }

        Y[row * N_out + col] = acc;
    }
}

void mlp_forward_cuda(
    const float* X, const float* W1, const float* B1, 
    const float* W2, const float* B2, 
    float* H, float* O, 
    int B, int N, int E, int E_expand
) {
    int M = B * N;
    dim3 block(TILE, TILE);
    
    // Pass 1: X @ W1^T + B1 -> H, followed by GELU
    dim3 grid1((E_expand + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    mlp_linear_kernel<true><<<grid1, block>>>(X, W1, B1, H, M, E, E_expand);

    // Pass 2: H @ W2^T + B2 -> O
    dim3 grid2((E + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    mlp_linear_kernel<false><<<grid2, block>>>(H, W2, B2, O, M, E_expand, E);
}
