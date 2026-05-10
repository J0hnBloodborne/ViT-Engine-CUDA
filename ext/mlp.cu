#include <cuda_runtime.h>
#include <math.h>

#define TILE 32

__inline__ __device__ float gelu_stable(float x) {
    // To achieve 1e-7 precision matching PyTorch's default F.gelu(x), we use erff().
    const float M_SQRT1_2 = 0.70710678118654752440f; // 1/sqrt(2)
    return 0.5f * x * (1.0f + erff(x * M_SQRT1_2));
}

// 256 threads per block, float4 vectorized
template <bool UseGeLU>
__global__ void mlp_linear_kernel(
    const float* __restrict__ X,      // [M, K]
    const float* __restrict__ W,      // [N_out, K]
    const float* __restrict__ B_bias, // [N_out]
    float* __restrict__ Y,            // [M, N_out]
    int M, int K, int N_out
) {
    int tid = threadIdx.x;
    
    // [32][33] to prevent any possible bank conflict
    __shared__ float s_X[TILE][TILE + 1]; 
    __shared__ float s_W_T[TILE][TILE + 1];

    int row_load = tid / 8;
    int col_load = (tid % 8) * 4;

    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    int global_row = blockIdx.y * TILE + row_load;
    int global_col = blockIdx.x * TILE + col_load;

    for (int t = 0; t < K; t += TILE) {
        // Coalesced float4 load for X tile
        if (blockIdx.y * TILE + row_load < M && t + col_load < K) {
            float4 x_vec = reinterpret_cast<const float4*>(&X[(blockIdx.y * TILE + row_load) * K + t + col_load])[0];
            s_X[row_load][col_load + 0] = x_vec.x;
            s_X[row_load][col_load + 1] = x_vec.y;
            s_X[row_load][col_load + 2] = x_vec.z;
            s_X[row_load][col_load + 3] = x_vec.w;
        } else {
            s_X[row_load][col_load + 0] = 0.0f;
            s_X[row_load][col_load + 1] = 0.0f;
            s_X[row_load][col_load + 2] = 0.0f;
            s_X[row_load][col_load + 3] = 0.0f;
        }

        // Coalesced float4 load for W tile. Transposed on the fly into shared memory.
        if (blockIdx.x * TILE + row_load < N_out && t + col_load < K) {
            float4 w_vec = reinterpret_cast<const float4*>(&W[(blockIdx.x * TILE + row_load) * K + t + col_load])[0];
            s_W_T[col_load + 0][row_load] = w_vec.x;
            s_W_T[col_load + 1][row_load] = w_vec.y;
            s_W_T[col_load + 2][row_load] = w_vec.z;
            s_W_T[col_load + 3][row_load] = w_vec.w;
        } else {
            s_W_T[col_load + 0][row_load] = 0.0f;
            s_W_T[col_load + 1][row_load] = 0.0f;
            s_W_T[col_load + 2][row_load] = 0.0f;
            s_W_T[col_load + 3][row_load] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            // Broadcasts across the warp
            float x_val = s_X[row_load][k];
            
            // Reads from distinct banks
            float w0 = s_W_T[k][col_load + 0];
            float w1 = s_W_T[k][col_load + 1];
            float w2 = s_W_T[k][col_load + 2];
            float w3 = s_W_T[k][col_load + 3];

            acc.x = __fmaf_rn(x_val, w0, acc.x);
            acc.y = __fmaf_rn(x_val, w1, acc.y);
            acc.z = __fmaf_rn(x_val, w2, acc.z);
            acc.w = __fmaf_rn(x_val, w3, acc.w);
        }

        __syncthreads();
    }

    if (global_row < M && global_col < N_out) {
        if (B_bias != nullptr) {
            float4 b_vec = reinterpret_cast<const float4*>(&B_bias[global_col])[0];
            acc.x += b_vec.x;
            acc.y += b_vec.y;
            acc.z += b_vec.z;
            acc.w += b_vec.w;
        }

        if (UseGeLU) {
            acc.x = gelu_stable(acc.x);
            acc.y = gelu_stable(acc.y);
            acc.z = gelu_stable(acc.z);
            acc.w = gelu_stable(acc.w);
        }

        // Single 16-byte aligned float4 write per thread!
        reinterpret_cast<float4*>(&Y[global_row * N_out + global_col])[0] = acc;
    }
}

void mlp_forward_cuda(const float* X, const float* W1, const float* B1, const float* W2, const float* B2, float* H, float* O, int B, int N, int E, int E_expand) {
    int M = B * N;
    dim3 block(256); // Exactly 256 threads
    
    // Pass 1: X @ W1^T + B1 -> H, followed by GELU
    dim3 grid1((E_expand + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    mlp_linear_kernel_vec<true><<<grid1, block>>>(X, W1, B1, H, M, E, E_expand);

    // Pass 2: H @ W2^T + B2 -> O
    dim3 grid2((E + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    mlp_linear_kernel_vec<false><<<grid2, block>>>(H, W2, B2, O, M, E_expand, E);
}
