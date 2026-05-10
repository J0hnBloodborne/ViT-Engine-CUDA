#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <math.h>

// ViT & Ampere Specific Constants
#define E 768 // Total embedding dimension
#define H_DIM 12  // Number of heads
#define D 64  // Head dimension (768 / 12)
#define VEC_SIZE 4
#define VECS_PER_ROW 16 // 64 / 4
#define THREADS_PER_BLOCK 128
#define WARPS_PER_BLOCK 4
#define ROWS_PER_WARP 2
#define BR 8 // WARPS_PER_BLOCK * ROWS_PER_WARP
#define BC 16

__global__ void flash_attn_2_ampere_768(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O, int B, int N,float scale) {
    int b = blockIdx.x / H_DIM;
    int h = blockIdx.x % H_DIM;
    int i_start = blockIdx.y * BR;
    
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    // 1 warp handles 2 rows. 1 half-warp (16 threads) handles 1 row of D=64.
    int local_row_idx = warp_id * 2 + (lane_id >= 16 ? 1 : 0);
    int global_row_idx = i_start + local_row_idx;
    int vec_idx = lane_id % 16; 

    // Base offset for this batch (b)
    int base_offset = b * N * E;
    
    // Load Q into registers
    float4 q_vec = {0.0f, 0.0f, 0.0f, 0.0f};
    if (global_row_idx < N) {
        int q_offset = base_offset + (global_row_idx * E) + (h * D) + (vec_idx * VEC_SIZE);
        q_vec = reinterpret_cast<const float4*>(Q + q_offset)[0];
    }

    // Shared memory: 4 tiles total for double buffering
    extern __shared__ float4 s_mem[];
    float4* s_K[2];
    float4* s_V[2];
    s_K[0] = s_mem;                                  // [BC][16]
    s_V[0] = s_mem + BC * VECS_PER_ROW;              // [BC][16]
    s_K[1] = s_mem + 2 * BC * VECS_PER_ROW;          // [BC][16]
    s_V[1] = s_mem + 3 * BC * VECS_PER_ROW;          // [BC][16]

    float m = -INFINITY;
    float l = 0.0f;
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // Helper lambda for async loading
    auto load_tile_async = [&](int j_start_val, int buf_idx) {
        for (int load_iter = 0; load_iter < 2; ++load_iter) {
            int flat_idx = tid + load_iter * THREADS_PER_BLOCK;
            int k_row = flat_idx / VECS_PER_ROW;
            int k_vec = flat_idx % VECS_PER_ROW;
            
            if (j_start_val + k_row < N) {
                int offset = base_offset + ((j_start_val + k_row) * E) + (h * D) + (k_vec * VEC_SIZE);
                __pipeline_memcpy_async(&s_K[buf_idx][flat_idx], &K[offset], sizeof(float4));
                __pipeline_memcpy_async(&s_V[buf_idx][flat_idx], &V[offset], sizeof(float4));
            } else {
                s_K[buf_idx][flat_idx] = {0.0f, 0.0f, 0.0f, 0.0f};
                s_V[buf_idx][flat_idx] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
    };

    // Prologue: start loading tile 0
    load_tile_async(0, 0);
    __pipeline_commit();

    int num_tiles = (N + BC - 1) / BC;

    for (int j_step = 0; j_step < num_tiles; ++j_step) {
        int j_start_curr = j_step * BC;
        int j_start_next = (j_step + 1) * BC;
        int buf_curr = j_step % 2;
        int buf_next = (j_step + 1) % 2;

        // Issue async load for NEXT tile
        if (j_start_next < N) {
            load_tile_async(j_start_next, buf_next);
        }
        __pipeline_commit();

        // Wait for CURRENT tile to be fully loaded (1 stage remaining in pipeline)
        __pipeline_wait_prior(1);
        __syncthreads();

        float S[BC];
        float m_tile = -INFINITY;

        // Compute S_j = Q_i @ K_j^T
        for (int j = 0; j < BC; ++j) {
            float4 k_vec = s_K[buf_curr][j * VECS_PER_ROW + vec_idx];
            float dot = 0.0f;
            dot = __fmaf_rn(q_vec.x, k_vec.x, dot);
            dot = __fmaf_rn(q_vec.y, k_vec.y, dot);
            dot = __fmaf_rn(q_vec.z, k_vec.z, dot);
            dot = __fmaf_rn(q_vec.w, k_vec.w, dot);
            
            // Fast Ampere intra-warp reduction
            dot += __shfl_down_sync(0xffffffff, dot, 8);
            dot += __shfl_down_sync(0xffffffff, dot, 4);
            dot += __shfl_down_sync(0xffffffff, dot, 2);
            dot += __shfl_down_sync(0xffffffff, dot, 1);
            
            // Broadcast
            if (lane_id < 16) dot = __shfl_sync(0xffffffff, dot, 0);
            else dot = __shfl_sync(0xffffffff, dot, 16);
            
            dot *= scale;
            if (j_start_curr + j >= N) dot = -INFINITY;

            S[j] = dot;
            if (dot > m_tile) m_tile = dot;
        }

        // Online Softmax
        float m_new = max(m, m_tile);
        // Using __expf for faster Ampere hardware execution
        float exp_prev = __expf(m - m_new);
        float l_tile = 0.0f;

        for (int j = 0; j < BC; ++j) {
            S[j] = __expf(S[j] - m_new);
            l_tile += S[j];
        }
        float l_new = l * exp_prev + l_tile;

        acc.x *= exp_prev;
        acc.y *= exp_prev;
        acc.z *= exp_prev;
        acc.w *= exp_prev;

        for (int j = 0; j < BC; ++j) {
            float4 v_vec = s_V[buf_curr][j * VECS_PER_ROW + vec_idx];
            acc.x = __fmaf_rn(S[j], v_vec.x, acc.x);
            acc.y = __fmaf_rn(S[j], v_vec.y, acc.y);
            acc.z = __fmaf_rn(S[j], v_vec.z, acc.z);
            acc.w = __fmaf_rn(S[j], v_vec.w, acc.w);
        }

        m = m_new;
        l = l_new;
        
        // Sync before the next loop iteration potentially overwrites buf_curr
        __syncthreads();
    }

    // Finalize output
    if (global_row_idx < N) {
        acc.x = __fdividef(acc.x, l);
        acc.y = __fdividef(acc.y, l);
        acc.z = __fdividef(acc.z, l);
        acc.w = __fdividef(acc.w, l);
        
        int o_offset = base_offset + (global_row_idx * E) + (h * D) + (vec_idx * VEC_SIZE);
        reinterpret_cast<float4*>(O + o_offset)[0] = acc;
    }
}

void flash_attn_2_forward_cuda(const float* Q, const float* K, const float* V, float* O, int B, int N, float scale) {
    dim3 grid(B * H_DIM, (N + BR - 1) / BR);
    dim3 block(THREADS_PER_BLOCK);

    // Shared memory: 4 tiles (2 K, 2 V) for double buffering
    size_t shared_mem_size = 4 * BC * VECS_PER_ROW * sizeof(float4);
    
    flash_attn_2_ampere_768<<<grid, block, shared_mem_size>>>(
        Q, K, V, O, B, N, scale
    );
}
