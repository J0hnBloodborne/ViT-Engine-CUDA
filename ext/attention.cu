#include <cuda_runtime.h>
#include <math.h>

// Flash Attention 2 Constants
#define BR 16
#define BC 16

__global__ void flash_attn_2_forward_kernel(
    const float* Q,    // [B, H, N, D]
    const float* K,    // [B, H, N, D]
    const float* V,    // [B, H, N, D]
    float* O,          // [B, H, N, D]
    int B, int H, int N, int D,
    float scale
) {
    // Each thread block handles one tile of Q
    int b = blockIdx.x / H;
    int h = blockIdx.x % H;
    int i_start = blockIdx.y * BR;

    int tid = threadIdx.x; // We use BR threads per block

    // Offsets for this batch and head
    int offset = (b * H + h) * N * D;
    const float* Qi = Q + offset + i_start * D;
    const float* Ki = K + offset;
    const float* Vi = V + offset;
    float* Oi = O + offset + i_start * D;

    // Shared memory for tiles
    extern __shared__ float s_mem[];
    float* s_Q = s_mem;               // [BR][D]
    float* s_K = s_mem + BR * D;      // [BC][D]
    float* s_V = s_mem + (BR + BC) * D; // [BC][D]

    // Local statistics for online softmax
    float m[BR]; // Local max per row
    float l[BR]; // Local sum-exp per row
    float acc[BR][128]; // Accumulator for output (assuming D <= 128 for simplicity)

    // Initialize stats
    for (int i = 0; i < BR; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        for (int d = 0; d < D; ++d) {
            acc[i][d] = 0.0f;
        }
    }

    // Load Qi tile into shared memory (coalesced)
    for (int d = tid; d < D; d += BR) {
        for (int i = 0; i < BR; ++i) {
            if (i_start + i < N) {
                s_Q[i * D + d] = Qi[i * D + d];
            } else {
                s_Q[i * D + d] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Iterate over tiles of K and V
    int Tr = (N + BR - 1) / BR;
    int Tc = (N + BC - 1) / BC;

    for (int j = 0; j < Tc; ++j) {
        // Load Ki and Vi tiles into shared memory
        int j_start = j * BC;
        for (int d = tid; d < D; d += BR) {
            for (int k = 0; k < BC; ++k) {
                if (j_start + k < N) {
                    s_K[k * D + d] = Ki[j_start * D + d + k * D]; // This indexing might be wrong if Ki is [N,D]
                    s_V[k * D + d] = Vi[j_start * D + d + k * D];
                } else {
                    s_K[k * D + d] = 0.0f;
                    s_V[k * D + d] = 0.0f;
                }
            }
        }
        // Correction on indexing: Ki[ (j_start + k)*D + d ]
        __syncthreads();

        // Each thread handles its own row(s) in the BR tile
        // For simplicity here, let's say each thread handles exactly one row i = tid
        int i = tid;
        if (i < BR && i_start + i < N) {
            float row_m_prev = m[i];
            float row_l_prev = l[i];
            
            // Compute S_ij = Q_i @ K_j^T
            float S[BC];
            float row_m_tile = -INFINITY;
            for (int k = 0; k < BC; ++k) {
                float score = 0.0f;
                for (int d = 0; d < D; ++d) {
                    score += s_Q[i * D + d] * s_K[k * D + d];
                }
                score *= scale;
                S[k] = score;
                if (score > row_m_tile) row_m_tile = score;
            }

            // Update stats
            float row_m_new = max(row_m_prev, row_m_tile);
            float exp_prev = expf(row_m_prev - row_m_new);
            float row_l_tile = 0.0f;
            for (int k = 0; k < BC; ++k) {
                S[k] = expf(S[k] - row_m_new);
                row_l_tile += S[k];
            }
            float row_l_new = row_l_prev * exp_prev + row_l_tile;

            // Update accumulator: acc = acc * exp(m_prev - m_new) + P_tile @ V_tile
            for (int d = 0; d < D; ++d) {
                float p_v = 0.0f;
                for (int k = 0; k < BC; ++k) {
                    p_v += S[k] * s_V[k * D + d];
                }
                acc[i][d] = acc[i][d] * exp_prev + p_v;
            }

            m[i] = row_m_new;
            l[i] = row_l_new;
        }
        __syncthreads();
    }

    // Finalize output: O_i = acc_i / l_i
    int i = tid;
    if (i < BR && i_start + i < N) {
        for (int d = 0; d < D; ++d) {
            Oi[i * D + d] = acc[i][d] / l[i];
        }
    }
}

void flash_attn_2_forward_cuda(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int D, float scale
) {
    dim3 grid(B * H, (N + BR - 1) / BR);
    dim3 block(BR); // One thread per row in Q tile

    size_t shared_mem_size = (BR * D + BC * D + BC * D) * sizeof(float);
    
    flash_attn_2_forward_kernel<<<grid, block, shared_mem_size>>>(
        Q, K, V, O, B, H, N, D, scale
    );
}
