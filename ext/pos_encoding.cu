#include <cuda_runtime.h>
#define SEQ_LEN 197 // 196 patches + 1 CLS token
#define EMBED_DIM 768
#define VEC_SIZE 4
#define THREADS_PER_BLOCK 192 // 768 / 4

__global__ void pos_encoding_kernel(float* patches, float* cls_token, float* pos_embed, float* out) {
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;

    int out_offset = (batch_idx * SEQ_LEN * EMBED_DIM) + (seq_idx * EMBED_DIM) + (tid * VEC_SIZE);

    float4* pos_vec = reinterpret_cast<float4*>(&pos_embed[seq_idx * EMBED_DIM]);
    float4 p_val = pos_vec[tid];

    if (seq_idx == 0) {
        float4* cls_vec = reinterpret_cast<float4*>(&cls_token[batch_idx * EMBED_DIM]);
        float4 c_val = cls_vec[tid];
        
        float4 res;
        res.x = c_val.x + p_val.x;
        res.y = c_val.y + p_val.y;
        res.z = c_val.z + p_val.z;
        res.w = c_val.w + p_val.w;
        
        reinterpret_cast<float4*>(&out[out_offset])[0] = res;
    } 
    else {
        int patch_offset = (batch_idx * 196 * EMBED_DIM) + ((seq_idx - 1) * EMBED_DIM) + (tid * VEC_SIZE);
        float4* patch_vec = reinterpret_cast<float4*>(&patches[patch_offset]);
        float4 patch_val = patch_vec[0];
        
        float4 res;
        res.x = patch_val.x + p_val.x;
        res.y = patch_val.y + p_val.y;
        res.z = patch_val.z + p_val.z;
        res.w = patch_val.w + p_val.w;
        
        reinterpret_cast<float4*>(&out[out_offset])[0] = res;
    }
}

void launch_pos_encoding(float* patches, float* cls_token, float* pos_embed, float* out, int batch_size) {
    dim3 grid(SEQ_LEN, batch_size);
    dim3 block(THREADS_PER_BLOCK);
    pos_encoding_kernel<<<grid, block>>>(patches, cls_token, pos_embed, out);
}