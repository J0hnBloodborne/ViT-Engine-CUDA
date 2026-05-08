#include <cuda_runtime.h>
#define IMG_SIZE 224
#define PATCH_SIZE 16
#define PATCHES_PER_SIDE (IMG_SIZE / PATCH_SIZE)
#define CHANNELS 3
#define COARSE_FACTOR 3 // Fixed to prevent out-of-bounds (256 * 3 = 768)
#define EMBED_DIM 768
#define NUM_PATCHES 196
#define PATCH_AREA 256
#define PATCH_VOLUME 768

__global__ void patch_embed_kernel(float* img, float* weights, float* out) {
    int patch_idx = blockIdx.x; 
    int batch_idx = blockIdx.y; // One batch per block
    int tid = threadIdx.x;

    __shared__ float patch_s[PATCH_VOLUME];

    int patch_row = (patch_idx / PATCHES_PER_SIDE) * PATCH_SIZE;
    int patch_col = (patch_idx % PATCHES_PER_SIDE) * PATCH_SIZE;

    // Find the starting point of the current batch in the input image
    float* batch_img = img + (batch_idx * CHANNELS * IMG_SIZE * IMG_SIZE);

    // Collaboratively load the patch into shared memory. Each thread loads one pixel (across all channels).
    #pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
        int py = tid / PATCH_SIZE;
        int px = tid % PATCH_SIZE;
        int global_idx = c * IMG_SIZE * IMG_SIZE + (patch_row + py) * IMG_SIZE + (patch_col + px);
        patch_s[c * PATCH_AREA + tid] = batch_img[global_idx];
    }

    __syncthreads();

    #pragma unroll
    for (int step = 0; step < COARSE_FACTOR; step++) {
        int out_dim = tid * COARSE_FACTOR + step;
        
        // Ampere architecture can do 128-bit float4 operations, so we can process 4 elements at a time.
         float4* weights_vec = reinterpret_cast<float4*>(&weights[out_dim * PATCH_VOLUME]);
         float4* patch_vec = reinterpret_cast<float4*>(patch_s);
        
        float sum = 0.0f;
        
        // Loop runs 192 times instead of 768. Partially unrolled.
        #pragma unroll 4
        for (int i = 0; i < PATCH_VOLUME / 4; i++) {
            float4 w = weights_vec[i];
            float4 p = patch_vec[i];
            sum += p.x * w.x + p.y * w.y + p.z * w.z + p.w * w.w;
        }

        // Find the starting point of the current batch in the output tensor
        int out_batch_offset = batch_idx * NUM_PATCHES * EMBED_DIM;
        out[out_batch_offset + (patch_idx * EMBED_DIM) + out_dim] = sum;
    }
}

void launch_patch_embed(float* img, float* weights, float* out, int batch_size) {
    dim3 grid(NUM_PATCHES, batch_size); 
    dim3 block(PATCH_AREA);
    patch_embed_kernel<<<grid, block>>>(img, weights, out);
}