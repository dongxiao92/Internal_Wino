#include "config.h"
//blockDim:(tile_h_per_kernel, tile_w_per_kernel, c_groups)
//gridDim:(tile_num_h/tile_h_per_kernel,tile_num_w/tile_w_per_kernel, K/k_per_kernel)
__global__
void transform_output(const float*input, float*output){
#if c_groups > 1
    __shared__ float local_sum_cache[tile_h_per_kernel*tile_w_per_kernel*c_groups*tile_height*tile_width];
#endif
    //compute positon in tile space.
    const int k = blockIdx.z;
    const int c = threadIdx.z;
    const int h = blockIdx.y*blockDim.y+threadIdx.y;
    const int w = blockIdx.x*blockDim.x+threadIdx.x;
    if(h<tile_num_h && w<tile_num_w){
	const float*input_src = input+tile_height*tile_width*(((k*k_per_kernel*C+c)*tile_num_h+h)*tile_num_w+w);
	float* output_pos = output+k*H*W*k_per_kernel+h*2*W+w*2;
	const int channel_input_stride = tile_height*tile_width*(tile_num_h*tile_num_w)*c_groups;
	const int kernel_input_stride = C*channel_input_stride;
	const int kernel_output_stride = H*W;
	
	float local_sum[tile_height][tile_width];
	float result[2][2];
#pragma unroll
	for(int k_out=0; k_out<k_per_kernel; ++k_out){
#pragma unroll
	    for(int i=0; i<tile_height; ++i){
#pragma unroll
		for(int j=0; j<tile_width; ++j){
		    local_sum[i][j] = input_src[i*tile_width+j];
		}
	    }
#pragma unroll
	    for(int c_in=1; c_in<C/c_groups; ++c_in){
		input_src += channel_input_stride;
		//load input
#pragma unroll
		for(int i=0; i<tile_height; ++i){
#pragma unroll
		    for(int j=0; j<tile_width; ++j){
			local_sum[i][j] += input_src[i*tile_width+j];
		    }
		}
	    }
	    input_src += channel_input_stride;
	    //do reduction cross channel
#if c_groups>1
	    //write local sum to shared memory
	    float*cache_pos = local_sum_cache+(c*tile_h_per_kernel*tile_w_per_kernel+(threadIdx.y*blockDim.x+threadIdx.x))*tile_height*tile_width;
#pragma unroll
	    for(int i=0 ;i<tile_height; ++i){
		for(int j=0; j<tile_width; ++j){
		    cache_pos[i*tile_width+j] = local_sum[i][j];
		}
	    }
	    __syncthreads();
	    int left_channel = c_groups;
	    int stride = (left_channel/2)*tile_h_per_kernel*tile_w_per_kernel*tile_height*tile_width;
	    while(left_channel > 1){
		if(c<left_channel/2){
#pragma unroll
		    for(int i=0; i<tile_width*tile_height; ++i)
			cache_pos[i] += cache_pos[i+stride];
		}
		stride /= 2;
		left_channel /= 2;
		__syncthreads();
	    }
	    if(c==0){
#pragma unroll
		for(int i=0; i<tile_height; ++i){
#pragma unroll
		    for(int j=0; j<tile_width; ++j){
			local_sum[i][j] = cache_pos[i*tile_width+j];
		    }
		}
	    }
#endif //c_groups>1
	    //transform
	    if(c == 0){
		result[0][0] = local_sum[0][0] + local_sum[0][1] + local_sum[0][2] + local_sum[1][0] + local_sum[1][1] + local_sum[1][2] + local_sum[2][0] + local_sum[2][1] + local_sum[2][2];
		result[0][1] = local_sum[0][1] - local_sum[0][2] - local_sum[0][3] + local_sum[1][1] - local_sum[1][2] - local_sum[1][3] + local_sum[2][1] - local_sum[2][2] - local_sum[2][3];
		result[1][0] = local_sum[1][0] + local_sum[1][1] + local_sum[1][2] - local_sum[2][0] - local_sum[2][1] - local_sum[2][2] - local_sum[3][0] - local_sum[3][1] - local_sum[3][2];
		result[1][1] = local_sum[1][1] - local_sum[1][2] - local_sum[1][3] - local_sum[2][1] + local_sum[2][2] + local_sum[2][3] - local_sum[3][1] + local_sum[3][2] + local_sum[3][3];
		//write to output
#pragma unroll
		for(int h_out=0;h_out<2; ++h_out){
#pragma unroll
		    for(int w_out=0; w_out<2; ++w_out){
			output_pos[h_out*W+w_out] = result[h_out][w_out];
		    }
		}
	    }
	    //change pointer
	    output_pos += kernel_output_stride;
	}
    }
}
