#include "config.h"

#define tile(p,intile_y, intile_x) (p[intile_x+intile_y*block_width])
//blockDim=(img_block_h, img_block_w),img_block_h*img_block_w<=1024
//gridDim=(W/img_block_w, H/img_block_h, C) or multi-kernel
//transform the whole image once.
__global__
void transform_input_whole(const float*input, float*output){
    const int h_input = threadIdx.y;
    const int w_input = threadIdx.x;
    if(h_input<H && w_input<W){
	__shared__ float img_block_cache[block_width*block_height];
	float*img_block_cache_start=img_block_cache+(PAD*block_width+PAD);
	
	const int tid = threadIdx.y*blockDim.x+threadIdx.x;
	//stage1:load image block into img_block
	const int channel = blockIdx.z;
	const float*input_src=input+channel*H*W+h_input*W+w_input;
	const int cache_pos_h = tid/img_block_w;
	const int cache_pos_w = tid%img_block_w;
	if(tid<img_block_h*img_block_w){
	    float val = *input_src;
	    img_block_cache_start[cache_pos_h*block_width+cache_pos_w] = val;
	}
	if(cache_pos_w==0){
	    //pad left and right columns
	    img_block_cache_start[cache_pos_h*block_width-PAD] = 0;
	    img_block_cache_start[cache_pos_h*block_width+img_block_w] = 0;
	}
	if(cache_pos_h==0){
	    //pad top and bottom rows
	    img_block_cache[tid+PAD] = 0;
	    img_block_cache[tid+block_width*(PAD+img_block_h)+PAD] = 0;
	}
	if(tid==0){
	    //pad four corners
	    img_block_cache[0] = 0;
	    img_block_cache[img_block_w+PAD] = 0;
	    img_block_cache[(PAD+img_block_h)*block_width] = 0;
	    img_block_cache[img_block_w+PAD+(PAD+img_block_h)*block_width] = 0;
	}
	__syncthreads();
	//stage2:compute tiles,the tile size is 4*4,stride between consecutive tile is 2.
	const int tile_num_x = img_block_w/2;
	const int tile_num_y = img_block_h/2;
	if(tid<tile_num_x*tile_num_y){
	    const int tile_idx = tid%tile_num_x;
	    const int tile_idy = tid/tile_num_x;
	    float* start_pos = &(img_block_cache[tile_idx*2+(tile_idy*2*block_width)]);
	    float* output_pos = &(output[((channel*H/2*W/2)+tid)*16]);
	    //float* output_pos = &(output[tid*16]);
	    output_pos[0] = tile(start_pos,0,0)+tile(start_pos,2,2)-tile(start_pos,2,0)-tile(start_pos,0,2);
	    output_pos[1] = tile(start_pos,0,1)+tile(start_pos,0,2)-tile(start_pos,2,1)-tile(start_pos,2,2);
	    output_pos[2] = tile(start_pos,0,2)+tile(start_pos,2,1)-tile(start_pos,2,2)-tile(start_pos,0,1);
	    output_pos[3] = tile(start_pos,0,1)+tile(start_pos,2,3)-tile(start_pos,2,1)-tile(start_pos,0,3);
	    output_pos[4] = tile(start_pos,1,0)+tile(start_pos,2,0)-tile(start_pos,1,2)-tile(start_pos,2,2);
	    output_pos[5] = tile(start_pos,1,1)+tile(start_pos,2,1)+tile(start_pos,1,2)+tile(start_pos,2,2);
	    output_pos[6] = tile(start_pos,1,2)+tile(start_pos,2,2)-tile(start_pos,1,1)-tile(start_pos,2,1);
	    output_pos[7] = tile(start_pos,1,1)+tile(start_pos,2,1)-tile(start_pos,1,3)-tile(start_pos,2,3);
	    output_pos[8] = tile(start_pos,2,0)+tile(start_pos,1,2)-tile(start_pos,1,0)-tile(start_pos,2,2);
	    output_pos[9] = tile(start_pos,2,1)+tile(start_pos,2,2)-tile(start_pos,1,1)-tile(start_pos,1,2);
	    output_pos[10] = tile(start_pos,2,2)+tile(start_pos,1,1)-tile(start_pos,1,2)-tile(start_pos,2,1);
	    output_pos[11] = tile(start_pos,2,1)+tile(start_pos,1,3)-tile(start_pos,1,1)-tile(start_pos,2,3);
	    output_pos[12] = tile(start_pos,1,0)+tile(start_pos,3,2)-tile(start_pos,3,0)-tile(start_pos,1,2);
	    output_pos[13] = tile(start_pos,1,1)+tile(start_pos,1,2)-tile(start_pos,3,1)-tile(start_pos,3,2);
	    output_pos[14] = tile(start_pos,1,2)+tile(start_pos,3,1)-tile(start_pos,1,1)-tile(start_pos,3,2);
	    output_pos[15] = tile(start_pos,1,1)+tile(start_pos,3,3)-tile(start_pos,3,1)-tile(start_pos,1,3);
	}
	__syncthreads();
    }
}
/*
   input:pointer of  to the very first positon in img_block
   output: always points to the starting position 
*/
__global__
void transform_input_top_left(const float*input, float*output){
    if(threadIdx.y<img_block_h && threadIdx.x<img_block_w){
    __shared__ float img_block_cache[block_width*block_height];
    //points to starting positon if valid image content.
    float*img_block_cache_start=img_block_cache+(PAD*block_width+PAD);
    
    const int tid = threadIdx.y*blockDim.x+threadIdx.x;
    //stage1:load image block into img_block
    const int channel = blockIdx.z;
    const int cache_pos_h = tid/img_block_w;
    const int cache_pos_w = tid%img_block_w;
    const float*input_src=input+channel*H*W+cache_pos_h*W+cache_pos_w;
    if(tid<img_block_h*img_block_w){
	float val = *input_src;
	img_block_cache_start[cache_pos_h*block_width+cache_pos_w] = val;
    }
    if(cache_pos_w==0){
	//pad left column
	img_block_cache_start[cache_pos_h*block_width-PAD] = 0;
	//pad right columns
	img_block_cache_start[cache_pos_h*block_width+img_block_w] = *(input_src+img_block_w);
    }
    if(cache_pos_h==0){
	//pad top rows
	img_block_cache[tid+PAD] = 0;
	//pad bottom rows
	img_block_cache[tid+block_width*(PAD+img_block_h)+PAD] = *(input_src+img_block_h*W);
    }
    if(tid==0){
	//pad top left corner
	img_block_cache[0] = 0;
	//pad top right corner
	img_block_cache[img_block_w+PAD] = 0;
	//pad down left corner
	img_block_cache[(PAD+img_block_h)*block_width] = 0;
	//pad down right corner
	img_block_cache[img_block_w+PAD+(PAD+img_block_h)*block_width] = *(input_src+img_block_h*W+img_block_w);
    }
    __syncthreads();
    //stage2:compute tile,tile size is 4*4,stride between tile is 2
    const int tile_num_x = img_block_w/2;
    const int tile_num_y = img_block_h/2;
    if(tid<tile_num_x*tile_num_y){
	const int tile_idx = tid%tile_num_x;
	const int tile_idy = tid/tile_num_x;
        float* start_pos = &(img_block_cache[tile_idx*2+(tile_idy*2*block_width)]);
	float* output_pos = &(output[((channel*H/2*W/2)+tile_idy*W/2+tile_idx)*16]);
	//float* output_pos = &(output[tid*16]);
	output_pos[0] = tile(start_pos,0,0)+tile(start_pos,2,2)-tile(start_pos,2,0)-tile(start_pos,0,2);
	output_pos[1] = tile(start_pos,0,1)+tile(start_pos,0,2)-tile(start_pos,2,1)-tile(start_pos,2,2);
	output_pos[2] = tile(start_pos,0,2)+tile(start_pos,2,1)-tile(start_pos,2,2)-tile(start_pos,0,1);
	output_pos[3] = tile(start_pos,0,1)+tile(start_pos,2,3)-tile(start_pos,2,1)-tile(start_pos,0,3);
	output_pos[4] = tile(start_pos,1,0)+tile(start_pos,2,0)-tile(start_pos,1,2)-tile(start_pos,2,2);
	output_pos[5] = tile(start_pos,1,1)+tile(start_pos,2,1)+tile(start_pos,1,2)+tile(start_pos,2,2);
	output_pos[6] = tile(start_pos,1,2)+tile(start_pos,2,2)-tile(start_pos,1,1)-tile(start_pos,2,1);
	output_pos[7] = tile(start_pos,1,1)+tile(start_pos,2,1)-tile(start_pos,1,3)-tile(start_pos,2,3);
	output_pos[8] = tile(start_pos,2,0)+tile(start_pos,1,2)-tile(start_pos,1,0)-tile(start_pos,2,2);
	output_pos[9] = tile(start_pos,2,1)+tile(start_pos,2,2)-tile(start_pos,1,1)-tile(start_pos,1,2);
	output_pos[10] = tile(start_pos,2,2)+tile(start_pos,1,1)-tile(start_pos,1,2)-tile(start_pos,2,1);
	output_pos[11] = tile(start_pos,2,1)+tile(start_pos,1,3)-tile(start_pos,1,1)-tile(start_pos,2,3);
	output_pos[12] = tile(start_pos,1,0)+tile(start_pos,3,2)-tile(start_pos,3,0)-tile(start_pos,1,2);
	output_pos[13] = tile(start_pos,1,1)+tile(start_pos,1,2)-tile(start_pos,3,1)-tile(start_pos,3,2);
	output_pos[14] = tile(start_pos,1,2)+tile(start_pos,3,1)-tile(start_pos,1,1)-tile(start_pos,3,2);
	output_pos[15] = tile(start_pos,1,1)+tile(start_pos,3,3)-tile(start_pos,3,1)-tile(start_pos,1,3);
    }
    __syncthreads();
    }
}
__global__
void transform_input_top_right(const float*input, float*output){
    if(threadIdx.y<img_block_h && threadIdx.x<img_block_w){
    __shared__ float img_block_cache[block_width*block_height];
    float*img_block_cache_start=img_block_cache+(PAD*block_width+PAD);
    
    const int tid = threadIdx.y*blockDim.x+threadIdx.x;
    //stage1:load image block into img_block
    const int channel = blockIdx.z;
    const int cache_pos_h = tid/img_block_w;
    const int cache_pos_w = tid%img_block_w;
    const float*input_src=input+channel*H*W+cache_pos_h*W+cache_pos_w;
    if(tid<img_block_h*img_block_w){
	float val = *input_src;
	img_block_cache_start[cache_pos_h*block_width+cache_pos_w] = val;
    }
    if(cache_pos_w==0){
	//pad left column
	img_block_cache_start[cache_pos_h*block_width-PAD] = *(input_src-1);
	//pad right columns
	img_block_cache_start[cache_pos_h*block_width+img_block_w] = 0;
    }
    if(cache_pos_h==0){
	//pad top rows
	img_block_cache[tid+PAD] = 0;
	//pad bottom rows
	img_block_cache[tid+block_width*(PAD+img_block_h)+PAD] = *(input_src+img_block_h*W);
    }
    if(tid==0){
	//pad top left corner
	img_block_cache[0] = 0;
	//pad top right corner
	img_block_cache[img_block_w+PAD] = 0;
	//pad down left corner
	img_block_cache[(PAD+img_block_h)*block_width] = *(input_src+img_block_h*W-1);
	//pad down right corner
	img_block_cache[img_block_w+PAD+(PAD+img_block_h)*block_width] = 0;
    }
    __syncthreads();
    //stage2:compute tile,tile size is 4*4,stride between tile is 2
    const int tile_num_x = img_block_w/2;
    const int tile_num_y = img_block_h/2;
    if(tid<tile_num_x*tile_num_y){
	const int tile_idx = tid%tile_num_x;
	const int tile_idy = tid/tile_num_x;
        float* start_pos = &(img_block_cache[tile_idx*2+(tile_idy*2*block_width)]);
	float* output_pos = &(output[((channel*H/2*W/2)+img_block_w/2+tile_idx+tile_idy*W/2)*16]);
	//float* output_pos = &(output[tid*16]);
	output_pos[0] = tile(start_pos,0,0)+tile(start_pos,2,2)-tile(start_pos,2,0)-tile(start_pos,0,2);
	output_pos[1] = tile(start_pos,0,1)+tile(start_pos,0,2)-tile(start_pos,2,1)-tile(start_pos,2,2);
	output_pos[2] = tile(start_pos,0,2)+tile(start_pos,2,1)-tile(start_pos,2,2)-tile(start_pos,0,1);
	output_pos[3] = tile(start_pos,0,1)+tile(start_pos,2,3)-tile(start_pos,2,1)-tile(start_pos,0,3);
	output_pos[4] = tile(start_pos,1,0)+tile(start_pos,2,0)-tile(start_pos,1,2)-tile(start_pos,2,2);
	output_pos[5] = tile(start_pos,1,1)+tile(start_pos,2,1)+tile(start_pos,1,2)+tile(start_pos,2,2);
	output_pos[6] = tile(start_pos,1,2)+tile(start_pos,2,2)-tile(start_pos,1,1)-tile(start_pos,2,1);
	output_pos[7] = tile(start_pos,1,1)+tile(start_pos,2,1)-tile(start_pos,1,3)-tile(start_pos,2,3);
	output_pos[8] = tile(start_pos,2,0)+tile(start_pos,1,2)-tile(start_pos,1,0)-tile(start_pos,2,2);
	output_pos[9] = tile(start_pos,2,1)+tile(start_pos,2,2)-tile(start_pos,1,1)-tile(start_pos,1,2);
	output_pos[10] = tile(start_pos,2,2)+tile(start_pos,1,1)-tile(start_pos,1,2)-tile(start_pos,2,1);
	output_pos[11] = tile(start_pos,2,1)+tile(start_pos,1,3)-tile(start_pos,1,1)-tile(start_pos,2,3);
	output_pos[12] = tile(start_pos,1,0)+tile(start_pos,3,2)-tile(start_pos,3,0)-tile(start_pos,1,2);
	output_pos[13] = tile(start_pos,1,1)+tile(start_pos,1,2)-tile(start_pos,3,1)-tile(start_pos,3,2);
	output_pos[14] = tile(start_pos,1,2)+tile(start_pos,3,1)-tile(start_pos,1,1)-tile(start_pos,3,2);
	output_pos[15] = tile(start_pos,1,1)+tile(start_pos,3,3)-tile(start_pos,3,1)-tile(start_pos,1,3);
    }
    __syncthreads();
    }
}
__global__
void transform_input_down_left(const float*input, float*output){
    if(threadIdx.y<img_block_h && threadIdx.x<img_block_w){
    __shared__ float img_block_cache[block_width*block_height];
    float*img_block_cache_start=img_block_cache+(PAD*block_width+PAD);
    
    const int tid = threadIdx.y*blockDim.x+threadIdx.x;
    //stage1:load image block into img_block
    const int channel = blockIdx.z;
    const int cache_pos_h = tid/img_block_w;
    const int cache_pos_w = tid%img_block_w;
    const float*input_src=input+channel*H*W+cache_pos_h*W+cache_pos_w;
    if(tid<img_block_h*img_block_w){
	float val = *input_src;
	img_block_cache_start[cache_pos_h*block_width+cache_pos_w] = val;
    }
    if(cache_pos_w==0){
	//pad left column
	img_block_cache_start[cache_pos_h*block_width-PAD] = 0;
	//pad right columns
	img_block_cache_start[cache_pos_h*block_width+img_block_w] = *(input_src+img_block_w);
    }
    if(cache_pos_h==0){
	//pad top rows
	img_block_cache[tid+PAD] = *(input_src-W);
	//pad bottom rows
	img_block_cache[tid+block_width*(PAD+img_block_h)+PAD] = 0;
    }
    if(tid==0){
	//pad top left corner
	img_block_cache[0] = 0;
	//pad top right corner
	img_block_cache[img_block_w+PAD] = *(input_src+img_block_w-W);
	//pad down left corner
	img_block_cache[(PAD+img_block_h)*block_width] = 0;
	//pad down right corner
	img_block_cache[img_block_w+PAD+(PAD+img_block_h)*block_width] = 0;
    }
    __syncthreads();
    //stage2:compute tile,tile size is 4*4,stride between tile is 2
    const int tile_num_x = img_block_w/2;
    const int tile_num_y = img_block_h/2;
    if(tid<tile_num_x*tile_num_y){
	const int tile_idx = tid%tile_num_x;
	const int tile_idy = tid/tile_num_x;
        float* start_pos = &(img_block_cache[tile_idx*2+(tile_idy*2*block_width)]);
	float* output_pos = &(output[((channel*H/2*W/2)+tile_num_y*W/2+tile_idx+tile_idy*W/2)*16]);
	//float* output_pos = &(output[tid*16]);
	output_pos[0] = tile(start_pos,0,0)+tile(start_pos,2,2)-tile(start_pos,2,0)-tile(start_pos,0,2);
	output_pos[1] = tile(start_pos,0,1)+tile(start_pos,0,2)-tile(start_pos,2,1)-tile(start_pos,2,2);
	output_pos[2] = tile(start_pos,0,2)+tile(start_pos,2,1)-tile(start_pos,2,2)-tile(start_pos,0,1);
	output_pos[3] = tile(start_pos,0,1)+tile(start_pos,2,3)-tile(start_pos,2,1)-tile(start_pos,0,3);
	output_pos[4] = tile(start_pos,1,0)+tile(start_pos,2,0)-tile(start_pos,1,2)-tile(start_pos,2,2);
	output_pos[5] = tile(start_pos,1,1)+tile(start_pos,2,1)+tile(start_pos,1,2)+tile(start_pos,2,2);
	output_pos[6] = tile(start_pos,1,2)+tile(start_pos,2,2)-tile(start_pos,1,1)-tile(start_pos,2,1);
	output_pos[7] = tile(start_pos,1,1)+tile(start_pos,2,1)-tile(start_pos,1,3)-tile(start_pos,2,3);
	output_pos[8] = tile(start_pos,2,0)+tile(start_pos,1,2)-tile(start_pos,1,0)-tile(start_pos,2,2);
	output_pos[9] = tile(start_pos,2,1)+tile(start_pos,2,2)-tile(start_pos,1,1)-tile(start_pos,1,2);
	output_pos[10] = tile(start_pos,2,2)+tile(start_pos,1,1)-tile(start_pos,1,2)-tile(start_pos,2,1);
	output_pos[11] = tile(start_pos,2,1)+tile(start_pos,1,3)-tile(start_pos,1,1)-tile(start_pos,2,3);
	output_pos[12] = tile(start_pos,1,0)+tile(start_pos,3,2)-tile(start_pos,3,0)-tile(start_pos,1,2);
	output_pos[13] = tile(start_pos,1,1)+tile(start_pos,1,2)-tile(start_pos,3,1)-tile(start_pos,3,2);
	output_pos[14] = tile(start_pos,1,2)+tile(start_pos,3,1)-tile(start_pos,1,1)-tile(start_pos,3,2);
	output_pos[15] = tile(start_pos,1,1)+tile(start_pos,3,3)-tile(start_pos,3,1)-tile(start_pos,1,3);
    }
    __syncthreads();
    }
}
__global__
void transform_input_down_right(const float*input, float*output){
    if(threadIdx.y<img_block_h && threadIdx.x<img_block_w){
    __shared__ float img_block_cache[block_width*block_height];
    float*img_block_cache_start=img_block_cache+(PAD*block_width+PAD);
    
    const int tid = threadIdx.y*blockDim.x+threadIdx.x;
    //stage1:load image block into img_block
    const int channel = blockIdx.z;
    const int cache_pos_h = tid/img_block_w;
    const int cache_pos_w = tid%img_block_w;
    const float*input_src=input+channel*H*W+cache_pos_h*W+cache_pos_w;
    if(tid<img_block_h*img_block_w){
	float val = *input_src;
	img_block_cache_start[cache_pos_h*block_width+cache_pos_w] = val;
    }
    if(cache_pos_w==0){
	//pad left column
	img_block_cache_start[cache_pos_h*block_width-PAD] = *(input_src-1);
	//pad right columns
	img_block_cache_start[cache_pos_h*block_width+img_block_w] = 0;
    }
    if(cache_pos_h==0){
	//pad top rows
	img_block_cache[tid+PAD] = *(input_src-W);
	//pad bottom rows
	img_block_cache[tid+block_width*(PAD+img_block_h)+PAD] = 0;
    }
    if(tid==0){
	//pad top left corner
	img_block_cache[0] = *(input_src-W-1);
	//pad top right corner
	img_block_cache[img_block_w+PAD] = 0;
	//pad down left corner
	img_block_cache[(PAD+img_block_h)*block_width] = 0;
	//pad down right corner
	img_block_cache[img_block_w+PAD+(PAD+img_block_h)*block_width] = 0;
    }
    __syncthreads();
    //stage2:compute tile,tile size is 4*4,stride between tile is 2
    const int tile_num_x = img_block_w/2;
    const int tile_num_y = img_block_h/2;
    if(tid<tile_num_x*tile_num_y){
	const int tile_idx = tid%tile_num_x;
	const int tile_idy = tid/tile_num_x;
        float* start_pos = &(img_block_cache[tile_idx*2+(tile_idy*2*block_width)]);
	float* output_pos = &(output[((channel*H/2*W/2)+(tile_num_y*W/2+tile_num_x)+tile_idx+tile_idy*W/2)*16]);
	//float* output_pos = &(output[tid*16]);
	output_pos[0] = tile(start_pos,0,0)+tile(start_pos,2,2)-tile(start_pos,2,0)-tile(start_pos,0,2);
	output_pos[1] = tile(start_pos,0,1)+tile(start_pos,0,2)-tile(start_pos,2,1)-tile(start_pos,2,2);
	output_pos[2] = tile(start_pos,0,2)+tile(start_pos,2,1)-tile(start_pos,2,2)-tile(start_pos,0,1);
	output_pos[3] = tile(start_pos,0,1)+tile(start_pos,2,3)-tile(start_pos,2,1)-tile(start_pos,0,3);
	output_pos[4] = tile(start_pos,1,0)+tile(start_pos,2,0)-tile(start_pos,1,2)-tile(start_pos,2,2);
	output_pos[5] = tile(start_pos,1,1)+tile(start_pos,2,1)+tile(start_pos,1,2)+tile(start_pos,2,2);
	output_pos[6] = tile(start_pos,1,2)+tile(start_pos,2,2)-tile(start_pos,1,1)-tile(start_pos,2,1);
	output_pos[7] = tile(start_pos,1,1)+tile(start_pos,2,1)-tile(start_pos,1,3)-tile(start_pos,2,3);
	output_pos[8] = tile(start_pos,2,0)+tile(start_pos,1,2)-tile(start_pos,1,0)-tile(start_pos,2,2);
	output_pos[9] = tile(start_pos,2,1)+tile(start_pos,2,2)-tile(start_pos,1,1)-tile(start_pos,1,2);
	output_pos[10] = tile(start_pos,2,2)+tile(start_pos,1,1)-tile(start_pos,1,2)-tile(start_pos,2,1);
	output_pos[11] = tile(start_pos,2,1)+tile(start_pos,1,3)-tile(start_pos,1,1)-tile(start_pos,2,3);
	output_pos[12] = tile(start_pos,1,0)+tile(start_pos,3,2)-tile(start_pos,3,0)-tile(start_pos,1,2);
	output_pos[13] = tile(start_pos,1,1)+tile(start_pos,1,2)-tile(start_pos,3,1)-tile(start_pos,3,2);
	output_pos[14] = tile(start_pos,1,2)+tile(start_pos,3,1)-tile(start_pos,1,1)-tile(start_pos,3,2);
	output_pos[15] = tile(start_pos,1,1)+tile(start_pos,3,3)-tile(start_pos,3,1)-tile(start_pos,1,3);
    }
    __syncthreads();
    }
}
