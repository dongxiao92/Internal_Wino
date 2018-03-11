#include "input_transform_kernels.h"
#define PAD 1
#define block_width (img_block_w+2*PAD)
#define block_height (img_block_h+2*PAD)
#define tile(p,intile_y, intile_x) (p[intile_x+intile_y*block_width])
/**
  *assumption:
    (1)input are stored in NCHW order;
    (2)output is stored in NCHW tile order;
 **/
template <typename Dtype>
__global__
void transform_input_whole(const Dtype*input, Dtype*output, const int num, const int channel, const int height, const int width){
    const int h_input = threadIdx.y;
    const int w_input = threadIdx.x;
    const int img_block_w = width;
    const int img_block_h = height;
    if(h_input<height && w_input<=width){
	//__shared__ float img_block_cache[block_width*block_height];
	extern __shared__ float img_block_cache[];
	float*img_block_cache_start=img_block_cache+(PAD*block_width+PAD);
	
	const int tid = threadIdx.y*blockDim.x+threadIdx.x;
	//stage1:load image block into img_block
	const int channel = blockIdx.z;
	const float*input_src=input+channel*height*width+h_input*width+w_input;
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
	    float* output_pos = &(output[((channel*height/2*width/2)+tid)*16]);
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

template <typename Dtype>
__global__ 
void transfrom_input_block(const Dtype*input, Dtype*output, const int num, const int channel, const int height, const int width,
	const int img_block_h, const int img_block_w,
	bool zero_padding_toph, bool zero_padding_bottomh, 
	bool zero_padding_leftw, bool zero_padding_rightw)
{
    const int h_input = threadIdx.y;
    const int w_input = threadIdx.x;
    if(h_input<img_block_h && w_input<img_block_w){
	//__shared__ float img_block_cache[block_width*block_height];
	extern __shared__ float img_block_cache[];
	float*img_block_cache_start=img_block_cache+(PAD*block_width+PAD);
	
	const int tid = threadIdx.y*blockDim.x+threadIdx.x;
	//stage1:load image block into img_block
	const int channel = blockIdx.z;
	const float*input_src=input+channel*height*width+h_input*width+w_input;
	const int cache_pos_h = tid/img_block_w;
	const int cache_pos_w = tid%img_block_w;
	if(tid<img_block_h*img_block_w){
	    float val = *input_src;
	    img_block_cache_start[cache_pos_h*block_width+cache_pos_w] = val;
	}
	if(cache_pos_w==0){
	    //pad left and right columns
	    Dtype pad_val_left = zero_padding_leftw?static_cast<Dtype>(0):
		input_src[cache_pos_h*block_width+cache_pos_w-1];
	    Dtype pad_val_right = zero_padding_rightw?static_cast<Dtype>(0):
		input_src[cache_pos_h*block_width+cache_pos_w+1];
	    img_block_cache_start[cache_pos_h*block_width-PAD] = pad_val_left;
	    img_block_cache_start[cache_pos_h*block_width+img_block_w] = pad_val_right;
	}
	if(cache_pos_h==0){
	    //pad top and bottom rows
	    Dtype pad_val_top = zero_padding_toph?static_cast<Dtype>(0):
		input_src[cache_pos_h*block_width+cache_pos_w-width];
	    Dtype pad_val_bottom = zero_padding_bottomh?static_cast<Dtype>(0):
		input_src[cache_pos_h*block_width+cache_pos_w+width];
	    img_block_cache[tid+PAD] = pad_val_top;
	    img_block_cache[tid+block_width*(PAD+img_block_h)+PAD] = pad_val_bottom;
	    if(tid==0){
		//pad four corners
		Dtype top_left_val = (zero_padding_toph&&zero_padding_leftw)?static_cast<Dtype>(0):
		    input_src[cache_pos_h*block_width+cache_pos_w-width-1];
		Dtype top_right_val = (zero_padding_toph&&zero_padding_rightw)?static_cast<Dtype>(0):
		    input_src[cache_pos_h*block_width+cache_pos_w-width+img_block_w];
		Dtype bottom_left_val = (zero_padding_bottomh&&zero_padding_leftw)?static_cast<Dtype>(0):
		    input_src[cache_pos_h*block_width+cache_pos_w+img_block_h*width-1];
		Dtype bottom_right_val = (zero_padding_bottomh&&zero_padding_rightw)?static_cast<Dtype>(0):
		    input_src[cache_pos_h*block_width+cache_pos_w+img_block_h*width+img_block_w];
		img_block_cache[0] = top_left_val;
		img_block_cache[img_block_w+PAD] = top_right_val;
		img_block_cache[(PAD+img_block_h)*block_width] = bottom_left_val;
		img_block_cache[img_block_w+PAD+(PAD+img_block_h)*block_width] = bottom_right_val;
	    }
	}
	__syncthreads();
	//stage2:compute tiles,the tile size is 4*4,stride between consecutive tile is 2.
	const int tile_num_x = img_block_w/2;
	const int tile_num_y = img_block_h/2;
	if(tid<tile_num_x*tile_num_y){
	    const int tile_idx = tid%tile_num_x;
	    const int tile_idy = tid/tile_num_x;
	    float* start_pos = &(img_block_cache[tile_idx*2+(tile_idy*2*block_width)]);
	    float* output_pos = &(output[((channel*height/2*width/2)+tid)*16]);
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
template __global__
void transform_input_whole<float>(const float*input, float*output, const int num, const int channel, const int height, const int width);
template __global__
void transfrom_input_block<float>(const float*input, float*output, const int num, const int channel, const int height, const int width,
	const int img_block_h, const int img_block_w,
	bool zero_padding_toph, bool zero_padding_bottomh, 
	bool zero_padding_leftw, bool zero_padding_rightw);

