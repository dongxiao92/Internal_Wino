#include <iostream>
#include "input_transform_kernels.h"
using std::cout;
using std::endl;
/**
  *transform input activation maps.
**/
template <typename Dtype>
void transform_input(const int num, const int channel, const int height, const int width,
	const Dtype*input, Dtype*output)
{
    if(height != width){
        cout<<"only square input is supported!"<<endl;
	return;
    }
    // TODO:support input with odd size.
    if(height%2 != 0){
	cout<<"winograd convolution can not compute input with odd size."<<endl;
	return;
    }
    if(height <= 32){
	//the transformation can be achieved once
	transform_block_32(num, channel, height, width, 
		height, width,
		input, output);
    }else{
	const int block_size = 32;
	unsigned int tile_count = 0;
	const int tile_size = 4*4;
	for(int start_h=0; start_h<height; start_h+=block_size){
	    unsigned int block_height = h-start_h>block_size?block_size:h-start_h;
	    bool padding_zero_toph = (start_h==0);
	    bool padding_zero_bottomh = (satrt_h+block_size>=height);
	    unsigned int input_offset_h = start_h*block_size;
	    for(int start_w=0; start_w<width; start_w+=block_size){
		unsigned int block_width = w-start_w>block_size?block_size:w-start_w;
		bool padding_zero_leftw = (start_w==0);
		bool padding_zero_rightw = (satrt_w+block_size>=width);
		unsigned int input_offset_w = start_w*block_size;

		tile_count += (block_height/2)*(block_width/2);
		transform_block_32(num, channel, height, width,
			block_height, block_width,
			input+(input_offset_h*width+input_offset_w), output+tile_count*tile_size,
			padding_zero_toph, padding_zero_bottomh,
			padding_zero_leftw, padding_zero_rightw);
	    }
	}
    }
}

/**
  * transform blocks with size<=32, using zero padding.
**/
template <typename Dtype>
static void transform_block_32(const int num, const int channel, const int height, const int width,
	const int block_height, const int block_width, 
	const Dtype*input, Dtype*output, 
        bool zero_padding_toph = true, bool zero_padding_bottomh = true,
        bool zero_padding_leftw = true, bool zero_padding_rightw = true)	
{
    //compute launch kernel parameters.
    dim3 block(block_width, block_height);
    dim3 grid(1, 1, channel);
    if(block_height == height){
	//zero padding
	transform_input_whole<<<grid, block>>>(input, output, num, channel, block_height, block_width);
    }else{
	transfrom_input<<<grid, block>>>(input, output, num, channel, height, width, 
		block_height, block_width, 
		zero_padding_toph, zero_padding_bottomh, 
		zero_padding_leftw, zero_padding_rightw);
    }
}
