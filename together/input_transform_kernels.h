#ifndef INPUT_TRANSFORM_KERNELS
#define INPUT_TRANSFORM_KERNELS

//transform input with size<=32 once.
template <typename Dtype>
__global__
void transform_input_whole(const Dtype*input, Dtype*output, const int block_height, const int block_width);

//transform input block with size block_height*block_width
template <typename Dtype>
__global__ 
void transfrom_input_block(const float*input, Dtype*output, const int block_height, const int block_width, 
		bool zero_padding_toph, bool zero_padding_bottomh, 
		bool zero_padding_leftw, bool zero_padding_rightw);

#endif
