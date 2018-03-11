#ifndef INPUT_TRANSFORM_H
#define  INPUT_TRANSFORM_H
#include "input_transform.h"
template <typename Dtype>
void transform_input(const int num, const int channel, const int height, const int width,
	const Dtype*input, Dtype*output);

#endif
