#include <cuda_runtime.h>
#include <memory>
#include <fstream>
#include <vector>
#include <ctime>
#include <sys/time.h>
#include <unistd.h>
#include "config_16_32_32_16_4_2_8_4.h" 
//#include "config.h"
__global__
 void input_filter_num_kernel0_0(const float*, float*); 
__global__
 void input_filter_num_kernel0_1(const float*, float*); 
__global__
 void input_filter_num_kernel0_2(const float*, float*); 
__global__
 void input_filter_num_kernel0_3(const float*, float*); 
__global__
 void input_filter_num_kernel0_4(const float*, float*); 
__global__
 void input_filter_num_kernel0_5(const float*, float*); 
__global__
 void input_filter_num_kernel0_6(const float*, float*); 
__global__
 void input_filter_num_kernel0_7(const float*, float*); 
__global__
 void input_filter_num_kernel1_0(const float*, float*); 
__global__
 void input_filter_num_kernel1_1(const float*, float*); 
__global__
 void input_filter_num_kernel1_2(const float*, float*); 
__global__
 void input_filter_num_kernel1_3(const float*, float*); 
__global__
 void input_filter_num_kernel1_4(const float*, float*); 
__global__
 void input_filter_num_kernel1_5(const float*, float*); 
__global__
 void input_filter_num_kernel1_6(const float*, float*); 
__global__
 void input_filter_num_kernel1_7(const float*, float*); 
__global__
 void input_filter_num_kernel2_0(const float*, float*); 
__global__
 void input_filter_num_kernel2_1(const float*, float*); 
__global__
 void input_filter_num_kernel2_2(const float*, float*); 
__global__
 void input_filter_num_kernel2_3(const float*, float*); 
__global__
 void input_filter_num_kernel2_4(const float*, float*); 
__global__
 void input_filter_num_kernel2_5(const float*, float*); 
__global__
 void input_filter_num_kernel2_6(const float*, float*); 
__global__
 void input_filter_num_kernel2_7(const float*, float*); 
__global__
 void input_filter_num_kernel3_0(const float*, float*); 
__global__
 void input_filter_num_kernel3_1(const float*, float*); 
__global__
 void input_filter_num_kernel3_2(const float*, float*); 
__global__
 void input_filter_num_kernel3_3(const float*, float*); 
__global__
 void input_filter_num_kernel3_4(const float*, float*); 
__global__
 void input_filter_num_kernel3_5(const float*, float*); 
__global__
 void input_filter_num_kernel3_6(const float*, float*); 
__global__
 void input_filter_num_kernel3_7(const float*, float*); 

#define ITER_COUNT 100
using namespace std;
//__global__
//void input_filter_mul_kernel0(const float*input, float*output);
//__global__
//void input_filter_mul_kernel2(const float*input, float*output);
//__global__
//void input_filter_mul_kernel4(const float*input, float*output);
//__global__
//void input_filter_mul_kernel6(const float*input, float*output);
#define upDiv(a, b) ((a+b-1)/b)
int main(const int argc, const char*argv[])
{
    cudaError_t error;
    //generate tile data
    const int tile_count = C*tile_count_h*tile_count_w*tile_height*tile_width;
    const int output_count = K*C*tile_count_h*tile_count_w*tile_height*tile_width;
    unique_ptr<float>tiles(new float[tile_count]);
    unique_ptr<float>output(new float[output_count]);
    memset(output.get(), 0, sizeof(float)*output_count);
    srand(time(0));
    for(int i=0;i<tile_count;++i)
	tiles.get()[i] = static_cast<float>(rand()*1.f/RAND_MAX);
    float*tile_dev;
    float*output_dev;
    error = cudaMalloc(&tile_dev, sizeof(float)*tile_count);
    if(error != cudaSuccess){
	printf("at __LINE__ cuda error with code=%d,error=%s\n", error, cudaGetErrorString(error));
    }
    error = cudaMalloc(&output_dev, sizeof(float)*output_count);
    if(error != cudaSuccess){
	printf("at __LINE__ cuda error with code=%d,error=%s\n", error, cudaGetErrorString(error));
    }
    error = cudaMemcpy(tile_dev, tiles.get(), sizeof(float)*tile_count, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
	printf("at __LINE__ cuda error with code=%d,error=%s\n", error, cudaGetErrorString(error));
    }
    error = cudaMemset(output_dev, 0, sizeof(float)*output_count);
    if(error != cudaSuccess){
	printf("at __LINE__ cuda error with code=%d,error=%s\n", error, cudaGetErrorString(error));
    }
    //call kernel
    dim3 block(tile_num_w_per_kernel, tile_num_h_per_kernel);
    dim3 grid(upDiv(tile_count_w, tile_num_w_per_kernel), upDiv(tile_count_h, tile_num_h_per_kernel));
const int kernel_count=32;
    //insert kernel count 
    //const int kernel_count = C;
    int stream_count = kernel_count>32?32:kernel_count;
    vector<cudaStream_t>streams(stream_count);
    for(int i=0; i<stream_count; ++i){
	error = cudaStreamCreate(&streams[i]);
	if(error != cudaSuccess){
	    printf("at __LINE__ cuda error with code=%d,error=%s\n", error, cudaGetErrorString(error));
	}
    }
    cudaDeviceSynchronize();
    struct timezone tz;
    struct timeval start_cpu[ITER_COUNT];
    struct timeval end_cpu[ITER_COUNT];
    for(int i=0; i<ITER_COUNT;++i){
	gettimeofday(&start_cpu[i], &tz);
input_filter_num_kernel0_0<<<block,grid,0,streams[0]>>>(tile_dev, output_dev);
input_filter_num_kernel0_1<<<block,grid,0,streams[1]>>>(tile_dev, output_dev);
input_filter_num_kernel0_2<<<block,grid,0,streams[2]>>>(tile_dev, output_dev);
input_filter_num_kernel0_3<<<block,grid,0,streams[3]>>>(tile_dev, output_dev);
input_filter_num_kernel0_4<<<block,grid,0,streams[4]>>>(tile_dev, output_dev);
input_filter_num_kernel0_5<<<block,grid,0,streams[5]>>>(tile_dev, output_dev);
input_filter_num_kernel0_6<<<block,grid,0,streams[6]>>>(tile_dev, output_dev);
input_filter_num_kernel0_7<<<block,grid,0,streams[7]>>>(tile_dev, output_dev);
input_filter_num_kernel1_0<<<block,grid,0,streams[8]>>>(tile_dev, output_dev);
input_filter_num_kernel1_1<<<block,grid,0,streams[9]>>>(tile_dev, output_dev);
input_filter_num_kernel1_2<<<block,grid,0,streams[10]>>>(tile_dev, output_dev);
input_filter_num_kernel1_3<<<block,grid,0,streams[11]>>>(tile_dev, output_dev);
input_filter_num_kernel1_4<<<block,grid,0,streams[12]>>>(tile_dev, output_dev);
input_filter_num_kernel1_5<<<block,grid,0,streams[13]>>>(tile_dev, output_dev);
input_filter_num_kernel1_6<<<block,grid,0,streams[14]>>>(tile_dev, output_dev);
input_filter_num_kernel1_7<<<block,grid,0,streams[15]>>>(tile_dev, output_dev);
input_filter_num_kernel2_0<<<block,grid,0,streams[16]>>>(tile_dev, output_dev);
input_filter_num_kernel2_1<<<block,grid,0,streams[17]>>>(tile_dev, output_dev);
input_filter_num_kernel2_2<<<block,grid,0,streams[18]>>>(tile_dev, output_dev);
input_filter_num_kernel2_3<<<block,grid,0,streams[19]>>>(tile_dev, output_dev);
input_filter_num_kernel2_4<<<block,grid,0,streams[20]>>>(tile_dev, output_dev);
input_filter_num_kernel2_5<<<block,grid,0,streams[21]>>>(tile_dev, output_dev);
input_filter_num_kernel2_6<<<block,grid,0,streams[22]>>>(tile_dev, output_dev);
input_filter_num_kernel2_7<<<block,grid,0,streams[23]>>>(tile_dev, output_dev);
input_filter_num_kernel3_0<<<block,grid,0,streams[24]>>>(tile_dev, output_dev);
input_filter_num_kernel3_1<<<block,grid,0,streams[25]>>>(tile_dev, output_dev);
input_filter_num_kernel3_2<<<block,grid,0,streams[26]>>>(tile_dev, output_dev);
input_filter_num_kernel3_3<<<block,grid,0,streams[27]>>>(tile_dev, output_dev);
input_filter_num_kernel3_4<<<block,grid,0,streams[28]>>>(tile_dev, output_dev);
input_filter_num_kernel3_5<<<block,grid,0,streams[29]>>>(tile_dev, output_dev);
input_filter_num_kernel3_6<<<block,grid,0,streams[30]>>>(tile_dev, output_dev);
input_filter_num_kernel3_7<<<block,grid,0,streams[31]>>>(tile_dev, output_dev);
	
	error = cudaDeviceSynchronize();
	gettimeofday(&end_cpu[i], &tz);
	if(error != cudaSuccess){
	    printf("at __LINE__ cuda error with code=%d,error=%s\n", error, cudaGetErrorString(error));
	}
    }
    float sumTime = 0.f;
    for(int i=0 ; i<ITER_COUNT; ++i){
	sumTime += 1000*(end_cpu[i].tv_sec - start_cpu[i].tv_sec) + (end_cpu[i].tv_usec-start_cpu[i].tv_usec)*1.f/1000;
    }
    printf("average time::%.4f ms\n", sumTime/ITER_COUNT);


    //input_filter_mul_kernel0<<<block,grid, 0, streams[0]>>>(tile_dev, output_dev);
    //input_filter_mul_kernel2<<<block,grid, 0, streams[1]>>>(tile_dev, output_dev);
    //input_filter_mul_kernel4<<<block,grid, 0, streams[2]>>>(tile_dev, output_dev);
    //input_filter_mul_kernel6<<<block,grid, 0, streams[3]>>>(tile_dev, output_dev);
    //get result
    error = cudaMemcpy(output.get(), output_dev, sizeof(float)*output_count, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
	printf("at __LINE__ cuda error with code=%d,error=%s\n", error, cudaGetErrorString(error));
    }
    //clean up
    error = cudaFree(tile_dev);
    if(error != cudaSuccess){
	printf("at __LINE__ cuda error with code=%d,error=%s\n", error, cudaGetErrorString(error));
    }
    error = cudaFree(output_dev);
    if(error != cudaSuccess){
	printf("at __LINE__ cuda error with code=%d,error=%s\n", error, cudaGetErrorString(error));
    }
    for(int i=0; i<stream_count; ++i){
	error = cudaStreamDestroy(streams[i]);
	if(error != cudaSuccess){
	    printf("at __LINE__ cuda error with code=%d,error=%s\n", error, cudaGetErrorString(error));
	}
    }
    return 0;
}
