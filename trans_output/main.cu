#include <cuda_runtime.h>
#include <memory>
#include <fstream>
#include <vector>
#include <ctime>
#include <sys/time.h>
#include <unistd.h>
#include "config.h"

#define ITER_COUNT 1
using namespace std;
__global__
void transform_output(const float*input, float*output);

int main(const int argc, const char*argv[])
{
    cudaError_t error;
    //generate tile data
    const int tile_count = K*C*tile_num_h*tile_num_w*tile_height*tile_width;
    const int output_count = K*H*W;
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
	printf("at line %d,cuda error with code=%d,error=%s\n", __LINE__, error, cudaGetErrorString(error));
    }
    error = cudaMalloc(&output_dev, sizeof(float)*output_count);
    if(error != cudaSuccess){
	printf("at line %d,cuda error with code=%d,error=%s\n", __LINE__, error, cudaGetErrorString(error));
    }
    error = cudaMemcpy(tile_dev, tiles.get(), sizeof(float)*tile_count, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
	printf("at line %d,cuda error with code=%d,error=%s\n", __LINE__, error, cudaGetErrorString(error));
    }
    error = cudaMemset(output_dev, 0, sizeof(float)*output_count);
    if(error != cudaSuccess){
	printf("at line %d,cuda error with code=%d,error=%s\n", __LINE__, error, cudaGetErrorString(error));
    }
    //call kernel
    dim3 block(tile_w_per_kernel, tile_h_per_kernel, c_groups);
    dim3 grid(tile_num_w/tile_w_per_kernel, tile_num_h/tile_h_per_kernel, K/k_per_kernel);

    //int stream_count = kernel_count>32?32:kernel_count;
    //vector<cudaStream_t>streams(stream_count);
    //for(int i=0; i<stream_count; ++i){
    //    error = cudaStreamCreate(&streams[i]);
    //    if(error != cudaSuccess){
    //        printf("at line %d,cuda error with code=%d,error=%s\n", __LINE__, error, cudaGetErrorString(error));
    //    }
    //}
    cudaDeviceSynchronize();
    struct timezone tz;
    struct timeval start_cpu[ITER_COUNT];
    struct timeval end_cpu[ITER_COUNT];
    for(int i=0; i<ITER_COUNT;++i){
	gettimeofday(&start_cpu[i], &tz);
	transform_output<<<grid, block>>>(tile_dev, output_dev);
	error = cudaDeviceSynchronize();
	if(error != cudaSuccess){
	    printf("at line %d,cuda error with code=%d,error=%s\n", __LINE__, error, cudaGetErrorString(error));
	}
	gettimeofday(&end_cpu[i], &tz);
    }
    float sumTime = 0.f;
    for(int i=0 ; i<ITER_COUNT; ++i){
        sumTime += 1000*(end_cpu[i].tv_sec - start_cpu[i].tv_sec) + (end_cpu[i].tv_usec-start_cpu[i].tv_usec)*1.f/1000;
    }
    printf("average time::%.4f ms\n", sumTime/ITER_COUNT);

    //get result
    error = cudaMemcpy(output.get(), output_dev, sizeof(float)*output_count, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
	printf("at line %d,cuda error with code=%d,error=%s\n", __LINE__, error, cudaGetErrorString(error));
    }
    /*
    //compute reference
    unique_ptr<float>output_ref(new float[output_count]);
    memset(output_ref.get(), 0, sizeof(float)*output_count);
    unique_ptr<float>local_sum(new float[K*tile_num_h*tile_num_w*tile_width*tile_height]);
    memset(local_sum.get(), 0, sizeof(float)*K*tile_num_h*tile_num_w*tile_width*tile_height);
    for(int k=0; k<K; ++k){
	for(int c=0; c<C;++c){
	    for(int tid=0; tid<tile_num_h*tile_num_w; ++tid){
		for(int eid=0; eid<tile_width*tile_height; ++eid)
		    local_sum.get()[(k*tile_num_h*tile_num_w+tid)*tile_width*tile_width+eid] += tiles.get()[((k*C+c)*tile_num_h*tile_num_w+tid)*tile_width*tile_width+eid];
	    }
	}
    }
    float tile_tmp[tile_height][tile_width];
    float result_tmp[2][2];
    for(int k=0; k<K; ++k){
	for(int c=0; c<C;++c){
	    for(int tid=0; tid<tile_num_h*tile_num_w; ++tid){
		for(int i=0; i<tile_height; ++i){
		    for(int j=0; j<tile_width; ++j){
			tile_tmp[i][j] = local_sum.get()[((k*tile_num_h*tile_num_w+tid)*tile_height+i)*tile_width+j];
		    }
		}
		result_tmp[0][0] = tile_tmp[0][0] + tile_tmp[0][1] + tile_tmp[0][2] + tile_tmp[1][0] + tile_tmp[1][1] + tile_tmp[1][2] + tile_tmp[2][0] + tile_tmp[2][1] + tile_tmp[2][2];
		result_tmp[0][1] = tile_tmp[0][1] - tile_tmp[0][2] - tile_tmp[0][3] + tile_tmp[1][1] - tile_tmp[1][2] - tile_tmp[1][3] + tile_tmp[2][1] - tile_tmp[2][2] - tile_tmp[2][3];
		result_tmp[1][0] = tile_tmp[1][0] + tile_tmp[1][1] + tile_tmp[1][2] - tile_tmp[2][0] - tile_tmp[2][1] - tile_tmp[2][2] - tile_tmp[3][0] - tile_tmp[3][1] - tile_tmp[3][2];
		result_tmp[1][1] = tile_tmp[1][1] - tile_tmp[1][2] - tile_tmp[1][3] - tile_tmp[2][1] + tile_tmp[2][2] + tile_tmp[2][3] - tile_tmp[3][1] + tile_tmp[3][2] + tile_tmp[3][3];
		float*output_pos = output_ref.get()+k*H*W+(tid/tile_num_w)*2*W+(tid%tile_num_w)*2;
		for(int i=0; i<2; ++i){
		    for(int j=0; j<2; ++j){
			output_pos[i*W+j] = result_tmp[i][j];
		    }
		}
	    }
	}
    }
    bool equal = true;
    for(int k=0; k<K; ++k){
	for(int h=0; h<H; ++h){
	    for(int w=0; w<W; ++w){
		float rst1 = output_ref.get()[(k*H+h)*W+w];
		float rst2 = output.get()[(k*H+h)*W+w];
		if(fabs(rst1-rst2)>0.001f){
		    printf("error at %d,%d,%d, %f vs %f \n", k, h, w, rst1, rst2);
		    equal = false;
		}
	    }
	}
    }
    if(equal){
	printf("pass\n");
    }
    */
    //clean up
    error = cudaFree(tile_dev);
    if(error != cudaSuccess){
	printf("at line %d,cuda error with code=%d,error=%s\n", __LINE__, error, cudaGetErrorString(error));
    }
    error = cudaFree(output_dev);
    if(error != cudaSuccess){
	printf("at line %d,cuda error with code=%d,error=%s\n", __LINE__, error, cudaGetErrorString(error));
    }
    //for(int i=0; i<stream_count; ++i){
    //    error = cudaStreamDestroy(streams[i]);
    //    if(error != cudaSuccess){
    //        printf("at line %d,cuda error with code=%d,error=%s\n", __LINE__, error, cudaGetErrorString(error));
    //    }
    //}
    return 0;
}
