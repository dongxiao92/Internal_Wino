#include <cuda_runtime.h>
#include <memory>
#include <fstream>
#include <ctime>
#include <sys/time.h>
#include <unistd.h>
#pragma include_config
#include "config.h"

#include "cudatimer.h"

#define ITER_COUNT 100
__global__
void transform_input_whole(const float*input, float*output);
__global__
void transform_input_top_left(const float*input, float*output);
__global__
void transform_input_top_right(const float*input, float*output);
__global__
void transform_input_down_left(const float*input, float*output);
__global__
void transform_input_down_right(const float*input, float*output);

using namespace std;
//#define DEBUG
int main(const int argc, const char*argv[])
{
    cudaError_t error;

    const int  image_count = H*W*C;
    const int tile_count = (H/2)*(W/2)*16*C;
    //malloc memory
    unique_ptr<float>image(new float[image_count]);
    unique_ptr<float>tiles(new float[tile_count]);
    float*image_dev;
    float*tile_dev;
    error = cudaMalloc(&image_dev, sizeof(float)*image_count);
    if(error != cudaSuccess){
	printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
    }
    error = cudaMalloc(&tile_dev, sizeof(float)*tile_count);
    if(error != cudaSuccess){
	printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
    }
    //gengerate image
    srand(time(0));
    for(int i=0;i<image_count;++i)
	image.get()[i] = static_cast<float>(rand()*1.f/RAND_MAX);
    error = cudaMemcpy(image_dev, image.get(), sizeof(float)*image_count, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
	printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
    }
    error = cudaMemset(tile_dev, 0, sizeof(float)*tile_count);
    if(error != cudaSuccess){
	printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
    }
    //call kernel
    dim3 block(img_block_w, img_block_h);
    dim3 grid(W/img_block_w, H/img_block_h, C);
    cudaDeviceSynchronize();
    struct timezone tz;
    struct timeval start_cpu[ITER_COUNT];
    struct timeval end_cpu[ITER_COUNT];

    CudaTimer timer;
    float total_time = 0.f;
    if(grid.x==1 && grid.y==1){
	for(int i=0; i<ITER_COUNT; ++i){
	    error = cudaMemset(tile_dev, 0, sizeof(float)*tile_count);
	    if(error != cudaSuccess){
	        printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
	    }
	    //gettimeofday(&start_cpu[i], &tz);
	    timer.start();
	    transform_input_whole<<<grid,block>>>(image_dev, tile_dev);
	    timer.stop();
	    total_time += timer.elapsedTime();
	    //cudaDeviceSynchronize();
	    //gettimeofday(&end_cpu[i], &tz);
	}
    }else if(grid.x == 2 && grid.y == 2){
	grid.x = 1;
	grid.y = 1;
	cudaStream_t stream0, stream1, stream2, stream3;
	error = cudaStreamCreate(&stream0);
	if(error != cudaSuccess){
	    printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
	}
	error = cudaStreamCreate(&stream1);
	if(error != cudaSuccess){
	    printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
	}
	error = cudaStreamCreate(&stream2);
	if(error != cudaSuccess){
	    printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
	}
	error = cudaStreamCreate(&stream3);
	if(error != cudaSuccess){
	    printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
	}
	for(int i=0; i<ITER_COUNT; ++i){
	    error = cudaMemset(tile_dev, 0, sizeof(float)*tile_count);
	    if(error != cudaSuccess){
	        printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
	    }
	    gettimeofday(&start_cpu[i], &tz);
	    transform_input_top_left<<<grid,block, 0, stream0>>>(image_dev, tile_dev);
	    transform_input_top_right<<<grid,block, 0, stream1>>>(image_dev+img_block_w, tile_dev);
	    transform_input_down_left<<<grid,block, 0, stream2>>>(image_dev+img_block_h*W, tile_dev);
	    transform_input_down_right<<<grid,block, 0, stream3>>>(image_dev+img_block_h*W+img_block_w, tile_dev);
	    cudaDeviceSynchronize();
	    gettimeofday(&end_cpu[i], &tz);
	    total_time += 1000*(end_cpu[i].tv_sec - start_cpu[i].tv_sec) + (end_cpu[i].tv_usec-start_cpu[i].tv_usec)*1.f/1000;
	}
        error = cudaStreamDestroy(stream0);
	if(error != cudaSuccess){
	    printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
	}
        error = cudaStreamDestroy(stream1);
	if(error != cudaSuccess){
	    printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
	}
        error = cudaStreamDestroy(stream2);
	if(error != cudaSuccess){
	    printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
	}
        error = cudaStreamDestroy(stream3);
	if(error != cudaSuccess){
	    printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
	}
    }else{
	printf("invalid kernel launch parameter\n");
	cudaFree(image_dev);
	cudaFree(tile_dev);
	return -1;
    }
    //float sumTime = 0.f;
    printf("average time:%.4f \n", total_time/ITER_COUNT);
    error = cudaMemcpy(tiles.get(), tile_dev, sizeof(float)*tile_count, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
	printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
    }
    error = cudaDeviceSynchronize();
    if(error != cudaSuccess){
	printf("at line %d,cuda error with code=%d,error=%s\n",  __LINE__, error, cudaGetErrorString(error));
    }
    //compute reference
    unique_ptr<float>tiles_ref(new float[tile_count]);
    const int padded_h = H+2*PAD;
    const int padded_w = W+2*PAD;
    unique_ptr<float>image_padded(new float[padded_h*padded_w*C]);
    memset(image_padded.get(), 0, sizeof(float)*padded_h*padded_w*C);
    for(int c=0; c<C;++c){
	for(int i=0;i<H;++i){
	    for(int j=0; j<W;++j){
		image_padded.get()[c*padded_h*padded_w+padded_w+PAD+(i*padded_w+j)] = image.get()[c*H*W+i*W+j];
	    }
	}
    }
    const int tile_num_x = W/2;
    const int tile_num_y = H/2;
    float image_tile[4][4];
    for(int c=0;c<C;++c){
        for(int ty=0; ty<tile_num_y; ++ty){
            for(int tx=0; tx<tile_num_x; ++tx){
                int start_h = ty*2;
                int start_w = tx*2;
                float*image_start = &(image_padded.get()[c*padded_h*padded_w+start_h*padded_w+start_w]);
                for(int row=0;row<4;++row){
		    for(int col=0;col<4;++col){
			//printf("image_tile[%d][%d]=%f \n", r, c, image_start[r*padded_w+c]);
			image_tile[row][col] = image_start[row*padded_w+col];
		    }
                }
                float trans_tmp1[4][4];
                for(int i=0;i<4;++i){
            	    trans_tmp1[0][i] = image_tile[0][i]-image_tile[2][i];
            	    trans_tmp1[1][i] = image_tile[1][i]+image_tile[2][i];
            	    trans_tmp1[2][i] = image_tile[2][i]-image_tile[1][i];
            	    trans_tmp1[3][i] = image_tile[1][i]-image_tile[3][i];
                }
                float trans_tmp2[4][4];
                for(int i=0;i<4;++i){
            	    trans_tmp2[i][0] = trans_tmp1[i][0]-trans_tmp1[i][2];
            	    trans_tmp2[i][1] = trans_tmp1[i][1]+trans_tmp1[i][2];
            	    trans_tmp2[i][2] = trans_tmp1[i][2]-trans_tmp1[i][1];
            	    trans_tmp2[i][3] = trans_tmp1[i][1]-trans_tmp1[i][3];
                }
		for(int row=0; row<4 ;++row){
		    for(int col=0; col<4; ++col){
			//printf("trans_rst[%d][%d]=%f\n", r, c, trans_tmp2[r][c]);
			tiles_ref.get()[16*(c*tile_num_x*tile_num_y+ty*tile_num_x+tx)+row*4+col]=trans_tmp2[row][col];
		    }
                }
            }
        }
    }
    //check
    bool equal = true;
    for(int i=0; i<tile_count; ++i){
	int channel = i/(16*tile_num_x*tile_num_y);
	int ty = (i/16/tile_num_x)%tile_num_y;
	int tx = (i/16)%tile_num_x;
	int idx = i%16;
	if(fabs(tiles_ref.get()[i]-tiles.get()[i])>0.00001f){
	    printf("error!(%d,%d,%d,%d):%f vs %f \n", channel, ty, tx, idx, tiles_ref.get()[i], tiles.get()[i]);
	    equal = false;
	}
    }
    if(equal)
	printf("pass\n");
#ifdef DEBUG
    fstream out("tile.gpu", fstream::out);
    for(int i=0 ;i<tile_count;++i){
	out<<tiles.get()[i]<<endl;
    }
    out.close();
#endif
    cudaFree(image_dev);
    cudaFree(tile_dev);
    return 0;
}
