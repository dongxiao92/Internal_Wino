#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <cstdlib>
#include <cstdio>
#include <cstring>
using namespace std;

static void help(){
    printf("Usage:main <param file path> <output kernel number> <input channel number> <height> <width>\n");
}

int main(const int argc, const char*argv[]){
   if(argc<6){
	help();
	return -1;
    }
    string param_path = string(argv[1]);
    int outfm = stoi(argv[2]);
    int infm = stoi(argv[3]);
    int height = stoi(argv[4]);
    int width = stoi(argv[5]);
    //int convsize = stoi(argv[4]);
    const int convsize = 3;
    unique_ptr<float>params(new float[outfm*infm*convsize*convsize]);
    memset(params.get(), 0, sizeof(float)*outfm*infm*convsize*convsize);
    ifstream in(param_path);
    string line;
    uint32_t param_count = 0;
    while(getline(in, line, '\n')){
	auto pos = line.rfind(" ");
	if(pos!=string::npos){
	    float val = stof(line.substr(pos+1));
	    params.get()[param_count++] = val;
	}else{
	    printf("invalid line:%s \n", line.c_str());
	}
    }
    if(param_count!=outfm*infm*convsize*convsize){
	printf("param count=%d,param needed=%d\n", param_count, outfm*infm*convsize*convsize);
	return -1;
    }
    //transform
    const int tile_count = outfm*infm;
    const int tile_size = 4;
    unique_ptr<float>param_tiles(new float[tile_count*tile_size*tile_size]);
    for(int tid=0; tid<tile_count; ++tid){
	float param_block[3][3];
	float* param_start = params.get()+tid*(3*3);
	for(int h=0; h<3; ++h){
	    for(int w=0; w<3; ++w){
		param_block[h][w] = param_start[h*3+w];
	    }
	}
	float tile[4][4];
	float trans_tmp[4][3];
	for(int i=0; i<3; ++i){
	    trans_tmp[0][i] = param_block[0][i];
	    trans_tmp[1][i] = 0.5*(param_block[0][i]+param_block[1][i]+param_block[2][i]);
	    trans_tmp[2][i] = 0.5*(param_block[0][i]-param_block[1][i]+param_block[2][i]);
	    trans_tmp[3][i] = param_block[2][i];
	}
	for(int i=0; i<4; ++i){
	    tile[i][0] = trans_tmp[i][0];
	    tile[i][1] = 0.5*(trans_tmp[i][0]+trans_tmp[i][1]+trans_tmp[i][2]);
	    tile[i][2] = 0.5*(trans_tmp[i][0]-trans_tmp[i][1]+trans_tmp[i][2]);
	    tile[i][3] = trans_tmp[i][2];
	}
	float*out_pos = param_tiles.get()+tid*(tile_size*tile_size);
	for(int h=0; h<4; ++h){
	    for(int w=0; w<4; ++w){
		out_pos[h*4+w] = tile[h][w];
	    }
	}
    }
    //write to file
    const int max_length_per_line = 100;
    char* content = new char[tile_count*tile_size*tile_size*max_length_per_line];
    int idx = 0;
    for(int k=0; k<outfm; ++k){
	for(int c=0; c<infm; ++c){
	    for(int i=0; i<tile_size; ++i){
		for(int j=0; j<tile_size; ++j){
		    sprintf(content+strlen(content), "#define TILE_%d_%d_%d_%d %.24ff\n",
			    k, c, i, j, param_tiles.get()[idx++]);
		}
	    }
	}
    }
    //string out_path = param_path.substr(0, param_path.rfind("."))+"_tile"+".h";
    //how to name output param file depends on multuply kernel.
    string out_path = "param_cuda_"+to_string(infm)+"_"+to_string(outfm)+"_"+to_string(height)+"_"+to_string(width)+"_tile"+".h";
    printf("out path:%s\n", out_path.c_str());
    FILE*out = fopen(out_path.c_str() ,"wb");
    fwrite(content, sizeof(char), strlen(content), out);
    fclose(out);
    delete[] content;
    return 0;
}
