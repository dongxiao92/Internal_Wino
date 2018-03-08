#unroll element-wise multiplication in kernel
import os
import sys
import re
#add code to explain why max tile_num_h_per_kernel=16
max_thread_per_block=1024
tile_height=4
tile_width=4

def up_div(a, b):
    return (a+b-1)/b

def sanity_check(k_per_kernel, c_per_kernel, tile_num_h_per_kernel, tile_num_w_per_kernel, tile_count_h, tile_count_w):
    tb_w = tile_num_w_per_kernel
    if tb_w > 1024:
	return False
    tb_h = tile_num_h_per_kernel
    if tb_h > 1024:
	return False
    thread_count = tb_w*tb_h
    if thread_count > max_thread_per_block:
	return False

    #grid_w = tile_count_w/tile_num_w_per_kernel
    grid_w = up_div(tile_count_w, tile_num_w_per_kernel)
    if grid_w > 2147483647:
	return False
    #grid_h = tile_count_h/tile_num_h_per_kernel
    grid_h = up_div(tile_count_h, tile_num_h_per_kernel)
    if grid_h > 65535:
	return False
    block_count = grid_w*grid_h
    return True
    
def generate_unroll_config(channel, height, width, kernel):
    #output feature maps have the same saptial size with input feature maps.
    #each tile generate a 2*2 block in output featrure maps.so the count of tile can be computed as follows.
    tile_count_h = height/2
    tile_count_w = width/2
    
    #k_per_kernel_start = min(4, kernel)
    k_per_kernel_start = 1
    k_per_kernel_end = kernel
    k_per_kernel = []
    while k_per_kernel_start<=k_per_kernel_end:
	k_per_kernel.append(k_per_kernel_start)
	k_per_kernel_start *= 2
    #k_per_kernel = [k for k in range(k_per_kernel_start, k_per_kernel_end+1)]
    
    c_per_kernel_start = 1
    c_per_kernel_end = channel
    c_per_kernel = []
    while c_per_kernel_start <= c_per_kernel_end:
	c_per_kernel.append(c_per_kernel_start)
	c_per_kernel_start *= 2
    #c_per_kernel = [c for c in range(c_per_kernel_start, c_per_kernel_end+1)]
    #we assume input data have a suqare shape.so time_num_h and time_num_w should beequal
    #maximum number of threads per block is 1024,so tile_num_h_per_kernel<=sqrt(1024)
    tile_num_h_per_kernel_max = min(tile_count_h, 32)
    tile_num_w_per_kernel_max = min(tile_count_w, 32)
    tile_num_h_per_kernel_start = min(1, tile_count_h)
    tile_num_w_per_kernel_start = min(1, tile_count_w)
    
    tile_num_h_per_kernel = []
    while tile_num_h_per_kernel_start<=tile_num_h_per_kernel_max:
	tile_num_h_per_kernel.append(tile_num_h_per_kernel_start)
	tile_num_h_per_kernel_start *= 2
    tile_num_w_per_kernel = []
    while tile_num_w_per_kernel_start<=tile_num_w_per_kernel_max:
	tile_num_w_per_kernel.append(tile_num_w_per_kernel_start)
	tile_num_w_per_kernel_start *= 2
    #tile_num_h_per_kernel = [ tile_num_h_per_kernel_max ]
    #tile_num_w_per_kernel = [ tile_num_w_per_kernel_max ]
    configs=[]
    for k in k_per_kernel:
	for c in c_per_kernel:
	    for h in tile_num_h_per_kernel:
		for w in tile_num_w_per_kernel:
		    #check computation, memory access, resource requriment
		    if sanity_check(k, c, h, w, tile_count_h, tile_count_w):
			configs.append((k, c, h, w))
    return tile_count_h, tile_count_w, configs

def generate_config_header(H, W, K, C, tile_count_h, tile_count_w, tile_num_h_per_kernel, tile_num_w_per_kernel, k_per_kernel, c_per_kernel):
    config=""
    config += "#define H "+str(H) + "\n"
    config += "#define W "+str(W) + "\n"
    config += "#define K "+str(K) + "\n"
    config += "#define C "+str(C) + "\n"

    config += "#define tile_count_h "+str(tile_count_h) + "\n"
    config += "#define tile_count_w "+str(tile_count_w) + "\n"

    config += "#define tile_num_h_per_kernel "+str(tile_num_h_per_kernel) + "\n"
    config += "#define tile_num_w_per_kernel "+str(tile_num_w_per_kernel) + "\n"
    config += "#define k_per_kernel "+str(k_per_kernel) + "\n"
    config += "#define c_per_kernel "+str(c_per_kernel) + "\n"
    config += "#define COUT "+str(C/c_per_kernel) + "\n"
    config += "#define tile_height "+str(tile_height) + "\n"
    config += "#define tile_width "+str(tile_width) + "\n"

    return config

base_kernel_name = "__global__ \nvoid input_filter_num_kernel[kidx]_[cidx](const float*input, float*output)"
def unroll(cfg_name, H,W,K,C,tile_count_h, tile_count_w, cfg):
    def gen_param_filename(C,H,W,K):
	return "param_cuda_"+str(C)+"_"+str(K)+"_"+str(H)+"_"+str(W)+"_tile.h"
    def gen_tile(k,c,h,w):
	return "TILE_"+str(k)+"_"+str(c)+"_"+str(h)+"_"+str(w)
    k_per_kernel, c_per_kernel, tile_num_h_per_kernel, tile_num_w_per_kernel=cfg
    header = ""
    param_header = gen_param_filename(C, H, W, K)
    #header += "#include \"param.h\" \n"
    header += "#include \""+param_header+"\" \n"
    header += "#include \""+cfg_name+"\" \n"
    k_count = K/k_per_kernel
    c_count = C/c_per_kernel
    base_indent="    "
    kernel_content = ""
    for i in range(k_count):
	for j in range(c_count):
	    kernel_name = base_kernel_name.replace("[kidx]", str(i)).replace("[cidx]", str(j))
	    kernel_name += "{\n"
	    kernel_content += kernel_name
	    kernel_content += base_indent + "const int h = blockIdx.y*tile_num_h_per_kernel+threadIdx.y; \n"
	    kernel_content += base_indent + "const int w = blockIdx.x*tile_num_w_per_kernel+threadIdx.x; \n"
	    kernel_content += base_indent + "if(h<tile_count_h&&w<tile_count_w){ \n"
	    kernel_content += 2*base_indent + "const int channel = "+ str(j) +"; \n"
	    kernel_content += 2*base_indent + "const int kernel = k_per_kernel*" + str(i) +"; \n"
	    kernel_content += 2*base_indent + "const float* input_src = input+tile_height*tile_width*((channel*tile_count_h+h)*tile_count_w+w); \n" 
	    kernel_content += 2*base_indent + "float tile_input[16]; \n"
	    kernel_content += "#pragma unroll \n"
	    kernel_content += 2*base_indent + "for(int i=0; i<tile_height*tile_width; ++i)\n"
	    kernel_content += 3*base_indent + "tile_input[i] = input_src[i]; \n"
	    kernel_content += 2*base_indent + "float*output_pos = output+tile_height*tile_width*(((kernel*COUT+channel)*tile_count_h+h)*tile_count_w+w); \n"
	    kernel_content += 2*base_indent + "const int out_kernel_stride = tile_height*tile_width*(COUT*tile_count_h*tile_count_w); \n"
	    if c_per_kernel !=1:
		kernel_content += 2*base_indent + "const int input_channel_stride = tile_height*tile_width*(tile_count_h*tile_count_w); \n"
	    kernel_content += 2*base_indent + "float local_result[k_per_kernel*tile_height*tile_width]; \n"
	    kernel_content += "#pragma unroll \n"
	    kernel_content += 2*base_indent + "for(int i=0; i<k_per_kernel*tile_height*tile_width; ++i){ \n"
	    kernel_content += 4*base_indent + "local_result[i] = 0; \n"
	    kernel_content += 2*base_indent + "} \n"
	    #start unroll
	    for c in range(c_per_kernel):
		for k in range(k_per_kernel):
	    	    for h in range(tile_height):
	    	        for w in range(tile_width):
			    kernel_content += 2*base_indent + "local_result["+str((k*tile_height+h)*tile_width+w)+"] += tile_input["+str(h*tile_width+w)+"]*"+gen_tile(i*k_per_kernel+k, j*c_per_kernel+c, h, w) +"; \n"
		    #if k != k_per_kernel-1:
			#kernel_content += 2*base_indent + "output_pos += out_kernel_stride; \n"
		    #else:
			#kernel_content += 2*base_indent + "output_pos = output_pos-(C*(k_per_kernel-1)-1)*tile_height*tile_width*tile_count_h*tile_count_w; \n"
		if c != c_per_kernel -1:
		    kernel_content += 2*base_indent + "input_src += input_channel_stride; \n"
		    kernel_content += "#pragma unroll \n"
		    kernel_content += 2*base_indent + "for(int i=0; i<tile_height*tile_width; ++i)\n"
		    kernel_content += 3*base_indent + "tile_input[i] = input_src[i]; \n"
	    #write to global memory
	    kernel_content += "#pragma unroll\n"
	    kernel_content += 2*base_indent+"for(int ko=0; ko<k_per_kernel; ++ko){ \n"
	    kernel_content += 3*base_indent+"for(int ho=0; ho<tile_height; ++ho){ \n"
	    kernel_content += 4*base_indent+"for(int wo=0; wo<tile_width; ++wo){ \n"
	    kernel_content += 5*base_indent+"output_pos[ko*out_kernel_stride+ho*tile_width+wo] = local_result[(ko*tile_height+ho)*tile_width+wo]; \n"
	    kernel_content += 4*base_indent+ "}\n"
	    kernel_content += 3*base_indent+ "}\n"
	    kernel_content += 2*base_indent+ "}\n"

	    kernel_content += base_indent+ "}\n" #end for if
	    kernel_content += "}\n"#end for kernel
    return header+kernel_content

def generate_kernel_call(kernel,channel, k_per_kernel, c_per_kernel, template_driver_path, new_driver_path, config_path):
    def gen_stream(kernel_idx):
	stream_count = 32
	base_stream="streams[{i}]"
	return base_stream.replace("{i}", str(kernel_idx%stream_count))
    k_count = kernel/k_per_kernel
    c_count = channel/c_per_kernel
    #generate 'compute kernel count'
    kernel_count_compute = "const int kernel_count="+str(k_count*c_count) +";\n"
    base_launch="<<<block,grid,0,[stream]>>>"
    call_param = "(tile_dev, output_dev)"
    kernel_call = ""
    base_kernel_call = "input_filter_num_kernel[kidx]_[cidx]"
    for i in range(k_count):
	for j in range(c_count):
	    kernel_call += base_kernel_call.replace("[kidx]", str(i)).replace("[cidx]", str(j))
	    launch_param = base_launch.replace("[stream]", gen_stream(i*c_count+j))
	    kernel_call += launch_param + call_param + ";\n"
    print kernel_count_compute
    print kernel_call
    #insert into main.cu
    lines = []
    with open(template_driver_path) as f:
	lines = f.readlines()
    line_idx = 0
    for line_idx in range(len(lines)):
	if lines[line_idx].find("#pragma insert_kernel_config") !=-1:
	    lines[line_idx] = kernel_count_compute
	elif lines[line_idx].find("#pragma insert_kernel_call") !=-1:
	    lines[line_idx] = kernel_call
	elif lines[line_idx].find("#pragma config_header") != -1:
	    lines[line_idx] = "#include \""+config_path+"\" \n" 
	elif lines[line_idx].find("#pragma kernel_del_header") != -1:
	    #lines[line_idx] = "#include kernel.h \n"
	    base_kernel_dec = "__global__\n void input_filter_num_kernel[kidx]_[cidx](const float*, float*); \n"
	    kernel_dec = ""
	    for i in range(k_count):
		for j in range(c_count):
		    kernel_dec += base_kernel_dec.replace("[kidx]", str(i)).replace("[cidx]", str(j))
	    lines[line_idx] = kernel_dec;
    with open(new_driver_path, "w") as f:
	for line in lines:
	    f.write(line)

def help():
    #print "Usage:unroll_kernel <channel> <height> <width> <kernels> <template driver path>"
    print "Usage:unroll_kernel <cfg_file_path> <template driver path> <target path>"

if __name__=='__main__':
    def gen_file_name(base_name, C,H,W,K, k_per_kernel, c_per_kernel, tile_num_h_per_kernel, tile_num_w_per_kernel, ext):
	return base_name+"_"+str(C)+"_"+str(H)+"_"+str(W)+"_"+str(K)+"_"+str(k_per_kernel)+"_"+str(c_per_kernel)+"_"+str(tile_num_h_per_kernel)+"_"+str(tile_num_w_per_kernel)+ext
    #if len(sys.argv)<6:
    #    help()
    #    exit(-1)
    #channel = int(sys.argv[1])
    #height = int(sys.argv[2])
    #width = int(sys.argv[3])
    #kernel = int(sys.argv[4])
    #template_driver_path = sys.argv[5]
    if len(sys.argv)<3:
        help()
	exit(-1)
    cfg_path = sys.argv[1]
    template_driver_path = sys.argv[2]
    target_path = sys.argv[3]
    #import pdb
    #pdb.set_trace()
    with open(cfg_path) as f:
	tmp_lines = f.readlines()
	lines=[line for line in tmp_lines if not line.startswith("#")]
    for line in lines:
	channel, height, width, kernel = tuple([int(e) for e in line.split(" ")])
	#channel = int(sys.argv[1])
	#height = int(sys.argv[2])
	#width = int(sys.argv[3])
	#kernel = int(sys.argv[4])
	tile_count_h, tile_count_w, configs = generate_unroll_config(channel, height, width, kernel)
	#cfg_idx = 0
	for cfg in configs:
	    k_per_kernel, c_per_kernel, tile_num_h_per_kernel, tile_num_w_per_kernel = cfg
	    header = generate_config_header(height, width, kernel, channel, tile_count_h, tile_count_w, tile_num_h_per_kernel, tile_num_w_per_kernel, k_per_kernel, c_per_kernel)
	    #cfg_name = "config_"+str(cfg_idx)+".h"
	    cfg_name = gen_file_name("config", channel, height, width, kernel, k_per_kernel, c_per_kernel, tile_num_h_per_kernel, tile_num_w_per_kernel,".h")
	    with open(os.path.join(target_path, cfg_name), "w")  as f:
	        f.write(header)
	    kernel_content = unroll(cfg_name, height , width , kernel , channel, tile_count_h, tile_count_w, cfg)
	    #kernel_name = "mul_kernel_"+str(cfg_idx)+".cu"
	    kernel_name = gen_file_name("mul_kernel", channel, height, width, kernel, k_per_kernel, c_per_kernel, tile_num_h_per_kernel, tile_num_w_per_kernel,".cu")
	    with open(os.path.join(target_path, kernel_name), "w") as f:
	        f.write(kernel_content)
	    main_name = gen_file_name(template_driver_path, channel, height, width, kernel,  k_per_kernel, c_per_kernel, tile_num_h_per_kernel, tile_num_w_per_kernel,".cu")
	    generate_kernel_call(kernel,channel, k_per_kernel, c_per_kernel, template_driver_path, os.path.join(target_path, main_name), cfg_name)
	    #cfg_idx += 1
