import os
import sys
import subprocess
processes = set()
max_processes = 8

def generate_config_header(c, h, w, k):
    img_block = 2
    while img_block < h:
	cfg_file_name = "config_"+c+"_"+h+"_"+w+"_"+k+"_"+str(img_block)+"_"+str(img_block)+"_"+".h"
	with open(cfg_file_name, "w") as f:
	    f.write("#define H"+ +"\n")
	    f.write("#define W"+ +"\n")
	    f.write("#define C"+ +"\n")
	    f.write("#define img_block_h "+ str(img_block) +"\n")
	    f.write("#define img_block_w "+ str(img_block) +"\n")
	    f.write("#define PAD 1\n")
	    f.write("#define block_width (img_block_w+2*PAD)\n")
	    f.write("#define block_height (img_block_h+2*PAD)\n")
	with open("main.cu.template") as f:
	    lines = f.readlines()
	for idx in range(len(lines)):
	    if lines[idx].find("") !=-1:
		lines[idx] = "#include \""+cfg_file_name +"\"\n"
		break
	main_name = "main_"+str()+"_"+str()+".cu"
	with open(main_name, "w") as f:
	    for line in lines:
		f.write(line)

	img_block *= 2

if __name__=='__main__':
    cfgs = [ line.strip() for line in open(sys.argv[1]).readlines() if not line.startswith("#") ]
    base_ptx_cmd = "nvcc -O3 -arch sm_35 -ptx -I./ -o ptx/[ptx_name] [cu_name]"
    base_main_cmd = "nvcc -O3 -std=c++11 -arch sm_35 [main_name] [cu_name] -I./ -o [run_name]"
    main_cmd_list = []
    ptx_cmd_list = []
    main_list = []
    for cfg in cfgs:
	c,h,w,k = tuple(cfg.split(" "))
	print "%s,%s,%s,%s"%(c, h, w, k)
	img_block = int(h)/2
	while img_block <= int(h):
	    if img_block % 2 != 0:
		img_block *= 2
		continue
	    cfg_file_name = "config_"+c+"_"+h+"_"+w+"_"+k+"_"+str(img_block)+"_"+str(img_block)+".h"
	    with open(cfg_file_name, "w") as f:
	        f.write("#define H "+ h +"\n")
	        f.write("#define W "+ w +"\n")
	        f.write("#define C "+ c +"\n")
	        f.write("#define img_block_h "+ str(img_block) +"\n")
	        f.write("#define img_block_w "+ str(img_block) +"\n")
	        f.write("#define PAD 1\n")
	        f.write("#define block_width (img_block_w+2*PAD)\n")
	        f.write("#define block_height (img_block_h+2*PAD)\n")
	    with open("main.cu.template") as f:
		lines = f.readlines()
	    for idx in range(len(lines)):
		if lines[idx].find("#pragma include_config") !=-1:
		    lines[idx] = "#include \""+cfg_file_name +"\"\n"
		    break
	    main_name = "main_"+c+"_"+h+"_"+w+"_"+k+"_"+str(img_block)+"_"+str(img_block)+".cu"
	    with open(main_name, "w") as f:
		for line in lines:
		    f.write(line)
	    with open("input_trans_kernel.cu.template") as f:
		lines = f.readlines()
	    for idx in range(len(lines)):
		if lines[idx].find("#pragma include_config") !=-1:
		    lines[idx] = "#include \""+cfg_file_name +"\"\n"
		    break
	    cu_name = "input_trans_kernel_"+c+"_"+h+"_"+w+"_"+k+"_"+str(img_block)+"_"+str(img_block)+".cu"
	    with open(cu_name, "w") as f:
		for line in lines:
		    f.write(line)
	    ptx_cmd = base_ptx_cmd.replace("[cu_name]", cu_name).replace("[ptx_name]", cu_name.replace(".cu", ".ptx"))
	    main_cmd = base_main_cmd.replace("[main_name]", main_name).replace("[run_name]", main_name.replace(".cu", "")).replace("[cu_name]", cu_name)
	    #print ptx_cmd
	    #print main_cmd
	    main_cmd_list.append(main_cmd)
	    ptx_cmd_list.append(ptx_cmd)
	    main_list.append(main_name.replace(".cu", ""))
	    img_block *= 2
    for ptx_cmd in ptx_cmd_list:
	#print ptx_cmd
	processes.add(subprocess.Popen(ptx_cmd, shell=True))
	if len(processes) >= max_processes:
	    os.wait()
	    processes.difference_update([p for p in processes if p.poll() is not None])
    while len(processes) > 0:
	os.wait()
	processes.difference_update([p for p in processes if p.poll() is not None])
    for main_cmd in main_cmd_list:
	#print main_cmd
	processes.add(subprocess.Popen(main_cmd, shell=True))
	if len(processes) >= max_processes:
	    os.wait()
	    processes.difference_update([p for p in processes if p.poll() is not None])
    os.wait()
    for main_name in main_list:
	print ("./"+main_name)
	os.system("./"+main_name)

