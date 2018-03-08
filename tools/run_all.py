import os
import sys

def gen_file_name(base_name, C,H,W,K, k_per_kernel, c_per_kernel, tile_num_h_per_kernel, tile_num_w_per_kernel, ext):
  return base_name+"_"+str(C)+"_"+str(H)+"_"+str(W)+"_"+str(K)+"_"+str(k_per_kernel)+"_"+str(c_per_kernel)+"_"+str(tile_num_h_per_kernel)+"_"+str(tile_num_w_per_kernel)+ext

if __name__=='__main__':
  if len(sys.argv) < 4:
    print ("Usage: run_all.py <file_path> <include_path> <output_file>")
    exit()
  file_ans = open(sys.argv[3], "w")
  main_file_path = os.path.abspath(sys.argv[1])
  include_path = sys.argv[2]
  for root, dirs, files in os.walk(main_file_path):
    for name in files:
      main_file = os.path.join(root, name) 
      main_file_name = os.path.basename(main_file)
      if main_file_name.find("main.cu_") < 0:
        continue
      #print (main_file_name)
      params = main_file_name.strip("main.cu_") 
      params = params.strip(".cu")
      params = params.split("_")

      channel = int(params[0])
      height = int(params[1])
      width = int(params[2])
      kernel = int(params[3])
      k_per_kernel = int(params[4])
      c_per_kernel = int(params[5])
      tile_num_h_per_kernel = int(params[6])
      tile_num_w_per_kernel = int(params[7])
      #print (params)
    
      kernel_name = gen_file_name("mul_kernel", channel, height, width, kernel, k_per_kernel, c_per_kernel, tile_num_h_per_kernel, tile_num_w_per_kernel,".cu")
      kernel_file = os.path.join(main_file_path, kernel_name)
      #print (kernel_name)
      command = "nvcc -std=c++11 -O3 -o test "
      command = command + "{0} {1} ".format(main_file, kernel_file, include_path)
      command = command + "-I{0} -I{1}".format(main_file_path, include_path)
      print (command)
      os.system(command)
      command = "./test > log.txt"
      print (command)
      os.system(command)
      f = open("log.txt")
      line = f.readline().strip()
      #print (line)
      line = line.strip("average time::")
      time = line.strip(" ms")
      f.close()
      print (time)
      file_ans.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(channel, height, width, kernel, k_per_kernel, c_per_kernel, tile_num_h_per_kernel, tile_num_w_per_kernel, time))
  file_ans.close()

