CXX:=g++
NVCC:=nvcc
RM=rm
CUDA_LIBS=cudart
LD_FLAGS = $(foreach lib,$(CUDA_LIBS), -l$(lib))
CU_SRCS := $(shell find ./ -name "*.cu")
CU_OBJS := ${CU_SRCS:.cu=.o}

main:$(CU_OBJS)
	$(NVCC) -O3 -std=c++11 -o $@ $^
	$(RM) $^

$(CU_OBJS): %.o : %.cu
	$(NVCC) -c -O3 -std=c++11 $(LD_FLAGS) -o $@ $<

.PHONY: clean

clean:
	- $(RM) main
	- $(RM) $(CU_OBJS)
