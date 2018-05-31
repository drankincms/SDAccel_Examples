#COMMON_REPO := ../../../
COMMON_REPO := /home/centos/src/project_data/aws-fpga/SDAccel/examples/xilinx/

include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/opencl/opencl.mk

# Host Application
host_SRCS=./src/host.cpp $(xcl2_SRCS)
host_HDRS=$(xcl2_HDRS)
host_CXXFLAGS=-I./src/ $(xcl2_CXXFLAGS) $(opencl_CXXFLAGS) -std=c++11
host_LDFLAGS=$(opencl_LDFLAGS) -I$(XILINX_SDACCEL)/Vivado_HLS/include/ -Wno-unknown-pragmas

# aws_hls4ml Kernels
aws_hls4ml_SRCS=./src/aws_hls4ml.cpp ./src/conv1d_small.cpp
aws_hls4ml_CLFLAGS=-k aws_hls4ml -I./src/

EXES=host
XCLBINS=aws_hls4ml

XOS=aws_hls4ml

aws_hls4ml_XOS=aws_hls4ml

# check
check_EXE=host
check_XCLBINS=aws_hls4ml

CHECKS=check

include $(COMMON_REPO)/utility/rules.mk
