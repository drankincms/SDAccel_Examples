#COMMON_REPO := ../../../
COMMON_REPO := /home/centos/src/project_data/aws-fpga/SDAccel/examples/xilinx/

#these need to be set by the user for their specific installation
HLS4ML_BASE := /home/centos/projects/hls4ml/
HLS4ML_PROJECT := my-hls-test
HLS4ML_NAME := myproject

include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/opencl/opencl.mk

# Host Application
host_SRCS=./src/host.cpp $(xcl2_SRCS)
host_HDRS=$(xcl2_HDRS)
host_CXXFLAGS=-I./src/ -I$(HLS4ML_BASE)/nnet_utils/ -I$(HLS4ML_BASE)/keras-to-hls/$(HLS4ML_PROJECT)/firmware/ $(xcl2_CXXFLAGS) $(opencl_CXXFLAGS) -std=c++11
host_LDFLAGS=$(opencl_LDFLAGS) -I$(XILINX_VIVADO)/include/ -I$(XILINX_SDACCEL)/include/ -Wno-unknown-pragmas

# aws_hls4ml Kernels
aws_hls4ml_SRCS=./src/aws_hls4ml.cpp $(HLS4ML_BASE)/keras-to-hls/$(HLS4ML_PROJECT)/firmware/$(HLS4ML_NAME).cpp 
aws_hls4ml_CLFLAGS=-k aws_hls4ml -I./src/ -I$(HLS4ML_BASE)/keras-to-hls/$(HLS4ML_PROJECT)/firmware/ -I$(HLS4ML_BASE)/keras-to-hls/$(HLS4ML_PROJECT)/firmware/weights -I$(HLS4ML_BASE)/nnet_utils/

EXES=host
XCLBINS=aws_hls4ml

XOS=aws_hls4ml

aws_hls4ml_XOS=aws_hls4ml

# check
check_EXE=host
check_XCLBINS=aws_hls4ml

CHECKS=check

include $(COMMON_REPO)/utility/rules.mk
