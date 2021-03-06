/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#include "xcl2.hpp"
#include <vector>
#include <parameters.h>
#include "kernel_params.h"

#define DATA_SIZE_IN N_INPUTS
#define DATA_SIZE_OUT N_OUTPUTS

int main(int argc, char** argv)
{

    int nevents = 1;
    if (argc > 1) nevents = atoi(argv[1]);

    size_t vector_size_in_bytes = sizeof(data32_t) * DATA_SIZE_IN * STREAMSIZE;
    size_t vector_size_out_bytes = sizeof(data32_t) * DATA_SIZE_OUT * STREAMSIZE;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr 
    // is used if it is properly aligned. when not aligned, runtime had no choice but to create
    // its own host side buffer. So it is recommended to use this allocator if user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
    // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR 
    std::vector<data32_t,aligned_allocator<data32_t>> source_in(DATA_SIZE_IN*STREAMSIZE);
    std::vector<data32_t,aligned_allocator<data32_t>> source_hw_results(DATA_SIZE_OUT*STREAMSIZE);

    //initialize
    for(int j = 0 ; j < DATA_SIZE_IN*STREAMSIZE ; j++){
        source_in[j] = 0;
    }
    for(int j = 0 ; j < DATA_SIZE_OUT*STREAMSIZE ; j++){
        source_hw_results[j] = 0;
    }

// OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    // find_binary_file() is a utility API which will search the xclbin file for
    // targeted mode (sw_emu/hw_emu/hw) and for targeted platforms.
    std::string binaryFile = xcl::find_binary_file(device_name,"aws_hls4ml");

    // import_binary_file() ia a utility API which will load the binaryFile
    // and will return Binaries.
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and 
    // Device-to-host communication
    cl::Buffer buffer_in   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vector_size_in_bytes, source_in.data());
    cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            vector_size_out_bytes, source_hw_results.data());

    std::vector<cl::Memory> inBufVec, outBufVec;
    inBufVec.push_back(buffer_in);
    outBufVec.push_back(buffer_output);

    cl::Kernel krnl_aws_hls4ml(program,"aws_hls4ml");

    //int size = DATA_SIZE;
    int narg = 0;
    krnl_aws_hls4ml.setArg(narg++, buffer_in);
    krnl_aws_hls4ml.setArg(narg++, buffer_output);
    //krnl_vector_add.setArg(narg++, size);
    //q.enqueueMigrateMemObjects(inBufVec,0/* 0 means from host*/);
    // Launch the Kernel
    // For HLS kernels global and local size is always (1,1,1). So, it is recommended
    // to always use enqueueTask() for invoking HLS kernel
    //q.enqueueTask(krnl_aws_hls4ml);
    // Copy Result from Device Global Memory to Host Local Memory
    //q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST);

    //q.finish();
    std::cout << "Output of HLS4ML algo is:"<<std::endl;
    for (int i = 0 ; i < nevents ; i++){
        // Create the test data 
        for(int j = 0 ; j < DATA_SIZE_IN*STREAMSIZE ; j++){
            source_in[j] = (data32_t)(12.34*(j+DATA_SIZE_IN*STREAMSIZE*(i+1)));
            //this is just a random number to produce dummy input data
        }
        for(int j = 0 ; j < DATA_SIZE_OUT*STREAMSIZE ; j++){
            source_hw_results[j] = 0;
        }
    
        // Copy input data to device global memory
        q.enqueueMigrateMemObjects(inBufVec,0/* 0 means from host*/);
        // Launch the Kernel
        // For HLS kernels global and local size is always (1,1,1). So, it is recommended
        // to always use enqueueTask() for invoking HLS kernel
        q.enqueueTask(krnl_aws_hls4ml);
        // Copy Result from Device Global Memory to Host Local Memory
        q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();
        for (int j = 0 ; j < STREAMSIZE ; j++){
            for (int k = 0 ; k < DATA_SIZE_OUT ; k++){
    	        std::cout << source_hw_results[j*DATA_SIZE_OUT + k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout<<"---- END EVENT "<<i+1<<" ----"<<std::endl;
    }
// OPENCL HOST CODE AREA END

    std::cout << "TEST PASSED" << std::endl; 
    return EXIT_SUCCESS;
}
