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

/*******************************************************************************
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce latency and 
    device resource utilization of the resulting RTL code
    This is vector addition example to demonstrate how HLS optimizations are used in kernel. 
*******************************************************************************/

#include "parameters.h"
#include "conv1d_small.h"
#include <hls_stream.h>

#define STREAMSIZE 8

// Read Data from Global Memory and write into Stream inStream
static void read_input(const data32_t *in, hls::stream<input_t> &inStream)
{
    mem_rd: for (int i = 0 ; i < STREAMSIZE*Y_INPUTS_1*N_CHAN_1 ; i++){
#pragma HLS pipeline
//#pragma HLS LOOP_TRIPCOUNT min=4096 max=4096
        //Blocking write command to inStream
        inStream << (input_t)in[i];
    }
}

// Read Input data from inStream and write the result into outStream
static void compute_cnn(hls::stream<input_t> &inStream ,
        hls::stream<result_t> &outStream)
{
    execute: for (int i = 0 ; i < STREAMSIZE ; i++){
#pragma HLS pipeline
//#pragma HLS LOOP_TRIPCOUNT min=4096 max=4096
        //Blocking read command from inStream and Blocking write command 
        //to outStream 
        input_t vin_buffer[Y_INPUTS_1][N_CHAN_1];    // Local memory to store vector1
        result_t vout_buffer[N_OUTPUTS];  // Local Memory to store result
//#pragma HLS UNROLL
        for (int j = 0 ; j < Y_INPUTS_1 ; j++){
//#pragma HLS UNROLL
            for(int k = 0 ; k < N_CHAN_1 ; k++){
                vin_buffer[j][k] = inStream.read();
            }
        }
        hls4ml: conv1d_small(vin_buffer,vout_buffer);
        for (int j = 0 ; j < N_OUTPUTS ; j++ ){
            outStream << vout_buffer[j];
        }
    }
}

// Read result from outStream and write the result to Global Memory
static void write_result(data32_t *out, hls::stream<result_t>
        &outStream)
{
    mem_wr: for (int i = 0 ; i < STREAMSIZE*N_OUTPUTS ; i++){
#pragma HLS pipeline
//#pragma HLS LOOP_TRIPCOUNT min=4096 max=4096
        //Blocking read command to inStream
        out[i] = (data32_t)outStream.read();
    }
}


/*
    Vector Addition Kernel Implementation 
    Arguments:
        in    (input)     --> Input Vector
        out   (output)    --> Output Vector
   */
extern "C" {
void aws_hls4ml(
        const data32_t *in, // Read-Only Vector
        data32_t *out       // Output Result
        )
{
// SDAccel kernel must have one and only one s_axilite interface which will be used by host application to configure the kernel.
// Here bundle control is defined which is s_axilite interface and associated with all the arguments (in1, in2, out and size),
// control interface must also be associated with "return".
// All the global memory access arguments must be associated to one m_axi(AXI Master Interface). Here all three arguments(in1, in2, out) are 
// associated to bundle gmem which means that a AXI master interface named "gmem" will be created in Kernel and all these variables will be 
// accessing global memory through this interface.
// Multiple interfaces can also be created based on the requirements. For example when multiple memory accessing arguments need access to
// global memory simultaneously, user can create multiple master interfaces and can connect to different arguments.
#pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=in   bundle=control
#pragma HLS INTERFACE s_axilite port=out  bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    hls::stream<input_t> inStream;
    hls::stream<result_t> outStream;
#pragma HLS STREAM variable=inStream  depth= 320
#pragma HLS STREAM variable=outStream depth= 40
//320 = STREAMSIZE (8) * Y_INPUTS_1 (10) * N_CHAN_1 (4)
//40  = STREAMSIZE (8) * N_OUTPUTS (5)

#pragma HLS dataflow
    read_input(in,inStream);
    compute_cnn(inStream,outStream);
    write_result(out,outStream);

}
}
