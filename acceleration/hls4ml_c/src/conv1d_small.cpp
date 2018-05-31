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

#include "nnet_utils/nnet_layer.h"
#include "nnet_utils/nnet_conv.h"
#include "nnet_utils/nnet_activation.h"

//hls-fpga-machine-learning insert weights
#include "weights/w1.h"
#include "weights/b1.h"
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w5.h"
#include "weights/b5.h"

void conv1d_small(
		  input_t data[Y_INPUTS_1][N_CHAN_1],
		  result_t res[N_OUTPUTS])
{

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=data complete dim=0 
    #pragma HLS ARRAY_RESHAPE variable=res complete dim=0 
    //#pragma HLS INTERFACE ap_vld port=data,res 

    //#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
    //#pragma HLS INTERFACE axis port=data
    //#pragma HLS INTERFACE axis port=res

    #pragma HLS PIPELINE 

    //const_size_in   = Y_INPUTS_1*N_CHAN_1;
    //const_size_out  = N_OUTPUTS;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    input_t layer1_out[Y_OUTPUTS_1*N_FILT_1];
    #pragma HLS ARRAY_PARTITION variable=layer1_out complete dim=0
    input_t conv_layer1_out[Y_OUTPUTS_1][N_FILT_1];
    #pragma HLS ARRAY_PARTITION variable=conv_layer1_out complete dim=0
    nnet::conv_1d<input_t, input_t, config1>(data, conv_layer1_out, w1, b1);
    input_t logits1[Y_OUTPUTS_1*N_FILT_1];
    #pragma HLS ARRAY_PARTITION variable=logits1 complete dim=0
    nnet::flatten<input_t, Y_OUTPUTS_1, N_FILT_1>(conv_layer1_out, logits1);
    nnet::relu<input_t, input_t, relu_config1>(logits1, layer1_out);

    input_t layer2_out[Y_OUTPUTS_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    input_t conv_layer2_in[Y_INPUTS_2][N_CHAN_2];
    #pragma HLS ARRAY_PARTITION variable=conv_layer2_in complete dim=0
    nnet::unflatten<input_t, Y_INPUTS_2, N_CHAN_2>(layer1_out, conv_layer2_in);
    input_t conv_layer2_out[Y_OUTPUTS_2][N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=conv_layer2_out complete dim=0
    nnet::conv_1d<input_t, input_t, config2>(conv_layer2_in, conv_layer2_out, w2, b2);
    input_t logits2[Y_OUTPUTS_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=logits2 complete dim=0
    nnet::flatten<input_t, Y_OUTPUTS_2, N_FILT_2>(conv_layer2_out, logits2);
    nnet::relu<input_t, input_t, relu_config2>(logits2, layer2_out);

    input_t layer3_out[Y_OUTPUTS_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    input_t conv_layer3_in[Y_INPUTS_3][N_CHAN_3];
    #pragma HLS ARRAY_PARTITION variable=conv_layer3_in complete dim=0
    nnet::unflatten<input_t, Y_INPUTS_3, N_CHAN_3>(layer2_out, conv_layer3_in);
    input_t conv_layer3_out[Y_OUTPUTS_3][N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=conv_layer3_out complete dim=0
    nnet::conv_1d<input_t, input_t, config3>(conv_layer3_in, conv_layer3_out, w3, b3);
    input_t logits3[Y_OUTPUTS_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=logits3 complete dim=0
    nnet::flatten<input_t, Y_OUTPUTS_3, N_FILT_3>(conv_layer3_out, logits3);
    nnet::relu<input_t, input_t, relu_config3>(logits3, layer3_out);

    layer4_t layer4_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    layer4_t logits4[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=logits4 complete dim=0
    nnet::compute_layer<input_t, layer4_t, config4>(layer3_out, logits4, w4, b4);
    nnet::relu<layer4_t, layer4_t, relu_config4>(logits4, layer4_out);

    result_t logits5[N_OUTPUTS];
    #pragma HLS ARRAY_PARTITION variable=logits5 complete dim=0
    nnet::compute_layer<layer4_t, result_t, config5>(layer4_out, logits5, w5, b5);
    nnet::softmax<result_t, result_t, softmax_config5>(logits5, res);


}

