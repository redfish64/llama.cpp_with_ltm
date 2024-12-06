#pragma once

#include "llama.h"

#include <vector> 
#include <string> 

struct ltm_entry_layer {
    const u_int16_t layer_index; // index of layer within LLM
    const u_int16_t n_layer; // size of layer
    const float * layer; //the elements within the layer
};


void write_ltm_layer_data(std::string data_dir, llama_model *model, std::vector<llama_token> current_context_window, 
                                         std::vector<ltm_entry_layer>);
