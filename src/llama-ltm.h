#pragma once

#include "llama.h"

#include <vector> 
#include <string> 


void timhack_write_ltm_layer_data(std::string data_dir, llama_model *model, std::vector<llama_token> current_context_window, 
                                         int extracted_layer_index,
                                         int index_len, float *index_data);