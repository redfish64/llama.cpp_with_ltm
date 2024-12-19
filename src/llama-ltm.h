#pragma once

#include "llama.h"

#include <vector> 
#include <string> 
#include <cstring> // For memcpy

struct ltm_entry_layer {
    u_int16_t layer_index; // index of layer within LLM
    u_int16_t n_layer; // size of layer
    float * f_layer_data; //the elements within the layer

    ltm_entry_layer() : layer_index(0), n_layer(0), f_layer_data(NULL)
    {
    }

    ltm_entry_layer(u_int16_t layer_index, u_int16_t n_layer, const float* layer_in) 
        : layer_index(layer_index), n_layer(n_layer) 
    {
        f_layer_data = new float[n_layer];
        std::memcpy(f_layer_data, layer_in, sizeof(float) * n_layer);
    }

    ~ltm_entry_layer() {
        delete[] f_layer_data;
    }
};

struct ltm_file_header {
    int magic;
    int version;
    char* model_name; 

    ltm_file_header(int magic, int version, const char* model_name_in) 
        : magic(magic), version(version) 
    {
        model_name = new char[strlen(model_name_in) + 1]; 
        strcpy(model_name, model_name_in); 
    }

    // Destructor
    ~ltm_file_header() {
        delete[] model_name;
    }
};

struct ltm_store_entry {
    int64_t date_ts;
    u_int16_t n_layers; //total number of elements in index (n_index/layer_size == number of layers saved)

    ltm_entry_layer *layers;

    //PERF we could possibly store tokens for data, but I really want to be able to run strings on the bin files and get a good result
    u_int16_t n_data; //number of elements in data
    char * data; // text output being stored

    // Constructor
    ltm_store_entry(int64_t date_ts, u_int16_t n_layers, 
                    const ltm_entry_layer* layers_in, 
                    u_int16_t n_data, const char* data_in) 
        : date_ts(date_ts), n_layers(n_layers), n_data(n_data) 
    {
        // Allocate memory for layers
        layers = new ltm_entry_layer[n_layers];
        for (int i = 0; i < n_layers; ++i) {
            layers[i].layer_index = layers_in[i].layer_index;
            layers[i].n_layer = layers_in[i].n_layer;
            layers[i].f_layer_data = new float[layers_in[i].n_layer];
            std::memcpy(layers[i].f_layer_data, layers_in[i].f_layer_data, sizeof(float) * layers_in[i].n_layer);
        }

        // Allocate memory for data
        data = new char[n_data + 1]; // +1 for null terminator (if needed)
        strcpy(data, data_in); 
    }

    // Destructor
    ~ltm_store_entry() {
        delete[] layers;
        delete[] data; 
    }
};



std::pair<ltm_file_header *, ltm_store_entry *> read_ltm_file(const std::string& filename);

void write_ltm_layer_data(std::string data_dir, llama_model *model, std::vector<llama_token> current_context_window, 
                                         std::vector<ltm_entry_layer>);
