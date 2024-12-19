#include "llama-ltm.h"
#include "llama-impl.h"
#include "llama.h"


#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <fstream> 
#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cstring> // For strcpy

#define MAX_MODEL_NAME_SIZE 256
#define MAX_LAYERS 2048
#define MAX_LAYER_SIZE 65536
#define MAX_DATA_SIZE 524288

const std::string kPathSeparator =
#ifdef _WIN32
                            "\\";
#else
                            "/";
#endif



#define READ_DATA(fs, mem) {fs.read(reinterpret_cast<char*>(&mem), sizeof(mem));}
#define READ_DATA_PTR(fs, mem, len) {fs.read(reinterpret_cast<char*>(mem), len);}

// Function to read into a char buffer with a maximum length
void read_str(std::ifstream& inputFile, char* buffer, size_t maxLength) {
    inputFile.getline(buffer, maxLength, '\0'); 

    // Check if the maximum length was reached (null terminator not found)
    if (inputFile.fail() && !inputFile.eof()) { 
        // Clear the error state
        inputFile.clear(); 

        // Discard remaining characters in the line
        inputFile.ignore(std::numeric_limits<std::streamsize>::max(), '\0'); 

        throw std::runtime_error("Line length exceeds maximum buffer size.");
    } 
}


// Function to read the binary file and populate the structs
// Returns a single ltm_file_header and single ltm_store_entry. Caller is expected to call delete on these structs when finished with them.
std::pair<ltm_file_header *, ltm_store_entry *> read_ltm_file(const std::string& filename) {
    // Open the file for reading
    std::ifstream inputFile(filename, std::ios::binary);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Unable to open file for reading: " + filename);
    }

    ltm_file_header *header = NULL;
    ltm_store_entry *store_entry = NULL;

    try {
        //file_header
        {
            
            int magic;
            int version;
            READ_DATA(inputFile, magic);
            READ_DATA(inputFile, version);

            char model_name[MAX_MODEL_NAME_SIZE];

            read_str(inputFile,model_name, MAX_MODEL_NAME_SIZE);

            header = new ltm_file_header(magic,version,model_name);
        }

        //file entry
        {
            int64_t date_ts;
            u_int16_t n_layers;

            u_int16_t n_data; //number of elements in data
            char * data; // text output being stored

            READ_DATA(inputFile, date_ts);
            READ_DATA(inputFile, n_layers);

            if(n_layers > MAX_LAYERS) {
                throw std::runtime_error("Too many layers: " + n_layers);
            }

            ltm_entry_layer *layers_in = new ltm_entry_layer [n_layers];

            for(int i = 0; i < n_layers; i++) {
                //entry layer
                {
                    ltm_entry_layer * e = &layers_in[i];

                    READ_DATA(inputFile, e->layer_index)
                    READ_DATA(inputFile, e->n_layer)
                    if(e->n_layer > MAX_LAYER_SIZE)
                        throw std::runtime_error("Layer size too big: " + e->n_layer);
                    e->f_layer_data = new float[e->n_layer];

                    READ_DATA_PTR(inputFile, e->f_layer_data, e->n_layer)
                }
            }

            READ_DATA(inputFile, n_data);

            char data_in[MAX_DATA_SIZE];

            read_str(inputFile,data, MAX_DATA_SIZE);

            store_entry = new ltm_store_entry(date_ts,n_layers,layers_in,n_data,data_in);

            delete [] layers_in;
        }

        return std::make_pair(header, store_entry);
    } catch (const std::exception& e) {
        //TODO 3 this can still leak memory if exceptions are thrown. c++ is annoying and archaic.
        if(header)
            delete header;
        if (store_entry)
        {
            delete store_entry;
        }
        throw;
    }

}

//TODO 3 make a new version that can store more than one entry in the file. It should put the number of entries in ltm_file_header

#define WRITE_DATA(fs, mem) {fs.write(reinterpret_cast<const char*>(&mem), sizeof(mem));}
#define WRITE_DATA_PTR(fs, mem, len) {fs.write(reinterpret_cast<const char*>(mem), len);}

static int write_ltm_header(std::ofstream &outputFile, ltm_file_header &h) {
    //TODO 2 handle write errors
    WRITE_DATA(outputFile,h.magic)
    WRITE_DATA(outputFile,h.version)
    WRITE_DATA_PTR(outputFile,h.model_name,sizeof(char) * (1+strlen(h.model_name))) //+1 for null char at end

    return 0;
}

static int write_ltm_entry(std::ofstream &outputFile, ltm_store_entry &se) {
    //TODO 2 handle write errors
    WRITE_DATA(outputFile,se.date_ts)
    WRITE_DATA(outputFile,se.n_layers)
    for(int i = 0; i < se.n_layers; i++) {
        const ltm_entry_layer *el = &se.layers[i];
        WRITE_DATA(outputFile,el->layer_index)
        WRITE_DATA(outputFile,el->n_layer)
        WRITE_DATA_PTR(outputFile,el->f_layer_data,sizeof(float) * el->n_layer)
    }
    WRITE_DATA(outputFile,se.n_data)
    WRITE_DATA_PTR(outputFile,se.data,sizeof(char) * se.n_data)

    return 0;
}

#define LTM_MAGIC 0x5f469a77
#define LTM_VERSION 1

void write_ltm_layer_data(std::string data_dir, llama_model *model, std::vector<llama_token> current_context_window, 
                                         std::vector<ltm_entry_layer> layers)
{
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::time_t now_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_local = *std::localtime(&now_t);
    std::stringstream ss;
    ss << std::put_time(&tm_local, "%Y-%m-%d_%H-%M-%S");
    std::string timestamp = ss.str();

    // Create the filename
    std::string filename = data_dir+ kPathSeparator + "ltm_" + timestamp + ".bin";

    // Open the file for writing
    std::ofstream outputFile(filename, std::ios::binary);

    if (!outputFile.is_open()) {
        LLAMA_LOG_ERROR("unable to open ltm data file for writing: %s\n",filename);
        return;
    }

    char model_desc[40];
    llama_model_desc(model,model_desc,40);

    ltm_file_header fh(LTM_MAGIC, LTM_VERSION, model_desc);

    //TODO 3, what should we do here, keep allocating more until we can hold it?
    int text_max_len = current_context_window.size() * 20; //20x should be more than enough
    char text [text_max_len];

    // LLAMA_API int32_t llama_detokenize(
    //     const struct llama_model * model,
    //            const llama_token * tokens,
    //                      int32_t   n_tokens,
    //                         char * text,
    //                      int32_t   text_len_max,
    //                         bool   remove_special,
    //                         bool   unparse_special);

    int32_t text_len = llama_detokenize(model,
                                        current_context_window.data(),
                                        current_context_window.size(),
                                        text,
                                        text_max_len,
                                        false,
                                        true);


    ltm_store_entry se(
        static_cast<uint64_t>(now_t), //     const int64_t date_ts;
        layers.size(),//     const u_int16_t n_layers; //total number of elements in index (n_index/layer_size == number of layers saved)
        layers.data(),//     const ltm_entry_layer *layers;
        text_len, // const u_int16_t n_data; //number of elements in data
        text//     const char * data; // text output being stored
    );

    write_ltm_header(outputFile,fh);
    write_ltm_entry(outputFile,se);

    outputFile.close();

}

