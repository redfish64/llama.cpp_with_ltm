#include "llama-ltm.h"
#include "log.h"
#include "llama.h"


#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <fstream> 
#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>

const std::string kPathSeparator =
#ifdef _WIN32
                            "\\";
#else
                            "/";
#endif

struct timhack_ltm_file_header 
{
    const int magic;
    const int version;
    const char * model;
};

struct timhack_ltm_store_entry {
    const int64_t date_ts;
    const u_int16_t layer_index; // indexed from the last layer backwards. So if there are 32 layers and this is 2 then it would be layer 30
    const u_int16_t n_index; //number of elements in index
    const u_int16_t n_data; //number of elements in data
    const float * index; //the elements within the layer
    //PERF we could possibly store tokens, but I really want to be able to run strings on the bin files and get a good result
    const char * data; // text output being stored
};

#define WRITE_DATA(fs, mem) {fs.write(reinterpret_cast<const char*>(mem), sizeof(mem));}
#define WRITE_DATA_LEN(fs, mem, len) {fs.write(reinterpret_cast<const char*>(mem), len);}

static int timhack_write_ltm_header(std::ofstream &outputFile, timhack_ltm_file_header &h) {
    //TODO 2 handle write errors
    WRITE_DATA(outputFile,h.magic)
    WRITE_DATA(outputFile,h.version)
    WRITE_DATA_LEN(outputFile,h.model,sizeof(char) * strlen(h.model))

    return 0;
}

static int timhack_write_ltm_entry(std::ofstream &outputFile, timhack_ltm_store_entry &se) {
    //TODO 2 handle write errors
    WRITE_DATA(outputFile,se.date_ts)
    WRITE_DATA(outputFile,se.layer_index)
    WRITE_DATA(outputFile,se.n_index)
    WRITE_DATA(outputFile,se.n_data)
    WRITE_DATA(outputFile,se.date_ts)
    WRITE_DATA_LEN(outputFile,se.index,sizeof(float) * se.n_index)
    WRITE_DATA_LEN(outputFile,se.data,sizeof(char) * se.n_data)

    return 0;
}

#define TIMHACK_MAGIC 0x5f469a77
#define TIMHACK_LTM_VERSION 1

void timhack_write_ltm_layer_data(std::string data_dir, llama_model *model, std::vector<llama_token> current_context_window, 
                                         int extracted_layer_index,
                                         int index_len, float *index_data)
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
        LOG_ERR("unable to open ltm data file for writing: %s\n",filename);
        return;
    }

    char model_desc[40];
    llama_model_desc(model,model_desc,40);

    timhack_ltm_file_header fh =
        {
            TIMHACK_MAGIC,
            TIMHACK_LTM_VERSION,
            model_desc
        };

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


    timhack_ltm_store_entry se = 
    { 
        static_cast<uint64_t>(now_t),    // const int64_t date_ts;
        extracted_layer_index, // const u_int16_t layer_index; // indexed from the last layer backwards. So if there are 32 layers and this is 2 then it would be layer 30
        index_len, // const u_int16_t n_index; //number of elements in index
        text_len, // const u_int16_t n_data; //number of elements in data
        index_data, // const float * layer;
        text// const char * data; //PERF we could possibly store tokens, but I really want to be able to run strings on the log files and get a good result
    };

    timhack_write_ltm_header(outputFile,fh);
    timhack_write_ltm_entry(outputFile,se);

    outputFile.close();

}

