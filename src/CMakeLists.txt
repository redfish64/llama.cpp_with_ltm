# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

#
# libraries
#

# Find Eigen (add this at the top of the library section)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find Faiss
find_package(Faiss REQUIRED)

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-vocab.cpp
            llama-grammar.cpp
            llama-sampling.cpp
            llama-ltm.cpp
            up_ltm.cpp
            unicode.h
            unicode.cpp
            unicode-data.cpp
            )

target_include_directories(llama PUBLIC . ../include ${FAISS_INCLUDE_DIR})
target_compile_features   (llama PUBLIC cxx_std_11) # don't bump

target_link_libraries(llama 
    PUBLIC 
        ggml
        Eigen3::Eigen
        faiss
)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
