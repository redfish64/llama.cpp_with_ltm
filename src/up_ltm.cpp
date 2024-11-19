#include "ggml.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <memory>

#ifdef DOOMSDAY
enum class ReductionMethod {
    RANDOM_PROJECTION,
    PCA
};

class DimensionalityReducer {
private:
    int input_dim;
    int output_dim;
    ReductionMethod method;
    struct ggml_context* ctx;
    struct ggml_tensor* projection_matrix;
    
    // For PCA
    Eigen::MatrixXf pca_components;
    Eigen::VectorXf mean_vector;
    bool pca_fitted = false;
    
    // Normalization parameters
    float* scale_factors = nullptr;
    float* shift_factors = nullptr;
    bool normalization_fitted = false;

    void initialize_random_projection() {
        float scale = sqrt(2.0f / (input_dim + output_dim));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        
        float* matrix_data = (float*)projection_matrix->data;
        for (int i = 0; i < input_dim * output_dim; i++) {
            matrix_data[i] = dist(gen);
        }
    }

    void fit_normalization(const std::vector<std::vector<float>>& training_data) {
        scale_factors = new float[input_dim];
        shift_factors = new float[input_dim];
        
        // Calculate mean and standard deviation for each dimension
        std::vector<float> sum(input_dim, 0.0f);
        std::vector<float> sum_sq(input_dim, 0.0f);
        
        for (const auto& sample : training_data) {
            for (int i = 0; i < input_dim; i++) {
                sum[i] += sample[i];
                sum_sq[i] += sample[i] * sample[i];
            }
        }
        
        int n = training_data.size();
        for (int i = 0; i < input_dim; i++) {
            shift_factors[i] = sum[i] / n;
            float variance = (sum_sq[i] / n) - (shift_factors[i] * shift_factors[i]);
            scale_factors[i] = 1.0f / sqrt(variance + 1e-6f);
        }
        
        normalization_fitted = true;
    }

    std::vector<float> normalize(const std::vector<float>& input) {
        if (!normalization_fitted) {
            return input;
        }
        
        std::vector<float> normalized(input_dim);
        for (int i = 0; i < input_dim; i++) {
            normalized[i] = (input[i] - shift_factors[i]) * scale_factors[i];
        }
        return normalized;
    }

    void fit_pca(const std::vector<std::vector<float>>& training_data) {
        // Convert training data to Eigen matrix
        Eigen::MatrixXf data(training_data.size(), input_dim);
        for (size_t i = 0; i < training_data.size(); i++) {
            for (int j = 0; j < input_dim; j++) {
                data(i, j) = training_data[i][j];
            }
        }
        
        // Compute mean
        mean_vector = data.colwise().mean();
        
        // Center the data
        data.rowwise() -= mean_vector.transpose();
        
        // Compute covariance matrix
        Eigen::MatrixXf covar = (data.transpose() * data) / float(data.rows() - 1);
        
        // Compute eigendecomposition
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(covar);
        
        // Sort eigenvectors by eigenvalues in descending order
        std::vector<std::pair<float, int>> eigenvalues(input_dim);
        for (int i = 0; i < input_dim; i++) {
            eigenvalues[i] = {eig.eigenvalues()(i), i};
        }
        std::sort(eigenvalues.begin(), eigenvalues.end(), std::greater<>());
        
        // Select top components
        pca_components = Eigen::MatrixXf(input_dim, output_dim);
        for (int i = 0; i < output_dim; i++) {
            pca_components.col(i) = eig.eigenvectors().col(eigenvalues[i].second);
        }
        
        pca_fitted = true;
    }

public:
    DimensionalityReducer(int in_dim, int out_dim, ReductionMethod m = ReductionMethod::RANDOM_PROJECTION) 
        : input_dim(in_dim), output_dim(out_dim), method(m) {
        // Initialize GGML context
        size_t ctx_size = in_dim * out_dim * sizeof(float) + 1024;
        ctx = ggml_init({.mem_size = ctx_size, .mem_buffer = NULL});
        
        // Create projection matrix
        projection_matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out_dim, in_dim);
        
        if (method == ReductionMethod::RANDOM_PROJECTION) {
            initialize_random_projection();
        }
    }
    
    void fit(const std::vector<std::vector<float>>& training_data) {
        // First fit normalization
        fit_normalization(training_data);
        
        // Then fit PCA if needed
        if (method == ReductionMethod::PCA) {
            fit_pca(training_data);
        }
    }
    
    std::vector<float> reduce(const std::vector<float>& input) {
        // First normalize the input
        std::vector<float> normalized = normalize(input);
        
        if (method == ReductionMethod::PCA && pca_fitted) {
            // Center the data
            Eigen::Map<const Eigen::VectorXf> input_eigen(normalized.data(), input_dim);
            Eigen::VectorXf centered = input_eigen - mean_vector;
            
            // Project using PCA components
            Eigen::VectorXf reduced = pca_components.transpose() * centered;
            
            return std::vector<float>(reduced.data(), reduced.data() + output_dim);
        } else {
            // Use random projection with GGML
            struct ggml_tensor* input_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_dim);
            memcpy(input_tensor->data, normalized.data(), input_dim * sizeof(float));
            
            struct ggml_tensor* result = ggml_mul_mat(ctx, projection_matrix, input_tensor);
            
            struct ggml_cgraph gf = ggml_build_forward(result);
            ggml_graph_compute(ctx, &gf);
            
            std::vector<float> reduced_vector(output_dim);
            memcpy(reduced_vector.data(), result->data, output_dim * sizeof(float));
            
            return reduced_vector;
        }
    }
    
    ~DimensionalityReducer() {
        if (scale_factors) delete[] scale_factors;
        if (shift_factors) delete[] shift_factors;
        ggml_free(ctx);
    }
};

class FAISSMemoryManager {
private:
    DimensionalityReducer reducer;
    std::unique_ptr<faiss::IndexIVFFlat> index;
    int nlist = 100;  // number of clusters for IVF
    bool is_trained = false;
    
public:
    FAISSMemoryManager(int layer_dim, int reduced_dim, ReductionMethod method = ReductionMethod::PCA) 
        : reducer(layer_dim, reduced_dim, method) {
        // Initialize FAISS index
        faiss::IndexFlatL2 quantizer(reduced_dim);
        index = std::make_unique<faiss::IndexIVFFlat>(&quantizer, reduced_dim, nlist);
        index->nprobe = 10;  // number of clusters to search at query time
    }
    
    void train(const std::vector<std::vector<float>>& training_data) {
        // First train the dimensionality reducer
        reducer.fit(training_data);
        
        // Reduce all training vectors
        std::vector<std::vector<float>> reduced_vectors;
        reduced_vectors.reserve(training_data.size());
        for (const auto& vector : training_data) {
            reduced_vectors.push_back(reducer.reduce(vector));
        }
        
        // Train FAISS index
        size_t n = reduced_vectors.size();
        float* training_data_raw = new float[n * reduced_vectors[0].size()];
        for (size_t i = 0; i < n; i++) {
            std::copy(reduced_vectors[i].begin(), reduced_vectors[i].end(), 
                     training_data_raw + i * reduced_vectors[i].size());
        }
        
        index->train(n, training_data_raw);
        delete[] training_data_raw;
        
        is_trained = true;
    }
    
    void add_memory(const std::vector<float>& layer_output) {
        if (!is_trained) {
            throw std::runtime_error("Index must be trained before adding vectors");
        }
        
        std::vector<float> reduced = reducer.reduce(layer_output);
        index->add(1, reduced.data());
    }
    
    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k) {
        if (!is_trained) {
            throw std::runtime_error("Index must be trained before searching");
        }
        
        std::vector<float> reduced = reducer.reduce(query);
        std::vector<float> distances(k);
        std::vector<faiss::Index::idx_t> indices(k);
        
        index->search(1, reduced.data(), k, distances.data(), indices.data());
        
        std::vector<std::pair<int, float>> results;
        for (int i = 0; i < k; i++) {
            results.emplace_back(indices[i], distances[i]);
        }
        
        return results;
    }
};

// Example usage
void example_usage() {
    const int layer_dim = 4096;  // Original LLM layer dimension
    const int reduced_dim = 256;  // Reduced dimension
    
    // Create memory manager with PCA reduction
    FAISSMemoryManager memory_manager(layer_dim, reduced_dim, ReductionMethod::PCA);
    
    // Training data (collect this from your LLM runs)
    std::vector<std::vector<float>> training_data;
    // ... fill training_data ...
    
    // Train the system
    memory_manager.train(training_data);
    
    // Add memories
    std::vector<float> layer_output(layer_dim);
    // ... fill layer_output from LLM ...
    memory_manager.add_memory(layer_output);
    
    // Search for similar memories
    std::vector<std::pair<int, float>> similar_memories = memory_manager.search(layer_output, 5);
}

#endif