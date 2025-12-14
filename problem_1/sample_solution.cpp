#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <omp.h>


std::vector<std::vector<double>> read_matrix() {
    size_t rows, cols;
    std::cin >> rows >> cols;

    size_t a, b, x, y, z, p;
    std::cin >> a >> b >> x >> y >> z >> p;
    std::vector<std::vector<size_t>> intermediate(rows, std::vector<size_t>(cols, b % p));
    intermediate[0][0] = a % p;
    
    // This part has dependencies, so we keep it serial
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (i > 0 && j > 0) {
                intermediate[i][j] = (intermediate[i][j] + intermediate[i - 1][j - 1] * x) % p;
            }
            if (i > 0) {
                intermediate[i][j] = (intermediate[i][j] + intermediate[i - 1][j] * y) % p;
            }
            if (j > 0) {
                intermediate[i][j] = (intermediate[i][j] + intermediate[i][j - 1] * z) % p;
            }
        }
    }
    
    // Parallelize finding max value with reduction
    size_t max_value = 0;
    #pragma omp parallel for collapse(2) reduction(max:max_value)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (intermediate[i][j] > max_value) {
                max_value = intermediate[i][j];
            }
        }
    }

    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
    // Parallelize normalization
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = static_cast<double>(intermediate[i][j]) / static_cast<double>(max_value);
        }
    }

    return result;
}


int main() {
    auto left = read_matrix();
    auto right = read_matrix();
    auto left_rows = left.size();
    auto left_cols = left[0].size();
    auto right_cols = right[0].size();

    if (left.empty() || right.empty() || left[0].size() != right.size()) {
        std::cerr << "Wrong matrices";
        return 1;
    }

    std::vector<std::vector<double>> result(left_rows, std::vector<double>(right_cols, 0.0));
    
    // Optimized matrix multiplication with better cache locality
    // Loop reordering: i-k-j instead of i-j-k for sequential memory access
    #pragma omp parallel for
    for (size_t i = 0; i < left_rows; ++i) {
        for (size_t k = 0; k < left_cols; ++k) {
            double left_ik = left[i][k];
            for (size_t j = 0; j < right_cols; ++j) {
                result[i][j] += left_ik * right[k][j];
            }
        }
    }

    std::cout << left_rows << ' ' << right_cols << "\n";
    for (size_t i = 0; i < left_rows; ++i) {
        for (size_t j = 0; j < right_cols; ++j) {
            std::cout << std::setprecision(12) << result[i][j] << ' ';
        }
        std::cout << "\n";
    }

    return 0;
}