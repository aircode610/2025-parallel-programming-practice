#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <omp.h>


std::vector<size_t> read_array() {
    size_t length, a, b, p;
    std::cin >> length >> a >> b >> p;
    std::vector<size_t> result(length);
    result[0] = a % p;
    for (size_t i = 1; i < result.size(); ++i) {
        result[i] = (result[i - 1] * a + b) % p;
    }
    return result;
}


// Partition function for quicksort
template<typename T>
size_t partition(std::vector<T>& arr, size_t left, size_t right) {
    // Use median-of-three for better pivot selection
    size_t mid = left + (right - left) / 2;
    if (arr[mid] < arr[left]) std::swap(arr[left], arr[mid]);
    if (arr[right - 1] < arr[left]) std::swap(arr[left], arr[right - 1]);
    if (arr[right - 1] < arr[mid]) std::swap(arr[mid], arr[right - 1]);
    
    T pivot = arr[mid];
    std::swap(arr[mid], arr[right - 1]);
    
    size_t i = left;
    for (size_t j = left; j < right - 1; ++j) {
        if (arr[j] < pivot) {
            std::swap(arr[i], arr[j]);
            ++i;
        }
    }
    std::swap(arr[i], arr[right - 1]);
    return i;
}


// Parallel quicksort implementation using OpenMP
template<typename T>
void parallel_quicksort(std::vector<T>& arr, size_t left, size_t right, int depth = 0) {
    if (right <= left) {
        return;
    }
    
    size_t size = right - left;
    
    // Use std::sort for smaller arrays - increased threshold for better performance
    if (size < 100000) {
        std::sort(arr.begin() + left, arr.begin() + right);
        return;
    }
    
    // Partition the array
    size_t pivot_idx = partition(arr, left, right);
    
    // Parallel recursion for large enough chunks and not too deep
    if (depth < 5 && size > 200000) {
        #pragma omp task shared(arr) firstprivate(left, pivot_idx, depth)
        parallel_quicksort(arr, left, pivot_idx, depth + 1);
        
        #pragma omp task shared(arr) firstprivate(pivot_idx, right, depth)
        parallel_quicksort(arr, pivot_idx + 1, right, depth + 1);
        
        #pragma omp taskwait
    } else {
        // Sequential recursion
        parallel_quicksort(arr, left, pivot_idx, depth + 1);
        parallel_quicksort(arr, pivot_idx + 1, right, depth + 1);
    }
}


int main() {
    // Set number of threads to maximum available
    omp_set_num_threads(omp_get_max_threads());
    
    auto array = read_array();
    
    // Use parallel quicksort with OpenMP
    #pragma omp parallel
    {
        #pragma omp single
        parallel_quicksort(array, 0, array.size());
    }

    size_t k;
    std::cin >> k;
    for (size_t i = k - 1; i < array.size(); i += k) {
        std::cout << array[i] << ' ';
    }
    std::cout << "\n";

    return 0;
}