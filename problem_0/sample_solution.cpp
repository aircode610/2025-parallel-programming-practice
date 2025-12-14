#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>


std::istream& operator>>(std::istream& in, __int128& value) {
    std::string s;
    in >> s;
    value = 0;
    bool negative = false;
    size_t i = 0;
    if (s[0] == '-') {
        negative = true;
        i = 1;
    }
    for (; i < s.size(); ++i) {
        value = value * 10 + (s[i] - '0');
    }
    if (negative) value = -value;
    return in;
}

std::ostream& operator<<(std::ostream& out, __int128 value) {
    if (value == 0) {
        out << '0';
        return out;
    }
    std::string s;
    bool negative = false;
    if (value < 0) {
        negative = true;
        value = -value;
    }
    while (value > 0) {
        s += '0' + static_cast<int>(value % 10);
        value /= 10;
    }
    if (negative) s += '-';
    std::reverse(s.begin(), s.end());
    out << s;
    return out;
}

struct SharedState {
    __int128 n;
    std::vector<__int128> factors;
    std::mutex mtx;
    std::atomic<bool> done;
    
    SharedState(__int128 initial_n) : n(initial_n), done(false) {}
};

void find_factors_in_range(__int128 start, __int128 end, SharedState& state) {
    // Get initial value of n (cached locally to avoid repeated locking)
    __int128 cached_n;
    {
        std::lock_guard<std::mutex> lock(state.mtx);
        cached_n = state.n;
    }
    
    // Make start odd if it's even
    if (start % 2 == 0) start++;
    
    for (__int128 p = start; p < end; p += 2) {
        // Check done flag without any synchronization
        if (state.done.load(std::memory_order_relaxed)) {
            return;
        }
        
        // Early exit if p is too large for our cached n
        if (p > cached_n / p) {
            break;
        }
        
        // Check if p divides cached_n
        if (cached_n % p == 0) {
            // We found a potential factor! Now lock and verify
            std::lock_guard<std::mutex> lock(state.mtx);
            
            // Re-check with current value of n
            __int128 current_n = state.n;
            
            if (current_n == 1) {
                state.done.store(true, std::memory_order_relaxed);
                return;
            }
            
            // Extract all instances of this factor
            while (current_n % p == 0 && current_n > 1) {
                state.factors.push_back(p);
                current_n /= p;
                state.n = current_n;
            }
            
            // Update our cached value
            cached_n = current_n;
            
            if (current_n == 1) {
                state.done.store(true, std::memory_order_relaxed);
                return;
            }
        }
    }
}

int main() {
    __int128 n;
    std::cin >> n;
    if (n <= 1) {
        return 0;
    }

    SharedState state(n);
    
    // Handle factor of 2
    while (state.n % 2 == 0) {
        state.factors.push_back(2);
        state.n /= 2;
    }
    
    if (state.n == 1) {
        for (const auto& factor : state.factors) {
            std::cout << factor << ' ';
        }
        std::cout << '\n';
        return 0;
    }
    
    // Calculate sqrt(n) using binary search
    __int128 sqrt_n = 1;
    __int128 current_n = state.n;
    
    __int128 left = 1, right = current_n;
    while (left <= right) {
        __int128 mid = left + (right - left) / 2;
        
        // Carefully compute mid^2 to avoid overflow issues
        if (mid <= current_n / mid) {
            sqrt_n = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    // Start from 3 (first odd number after 2)
    __int128 range_start = 3;
    __int128 total_range = sqrt_n - range_start + 1;
    __int128 range_size = (total_range + num_threads - 1) / num_threads;
    
    if (range_size > 0 && sqrt_n >= 3) {
        std::vector<std::thread> threads;
        
        for (unsigned int i = 0; i < num_threads; ++i) {
            __int128 start = range_start + i * range_size;
            __int128 end = std::min(start + range_size, sqrt_n + 1);
            
            if (start >= sqrt_n + 1) break;
            
            threads.emplace_back(find_factors_in_range, start, end, std::ref(state));
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    // If n > 1 after all divisions, it's a prime factor
    if (state.n > 1) {
        state.factors.push_back(state.n);
    }
    
    // Sort factors before output
    std::sort(state.factors.begin(), state.factors.end());

    for (const auto& factor : state.factors) {
        std::cout << factor << ' ';
    }
    std::cout << '\n';

    return 0;
}