#pragma once
#include <vector>

/* NOTE: math will conflict with other lib; so it's named mymath */
class mymath {
public:
    uint64_t static get_distribution(int r, std::vector<int>& distribution) {
        int sum = 0;
        for (int i = 0; i < distribution.size(); i++)
            sum += distribution[i];

        assert(sum > 0);
        r = r % sum;
        for (int i = 0; i < distribution.size(); i++) {
            if (r < distribution[i])
                return i;
            r -= distribution[i];
        }
        assert(false);
    }

    inline uint64_t static floor(uint64_t original, uint64_t n) {
        if (n == 0)
            assert(false);

        if (original % n == 0)
            return original;

        return original - original % n;
    }

    inline uint64_t static hash_mod(uint64_t n, uint64_t m) {
        if (m == 0)
            assert(false);

        return n % m;
    }

    static uint64_t hash(uint64_t key) {
        key = (~key) + (key << 21); // key = (key << 21) - key - 1;
        key = key ^ (key >> 24);
        key = (key + (key << 3)) + (key << 8); // key * 265
        key = key ^ (key >> 14);
        key = (key + (key << 2)) + (key << 4); // key * 21
        key = key ^ (key >> 28);
        key = key + (key << 31);

        return key;
    }


};

class mytuple {
    int static compare_tuple(int N, std::vector<int>& vec,
                             size_t i, std::vector<int>& vec2, size_t j) {
        // ture means less or equal
        for (int t = 0; t < N; t++) {
            if (vec[i * N + t] < vec2[j * N + t])
                return -1;

            if (vec[i * N + t] > vec2[j * N + t])
                return 1;
        }
        return 0;
    }

    inline void static swap_tuple(int N, std::vector<int>& vec,
                                  size_t i, size_t j) {
        for (int t = 0; t < N; t++)
            std::swap(vec[i * N + t], vec[j * N + t]);
    }

    void static qsort_tuple_recursive(int N, std::vector<int>& vec,
                                      size_t begin, size_t end) {
        if (begin + 1 >= end)
            return ;

        int middle = begin;
        for (int iter = begin + 1; iter < end; iter++) {
            if (compare_tuple(N, vec, iter, vec, begin) == -1 ) {
                middle++;
                swap_tuple(N, vec, iter, middle);
            }
        }

        swap_tuple(N, vec, begin, middle);
        qsort_tuple_recursive(N, vec, begin, middle);
        qsort_tuple_recursive(N, vec, middle + 1, end);
    }

    bool static binary_search_tuple_recursive(int N, std::vector<int>& vec,
            std::vector<int>& target,
            int begin, int end) {
        if (begin >= end)
            return false;

        int middle = (begin + end) / 2;
        int r = compare_tuple(N, target, 0, vec, middle);
        if (r == 0)
            return true;

        if (r < 0)
            return binary_search_tuple_recursive(N, vec, target, begin, middle);
        else
            return binary_search_tuple_recursive(N, vec, target, middle + 1, end);
    }


public:
    bool static binary_search_tuple(int N, std::vector<int>& vec,
                                    std::vector<int>& target) {
        binary_search_tuple_recursive(N, vec, target, 0, vec.size() / N);
    }

    void static qsort_tuple(int N, std::vector<int>& vec) {
        qsort_tuple_recursive(N, vec, 0, vec.size() / N);
    }
};
