/*
 * Copyright (c) 2016 Shanghai Jiao Tong University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://ipads.se.sjtu.edu.cn/projects/wukong
 *
 */

#pragma once

#include <iostream>
#include <vector>
#include <stdint.h>
#include <assert.h>

#include "type.hpp"

namespace wukong {

class math {
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

    inline int static hash_mod(uint64_t n, int m) {
        if (m == 0)
            assert(false);
        return n % m;
    }

    // TomasWang's 64 bit integer hash
    static uint64_t hash_u64(uint64_t key) {
        key = (~key) + (key << 21); // key = (key << 21) - key - 1;
        key = key ^ (key >> 24);
        key = (key + (key << 3)) + (key << 8); // key * 265
        key = key ^ (key >> 14);
        key = (key + (key << 2)) + (key << 4); // key * 21
        key = key ^ (key >> 28);
        key = key + (key << 31);
        return key;
    }

    static uint64_t inverse_hash_u64(uint64_t key) {
        uint64_t tmp;

        // Invert key = key + (key << 31)
        tmp = key - (key << 31);
        key = key - (tmp << 31);

        // Invert key = key ^ (key >> 28)
        tmp = key ^ key >> 28;
        key = key ^ tmp >> 28;

        // Invert key *= 21
        key *= 14933078535860113213u;

        // Invert key = key ^ (key >> 14)
        tmp = key ^ key >> 14;
        tmp = key ^ tmp >> 14;
        tmp = key ^ tmp >> 14;
        key = key ^ tmp >> 14;

        // Invert key *= 265
        key *= 15244667743933553977u;

        // Invert key = key ^ (key >> 24)
        tmp = key ^ key >> 24;
        key = key ^ tmp >> 24;

        // Invert key = (~key) + (key << 21)
        tmp = ~key;
        tmp = ~(key - (tmp << 21));
        tmp = ~(key - (tmp << 21));
        key = ~(key - (tmp << 21));

        return key;
    }

    static uint64_t hash_prime_u64(uint64_t upper) {
        if (upper >= (1l << 31)) {
            std::cout << "WARNING: " << upper << " is too large!"
                      << std::endl;
            return upper;
        }

        if (upper >= 1610612741l) return 1610612741l;     // 2^30 ~ 2^31
        else if (upper >= 805306457l) return 805306457l;  // 2^29 ~ 2^30
        else if (upper >= 402653189l) return 402653189l;  // 2^28 ~ 2^29
        else if (upper >= 201326611l) return 201326611l;  // 2^27 ~ 2^28
        else if (upper >= 100663319l) return 100663319l;  // 2^26 ~ 2^27
        else if (upper >= 50331653l) return 50331653l;    // 2^25 ~ 2^26
        else if (upper >= 25165843l) return 25165843l;    // 2^24 ~ 2^25
        else if (upper >= 12582917l) return 12582917l;    // 2^23 ~ 2^24
        else if (upper >= 6291469l) return 6291469l;      // 2^22 ~ 2^23
        else if (upper >= 3145739l) return 3145739l;      // 2^21 ~ 2^22
        else if (upper >= 1572869l) return 1572869l;      // 2^20 ~ 2^21
        else if (upper >= 786433l) return 786433l;        // 2^19 ~ 2^20
        else if (upper >= 393241l) return 393241l;        // 2^18 ~ 2^19
        else if (upper >= 196613l) return 196613l;        // 2^17 ~ 2^18
        else if (upper >= 98317l) return 98317l;          // 2^16 ~ 2^17

        std::cout << "WARNING: " << upper << " is too small!"
                  << std::endl;
        return upper;
    }
}; // end of class math

class tuple {
    int static compare_tuple(int N, std::vector<sid_t>& vec,
                             int i, std::vector<sid_t>& vec2, int j) {
        // ture means less or equal
        for (int t = 0; t < N; t++) {
            if (vec[i * N + t] < vec2[j * N + t])
                return -1;

            if (vec[i * N + t] > vec2[j * N + t])
                return 1;
        }
        return 0;
    }

    inline void static swap_tuple(int N, std::vector<sid_t> &vec, int i, int j) {
        for (int t = 0; t < N; t++)
            std::swap(vec[i * N + t], vec[j * N + t]);
    }

    void static qsort_tuple_recursive(int N, std::vector<sid_t> &vec, int begin, int end) {
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

    bool static binary_search_tuple_recursive(int N, std::vector<sid_t> &vec,
            std::vector<sid_t> &target,
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
    bool static binary_search_tuple(int N, std::vector<sid_t> &vec,
                                    std::vector<sid_t> &target) {
        binary_search_tuple_recursive(N, vec, target, 0, vec.size() / N);
    }

    void static qsort_tuple(int N, std::vector<sid_t>& vec) {
        qsort_tuple_recursive(N, vec, 0, vec.size() / N);
    }
}; // end of class tuple

} // end of namespace wukong
