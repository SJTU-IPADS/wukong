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

#include <stdint.h> // include this header for uint64_t

namespace wukong {

class atomic {
public:

    static uint64_t compare_and_swap(uint64_t *ptr, uint64_t old_val, uint64_t new_val) {
        return __sync_val_compare_and_swap(ptr, old_val, new_val);
    }

    static uint32_t compare_and_swap(uint32_t *ptr, uint32_t old_val, uint32_t new_val) {
        return __sync_val_compare_and_swap(ptr, old_val, new_val);
    }

    static uint64_t add_and_fetch(volatile uint64_t *ptr, uint64_t val) {
        return __sync_add_and_fetch(ptr, val);
    }

    static uint32_t add_and_fetch(volatile uint32_t *ptr, uint32_t val) {
        return __sync_add_and_fetch(ptr, val);
    }

}; // end of class atomic

} // end of namespace wukong
