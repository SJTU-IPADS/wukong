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

#include <stdint.h>

#ifdef DTYPE_64BIT

typedef uint64_t sid_t;  // data type for string-id
typedef int64_t ssid_t;  // signed string id

#else

typedef uint32_t sid_t;  // data type for string-id
typedef int32_t ssid_t;  // signed string id

#endif

enum dir_t { IN, OUT, CORUN }; // direction: IN=0, OUT=1, and optimization hints

