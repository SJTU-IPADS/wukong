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

#include <boost/variant.hpp>

using namespace std;

enum data_type { SID_t = 0, INT_t, FLOAT_t, DOUBLE_t };

//attr_t unions int, double, float
typedef boost::variant<int, double, float> attr_t;

// get the variant type
class variant_type : public boost::static_visitor<int> {
public:
    int operator ()(int i) const { return INT_t; }
    int operator ()(float f) const { return FLOAT_t; }
    int operator ()(double d) const { return DOUBLE_t; }
};

// get the size of variant type
static inline size_t get_sizeof(int type) {
    switch (type) {
    case INT_t: return sizeof(int);
    case FLOAT_t: return sizeof(float);
    case DOUBLE_t: return sizeof(double);
    default: return 0;
    }
}
