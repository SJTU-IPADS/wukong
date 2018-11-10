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

#include<vector>

#include "type.hpp"

using namespace std;

class LoaderInterface {
public:
    virtual void load(const string &src,
                      vector<vector<triple_t>> &triple_pso,
                      vector<vector<triple_t>> &triple_pos,
                      vector<vector<triple_attr_t>> &triple_sav) = 0;

    virtual ~LoaderInterface() {}
};
