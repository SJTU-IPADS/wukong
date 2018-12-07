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

// utils
#include "logger2.hpp"

static void conflict_detector(void)
{

#if defined(USE_GPU) and defined(DYNAMIC_GSTORE)
    logstream(LOG_ERROR) << "Currently, USE_GPU cannot work with DYNAMIC_GSTORE. "
                         << "Please disable USE_GPU or DYNAMIC_GSTORE, and then rebuild Wukong."
                         << LOG_endl;
    exit(-1);
#endif

}
