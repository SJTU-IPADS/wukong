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

#define KiB2B(_x)   ((_x) * 1024ul)
#define MiB2B(_x)   (KiB2B((_x)) * 1024ul)
#define GiB2B(_x)   (MiB2B((_x)) * 1024ul)

#define B2KiB(_x)   ((_x) / 1024.0)
#define B2MiB(_x)   (B2KiB((_x)) / 1024.0)
#define B2GiB(_x)   (B2MiB((_x)) / 1024.0)

#define USEC(_x)    ((_x) * 1ul)
#define MSEC(_x)    (USEC((_x)) * 1000ul)
#define SEC(_x)     (MSEC((_x)) * 1000ul)
