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

#include <sys/time.h>
#include <unistd.h>
#include <stdint.h>

class timer {
public:
    static uint64_t get_usec() {
        struct timespec tp;
        /* POSIX.1-2008: Applications should use the clock_gettime() function
           instead of the obsolescent gettimeofday() function. */
        /* NOTE: The clock_gettime() function is only available on Linux.
           The mach_absolute_time() function is an alternative on OSX. */
        clock_gettime(CLOCK_MONOTONIC, &tp);
        return ((tp.tv_sec * 1000 * 1000) + (tp.tv_nsec / 1000));
    }

    static void cpu_relax(int u) {
        int t = 166 * u;
        while ((t--) > 0)
            _mm_pause(); // a busy-wait loop
    }
};
