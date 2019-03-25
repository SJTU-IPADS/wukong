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

/**
 * Copyright (c) 2009 Carnegie Mellon University.
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
 *      http://www.graphlab.ml.cmu.edu
 *
 */


// Copyright (c) 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "errors.hpp"
#include "logger2.hpp"

// failure handling option
/*
 * if set WUKONG_LOGGER_THROW_ON_FAILURE
 * ---assertion will throw fail message
 * else
 * ---assertion will abort the program
 */
//#define WUKONG_LOGGER_FAIL_METHOD
#define WUKONG_LOGGER_THROW_ON_FAILURE

#ifdef WUKONG_LOGGER_THROW_ON_FAILURE
  #define WUKONG_LOGGER_FAIL_METHOD(status_code) \
      throw(WukongException(status_code))
#else
  #define WUKONG_LOGGER_FAIL_METHOD(status_code) abort()
#endif

/*
 * actual check
 * use __buildin_expect to provide the complier with branch prediction
 */
// check the condition
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (__builtin_expect(!(condition), 0)) {                                   \
      logstream(LOG_ERROR) << "Assertion: " << __FILE__ << "(" << __func__     \
                           << ":" << __LINE__ << ")"                           \
                           << ": \'" << #condition << "\' failed" << LOG_endl; \
      WUKONG_LOGGER_FAIL_METHOD(UNKNOWN_ERROR);                                \
    }                                                                          \
  } while (0)

// check the val1 op val2
#define CHECK_OP(op, val1, val2)                                            \
  do {                                                                      \
    const typeof(val1) _CHECK_OP_v1_ = (typeof(val1))val1;                  \
    const typeof(val2) _CHECK_OP_v2_ = (typeof(val2))val2;                  \
    if (__builtin_expect(!((_CHECK_OP_v1_)op(typeof(val1))(_CHECK_OP_v2_)), \
                         0)) {                                              \
      logstream(LOG_ERROR) << "Assertion: " << __FILE__ << "(" << __func__  \
                           << ":" << __LINE__ << ")"                        \
                           << ": \'" << #val1 << " " << #op << " " << #val2 \
                           << " [ " << val1 << " " << #op << " " << val2    \
                           << " ]\'"                                        \
                           << " failed" << LOG_endl;                        \
      WUKONG_LOGGER_FAIL_METHOD(UNKNOWN_ERROR);                             \
    }                                                                       \
  } while (0)

#define CHECK_EQ(val1, val2) CHECK_OP(==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(!=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(<=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(<, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(>=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(>, val1, val2)

// condition assert
#define ASSERT_TRUE(cond) CHECK(cond)
#define ASSERT_FALSE(cond) CHECK(!(cond))
// adapt to the original wukong assert
#define ASSERT(cond) CHECK(cond)

#define ASSERT_EQ(val1, val2) CHECK_EQ(val1, val2)
#define ASSERT_NE(val1, val2) CHECK_NE(val1, val2)
#define ASSERT_LE(val1, val2) CHECK_LE(val1, val2)
#define ASSERT_LT(val1, val2) CHECK_LT(val1, val2)
#define ASSERT_GE(val1, val2) CHECK_GE(val1, val2)
#define ASSERT_GT(val1, val2) CHECK_GT(val1, val2)

// string equal
#define ASSERT_STREQ(a, b) CHECK(strcmp(a, b) == 0)

// check the condition if wrong print out the message of variable parameters in
// fmt
#define ASSERT_MSG(condition, fmt, ...)                                        \
  do {                                                                         \
    if (__builtin_expect(!(condition), 0)) {                                   \
      logstream(LOG_ERROR) << "Assertion: " << __FILE__ << "(" << __func__     \
                           << ":" << __LINE__ << ")"                           \
                           << ": \'" << #condition << "\' failed" << LOG_endl; \
      logger(LOG_ERROR, fmt, ##__VA_ARGS__);                                   \
      WUKONG_LOGGER_FAIL_METHOD(fmt);                                          \
    }                                                                          \
  } while (0)

#define ASSERT_ERROR_CODE(condition, error_code)                     \
    do {                                                               \
        if (__builtin_expect(!(condition), 0)) {                       \
            logstream(LOG_DEBUG)                                       \
                << "Assertion: " << __FILE__ << "(" << __func__ << ":" \
                << __LINE__ << ")"                                     \
                << ": \'" << #condition << "\' failed" << LOG_endl;    \
            WUKONG_LOGGER_FAIL_METHOD(error_code);                    \
        }                                                              \
    } while (0)
