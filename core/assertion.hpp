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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "logger2.hpp"

// failed option control
/*
 * if set WUKONG_LOGGER_FAIL_METHOD
 * ---assertion won't abort or throw any message
 * else
 * ---if set WUKONG_LOGGER_THROW_ON_FAILURE
 * ------assertion will throw fail message
 * ---else
 * ------assertion will abort the program
 */
//#define WUKONG_LOGGER_FAIL_METHOD
//#define WUKONG_LOGGER_THROW_ON_FAILURE
// failed option
#ifndef WUKONG_LOGGER_FAIL_METHOD
#ifdef WUKONG_LOGGER_THROW_ON_FAILURE
#define WUKONG_LOGGER_FAIL_METHOD(str) throw(str)
#else
#define WUKONG_LOGGER_FAIL_METHOD(str) abort()
#endif
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
      WUKONG_LOGGER_FAIL_METHOD("assertion failure");                          \
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
      WUKONG_LOGGER_FAIL_METHOD("assertion failure");                       \
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
      WUKONG_LOGGER_FAIL_METHOD("assertion failure");                          \
    }                                                                          \
  } while (0)
