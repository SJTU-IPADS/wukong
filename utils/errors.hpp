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
#include <exception>

#define ERR_MSG(n) (err_msgs[n])

// error begin with 1, no error can be 0
enum {
    SUCCESS, 
    UNKNOWN_ERROR,
    SYNTAX_ERROR,
    UNKNOWN_PATTERN,
    ATTR_DISABLE,
    NO_REQUIRED_VAR,
    UNSUPPORT_UNION,
    OBJ_ERROR,
    VERTEX_INVALID,
    UNKNOWN_SUB,
    SETTING_ERROR,
    FIRST_PATTERN_ERROR,
    UNKNOWN_FILTER,
    ERROR_LAST
};

// error_messages
const char *err_msgs[ERROR_LAST] = {
    "Everythong is ok",
    "Something wrong happened",
    "Something wrong in the query syntax, fail to parse!",
    "Unsupported triple pattern.",
    "MUST enable attribute support!",
    "NO required variables!",
    "Unsupport UNION on attribute results",
    "Object should not be an index",
    "Subject or object is not valid",
    "Tripple pattern should not start from unknown subject.",
    "You may change SETTING files to avoid this error. (e.g. global.hpp/config/...)",
    "Const_X_X or index_X_X must be the first pattern.",
    "Unsupported filter type."};

// An exception
struct WukongException : public exception {
   private:
    int status_code;
    const char *message;

   public:
    /// Constructor
    WukongException(const int status_code) : status_code(status_code) {}
    WukongException(const char *message)
        : message(message), status_code(UNKNOWN_ERROR) {}

    const char *what() const throw() { return ERR_MSG(status_code); }

    int code() { return status_code; }

    /// Destructor
    ~WukongException() {}
};
