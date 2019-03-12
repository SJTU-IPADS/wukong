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

// number of errors
#define ERRS_NUM (ERROR_LAST - ERROR_FIRST + 1)

// copy first error no
#define ERROR_FIRST         1
#define UNKNOWN_ERROR       1
#define SYNTAX_ERROR        2
#define UNKNOWN_PATTERN     3
#define ATTR_DISABLE        4
#define NO_REQUIRED_VAR     5
#define UNSUPPORT_UNION     6
#define OBJ_ERROR           7
#define VERTEX_INVALID      8
#define UNKNOWN_SUB         9
#define SETTING_ERROR       10
#define FIRST_PATTERN_ERROR 11
#define UNKNOWN_FILTER      12
#define ERROR_LAST          12
// copy last error no

// error_messages
const char *err_msgs[ERRS_NUM] = {"Something wrong happened",
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

// check the condition
#define ERR_MSG(errno)    (err_msgs[(errno)-ERROR_FIRST])

// An exception
struct WukongException {
   private:
    int status_code;
    const char *message;

   public:
    /// Constructor
    WukongException(const int status_code) : status_code(status_code) {}
    WukongException(const char *message)
        : message(message), status_code(UNKNOWN_ERROR) {}
    const char *msg() {
        return ERR_MSG(status_code);
    }

    int get_status_code(){return status_code;}

    /// Destructor
    ~WukongException() {}
};
