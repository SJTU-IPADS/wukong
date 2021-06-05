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
#include <string>
#include <time.h>


namespace wukong {

class time_tool {
public:
    static bool is_date(std::string &s) {
        if(s.size() != 10) return false;
        if(s.at(0) >= '0' && s.at(0) <= '9' &&
           s.at(1) >= '0' && s.at(1) <= '9' &&
           s.at(2) >= '0' && s.at(2) <= '9' &&
           s.at(3) >= '0' && s.at(3) <= '9' &&
           s.at(4) == '-' &&
           s.at(5) >= '0' && s.at(5) <= '9' &&
           s.at(6) >= '0' && s.at(6) <= '9' &&
           s.at(7) == '-' &&
           s.at(8) >= '0' && s.at(8) <= '9' &&
           s.at(9) >= '0' && s.at(9) <= '9') {
               return true;
           } else return false;
    }

    static bool is_time(std::string &s) {
        if(s.size() != 19) return false;
        if(s.at(0) >= '0' && s.at(0) <= '2' &&
           s.at(1) >= '0' && s.at(1) <= '9' &&
           s.at(2) >= '0' && s.at(2) <= '9' &&
           s.at(3) >= '0' && s.at(3) <= '9' &&
           s.at(4) == '-' &&
           s.at(5) >= '0' && s.at(5) <= '1' &&
           s.at(6) >= '0' && s.at(6) <= '9' &&
           s.at(7) == '-' &&
           s.at(8) >= '0' && s.at(8) <= '3' &&
           s.at(9) >= '0' && s.at(9) <= '9' &&
           s.at(10) == 'T' &&
           s.at(11) >= '0' && s.at(11) <= '2' &&
           s.at(12) >= '0' && s.at(12) <= '9' &&
           s.at(13) == ':' &&
           s.at(14) >= '0' && s.at(14) <= '5' &&
           s.at(15) >= '0' && s.at(15) <= '9' &&
           s.at(16) == ':' &&
           s.at(17) >= '0' && s.at(17) <= '5' &&
           s.at(18) >= '0' && s.at(18) <= '9') {
            return true;
        } else return false;
    }

    static int64_t str2int(std::string &s) {
        struct tm tm;
        memset(&tm, 0, sizeof(tm));
        strptime(s.c_str(), "%Y-%m-%dT%H:%M:%S", &tm);
        return (int64_t)mktime(&tm);
    }

    static std::string int2str(int64_t num) {
        struct tm *ptm = gmtime(&num);
        return std::to_string(1900 + ptm->tm_year) + "-" + (ptm->tm_mon >= 9 ? std::to_string(1 + ptm->tm_mon) : "0" + std::to_string(1 + ptm->tm_mon)) + "-" +
        (ptm->tm_mday >= 10 ? std::to_string(ptm->tm_mday) : "0" + std::to_string(ptm->tm_mday)) + "T" +
        (ptm->tm_hour >= 10 ? std::to_string(ptm->tm_hour) : "0" + std::to_string(ptm->tm_hour)) + ":" +
        (ptm->tm_min >= 10 ? std::to_string(ptm->tm_min) : "0" + std::to_string(ptm->tm_min)) + ":" +
        (ptm->tm_sec >= 10 ? std::to_string(ptm->tm_sec) : "0" + std::to_string(ptm->tm_sec));
    }
};
}