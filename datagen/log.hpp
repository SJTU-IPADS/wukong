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
#include <stdio.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <string>

template <typename Record>
class Logger {
    std::string log_name;    // log file name
    std::string commit_name; // commit log file name
public:
    Logger(const std::string &file_path) : log_name(file_path + "/log"),
                                      commit_name(file_path + "/log_commit") { }

    void write_log(const Record &record) {
        std::ofstream file(log_name.c_str(), std::ofstream::out | std::ofstream::app);
        file << record.to_str() << " " << "commit" << std::endl;
        file.close();
    }

    void commit(const std::string &commit_info) {
        std::ofstream file(commit_name);
        file << commit_info << std::endl;
        file.close();
    }

    bool recover_from_failure(std::function<void(Record &)> func) {
        bool has_log = false;
        std::ifstream log_file(log_name.c_str());

        std::string tmp_name = log_name + ".tmp";
        std::ofstream tmp_file(tmp_name.c_str());

        while (!log_file.eof()) {
            std::string str_record;
            std::getline(log_file, str_record);
            // a valid record is:
            // record | "commit"
            size_t pos = str_record.find("commit");
            if (pos != std::string::npos) {
                Record record(str_record.substr(0, pos));
                func(record);

                tmp_file << str_record << std::endl;
                has_log = true;
            } else {
                break;
            }
        }
        log_file.close();
        tmp_file.close();
        std::remove(log_name.c_str());
        std::rename(tmp_name.c_str(), log_name.c_str());
        return has_log;
    }
};
