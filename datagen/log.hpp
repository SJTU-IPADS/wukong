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

#include "utils.hpp"

#include <stdio.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <string>

using namespace std;

template <typename Record>
class Logger {
    std::string log_name;    // log file name
    std::string commit_name; // commit log file name
public:
    Logger(const std::string &file_path) : log_name(file_path + "/log"),
                                      commit_name(file_path + "/log_commit") { }

    Logger(const std::string &file_path, string log, string commit)
        : log_name(file_path + "/" + log), commit_name(file_path + "/" + commit) { }

    void write_log(const Record &record) {
        std::ofstream file(log_name.c_str(), std::ofstream::out | std::ofstream::app);
        file << record.to_str() << " " << "commit" << std::endl;
        file.close();
    }

    void commit(const std::string &commit_info) {
        std::ofstream file(commit_name);
        file << commit_info << std::endl;
        file << "Commit." << std::endl;
        file.close();
    }

    bool recover_from_failure(std::function<void(Record &)> func) {
        bool has_log = false;

        auto read_handler = [&](const std::string &str_record) {
            // a valid record is:
            // record | "commit"
            size_t pos = str_record.find("commit");
            if (pos != std::string::npos) {
                Record record(str_record.substr(0, pos));
                func(record);

                has_log = true;
            }
        };
        auto start_delete = [](const std::string &line) {
            return line.find("commit") == std::string::npos;
        };
        FileSys::read_in_line_and_delete_last(log_name, read_handler, start_delete);

        return has_log;
    }

    bool already_commit() { 
        return (FileSys::file_exist(commit_name) && FileSys::file_contain(commit_name, "Commit."));
    }
};
