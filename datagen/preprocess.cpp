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

#include "log.hpp"
#include "utils.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

typedef int64_t i64;

struct Record {
    size_t file_idx;
    i64 pos;

    Record() : file_idx(0), pos(0) {}

    Record(size_t file_idx, i64 pos)
        : file_idx(file_idx), pos(pos) {}

    Record(const string &str) {
        stringstream ss(str);
        ss >> file_idx >> pos;
    }

    Record &operator= (const Record &r) {
        file_idx = r.file_idx;
        pos = r.pos;
        return *this;
    }

    string to_str() const {
        stringstream ss;
        ss << file_idx << " " << pos;
        return ss.str();
    }
};

class Encoder {
    string sdir_name;  // source directory
    string ddir_name;  // destination directory

    size_t partitions;

    unordered_set<string> created;  // file created

    int file_idx;  // the next hanlded file index after recovery
    Logger<Record> logger;

    bool recover_from_failure() {
        Record record;
        bool has_log = logger.recover_from_failure([&](Record &new_record) {
            record = new_record;
        });

        if (has_log) {
            file_idx = record.file_idx + 1;  // start from the next unprocessed file
            next_normal_id = 
                recover_table(normal_table, normal_name, record.normal_sz, next_normal_id);
            next_index_id = 
                recover_table(index_table, index_name, record.index_sz, next_index_id);
            recover_attr_table(record.type_sz);
        }
        return has_log;
    }

    void write_log() {
        Record r(file_idx, normal_table.size(), index_table.size(), type_table.size());
        logger.write_log(r);
    }

    void process_file(const string &name, i64 pos) {
        ifstream input(name.c_str());

        sid_t s, p, o;
        input.seekg(pos);
        while (input >> s >> p >> o) {
            for (size_t cnt = 0; cnt < CHUNK_SIZE; cnt++) {
                write_triple(dispatch(s), s, p, o);
                write_triple(dispatch(o), s, p, o);
            }
            write_log();
        }
    }

public:
    Encoder(string sdir_name, string ddir_name, size_t partitions)
        : sdir_name(sdir_name), ddir_name(ddir_name), partitions(partitions) 
        file_idx(0), logger(ddir_name) {

        // If failure happened before, reload log and tables and continue from failure point.
        // Otherwise check and create destination directory.
        if (!recover_from_failure()) {
            if (!FileSys::dir_exist(ddir_name)) {
                if (!FileSys::create_dir(ddir_name))
                    exit(-1);
            }
        }
    }

    void encode_data() {
        auto files = FileSys::get_files(sdir_name, [](const string &file) {
            return file.find("id_") != string::npos;
        });

        sort(files.begin(), files.end());

        while (file_idx < files.size()) {
            process_file(files[file_idx]);
            write_log();
            cout << "Process No." << file_idx << " input file: " << files[file_idx] << "." << endl;
            file_idx++;
        }

        // print info
        stringstream ss;
        ss << "#total_vertex = " << normal_table.size() + index_table.size() << endl;
        ss << "#normal_vertex = " << normal_table.size() << endl;
        ss << "#index_vertex = " << index_table.size() << endl;
        ss << "#attr_vertex = " << type_table.size() << endl;
        logger.commit(ss.str());
        cout << ss.str();
    }
};

int main(int argc, char** argv) {
    if (argc != 3 || argc != 4) {
        printf("Usage: ./Process src_dir dst_dir [partition_num (1024 as default)].\n");
        return -1;
    }
    string sdir_name = argv[1];
    string ddir_name = argv[2];
    if (sdir_name == ddir_name) {
        cout << "Dst_dir should be different from src_dir.\n";
        return -1;
    }

    size_t partitions = 1024;
    if (argc == 4)
        partitions = argv[3];

    Encoder encoder(sdir_name, ddir_name, partitions);
    encoder.encode_data();
    return 0;
}
