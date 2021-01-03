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

#include <stdint.h>

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
typedef uint64_t sid_t;

struct Record {
    size_t triples;          
    int in_file;  // the next hanlded file index after recovery
    i64 in_pos;   // next position of the next handled file
    vector<i64> spo_pos;     // output file positions
    vector<i64> ops_pos;     // output file positions

    Record(partitions) : triples(0), in_file(0), in_pos(0),
        spo_pos(vector<i64>(partitions, 0))
        pos_pos(vector<i64>(partitions, 0)) {}

    Record(size_t triples, size_t in_file, i64 in_pos,
            const vector<i64> &spo_pos, const vector<i64> &ops_pos)
        : triples(triples) in_file(in_file), in_pos(in_pos),
          spo_pos(spo_pos), ops_pos(ops_pos) {}

    Record(const string &str) {
        stringstream ss(str);
        ss >> triples >> in_file >> in_pos;
        size_t sz = 0;
        ss >> sz;
        spo_pos.resize(sz);
        for (size_t i = 0; i < sz; i++)
            ss >> spo_pos[i];
        for (size_t i = 0; i < sz; i++)
            ss >> ops_pos[i];
    }

    Record &operator= (Record &r) {
        triples = r.triples;
        in_file = r.in_file;
        in_pos = r.in_pos;
        spo_pos.swap(r.spo_pos);
        ops_pos.swap(r.ops_pos);
        return *this;
    }

    string to_str() const {
        stringstream ss;
        ss << triples << " " << in_file << " " << in_pos;
        ss << " " << spo_pos.size();
        for (size_t i = 0; i < spo_pos.size(); i++)
            ss << " " << spo_pos[i];
        for (size_t i = 0; i < ops_pos.size(); i++)
            ss << " " << ops_pos[i];
        return ss.str();
    }
};

class Encoder {
    string sdir_name;  // source directory
    string ddir_name;  // destination directory

    size_t partitions;

    Record rc;
    Logger<Record> logger;

    const size_t CHUNK_SIZE = 16 * 1024;

    enum TYPE_T { SPO = 0, OPS = 1};

    bool recover_from_failure() {
        bool has_log = logger.recover_from_failure([&, this](Record &new_rc) {
            rc = new_rc;
        });

        return has_log;
    }

    void write_log() {
        logger.write_log(rc);
    }

    size_t dispatch(sid_t id) {
        return id % partitions;
    }

    void write_triple(sid_t s, sid_t p, sid_t o, TYPE_T type) {
        size_t pid;
        string prefix;
        i64 &pos = rc.spo_pos[0];

        if (type == SPO) {
            pid = dispatch(s);
            pos = rc.spo_pos[pid];
            prefix = "id_spo_";
        } else {
            pid = dispatch(o);
            pos = rc.ops_pos[pid];
            prefix = "id_ops_";
        }

        string file = ddir_name + "/" + prefix + to_string(pid);
        ofstream output(file.c_str());
        output.seekg(pos);
        output << s << " " << p << " " << o << endl;
        pos = output.tellg();
        output.close();
    }

    void process_file(const string &name) {
        ifstream input(name.c_str());
        input.seekg(rc.in_pos);

        sid_t s, p, o;
        while (input >> s >> p >> o) {
            for (size_t cnt = 0; cnt < CHUNK_SIZE; cnt++) {
                write_triple(s, p, o, SPO);
                write_triple(s, p, o, OPS);
                rc.triples++;
            }
            rc.in_pos = input.tellg();
            write_log();
        }

        input.close();
    }

public:
    Encoder(string sdir_name, string ddir_name, size_t partitions)
        : sdir_name(sdir_name), ddir_name(ddir_name),
        partitions(partitions), rc(Record(partitions)), logger(ddir_name) {

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

        while (rc.in_file < files.size()) {
            process_file(files[rc.in_file]);
            cout << "Process No." << rc.in_file << " input file: " << files[rc.in_file] << "." << endl;
            rc.in_file++;
            rc.in_pos = 0;
        }

        // print info
        stringstream ss;
        ss << "Prerocess is done. Repartition "
            << rc.triples << " triples into " << partitions << " partitions." << endl;
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

    if (partitions < 1 || partitions > 10 * 1024 * 1024) {
        cout << "Number of partitions is too small or too large." << endl;
        return -1;
    }

    Encoder encoder(sdir_name, ddir_name, partitions);
    encoder.encode_data();
    return 0;
}
