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
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

typedef int64_t i64;
typedef uint64_t sid_t;

class triple_t {
public:
    sid_t s;
    sid_t p;
    sid_t o;

    triple_t() : s(0), p(0), o(0) {}
    triple_t(sid_t s, sid_t p, sid_t o) : s(s), p(p), o(o) {}
};

typedef unordered_map<int, vector<triple_t>> table_t;

struct Record {
    size_t triples;          
    int in_file;  // the next hanlded file index after recovery
    vector<i64> spo_pos;     // output file positions
    vector<i64> ops_pos;     // output file positions

    Record(int partitions) : triples(0), in_file(0),
        spo_pos(vector<i64>(partitions, 0)),
        ops_pos(vector<i64>(partitions, 0)) {}

    Record(size_t triples, size_t in_file,
            const vector<i64> &spo_pos, const vector<i64> &ops_pos)
        : triples(triples), in_file(in_file),
          spo_pos(spo_pos), ops_pos(ops_pos) {}

    Record(const string &str) {
        stringstream ss(str);
        ss >> triples >> in_file;
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
        spo_pos.swap(r.spo_pos);
        ops_pos.swap(r.ops_pos);
        return *this;
    }

    string to_str() const {
        stringstream ss;
        ss << triples << " " << in_file;
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

    int partitions;

    Record rc;
    Logger<Record> logger;


    table_t spo_table;
    table_t ops_table;

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

    int dispatch(sid_t id) {
        return id % partitions;
    }

    void flush_table(table_t &table, vector<i64> &pos, string prefix) {
        for (auto &it : table) {
            if (!it.second.empty()) {
                int pid = it.first;
                string file = ddir_name + "/" + prefix + to_string(pid) + ".nt";
                ofstream output(file.c_str(), std::ofstream::out | std::ofstream::app);
                // output.seekp(pos[pid]);

                cout << "before " << it.second.size() << endl;
                for (auto t : it.second) {
                    output << t.s << " " << t.p << " " << t.o << endl;
                }
                pos[pid] = output.tellp();
                output.close();

                it.second.clear();
                cout << it.second.size() << endl;
            }
        }
    }

    void insert(table_t &table, int pid, triple_t triple) {
        if (table.find(pid) == table.end()) {
            table[pid] = {triple};
        } else {
            table[pid].emplace_back(triple);
        }
    }

    void write_triple(sid_t s, sid_t p, sid_t o, TYPE_T type) {
        if (type == SPO) {
            insert(spo_table, dispatch(s), triple_t(s, p, o));
        } else {
            insert(ops_table, dispatch(o), triple_t(s, p, o));
        }
    }

    void process_file(const string &name) {
        ifstream input((sdir_name + "/" + name).c_str());

        sid_t s, p, o;
        while (input >> s >> p >> o) {
            write_triple(s, p, o, SPO);
            write_triple(s, p, o, OPS);
            rc.triples++;
        }
        flush_table(spo_table, rc.spo_pos, "id_spo_");
        flush_table(ops_table, rc.ops_pos, "id_ops_");
        write_log();

        input.close();
    }

public:
    Encoder(string sdir_name, string ddir_name, int partitions)
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
    if (argc != 3 && argc != 4) {
        printf("Usage: ./preprocess src_dir dst_dir [partition_num (1024 as default)].\n");
        return -1;
    }
    string sdir_name = argv[1];
    string ddir_name = argv[2];
    if (sdir_name == ddir_name) {
        cout << "Dst_dir should be different from src_dir.\n";
        return -1;
    }

    int partitions = 1024;
    if (argc == 4)
        partitions = atoi(argv[3]);

    if (partitions < 1 || partitions > 1024 * 1024) {
        cout << "Number of partitions is too small or too large." << endl;
        return -1;
    }

    Encoder encoder(sdir_name, ddir_name, partitions);
    encoder.encode_data();
    return 0;
}
