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
class Encoder {
    typedef uint64_t sid_t;
    struct triple_t {
        sid_t s;
        sid_t p;
        sid_t o;

        triple_t() : s(0), p(0), o(0) {}
        triple_t(sid_t s, sid_t p, sid_t o) : s(s), p(p), o(o) {}
    };
    typedef unordered_map<int, vector<triple_t>> table_t;

    struct Record {
        int in_file;                // the next hanlded file index after recovery
        vector<size_t> spo_num;      // processed spo triples
        vector<size_t> ops_num;      // processed ops triples

        Record(int partitions) : in_file(0), 
            spo_num(vector<size_t>(partitions, 0)), ops_num(vector<size_t>(partitions, 0)) {}

        Record(int in_file, const vector<size_t> &spo_num, const vector<size_t> &ops_num)
            : in_file(in_file), spo_num(spo_num), ops_num(ops_num) {}

        Record(const string &str) {
            stringstream ss(str);
            size_t sz = 0;
            ss >> in_file >> sz;

            spo_num.resize(sz);
            ops_num.resize(sz);
            for (auto &num : spo_num)
                ss >> num;
            for (auto &num : ops_num)
                ss >> num;
        }

        Record &operator= (Record &r) {
            in_file = r.in_file;
            spo_num.swap(r.spo_num);
            ops_num.swap(r.ops_num);
            return *this;
        }

        string to_str() const {
            stringstream ss;
            ss << in_file << " " << spo_num.size();
            for (auto num : spo_num)
                ss << " " << num;
            for (auto num : ops_num)
                ss << " " << num;
            return ss.str();
        }
    };

    string sdir_name;  // source directory
    string ddir_name;  // destination directory
    int partitions;

    Record rc;
    Logger<Record> logger;

    table_t spo_table;
    table_t ops_table;

    const size_t MAX_SIZE = 64 * 1024;

    enum type_t { SPO = 0, OPS = 1};

    string prefix(type_t type) { return type == SPO ? "id_spo_" : "id_ops_"; }

    vector<size_t> &triple_num(type_t type) { return type == SPO ? rc.spo_num : rc.ops_num; }

    int pid(type_t type, triple_t t) { return type == SPO ? t.s % partitions : t.o % partitions; }

    table_t &table(type_t type) { return type == SPO ? spo_table : ops_table; }

    void flush_table(type_t type) {
        auto &table_ = table(type);

        for (auto it : table_) {
            if (!it.second.empty()) {
                flush(table_, it.first, prefix(type), triple_num(type));
            }
        }
    }

    void flush(table_t &table, int pid, string prefix, vector<size_t> &nums) {
        string file = ddir_name + "/" + prefix + to_string(pid) + ".nt";
        ofstream output(file.c_str(), std::ofstream::out | std::ofstream::app);
        auto &triples = table[pid];
        for (auto t : triples) {
            output << t.s << " " << t.p << " " << t.o << endl;
        }
        output.close();
        nums[pid] += triples.size();
        triples.clear();
    }

    void write_triple(triple_t triple, type_t type) {
        auto &table_ = table(type);
        int pid_ = pid(type, triple);

        auto it = table_.find(pid_);
        if (it != table_.end()) {
            it->second.emplace_back(triple);
            if (it->second.size() > MAX_SIZE)
                flush(table_, pid_, prefix(type), triple_num(type));
        } else {
            table_[pid_] = {triple};
        }
    }

    void process_file(const string &name) {
        ifstream input((sdir_name + "/" + name).c_str());
        sid_t s, p, o;
        while (input >> s >> p >> o) {
            write_triple(triple_t(s, p, o), SPO);
            write_triple(triple_t(s, p, o), OPS);
        }
        flush_table(SPO);
        flush_table(OPS);
        input.close();
    }

    void recover_table(int pid, type_t type) {
        size_t num = (triple_num(type))[pid];
        size_t lines = 0;
        string fname = ddir_name + "/" + prefix(type) + to_string(pid) + ".nt";
        FileSys::read_in_line_and_delete_last(fname,
                [&] (const string &line) { lines++; },
                [&] (const string &line) { return lines >= num; }
        );
        // Report error if there are no more than #num lines in the file.
        if (lines < num) {
            cout << "Log error. Please clear the destination directory and retry." << endl;
            exit(0);
        }
    }

    bool recover_from_failure() {
        bool has_log = logger.recover_from_failure([&, this](Record &new_rc) {
            rc = new_rc;
        });
        for (int i = 0; i < partitions; i++) {
            recover_table(i, SPO);
            recover_table(i, OPS);
        }
        return has_log;
    }

public:
    Encoder(string sdir_name, string ddir_name, int partitions)
        : sdir_name(sdir_name), ddir_name(ddir_name), partitions(partitions),
        rc(Record(partitions)), logger(ddir_name, "log_encode", "log_commit_encode") {

        if (!logger.already_commit()) {
            bool has_log = recover_from_failure();
            if (!has_log) {
                if (!FileSys::dir_exist(ddir_name)) {
                    if (!FileSys::create_dir(ddir_name))
                    exit(-1);
                }
            }
        }
    }

    void encode_data() {
        if (!logger.already_commit()) {
            auto files = FileSys::get_files(sdir_name, [](const string &file) {
                return file.find("id_") != string::npos;
            });
            sort(files.begin(), files.end());
            while (rc.in_file < files.size()) {
                process_file(files[rc.in_file]);
                cout << "Process No." << rc.in_file << " input file: " << files[rc.in_file] << "." << endl;
                logger.write_log(rc);
                rc.in_file++;
            }
            string info = "Repartition " + to_string(rc.in_file) + " files into " + to_string(partitions) + " partitions.";
            logger.commit(info);
            cout << info << endl;
        }

        size_t triples = 0;
        for (auto num : rc.spo_num)
            triples += num;
        ofstream info_file(ddir_name + "/" + "metadata");
        info_file << partitions << " " << triples << " #partition_num | triple_num" << endl;
        info_file.close();

        cout << "Preprocess is done. " << endl;
    }
};

class Copyer {
    struct Record {
        size_t file_idx;    // next file to copy
        Record() : file_idx(0) {}
        Record(size_t file_idx) : file_idx(file_idx) {}
        Record(const string &str) : file_idx(stoi(str)) {}
        Record &operator= (Record &r) {
            file_idx = r.file_idx;
            return *this;
        }
        string to_str() const { return to_string(file_idx); }
    };

    string sdir_name;  // source directory
    string ddir_name;  // destination directory
    Logger<Record> logger;
    Record rc;

public:
    Copyer(string sdir_name, string ddir_name) : sdir_name(sdir_name), ddir_name(ddir_name), logger(ddir_name) {
        if (!logger.already_commit()) {
            logger.recover_from_failure([&, this](Record &new_rc) {
                rc = new_rc;
            });
        }
    }

    void copy_other_files() {
        if (!logger.already_commit()) {
            auto files = FileSys::get_files(sdir_name, [](const string &file) {
                return file.find("id_") == string::npos;
            });
            sort(files.begin(), files.end());
            while (rc.file_idx < files.size()) {
                auto &file = files[rc.file_idx];
                FileSys::copy_file(sdir_name + "/" + file, ddir_name + "/" + file);
                logger.write_log(rc);
                rc.file_idx++;
            }
            logger.commit("Copy other files is done.");
        }
        cout << "Copy other files is done." << endl;
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

    Copyer copyer(sdir_name, ddir_name);
    copyer.copy_other_files();
    return 0;
}
