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
#include <vector>

/**
 * transfer str-format RDF data into id-format RDF data (triple rows)
 *
 * A simple manual
 *  $g++ -std=c++11 generate_data.cpp -o generate_data
 *  $./generate_data lubm_raw_40 id_lubm_40
 */

using namespace std;

/* logical we split id-mapping table to normal-vertex and index-vertex table,
   but mapping all strings into the same id space. We reserve 2^NBITS_IDX ids
   for index vertices. */
enum { NBITS_IDX = 17 };

typedef int64_t i64;
typedef unordered_map<string, int64_t> table64_t;
typedef unordered_map<string, int> table_t;

struct Record {
    size_t file_idx;
    size_t normal_sz;
    size_t index_sz;
    size_t type_sz;

    Record() : file_idx(0), normal_sz(0), index_sz(0), type_sz(0) {}

    Record(size_t f, size_t n, size_t i, size_t t)
        : file_idx(f), normal_sz(n), index_sz(i), type_sz(t) {}

    Record(const string &str) {
        stringstream ss(str);
        ss >> file_idx >> normal_sz >> index_sz >> type_sz;
    }

    Record &operator= (const Record &r) {
        file_idx = r.file_idx;
        normal_sz = r.normal_sz;
        index_sz = r.index_sz;
        type_sz = r.type_sz;
        return *this;
    }

    string to_str() const {
        stringstream ss;
        ss << file_idx << " " << normal_sz << " " << index_sz << " " << type_sz;
        return ss.str();
    }
};

class Encoder {
    string sdir_name;  // source directory
    string ddir_name;  // destination directory

    string normal_name; // normal table file
    string index_name;  // index table file
    string attr_name;   // attr type table file

    table64_t normal_table;  // normal-vertex id table (vid)
    table64_t local_normal_table; // normal table in one step.

    table64_t index_table;   // index-vertex (i.e, predicate or type) id table (p/tid)
    table64_t local_index_table;

    table_t type_table; //store the attr_index type mapping
    table_t local_type_table;

    int64_t next_index_id;
    int64_t next_normal_id;

    int file_idx;  // the next hanlded file index after recovery

    Logger<Record> logger;

    int find_type (string str) {
        if (str.find("^^xsd:int") != string::npos
                || str.find("^^<http://www.w3.org/2001/XMLSchema#int>") != string::npos)
            return 1;
        else if (str.find("^^xsd:float") != string::npos
                || str.find("^^<http://www.w3.org/2001/XMLSchema#float>") != string::npos)
            return 2;
        else if (str.find("^^xsd:double") != string::npos
                || str.find("^^<http://www.w3.org/2001/XMLSchema#double>") != string::npos)
            return 3;
        else
            return 0;
    }

    string find_value(string str) {
        size_t begin, end;
        begin = str.find('"');
        if (begin == string::npos) {
            cout << "ERROR Format " << endl;
            return "";
        }
        end = str.find('"', begin + 1);

        if (end == string::npos) {
            cout << "ERROR Format " << endl;
            return "";
        }
        return str.substr(begin + 1, end - begin - 1);
    }

    void write_table(table64_t &global, table64_t &local, string &fname) {
        ofstream file(fname.c_str(), std::ofstream::out | std::ofstream::app);
        for (auto it : local) {
            file << it.first << "\t" << it.second << endl;
            global[it.first] = it.second;
        }
        file.close();
        local.clear();
    }

    void write_attr_table() {
        ofstream f_attr(attr_name.c_str(), std::ofstream::out | std::ofstream::app);
        for (auto it : local_type_table) {
            f_attr << it.first << "\t"
                   << index_table[it.first] << "\t"
                   << it.second << endl;
            type_table[it.first] = it.second;
        }
        f_attr.close();
        local_type_table.clear();
    }

    void recover_table(const string &name, size_t size, function<void(const string &)> func) {
        size_t lines = 0;

        // Recover #size lines from files and delete the other lines in the file.
        FileSys::read_in_line_and_delete_last(name,
            [&](const string &line) {
                func(line);
                lines++;
            },
            [&](const string &line) {
                return lines >= size;
        });

        // Report error if there are no more than #size lines in the file.
        if (lines < size) {
            cout << "Log error. Please clear the destination directory and retry." << endl;
            exit(0);
        }
    }

    i64 recover_table(unordered_map<string, i64> &table, const string &name, size_t size, i64 next_id) {
        recover_table(name, size, [&](const string &line) {
            stringstream ss(line);
            string str;
            i64 id;
            ss >> str >> id;
            table[str] = id;
            next_id = max(id + 1, next_id);
            
        });
        return next_id;
    }

    void recover_attr_table(size_t size) {
        recover_table(attr_name, size, [&, this](const string &line) {
            stringstream ss(line);
            string str;
            i64 index;
            int type;
            ss >> str >> index >> type;
            this->type_table[str] = type;
        });
    }

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

    i64 insert(table64_t &global, table64_t &local, string &str, i64 &id) {
        if (global.find(str) != global.end())
            return global[str];
        if (local.find(str) != local.end())
            return local[str];

        // new string
        local[str] = id;
        return id++;
    }

    void insert_type(string &predicate, int type) {
        if (type_table.find(predicate) == type_table.end()) {
            local_type_table[predicate] = type;
        }
    }

    void process_file(string fname) {
        ifstream ifile((string(sdir_name) + "/" + fname).c_str());
        ofstream ofile((string(ddir_name) + "/id_" + fname).c_str());
        ofstream attr_file((string(ddir_name) + "/attr_" + fname).c_str());
        // prefix mapping
        unordered_map<string, string> str_to_str;

        // str-format: subject predicate object .
        string subject, predicate, object, dot;
        // read (str-format) input file
        while (ifile >> subject >> predicate >> object >> dot) {
            // handle prefix
            if (subject == "@prefix") {
                size_t sindex = predicate.find(':');
                string prekey = predicate.substr(0, sindex);
                str_to_str[prekey] = object;
                continue;
            }

            int type = 0;
            // the attr triple
            if ((type = find_type(object)) != 0) {
                i64 sid = insert(normal_table, local_normal_table, subject, next_normal_id);
                i64 pid = insert(index_table, local_index_table, predicate, next_index_id);
                insert_type(predicate, type);

                string obj = find_value(object);

                attr_file << sid << "\t"
                          << pid << "\t"
                          << type << "\t" << obj << endl;

            //the normal triple
            } else {
                // replace prefix
                if (str_to_str.size() != 0) {
                    size_t pindex;
                    if ((pindex = subject.find(':')) != string::npos) {
                        string prekey = subject.substr(0, pindex);
                        string lefts = subject.substr(pindex+1, subject.size()-pindex-1);
                        string prev = str_to_str[prekey];
                        subject = prev.substr(0, prev.size()-1) + lefts + '>';
                    }
                    if ((pindex = predicate.find(':')) != string::npos) {
                        string prekey = predicate.substr(0, pindex);
                        string lefts = predicate.substr(pindex+1, predicate.size()-pindex-1);
                        string prev = str_to_str[prekey];
                        predicate = prev.substr(0, prev.size()-1) + lefts + '>';
                    }
                    if ((pindex = object.find(':')) != string::npos) {
                        string prekey = object.substr(0, pindex);
                        string lefts = object.substr(pindex+1, object.size()-pindex-1);
                        string prev = str_to_str[prekey];
                        object = prev.substr(0, prev.size()-1) + lefts + '>';
                    }
                }

                // add a new normal vertex (i.e., vid)
                i64 sid = insert(normal_table, local_normal_table, subject, next_normal_id);
                // add a new (predicate) index vertex (i.e., pid)
                i64 pid = insert(index_table, local_index_table, predicate, next_index_id);

                bool is_type_index = (predicate == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>");
                // treat different types as individual indexes
                i64 oid;
                if (is_type_index) {
                    // add a new (type) index vertex (i.e., tidx)
                    oid = insert(index_table, local_index_table, object, next_index_id);
                } else {
                    // add a new normal vertex (i.e., vid)
                    oid = insert(normal_table, local_normal_table, object, next_normal_id);
                }

                // write (id-format) output file
                ofile << sid << "\t" << pid << "\t" << oid << endl;
            }
        }
        ifile.close();
        ofile.close();
        attr_file.close();
    }

public:
    Encoder(string sdir_name, string ddir_name) : sdir_name(sdir_name), ddir_name(ddir_name), 
        normal_name(ddir_name + "/str_normal"),
        index_name(ddir_name + "/str_index"),
        attr_name(ddir_name + "/str_attr_index"),
        file_idx(0), logger(ddir_name) {
        // reserve the first two ids for the class of index vertex (i.e, predicate and type)
        next_index_id = 2;
        // reserve 2^NBITS_IDX ids for index vertices
        next_normal_id = 1 << NBITS_IDX;
        // reserve t/pid[0] to predicate-index
        local_index_table["__PREDICATE__"] = 0;
        // reserve t/pid[1] to type-index
        local_index_table["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"] = 1;

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
            return file.find(".nt") != string::npos;
        });

        sort(files.begin(), files.end());

        while (file_idx < files.size()) {
            process_file(files[file_idx]);
            write_table(normal_table, local_normal_table, normal_name);
            write_table(index_table, local_index_table, index_name);
            write_attr_table();
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
    if (argc != 3) {
        printf("Usage: ./generate_data src_dir dst_dir.\n");
        return -1;
    }
    string sdir_name = argv[1];
    string ddir_name = argv[2];
    if (sdir_name == ddir_name) {
        cout << "Dst_dir should be different from src_dir.\n";
        return -1;
    }
    Encoder encoder(sdir_name, ddir_name);
    encoder.encode_data();
    return 0;
}
