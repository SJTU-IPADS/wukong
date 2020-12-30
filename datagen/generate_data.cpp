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

bool dir_exist(string pathname) {
    struct stat info;

    if (stat(pathname.c_str(), &info) != 0)
        return false;
    else if (info.st_mode & S_IFDIR)  // S_ISDIR() doesn't exist on Windows 
        return true;
    return false;
}

class Encoder{
    // Skip strings which indicate comments in RDF/XML files.
    const vector<string> skip_strs = 
            {"<http://www.w3.org/2002/07/owl#Ontology>",
             "<http://www.w3.org/2002/07/owl#imports>"};

    string sdir_name;  // source directory
    string ddir_name;  // destination directory

    string log_name;    // log file name
    string commit_name; // commit log file name
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

    int count;  // the next hanlded file index after recovery 

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
        ofstream f_attr(attr_name.c_str());
        for (auto it : local_type_table)
            f_attr << it.first << "\t"
                   << index_table[it.first] << "\t"
                   << it.second << endl;
        f_attr.close();
    }

    i64 recover_table(unordered_map<string, i64> &table, string name, i64 size) {
        ifstream file(name.c_str());
        i64 next_id = 0;
        for (i64 i = 0; i < size; i++) {
            if(file.eof()) {
                cout << "Log error. Please clear the destination directory and retry." << endl;
                exit(0);
            }
            string str;
            i64 id;
            file >> str >> id;
            table[str] = id;
            next_id = max(id + 1, next_id);
        }
        file.close();
        return next_id;
    }

    void recover_attr_table(int size) {
        ifstream file(attr_name.c_str());
        for (i64 i = 0; i < size; i++) {
            if(file.eof()) {
                cout << "Log error. Please clear the destination directory and retry." << endl;
                exit(0);
            }
            string str;
            i64 index;
            int type;
            file >> str >> index >> type;
            type_table[str] = type;
        }
        file.close();
    }

    bool recover_from_failure() {
        ifstream log_file(log_name.c_str());
        // This is a new encoder.
        if (!log_file.good())
            return false;
        // failure happened before
        if (log_file.good()) {
            int cnt = -1;
            i64 normal_sz = 0, index_sz = 0, type_sz = 0;
            // reload all tables valid size
            while (!log_file.eof()) {
                string record;
                getline(log_file, record);
                // a valid record is:
                // count | normal_table table size | index_table table size | type_table table size | "commit"
                if (record.find("commit") != string::npos) {
                    stringstream ss(record);
                    ss >> cnt >> normal_sz >> index_sz >> type_sz;
                } else {
                    break;
                }
            }
            count = cnt + 1; // start from the next invalid file index

            next_normal_id = recover_table(normal_table, normal_name, normal_sz);
            next_index_id = recover_table(index_table, index_name, index_sz);
            recover_attr_table(type_sz);
        }
        log_file.close();
        return true;
    }

    void write_log(int file_idx) {
        ofstream log_file(log_name.c_str(), std::ofstream::out | std::ofstream::app);
        log_file << file_idx << " " << normal_table.size()
                             << " " << index_table.size()
                             << " " << type_table.size()
                             << " " << "commit" << endl;
        log_file.close();
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

    bool skip_triple(string &sub, string &pre, string &obj) {
        for (auto s : skip_strs) {
            if (sub == s || pre == s || obj == s)
                return true;
        }
        return false;
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
            // Original generate_data code does not skip these triples, so add comment here.
            //if (skip_triple(subject, predicate, object))
            //    continue;

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

                // Original generate_data code does not skip these triples, so add comment here.
                // if (skip_triple(subject, predicate, object))
                //    continue;

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
    Encoder(string sdir_name, string ddir_name) : sdir_name(sdir_name), ddir_name(ddir_name), count(0) {
        log_name    = ddir_name + "/log";
        commit_name    = ddir_name + "/log_commit";
        normal_name = ddir_name + "/str_normal";
        index_name  = ddir_name + "/str_index";
        attr_name   = ddir_name + "/str_attr_index";
    }

    // If failure happened before, reload log and tables and continue from failure point.
    // Otherwise init a new encoder.
    void init() {
        // reserve the first two ids for the class of index vertex (i.e, predicate and type)
        next_index_id = 2;
        next_normal_id = 1 << NBITS_IDX; // reserve 2^NBITS_IDX ids for index vertices
        if (!recover_from_failure()) {
            if (!dir_exist(ddir_name)) {
                // create destination directory
                if (mkdir(ddir_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) < 0) {
                    cout << "Error: Creating dst_dir (" << ddir_name << ") failed." << endl;
                    exit(-1);
                }
            }

            // reserve t/pid[0] to predicate-index
            local_index_table["__PREDICATE__"] = 0;
            // reserve t/pid[1] to type-index
            local_index_table["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"] = 1;
        }
    }

    void encode_data() {
        // open source directory
        DIR *sdir = opendir(sdir_name.c_str());
        if (!sdir) {
            cout << "Error: Opening src_dir (" << sdir_name << ") failed." << endl;
            exit(-1);
        }

        struct dirent *dent;
        int file_idx = 0;
        while ((dent = readdir(sdir)) != NULL) {
            if (dent->d_name[0] == '.')
                continue;
            // skip some files such as log files
            if (string(dent->d_name).find(".nt") == string::npos)
                continue;

            if (file_idx >= count) {
                process_file(string(dent->d_name));
                write_table(normal_table, local_normal_table, normal_name);
                write_table(index_table, local_index_table, index_name);
                write_attr_table();
                write_log(file_idx);
                cout << "Process No." << file_idx << " input file: " << dent->d_name << "." << endl;
            }
            file_idx++;
        }
        closedir(sdir);
    }

    void print() {
        cout << "#total_vertex = " << normal_table.size() + index_table.size() << endl;
        cout << "#normal_vertex = " << normal_table.size() << endl;
        cout << "#index_vertex = " << index_table.size() << endl;
        cout << "#attr_vertex = " << type_table.size() << endl;

        ofstream output(commit_name);
        output << "Encoding nt_triple format file to id format completes." << endl;
        output << "#total_vertex = " << normal_table.size() + index_table.size() << endl;
        output << "#normal_vertex = " << normal_table.size() << endl;
        output << "#index_vertex = " << index_table.size() << endl;
        output << "#attr_vertex = " << type_table.size() << endl;
        output.close();
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("usage: ./generate_data src_dir dst_dir\n");
        return -1;
    }

    string sdir_name = argv[1];
    string ddir_name = argv[2];

    Encoder encoder(sdir_name, ddir_name);
    encoder.init();
    encoder.encode_data();
    encoder.print();

    return 0;
}
