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

#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <unordered_map>
#include <tbb/concurrent_unordered_map.h>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <stdint.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <pthread.h>


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

typedef uint64_t u64;
typedef tbb::concurrent_unordered_map<string, uint64_t> cmap_64;
typedef tbb::concurrent_unordered_map<string, int> cmap;

class Transfer {
    int num_threads;
    int tid;
    char *sdir_name;
    char *ddir_name;
    cmap_64 &normal_table;  // normal-vertex id table(vid)
    cmap_64 &index_table;   // index-vertex (i.e, predicate or type) id table (p/tid)
    cmap &attr_type;     // store the attr_index type mapping

    u64 *index_id;
    u64 *normal_id;

    u64 fetch_and_add(volatile u64 *ptr, u64 val) {
        return __sync_fetch_and_add(ptr, val);
    }

    void handle_attr_triple(string &subject, string &predicate, string &object, int type, ofstream &attr_file) {
        if (normal_table.find(subject) == normal_table.end()) {
            u64 id = fetch_and_add(normal_id, 1);
            normal_table[subject] = id;
        }
        if (index_table.find(predicate) == index_table.end()) {
            u64 id = fetch_and_add(index_id, 1);
            index_table[predicate] = id;
            attr_type[predicate] = type;
        }
        string obj = find_value(object);

        attr_file << normal_table[subject] << "\t"
            << index_table[predicate] << "\t"
            << type << "\t" << obj << endl;

    }

    void handle_normal_triple(string &subject, string &predicate, string &object,
            unordered_map<string, string> &str_to_str, ofstream &ofile) {
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
        if (normal_table.find(subject) == normal_table.end()) {
            u64 id = fetch_and_add(normal_id, 1);
            normal_table[subject] = id;
        }
        // add a new (predicate) index vertex (i.e., pid)
        if (index_table.find(predicate) == index_table.end()) {
            u64 id = fetch_and_add(index_id, 1);
            index_table[predicate] = id;
            //index_str.push_back(predicate);
        }

        bool is_type_index = (predicate == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>");
        // treat different types as individual indexes
        if (is_type_index) {
            // add a new (type) index vertex (i.e., tidx)
            if (index_table.find(object) == index_table.end()) {
                u64 id = fetch_and_add(index_id, 1);
                index_table[object] = id;
                //index_str.push_back(object);
            }
        } else {
            // add a new normal vertex (i.e., vid)
            if (normal_table.find(object) == normal_table.end()) {
                u64 id = fetch_and_add(normal_id, 1);
                normal_table[object] = id;
                //normal_str.push_back(object);
            }
        }

        // write (id-format) output file
        int64_t triple[3];
        triple[0] = normal_table[subject];
        triple[1] = index_table[predicate];
        triple[2] = (is_type_index) ? index_table[object] : normal_table[object];
        ofile << triple[0] << "\t" << triple[1] << "\t" << triple[2] << endl;
    }

    void transfer_file(string fname) {
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
                handle_attr_triple(subject, predicate, object, type, attr_file);
            } else {
                handle_normal_triple(subject, predicate, object, str_to_str, ofile);
            }
        }
        ifile.close();
        ofile.close();
        attr_file.close();
    }

public:
    Transfer(int tid, int num_threads, char *sdir_name, char *ddir_name,
            cmap_64 &normal_table, cmap_64 &index_table, cmap &attr_type,
            u64 *index_id, u64 *normal_id)
            : tid(tid), num_threads(num_threads), sdir_name(sdir_name), ddir_name(ddir_name),
            normal_table(normal_table), index_table(index_table), attr_type(attr_type),
            index_id(index_id), normal_id(normal_id) { }
    
    void run() {
        int count = 0;
        DIR *sdir = opendir(sdir_name);
        struct dirent *dent;
        while ((dent = readdir(sdir)) != NULL) {
            if (dent->d_name[0] == '.')
                continue;

            if (count % num_threads == tid) {
                transfer_file(string(dent->d_name));
                cout << "#" << tid << " process No." << count << " input file: " << dent->d_name << "." << endl;
            }
            count++;
        }
        closedir(sdir);
    }
};

void *transfer_thread(void *arg) {
    Transfer *t = (Transfer *)arg;
    t->run();
} 

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("usage: ./generate_data src_dir dst_dir\n");
        return -1;
    }

    char *sdir_name = argv[1];
    char *ddir_name = argv[2];

    // create destination directory
    if (mkdir(ddir_name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) < 0) {
        cout << "Error: Creating dst_dir (" << ddir_name << ") failed." << endl;
        exit(-1);
    }

    // open source directory
    DIR *sdir = opendir(sdir_name);
    if (!sdir) {
        cout << "Error: Opening src_dir (" << sdir_name << ") failed." << endl;
        exit(-1);
    }
    closedir(sdir);

    cmap_64 normal_table;  // normal-vertex id table(vid)
    cmap_64 index_table;   // index-vertex (i.e, predicate or type) id table (p/tid)
    cmap attr_type;     // store the attr_index type mapping
    
    // reserve t/pid[0] to predicate-index
    index_table["__PREDICATE__"] = 0;

    // reserve t/pid[1] to type-index
    index_table["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"] = 1;

    // reserve the first two ids for the class of index vertex (i.e, predicate and type)
    u64 index_id = 2;
    u64 normal_id = 1 << NBITS_IDX; // reserve 2^NBITS_IDX ids for index vertices

    int num_threads = 1;
    pthread_t *threads = new pthread_t[num_threads];
    for (int tid = 0; tid < num_threads; tid++) {
        Transfer *trans = new Transfer(tid, num_threads, sdir_name, ddir_name,
                normal_table, index_table, attr_type, &index_id, &normal_id);
        pthread_create(&(threads[tid]), NULL, transfer_thread, trans);
    }

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

    /* build ID-mapping (str2id) table file for normal vertices */
    {
        ofstream f_normal((string(ddir_name) + "/str_normal").c_str());
        for (auto it = normal_table.begin(); it != normal_table.end(); ++it) 
            f_normal << it->first << "\t" << it->second << endl;
    }

    /* build ID-mapping (str2id) table file for index vertices */
    {
        ofstream f_index((string(ddir_name) + "/str_index").c_str());
        for (auto it = index_table.begin(); it != index_table.end(); ++it)
            f_index << it->first << "\t" << it->second << endl;
    }

    /* build ID-mapping (str2id) table file for attr vertices */
    if (!index_table.empty())
    {
        ofstream f_attr((string(ddir_name) + "/str_attr_index").c_str());
        for (auto it = attr_type.begin(); it != attr_type.end(); ++it)
            f_attr << it->first << "\t"
                   << index_table[it->first] << "\t"
                   << it->second << endl;
    }

    cout << "#total_vertex = " << normal_table.size() + index_table.size() << endl;
    cout << "#normal_vertex = " << normal_table.size() << endl;
    cout << "#index_vertex = " << index_table.size() << endl;
    cout << "#attr_vertex = " << attr_type.size() << endl;

    return 0;
}
