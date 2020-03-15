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
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <assert.h>

#include <sys/stat.h>
#include <sys/types.h>


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

int
main(int argc, char** argv)
{
    unordered_map<string, int64_t> str2id_normal;  // normal-vertex id table (vid)
    unordered_map<string, int64_t> str2id_index;   // index-vertex (i.e, predicate or type) id table (p/tid)
    unordered_map<string, int> index_to_type; //store the attr_index type mapping

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

    // reserve t/pid[0] to predicate-index
    str2id_index["__PREDICATE__"] = 0;

    // reserve t/pid[1] to type-index
    str2id_index["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"] = 1;

    // reserve the first two ids for the class of index vertex (i.e, predicate and type)
    int64_t next_index_id = 2;
    int64_t next_normal_id = 1 << NBITS_IDX; // reserve 2^NBITS_IDX ids for index vertices
    int count = 0;

    struct dirent *dent;
    while ((dent = readdir(sdir)) != NULL) {
        if (dent->d_name[0] == '.')
            continue;

        ifstream ifile((string(sdir_name) + "/" + string(dent->d_name)).c_str());
        ofstream ofile((string(ddir_name) + "/id_" + string(dent->d_name)).c_str());
        ofstream attr_file((string(ddir_name) + "/attr_" + string(dent->d_name)).c_str());
        cout << "Process No." << ++count << " input file: " << dent->d_name << "." << endl;

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
                if (str2id_normal.find(subject) == str2id_normal.end()) {
                    str2id_normal[subject] = next_normal_id;
                    next_normal_id ++;
                }
                if (str2id_index.find(predicate) == str2id_index.end()) {
                    str2id_index[predicate] = next_index_id;
                    next_index_id ++;
                    index_to_type[predicate] = type;
                }
                string obj = find_value(object);

                attr_file << str2id_normal[subject] << "\t"
                          << str2id_index[predicate] << "\t"
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
                if (str2id_normal.find(subject) == str2id_normal.end()) {
                    str2id_normal[subject] = next_normal_id;
                    next_normal_id ++;
                    //normal_str.push_back(subject);
                }
                // add a new (predicate) index vertex (i.e., pid)
                if (str2id_index.find(predicate) == str2id_index.end()) {
                    str2id_index[predicate] = next_index_id;
                    next_index_id ++;
                    //index_str.push_back(predicate);
                }

                bool is_type_index = (predicate == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>");
                // treat different types as individual indexes
                if (is_type_index) {
                    // add a new (type) index vertex (i.e., tidx)
                    if (str2id_index.find(object) == str2id_index.end()) {
                        str2id_index[object] = next_index_id;
                        next_index_id ++;
                        //index_str.push_back(object);
                    }
                } else {
                    // add a new normal vertex (i.e., vid)
                    if (str2id_normal.find(object) == str2id_normal.end()) {
                        str2id_normal[object] = next_normal_id;
                        next_normal_id ++;
                        //normal_str.push_back(object);
                    }
                }

                // write (id-format) output file
                int64_t triple[3];
                triple[0] = str2id_normal[subject];
                triple[1] = str2id_index[predicate];
                triple[2] = (is_type_index) ? str2id_index[object] : str2id_normal[object];
                ofile << triple[0] << "\t" << triple[1] << "\t" << triple[2] << endl;
            }
        }
    }
    closedir(sdir);

    /* build ID-mapping (str2id) table file for normal vertices */
    {
        ofstream f_normal((string(ddir_name) + "/str_normal").c_str());
        for (auto it : str2id_normal) 
            f_normal << it.first << "\t" << it.second << endl;
    }

    /* build ID-mapping (str2id) table file for index vertices */
    {
        ofstream f_index((string(ddir_name) + "/str_index").c_str());
        for (auto it : str2id_index)
            f_index << it.first << "\t" << it.second << endl;
    }

    /* build ID-mapping (str2id) table file for attr vertices */
    if (!str2id_index.empty())
    {
        ofstream f_attr((string(ddir_name) + "/str_attr_index").c_str());
        for (auto it : index_to_type)
            f_attr << it.first << "\t"
                   << str2id_index[it.first] << "\t"
                   << it.second << endl;
    }

    cout << "#total_vertex = " << str2id_normal.size() + str2id_index.size() << endl;
    cout << "#normal_vertex = " << str2id_normal.size() << endl;
    cout << "#index_vertex = " << str2id_index.size() << endl;
    cout << "#attr_vertex = " << index_to_type.size() << endl;

    return 0;
}
