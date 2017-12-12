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

#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <algorithm>
#include <assert.h>
#include <boost/mpi.hpp>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "config.hpp"
#include "hdfs.hpp"

using namespace std;


class String_Server {
public:
    boost::unordered_map<string, int64_t> str2id;
    boost::unordered_map<int64_t, string> id2str;
    // predicate type, 0 is sid, 1 is int, 2 is float, 3 is double
    boost::unordered_map<int64_t, int32_t> pred_type;

    
    String_Server(string dname) {
        if (boost::starts_with(dname, "hdfs:")) {
            if (!wukong::hdfs::has_hadoop()) {
                cout << "ERORR: attempting to load ID-mapping files from HDFS "
                     << "but Wukong was built without HDFS."
                     << endl;
                exit(-1);
            }
            load_from_hdfs(dname);
        } else
            load_from_posixfs(dname);

        cout << "loading String Server is finished." << endl;
    }

private:
    /* load ID mapping files from a shared filesystem (e.g., NFS) */
    void load_from_posixfs(string dname) {
        DIR *dir = opendir(dname.c_str());
        if (dir == NULL) {
            cout << "ERROR: failed to open the directory of ID-mapping files ("
                 << dname << ")." << endl;
            exit(-1);
        }

        struct dirent *ent;
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name[0] == '.')
                continue;

            string fname(dname + ent->d_name);
            if ((boost::ends_with(fname, "/str_index"))
                    || (boost::ends_with(fname, "/str_normal") && !global_load_minimal_index)
                    || (boost::ends_with(fname, "/str_normal_minimal") && global_load_minimal_index)) {
                cout << "loading ID-mapping file: " << fname << endl;

                ifstream file(fname.c_str());
                string str;
                int64_t id;
                while (file >> str >> id) {
                    // both string and ID are unique
                    assert(str2id.find(str) == str2id.end());
                    assert(id2str.find(id) == id2str.end());

                    str2id[str] = id;
                    id2str[id] = str;
                    if (boost::ends_with(fname, "/str_index")) {
                        pred_type[id] = 0;
                    }
                }
                file.close();
            }

            if  (boost::ends_with(fname,"/str_attr_index")) {
                cout << "loading attr-ID-mapping file: " << fname << endl;
                ifstream file(fname.c_str());
                string str;
                int64_t id;
                int32_t type;
                while (file >> str >> id >> type) {
                    // both string and ID are unique
                    assert(str2id.find(str) == str2id.end());
                    assert(id2str.find(id) == id2str.end());

                    str2id[str] = id;
                    id2str[id] = str;
                    cout << " add attr_index " << id<<endl;
                    pred_type[id] = type;
                }
                file.close();
            }
        }
    }

    /* load ID mapping files from HDFS */
    void load_from_hdfs(string dname) {
        wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
        vector<string> files = hdfs.list_files(dname);

        for (int i = 0; i < files.size(); i++) {
            string fname = files[i];
            // NOTE: users may use a short path (w/o ip:port)
            // e.g., hdfs:/xxx/xxx/
            if ((boost::ends_with(fname, "/str_index"))
                    || (boost::ends_with(fname, "/str_normal") && !global_load_minimal_index)
                    || (boost::ends_with(fname, "/str_normal_minimal") && global_load_minimal_index)) {
                cout << "loading ID-mapping file from HDFS: " << fname << endl;

                wukong::hdfs::fstream file(hdfs, fname);
                string str;
                int64_t id;
                while (file >> str >> id) {
                    // both string and ID are unique
                    assert(str2id.find(str) == str2id.end());
                    assert(id2str.find(id) == id2str.end());

                    str2id[str] = id;
                    id2str[id] = str;
                    
                    if (boost::ends_with(fname, "/str_index")) {
                        pred_type[id] = 0;
                    }
                }
            }
            
            if  (boost::ends_with(fname,"/str_attr_index")) {
                cout << "loading attr-ID-mapping file: " << fname << endl;
                
                wukong::hdfs::fstream file(hdfs, fname);
                string str;
                int64_t id;
                int32_t type;
                while (file >> str >> id >> type) {
                    // both string and ID are unique
                    assert(str2id.find(str) == str2id.end());
                    assert(id2str.find(id) == id2str.end());
                    
                    str2id[str] = id;
                    id2str[id] = str;

                    pred_type[id] = type;
                }
                file.close();
            }

        }
    }

};
