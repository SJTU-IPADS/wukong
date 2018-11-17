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
#include <boost/mpi.hpp>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "global.hpp"
#include "hdfs.hpp"
#include "type.hpp"

// utils
#include "assertion.hpp"

using namespace std;


class String_Server {
public:
    boost::unordered_map<string, sid_t> str2id;
    boost::unordered_map<sid_t, string> id2str;

    // the data type of predicate/attribute: sid=0, integer=1, float=2, double=3
    boost::unordered_map<sid_t, int32_t> pid2type;

    uint64_t next_index_id;
    uint64_t next_normal_id;

    String_Server(string dname) {
        uint64_t start = timer::get_usec();

        next_index_id = 0;
        next_normal_id = 0;

        if (boost::starts_with(dname, "hdfs:")) {
            if (!wukong::hdfs::has_hadoop()) {
                logstream(LOG_ERROR) << "attempting to load ID-mapping files from HDFS "
                                     << "but Wukong was built without HDFS."
                                     << LOG_endl;
                exit(-1);
            }
            load_from_hdfs(dname);
        } else
            load_from_posixfs(dname);

        uint64_t end = timer::get_usec();
        logstream(LOG_INFO) << "loading string server is finished ("
                            << (end - start) / 1000 << " ms)" << LOG_endl;
    }

    bool exist(sid_t sid) { return id2str.find(sid) != id2str.end(); }

    bool exist(string str) { return str2id.find(str) != str2id.end(); }

private:
    /* load ID mapping files from a shared filesystem (e.g., NFS) */
    void load_from_posixfs(string dname) {
        DIR *dir = opendir(dname.c_str());
        if (dir == NULL) {
            logstream(LOG_ERROR) << "failed to open the directory of ID-mapping files ("
                                 << dname << ")." << LOG_endl;
            exit(-1);
        }

        struct dirent *ent;
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name[0] == '.')
                continue;

            string fname(dname + ent->d_name);
            if (boost::ends_with(fname, "/str_index")
                    || boost::ends_with(fname, "/str_normal")) {
                logstream(LOG_INFO) << "loading ID-mapping file: " << fname << LOG_endl;
                ifstream file(fname.c_str());
                string str;
                sid_t id;
                while (file >> str >> id) {
                    str2id[str] = id;
                    id2str[id] = str;
                    if (boost::ends_with(fname, "/str_index"))
                        pid2type[id] = SID_t;
                }
                if (boost::ends_with(fname, "/str_index"))
                    next_index_id = ++id;
                else
                    next_normal_id = ++id;
                file.close();
            }

            // load the attribute index from the str_attr_index file
            // it contains by (string, predicate-ID, predicate-type)
            // predicate type: SID_t, INT_t, FLOAT_t, DOUBLE_t
            //
            // NOTE: the predicates/attributes in str_attr_index should be exclusive
            //       to the predicates/attributes in str_index
            if (boost::ends_with(fname, "/str_attr_index")) {
                logstream(LOG_INFO) << "loading ID-mapping (attribute) file: " << fname << LOG_endl;
                ifstream file(fname.c_str());
                string str;
                sid_t id;
                int32_t type;
                while (file >> str >> id >> type) {
                    str2id[str] = id;
                    id2str[id] = str;
                    pid2type[id] = type;

                    // FIXME: dynamic loading (next_index_id)
                    logstream(LOG_INFO) << " attribute[" << id << "] = " << type << LOG_endl;
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
            if (boost::ends_with(fname, "/str_index")
                    || boost::ends_with(fname, "/str_normal")) {
                logstream(LOG_INFO) << "loading ID-mapping file from HDFS: " << fname << LOG_endl;
                wukong::hdfs::fstream file(hdfs, fname);
                string str;
                sid_t id;
                while (file >> str >> id) {
                    str2id[str] = id;
                    id2str[id] = str;
                    if (boost::ends_with(fname, "/str_index"))
                        pid2type[id] = SID_t;
                }
                if (boost::ends_with(fname, "/str_index"))
                    next_index_id = ++id;
                else
                    next_normal_id = ++id;
                file.close();
            }

            // load the attribute index from the str_attr_index file
            // it contains by (string, predicate-ID, predicate-type)
            // predicate type: SID_t, INT_t, FLOAT_t, DOUBLE_t
            //
            // NOTE: the predicates/attributes in str_attr_index should be exclusive
            //       to the predicates/attributes in str_index
            if (boost::ends_with(fname, "/str_attr_index")) {
                logstream(LOG_INFO) << "loading ID-mapping (attribute) file: " << fname << LOG_endl;
                wukong::hdfs::fstream file(hdfs, fname);
                string str;
                sid_t id;
                int32_t type;
                while (file >> str >> id >> type) {
                    str2id[str] = id;
                    id2str[id] = str;
                    pid2type[id] = type;

                    // FIXME: dynamic loading (next_index_id)
                    logstream(LOG_INFO) << " attribute[" << id << "] = " << type << LOG_endl;
                }
                file.close();
            }
        }
    }
};
