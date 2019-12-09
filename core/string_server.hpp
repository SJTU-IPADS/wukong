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

// #define USE_BITRIE  // use bi-tire to store ID-STR mapping (reduce memory usage)
#ifdef USE_BITRIE
#include "bitrie.hpp"
#endif


using namespace std;

class StringServer {
private:
#ifdef USE_BITRIE
    bitrie<char, sid_t> bimap;  // ID-STRING (bi-)map
#else
    boost::unordered_map<string, sid_t> simap;  // STRING to ID
    boost::unordered_map<sid_t, string> ismap;  // ID to STRING
#endif

public:
    // the data type of predicate/attribute: sid=0, integer=1, float=2, double=3
    boost::unordered_map<sid_t, char> pid2type;

    uint64_t next_index_id;
    uint64_t next_normal_id;

    StringServer(string dname) {
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

#ifdef USE_BITRIE
    bool exist(sid_t sid) { return bimap.exist(sid); }

    bool exist(string str) { return bimap.exist(str); }

    string id2str(sid_t sid) { return bimap[sid]; }

    sid_t str2id(string str) { return bimap[str]; }

    void add(string str, sid_t sid) { bimap.insert_kv(str, sid); }

    void shrink() { bimap.storage_resize(); }
#else
    bool exist(sid_t sid) { return ismap.find(sid) != ismap.end(); }

    bool exist(string str) { return simap.find(str) != simap.end(); }

    string id2str(sid_t sid) { return ismap[sid]; }

    sid_t str2id(string str) { return simap[str]; }

    void add(string str, sid_t sid) { simap[str] = sid; ismap[sid] = str; }

    void shrink() { }
#endif


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
                    add(str, id);  // add a new ID-STRING (bi-direction) pair
                    if (boost::ends_with(fname, "/str_index"))
                        pid2type[id] = (char)SID_t;
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
                int type;
                while (file >> str >> id >> type) {
                    add(str, id);  // add a new ID-STRING (bi-direction) pair
                    pid2type[id] = (char)type;

                    // FIXME: dynamic loading (next_index_id)
                    logstream(LOG_INFO) << " attribute[" << id << "] = " << type << LOG_endl;
                }
                file.close();
            }
        }

        shrink();  // save memory
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
                    add(str, id);  // add a new ID-STRING (bi-direction) pair
                    if (boost::ends_with(fname, "/str_index"))
                        pid2type[id] = (char)SID_t;
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
                char type;
                while (file >> str >> id >> type) {
                    add(str, id); // add a new ID-STRING (bi-direction) pair
                    pid2type[id] = (char)type;

                    // FIXME: dynamic loading (next_index_id)
                    logstream(LOG_INFO) << " attribute[" << id << "] = " << type << LOG_endl;
                }
                file.close();
            }
        }

        shrink();  // save memory
    }
};
