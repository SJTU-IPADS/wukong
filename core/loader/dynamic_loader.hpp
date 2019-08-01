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
#include <atomic>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include <tbb/concurrent_vector.h>
#include <boost/mpi.hpp>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "omp.h"

#include "global.hpp"
#include "type.hpp"
#include "rdma.hpp"

#include "store/dynamic_gstore.hpp"

// utils
#include "timer.hpp"
#include "assertion.hpp"
#include "math.hpp"

using namespace std;

class DynamicLoader {
private:
    int sid;
    StringServer *str_server;
    DynamicGStore *gstore;

    inline vector<string> list_files(string dname, string prefix) {
        if (boost::starts_with(dname, "hdfs:")) {
            if (!wukong::hdfs::has_hadoop()) {
                logstream(LOG_ERROR) << "attempting to load data files from HDFS "
                                     << "but Wukong was built without HDFS."
                                     << LOG_endl;
                exit(-1);
            }

            wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
            return vector<string>(hdfs.list_files(dname, prefix));
        } else {
            // files located on a shared filesystem (e.g., NFS)
            DIR *dir = opendir(dname.c_str());
            if (dir == NULL) {
                logstream(LOG_ERROR) << "failed to open directory (" << dname
                                     << ") at server " << sid << LOG_endl;
                exit(-1);
            }

            vector<string> files;
            struct dirent *ent;
            while ((ent = readdir(dir)) != NULL) {
                if (ent->d_name[0] == '.')
                    continue;

                string fname(dname + ent->d_name);
                // Assume the fnames (ID-format) start with the prefix.
                if (boost::starts_with(fname, dname + prefix))
                    files.push_back(fname);
            }
            return files;
        }
    }

    // FIXME: move mapping code to string_server
    boost::unordered_map<sid_t, sid_t> id2id;

    void flush_convertmap() { id2id.clear(); }

    void convert_sid(sid_t &sid) {
        if (id2id.find(sid) != id2id.end())
            sid = id2id[sid];
    }

    bool check_sid(const sid_t id) {
        if (str_server->exist(id))
            return true;

        logstream(LOG_WARNING) << "Unknown SID: " << id << LOG_endl;
        return false;
    }

    void dynamic_load_mappings(string dname) {
        unordered_set<sid_t> dynamic_loaded_preds;

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
                    if (str_server->exist(str)) {
                        id2id[id] = str_server->str2id(str);
                    } else {
                        if (boost::ends_with(fname, "/str_index")) {
                            id2id[id] = str_server->next_index_id ++;
                            // if this is a new pred, we should create a new segment for it
                            dynamic_loaded_preds.insert(id2id[id]);
                        }
                        else
                            id2id[id] = str_server->next_normal_id ++;

                        // add a new string-ID (bi-direction) pair to string server
                        str_server->add(str, id2id[id]);
                    }
                }
                file.close();
            }
        }
        // create new segments for new preds
        for (auto pid : dynamic_loaded_preds)
            gstore->create_new_seg(pid, dynamic_loaded_preds.size());
    }

public:
    DynamicLoader(int sid, StringServer *str_server, DynamicGStore *gstore)
        : sid(sid), str_server(str_server), gstore(gstore) { }

    int64_t dynamic_load_data(string dname, bool check_dup) {
        uint64_t start, end;
        // step 1: load ID-mapping files and construct id2id mapping
        dynamic_load_mappings(dname);

        // step 2: list files to load
        vector<string> dfiles(list_files(dname, "id_"));   // ID-format data files
        vector<string> afiles(list_files(dname, "attr_")); // ID-format attribute files

        if (dfiles.size() == 0 && afiles.size() == 0) {
            logstream(LOG_WARNING) << "no files found in directory (" << dname
                                   << ") at server " << sid << LOG_endl;
            return 0;
        }
        logstream(LOG_INFO) << dfiles.size() << " data files and " << afiles.size()
                            << " attribute files found in directory (" << dname
                            << ") at server " << sid << LOG_endl;

        sort(dfiles.begin(), dfiles.end());

        int num_dfiles = dfiles.size();

        // step 3: load triples into gstore
        // FIXME: dynamic loading triples doesn't update segment metadata. eg: num_keys, num_edges
        start = timer::get_usec();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < num_dfiles; i++) {
            int64_t cnt = 0;

            int64_t tid = omp_get_thread_num();
            /// FIXME: support HDFS
            ifstream file(dfiles[i]);
            sid_t s, p, o;
            while (file >> s >> p >> o) {
                convert_sid(s); convert_sid(p); convert_sid(o); //convert origin ids to new ids
                /// FIXME: just check and print warning
                check_sid(s); check_sid(p); check_sid(o);

                if (sid == wukong::math::hash_mod(s, Global::num_servers)) {
                    gstore->insert_triple_out(triple_t(s, p, o), check_dup, tid);
                    cnt ++;
                }

                if (sid == wukong::math::hash_mod(o, Global::num_servers)) {
                    gstore->insert_triple_in(triple_t(s, p, o), check_dup, tid);
                    cnt ++;
                }
            }
            file.close();

            logstream(LOG_INFO) << "load " << cnt << " triples from file " << dfiles[i]
                                << " at server " << sid << LOG_endl;
        }
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting into gstore" << LOG_endl;

        flush_convertmap(); //clean the id2id mapping

        sort(afiles.begin(), afiles.end());
        int num_afiles = afiles.size();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < num_afiles; i++) {
            int64_t cnt = 0;

            /// FIXME: support HDFS
            ifstream file(afiles[i]);
            sid_t s, a;
            attr_t v;
            int type;
            while (file >> s >> a >> type) {
                /// FIXME: just check and print warning
                check_sid(s); check_sid(a);

                switch (type) {
                case 1:
                    int i;
                    file >> i;
                    v = i;
                    break;
                case 2:
                    float f;
                    file >> f;
                    v = f;
                    break;
                case 3:
                    double d;
                    file >> d;
                    v = d;
                    break;
                default:
                    logstream(LOG_ERROR) << "Unsupported value type" << LOG_endl;
                    break;
                }

                if (sid == wukong::math::hash_mod(s, Global::num_servers)) {
                    /// Support attribute files
                    // gstore->insert_triple_attribute(triple_sav_t(s, a, v));
                    cnt ++;
                }
            }
            file.close();

            logstream(LOG_INFO) << "load " << cnt << " attributes from file " << afiles[i]
                                << " at server " << sid << LOG_endl;
        }

        gstore->sync_metadata();

        return 0;
    }
};
