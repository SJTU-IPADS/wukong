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

// loader
#include "base_loader.hpp"

// utils
#include "hdfs.hpp"

class HDFSLoader : public BaseLoader {
protected:
    istream *init_istream(const string &src) {
        wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
        return new wukong::hdfs::fstream(hdfs, src);
    }

    void close_istream(istream *stream) {
        static_cast<wukong::hdfs::fstream *>(stream)->close();
        delete stream;
    }

    vector<string> list_files(const string &src, string prefix) {
        if (!wukong::hdfs::has_hadoop()) {
            logstream(LOG_ERROR) << "attempting to load data files from HDFS "
                                 << "but Wukong was built without HDFS." << LOG_endl;
            exit(-1);
        }
        
        wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
        return vector<string>(hdfs.list_files(src, prefix));
    }

public:
    HDFSLoader(int sid, Mem *mem, StringServer *str_server, GStore *gstore)
        : BaseLoader(sid, mem, str_server, gstore) { }

    ~HDFSLoader() { }
};
