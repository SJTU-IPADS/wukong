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

class PosixLoader : public BaseLoader {
protected:
    istream *init_istream(const string &src) {
        return new ifstream(src.c_str());
    }

    void close_istream(istream *stream) {
        static_cast<ifstream *>(stream)->close();
        delete stream;
    }

    vector<string> list_files(const string &src, string prefix) {
        DIR *dir = opendir(src.c_str());
        if (dir == NULL) {
            logstream(LOG_ERROR) << "failed to open directory (" << src
                                 << ") at server " << sid << LOG_endl;
            exit(-1);
        }

        vector<string> files;
        struct dirent *ent;
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name[0] == '.')
                continue;

            string fname(src + ent->d_name);
            // Assume the fnames (ID-format) start with the prefix.
            if (boost::starts_with(fname, src + prefix))
                files.push_back(fname);
        }
        return files;
    }

public:
    PosixLoader(int sid, Mem *mem, StringServer *str_server, GStore *gstore)
        : BaseLoader(sid, mem, str_server, gstore) { }

    ~PosixLoader() { }
};
