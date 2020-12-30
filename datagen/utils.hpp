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

#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <functional>
#include <string>
#include <vector>

using namespace std;

class FileSys {
public:

    static bool dir_exist(const string &pathname) {
        struct stat info;

        if (stat(pathname.c_str(), &info) != 0)
            return false;
        else if (info.st_mode & S_IFDIR)  // S_ISDIR() doesn't exist on Windows 
            return true;
        return false;
    }

    static bool create_dir(const string &dir_name) {
        if (mkdir(dir_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) < 0) {
            cout << "Error: Creating dir (" << dir_name << ") failed." << endl;
            return false;
        }
        return true;
    }

    static vector<string> get_files(const string &dir_name, function<bool(const string &)> func) {
        DIR *dir = opendir(dir_name.c_str());
        if (!dir) {
            cout << "Error: Opening dir (" << dir_name << ") failed." << endl;
            exit(-1);
        }

        vector<string> files;
        struct dirent *dent;
        while ((dent = readdir(dir)) != NULL) {
            if (dent->d_name[0] == '.')
                continue;
            // skip some files
            string file(dent->d_name);
            if (func(file)) {
                files.emplace_back(file);
            }
        }

        return files;
    }
};
