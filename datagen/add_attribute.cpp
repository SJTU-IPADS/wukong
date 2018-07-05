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
#include <vector>
#include <algorithm>
#include <assert.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

/**
 * add str-format RDF data with attribute value (triple rows)
 *
 * A simple manual
 *  $g++ -std=c++11 add_attribute.cpp -o add_attribute
 *  $./add_attribute  lubm_raw_40 lubm_raw_40_attr
 */

using namespace std;
#define ID_PREDICATE "<http://swat.cse.lehigh.edu/onto/univ-bench.owl#id>"
#define NAME_PREDICATE "<http://swat.cse.lehigh.edu/onto/univ-bench.owl#name>"

int get_id(string str)
{
    istringstream is(str);
    stringstream ss;
    char ch;
    int id;
    while (is>>ch)
    {
        if (ch>='0'&&ch<='9')
        {
            ss << ch;
        }
    }
    ss >> id;
    return  id;
}

bool is_name(string str) {
    return str == NAME_PREDICATE;
}

string structure_id(int id)
{
    stringstream ss;
    ss << "\"" << id << "\"" << "^^<http://www.w3.org/2001/XMLSchema#int>";
    return ss.str();
}

int
main(int argc, char** argv)
{

    if (argc != 3) {
        printf("usage: ./add_attribute src_dir dst_dir\n");
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

    int count = 0;

    struct dirent *dent;
    while ((dent = readdir(sdir)) != NULL) {
        if (dent->d_name[0] == '.')
            continue;

        ifstream ifile((string(sdir_name) + "/" + string(dent->d_name)).c_str());
        ofstream ofile((string(ddir_name) +  "/" + string(dent->d_name)).c_str()) ;
        cout << "Process No." << ++count << " input file: " << dent->d_name << "." << endl;

        // str-format: subject predicate object .
        string subject, predicate, object, dot;
        // read (str-format) input file
        while (ifile >> subject >> predicate >> object >> dot) {
            ofile << subject << "\t" << predicate << "\t" << object << "\t" << dot << endl;

            // find course subject
            // add id attribute predicate
            if(is_name(predicate)){
                int id = get_id(object);
                string id_object = structure_id(id); 
                ofile << subject << "\t" << ID_PREDICATE <<"\t" << id_object << "\t" << dot <<endl;
            }
        }
    }
    closedir(sdir);
    return 0;
}

