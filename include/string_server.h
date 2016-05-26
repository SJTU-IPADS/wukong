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

#include "global_cfg.h"


using namespace std;

class string_server {
    void load_index(string filename);

public:
    boost::unordered_map<string, int> subject_to_id;
    boost::unordered_map<string, int> predict_to_id;
    boost::unordered_map<int, string> id_to_subject;
    boost::unordered_map<int, string> id_to_predict;

    boost::unordered_map<string, int> str2id;
    boost::unordered_map<int, string> id2str;

    string_server(string dir_name);
    void load_index_predict(string filename, boost::unordered_map<string, int>& str2id,
                            boost::unordered_map<int, string>& id2str);

};
