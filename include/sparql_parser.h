#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string.hpp>

#include "query_basic_types.h"
#include "string_server.h"

using namespace std;

class sparql_parser{
    string_server* str_server;
    boost::unordered_map<string,string> prefix_map;
    boost::unordered_map<string,int> variable_map;
    vector<int> cmd_chains;
    void clear();
    bool readFile(string filename);
public:
    sparql_parser(string_server* _str_server);
    bool parse(string filename,request_or_reply& r);
};
