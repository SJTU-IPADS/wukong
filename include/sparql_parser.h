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

    const static int place_holder=INT_MIN;

    request_template req_template;
    void clear();
    bool readFile(string filename);
public:
    boost::unordered_map<string,vector<int>* > type_to_idvec; // translate %type to a vector

    bool find_type_of(string type,request_or_reply& r);

    sparql_parser(string_server* _str_server);
    bool parse(string filename,request_or_reply& r);
    bool parse_template(string filename,request_template& r);
};
