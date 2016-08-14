#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <assert.h>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string.hpp>

#include "query_basic_types.h"
#include "string_server.h"


using namespace std;

/**
 * Three types of tokens
 * 1. pattern constant e.g., <http://www.Department0.University0.edu>
 * 2. pattern variable e.g., ?X
 * 3. pattern group e.g., %ub:GraduateCourse
 */
class sparql_parser {
    string_server *str_server;

    boost::unordered_map<string, string> prefix_map;
    boost::unordered_map<string, int64_t> pvars;

    const static int64_t place_holder = (INT64_MIN + 1); // pattern group (a group of constants with a certain type)
    const static int64_t INVALID_ID = (INT64_MIN);

    request_template req_template;
    bool valid;

    int fork_step;
    int join_step;

    vector<string> get_tokens(string fname);
    bool extract_patterns(vector<string> &tokens);
    void replace_prefix(vector<string> &tokens);

    int64_t token2id(string &token);

    bool do_parse(vector<string> &tokens);

    void dump_cmd_chains(void);
    void clear(void);

public:
    boost::unordered_map<string, vector<int64_t> *> type_to_idvec; // translate %type to a vector

    sparql_parser(string_server *_str_server);

    bool parse(string fname, request_or_reply &r);
    bool parse_string(string input_str, request_or_reply &r);
    bool parse_template(string fname, request_template &r);

    bool find_type_of(string type, request_or_reply &r);
};
