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
 *      http://ipads.se.sjtu.edu.cn/projects/wukong.html
 *
 */

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
 * Q := SELECT RD WHERE GP
 *
 * The types of tokens (supported)
 * 0. SPARQL's Prefix e.g., PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
 * 1. SPARQL's Keyword (incl. SELECT, WHERE)
 *
 * 2. pattern's constant e.g., <http://www.Department0.University0.edu>
 * 3. pattern's variable e.g., ?X
 * 4. pattern's random-constant e.g., %ub:GraduateCourse (extended by Wukong in batch-mode)
 *
 */
class sparql_parser {
private:
    const static int64_t PTYPE_PH = (INT64_MIN + 1); // place holder of pattern type (a special group of objects)
    const static int64_t DUMMY_ID = (INT64_MIN);

    // str2ID mapping for pattern constants (e.g., <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> 1)
    string_server *str_server;

    // str2ID mapping for pattern variables (e.g., ?X -1)
    boost::unordered_map<string, int64_t> pvars;

    // abbr2str mapping for prefixes (e.g., rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>)
    boost::unordered_map<string, string> prefixes;


    request_template req_template;

    // support at most one fetch-result optimization
    int corun_step;
    int fetch_step;

    vector<string> get_tokens(istream &is);

    bool extract(vector<string> &tokens);

    void resolve(vector<string> &tokens);

    int64_t token2id(string &token);

    bool do_parse(vector<string> &tokens);

    void dump_cmd_chains(void);

    void clear(void);

public:
    // the stat of query parsing
    bool valid;
    std::string strerror;

    sparql_parser(string_server *_str_server);

    bool parse(istream &is, request_or_reply &r);

    bool parse_template(istream &is, request_template &r);

    //boost::unordered_map<string, vector<int64_t> *> type2grp; // mapping table from %type to a group of IDs
    bool add_type_pattern(string type, request_or_reply &r);
};
