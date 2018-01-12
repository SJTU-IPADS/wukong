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
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <assert.h>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string.hpp>

#include "query.hpp"
#include "type.hpp"
#include "string_server.hpp"

#include "SPARQLParser.hpp"

using namespace std;

// Read a stream into a string
static string read_input(istream& in) {
    string result;
    while (true) {
        string s;
        getline(in, s);
        result += s;
        if (!in.good())
            break;
        result += '\n';
    }

    return result;
}

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
class Parser {
private:
    // place holder of pattern type (a special group of objects)
    const static ssid_t PTYPE_PH = std::numeric_limits<ssid_t>::min() + 1;
    const static ssid_t DUMMY_ID = std::numeric_limits<ssid_t>::min();
    const static ssid_t PREDICATE_ID = 0;

    // str2ID mapping for pattern constants (e.g., <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> 1)
    String_Server *str_server;

    //_H_ means helper
    boost::unordered_map<unsigned, ssid_t> _H_incVarIdMap;
    ssid_t varId = -1;

    ssid_t _H_inc_var_id(unsigned ori_id) {
        if (_H_incVarIdMap.find(ori_id) == _H_incVarIdMap.end()) {
            _H_incVarIdMap[ori_id] = varId;
            return varId--;
        } else {
            return _H_incVarIdMap[ori_id];
        }
    }

    ssid_t _H_encode(const SPARQLParser::Element& element) {//const
        switch (element.type) {
        case SPARQLParser::Element::Variable:
            return _H_inc_var_id(element.id);
        case SPARQLParser::Element::Literal:
            cout << "Not Support Literal" << endl;
            return DUMMY_ID;
        case SPARQLParser::Element::IRI:
        {
            string strIRI = "<" + element.value + ">" ;
            if (!str_server->exist(strIRI)) {
                cout << "Unknown IRI: " + strIRI << endl;
                return DUMMY_ID;
            }
            return str_server->str2id[strIRI];
        }
        case SPARQLParser::Element::Template:
            return PTYPE_PH;
        case SPARQLParser::Element::Predicate:
            return PREDICATE_ID;
        default:
            return DUMMY_ID;
        }
        return DUMMY_ID;
    }

    void _H_simplist_transfer(const SPARQLParser &parser, SPARQLQuery &r) {
        vector<ssid_t> temp_cmd_chains ;
        vector<int> temp_pred_type_chains;
        SPARQLParser::PatternGroup group = parser.getPatterns();
        for (std::vector<SPARQLParser::Pattern>::const_iterator iter = group.patterns.begin(),
                limit = group.patterns.end(); iter != limit; ++iter) {
            temp_cmd_chains.push_back(_H_encode(iter->subject));
            temp_cmd_chains.push_back(_H_encode(iter->predicate));
            temp_cmd_chains.push_back(iter->direction);
            temp_cmd_chains.push_back(_H_encode(iter->object));

            int type =  str_server->id2type[_H_encode(iter->predicate)];
            if (type > 0 && (!global_enable_vattr)) {
                cout << "Need to change config to enable vertex_attr " << endl;
                assert(false);
            }
            temp_pred_type_chains.push_back(str_server->id2type[_H_encode(iter->predicate)]);
        }
        r.cmd_chains = temp_cmd_chains;
        r.pred_type_chains = temp_pred_type_chains;
        // init the var_map
        r.nvars = parser.getVariableCount();

        if (!global_use_rdma) {
            // TODO: corun optimization is not supported w/o RDMA
            cout << "[WARNING]: RDMA is not enabled, skip corun optimization!" << endl;
        } else {
            r.corun_step = parser.getCorunStep();
            r.fetch_step = parser.getFetchStep();
        }
        
    }

    void _H_push(const SPARQLParser::Element &element, request_template &r, int pos) {
        ssid_t id = _H_encode(element);
        if (id == PTYPE_PH) {
            string strIRI = "<" + element.value + ">";
            r.ptypes_str.push_back(strIRI);
            r.ptypes_pos.push_back(pos);
        }
        r.cmd_chains.push_back(id);
    }

    void _H_template_transfer(const SPARQLParser &parser, request_template &r) {
        SPARQLParser::PatternGroup group = parser.getPatterns();
        int pos = 0;
        for (std::vector<SPARQLParser::Pattern>::const_iterator iter = group.patterns.begin(),
                limit = group.patterns.end(); iter != limit; ++iter) {
            _H_push(iter->subject, r, pos++);
            r.cmd_chains.push_back(_H_encode(iter->predicate)); pos++;
            r.cmd_chains.push_back(OUT); pos++;
            _H_push(iter->object, r, pos++);

            int type =  str_server->id2type[_H_encode(iter->predicate)];
            if (type > 0 && (!global_enable_vattr)) {
                cout << "Need to change config to enable vertex_attr " << endl;
                assert(false);
            }
            r.pred_type_chains.push_back(type);
        }

        // set the number of variables in triple patterns
        r.nvars = parser.getVariableCount();
    }

public:
    // the stat of query parsing
    std::string strerror;

    Parser(String_Server *_ss): str_server(_ss) { clear(); }

    /* Used in single-mode */
    bool parse(istream &is, SPARQLQuery &r) {
        // clear intermediate states of parser
        string query = read_input(is);
        SPARQLLexer lexer(query);
        SPARQLParser parser(lexer);
        varId = -1;
        _H_incVarIdMap.clear();
        try {
            parser.parse();//sparql -f query/lubm_q1
            _H_simplist_transfer(parser, r);
        } catch (const SPARQLParser::ParserException &e) {
            cerr << "parse error: " << e.message << endl;
            return false;
        }
        // check if using custom grammar when planner is on
        if(parser.isUsingCustomGrammar() && global_enable_planner){
            cerr << "custom grammar can only be used when planner is off! " << endl;
            return false;
        }
        cout << "parsing triples is finished." << endl;
        return true;
    }

    /* Used in batch-mode */
    bool parse_template(istream &is, request_template &r) {
        string query = read_input(is);
        SPARQLLexer lexer(query);
        SPARQLParser parser(lexer);
        varId = -1;
        _H_incVarIdMap.clear();
        try {
            parser.parse();
            _H_template_transfer(parser, r);
        } catch (const SPARQLParser::ParserException &e) {
            cerr << "parse error: " << e.message << endl;
            return false;
        }
        return true;
    }

    bool add_type_pattern(string type, SPARQLQuery &r) {
        clear();
        r = SPARQLQuery();

        // add an additonal pattern cmd to collect pattern constants with a certain type
        r.cmd_chains.push_back(str_server->str2id[type]); // type ID
        r.cmd_chains.push_back(TYPE_ID);  // reserved ID for "rdf:type"
        r.cmd_chains.push_back(IN);
        r.cmd_chains.push_back(-1);

        r.pred_type_chains.push_back(0);
        r.nvars = 1;
        return true;
    }

};
