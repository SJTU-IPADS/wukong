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

inline bool is_upper(string str1, string str2) {
    return boost::to_upper_copy<std::string>(str1) == str2;
}

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

    // str2ID mapping for pattern constants (e.g., <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> 1)
    String_Server *str_server;

    // str2ID mapping for pattern variables (e.g., ?X -1)
    boost::unordered_map<string, ssid_t> pvars;

    // abbr2str mapping for prefixes (e.g., rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>)
    boost::unordered_map<string, string> prefixes;


    request_template req_template;

    // support at most one fetch-result optimization
    int corun_step;
    int fetch_step;

    void clear(void) {
        prefixes.clear();
        pvars.clear();

        req_template = request_template();
        valid = true;
        fetch_step = -1;
        corun_step = -1;
    }

    vector<string> get_tokens(istream &is) {
        vector<string> tokens;
        string t;
        while (is >> t)
            tokens.push_back(t);
        return tokens;
    }

    bool extract(vector<string> &tokens) {
        int idx = 0;

        // prefixes (e.g., PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>)
        while (tokens.size() > idx && tokens[idx] == "PREFIX") {
            if (tokens.size() < idx + 3) {
                valid = false;
                strerror = "Invalid PREFIX";
                return valid;
            }

            prefixes[tokens[idx + 1]] = tokens[idx + 2];
            idx += 3;
        }

        /// TODO: support more (extended) clauses (e.g., PROCEDURE)

        // SELECT clause
        if ((tokens.size() > idx) && (tokens[idx++] != "SELECT")) {
            valid = false;
            strerror = "Invalid keyword";
            return valid;
        }

        /// TODO: result description (e.g., ?X ?Z)
        while ((tokens.size() > idx) && (tokens[idx++] != "WHERE"));

        if (tokens[idx++] != "{") {
            valid = false;
            strerror = "Invalid bracket";
            return valid;
        }

        // triple-patterns in WHERE clause
        vector<string> patterns;
        while (tokens[idx] != "}") {
            // CORUN and FETCH are two extend keywork by Wukong to support
            // collaborative execution. Different to fork-join execution,
            // the co-run execution will not send full-history. The patterns
            // within CORUN and FETCH will be executed on remote workers separatly
            // the results will be fetched back in the end.

            // Since they are not patterns, we just record the range of patterns.
            if (tokens[idx] == "CORUN")
                corun_step = patterns.size() / 4;
            else if (tokens[idx] == "FETCH")
                fetch_step = patterns.size() / 4;
            else
                patterns.push_back(tokens[idx]);
            idx++;
        }

        // 4-element tuple for each pattern
        // e.g., ?Y rdf:type ub:University .
        if (patterns.size() % 4 != 0) {
            valid = false;
            strerror = "Invalid pattern";
            return valid;
        }

        tokens.swap(patterns);
        return true;
    }

    /* The abbreviated forms in the SPARQL syntax are resolved to produce absolute IRIs */
    void resolve(vector<string> &tokens) {
        for (int i = 0; i < tokens.size(); i++) {
            for (auto iter : prefixes) {
                if (tokens[i].find(iter.first) == 0) {
                    string s = iter.second;
                    s.insert(s.find("#") + 1,
                             tokens[i],
                             iter.first.size(),
                             string::npos);
                    tokens[i] = s;
                    break;
                } else if (tokens[i][0] == '%'
                           && tokens[i].find(iter.first) == 1) {
                    // random-constants (start with '%') with a certain type,
                    // which is extended by Wukong in batch-mode
                    // e.g., %ub:University (incl. <http://www.Department0.University0.edu>, ..)
                    string s = "%" + iter.second;
                    s.insert(s.find("#") + 1,
                             tokens[i], iter.first.size() + 1,
                             string::npos);
                    tokens[i] = s;
                    break;
                }
            }
        }
    }

    ssid_t token2id(string &token) {
        if (token[0] == '?') {  // pattern variable
            if (pvars.find(token) == pvars.end()) {
                // use negatie ID for variable
                ssid_t id = - (pvars.size() + 1); // starts from -1
                pvars[token] = id;
            }
            return pvars[token];
        } else if (token[0] == '%') {  // pattern random-constant (batch mode)
            req_template.ptypes_str.push_back(token.substr(1));
            return PTYPE_PH;
        } else {  // pattern constant
            if (!str_server->exist(token)) {
                strerror = "Unknown constant: " + token;
                valid = false;
                return DUMMY_ID;
            }
            return str_server->str2id[token];
        }
    }

    void dump_cmd_chains(void) {
        cout << "cmd_chain size: " << req_template.cmd_chains.size() << endl;
        for (int i = 0; i < req_template.cmd_chains.size(); i += 4) {
            cout << "pattern#" << i / 4 << ": "
                 << req_template.cmd_chains[i] << "\t"
                 << req_template.cmd_chains[i + 1] << "\t"
                 << req_template.cmd_chains[i + 2] << "\t"
                 << req_template.cmd_chains[i + 3] << "\t"
                 << endl;
        }
    }

    bool do_parse(vector<string> &tokens) {
        if (!extract(tokens))
            return false;

        resolve(tokens);

        // generate ID-format patterns
        for (int i = 0; (i + 3) < tokens.size(); i += 4) {
            // SPO
            string triple[3] = {tokens[i + 0], tokens[i + 1], tokens[i + 2]};

            dir_t d;
            if (tokens[i + 3] == "." || tokens[i + 3] == "->") {
                d = OUT;
            } else if (tokens[i + 3] == "<-") {
                d = IN;
                swap(triple[0], triple[2]);
            } else {
                valid = false;
                strerror = "Invalid seperator";
                return valid;
            }

            req_template.cmd_chains.push_back(token2id(triple[0]));
            req_template.cmd_chains.push_back(token2id(triple[1]));
            req_template.cmd_chains.push_back(d);
            req_template.cmd_chains.push_back(token2id(triple[2]));

            /// FIXME: support unknown predicate/attributed (token2id(triple[1]) < 0)
            int type = str_server->id2type[token2id(triple[1])];
            if (type > 0 && (!global_enable_vattr)) {
                cout << "Need to change config to enable vertex_attr " << endl;
                assert(false);
            }
            req_template.pred_type_chains.push_back(type);

            req_template.nvars = pvars.size();
        }

        // insert a new CORUN pattern
        if (fetch_step >= 0) {
            if (!global_use_rdma) {
                // TODO: corun optimization is not supported w/o RDMA
                cout << "[WARNING]: RDMA is not enabled, skip corun optimization!" << endl;
            } else {
                // TODO: support corun optimization smoothly
                vector<ssid_t> corun_pattern;
                corun_pattern.push_back((ssid_t)DUMMY_ID); // unused
                corun_pattern.push_back((ssid_t)DUMMY_ID); // unused
                corun_pattern.push_back(CORUN);
                corun_pattern.push_back(fetch_step + 1); // because we insert a new cmd in the middle

                req_template.cmd_chains.insert(req_template.cmd_chains.begin() + corun_step * 4,
                                               corun_pattern.begin(), corun_pattern.end());
                req_template.pred_type_chains.insert(req_template.pred_type_chains.begin() + corun_step, 0);
            }
        }

        // record positions of patterns with random-constants (batch mode)
        for (int i = 0; i < req_template.cmd_chains.size(); i++)
            if (req_template.cmd_chains[i] == PTYPE_PH)
                req_template.ptypes_pos.push_back(i);

        // dump_cmd_chains();
        return valid;
    }

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
            temp_cmd_chains.push_back(OUT);
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
    bool _H_do_parse(istream &is, SPARQLQuery &r) {
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
        return true;
    }

    bool _H_do_parse_template(istream &is, request_template &r) {
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

public:
    // the stat of query parsing
    bool valid;
    std::string strerror;

    Parser(String_Server *_ss): str_server(_ss) { clear(); }

    /* Used in single-mode */
    bool parse(istream &is, SPARQLQuery &r) {
        // clear intermediate states of parser
        clear();

        if (!_H_do_parse(is, r))
            return false;

        cout << "parsing triples is finished." << endl;
        return true;

        // if (global_enable_planner) {
        //     // ASSUMPTION: a normal SPARQL query

        //     if (!_H_do_parse(is, r))
        //         return false;

        //     cout << "parsing triples is finished." << endl;
        //     return true;
        // } else {
        //     // ASSUMPTION: an extended SPARQL query w/o planning
        //     // TODO: only support the clause "SELECT ... WHERE { ... }"

        //     // spilt stream into tokens
        //     vector<string> tokens = get_tokens(is);

        //     // parse the tokens
        //     if (!do_parse(tokens))
        //         return false;

        //     if (req_template.ptypes_pos.size() != 0) {
        //         cout << "ERROR: there is unsupported template pattern." << endl;
        //         return false;
        //     }

        //     r.cmd_chains = req_template.cmd_chains;
        //     r.pred_type_chains = req_template.pred_type_chains;
        //     //init the var map in the req
        //     r.nvars = req_template.nvars;
        //     return true;
        // }
    }

    /* Used in batch-mode */
    bool parse_template(istream &is, request_template &r) {
        if (global_enable_planner) {
            if (!_H_do_parse_template(is, r))
                return false;
            // cout << "parsing template is finished." << endl;
            return true;
        }

        // clear intermediate states of parser
        clear();

        vector<string> tokens = get_tokens(is);
        if (!do_parse(tokens))
            return false;

        if (req_template.ptypes_pos.size() == 0) {
            cout << "ERROR: there is no template pattern!" << endl;
            return false;
        }

        r = req_template;
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
