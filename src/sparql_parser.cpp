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

#include "sparql_parser.h"

inline static bool is_upper(string str1, string str2) {
    return boost::to_upper_copy<std::string>(str1) == str2;
}

sparql_parser::sparql_parser(string_server *_str_server)
    : str_server(_str_server) {
    valid = true;
}


void
sparql_parser::clear(void)
{
    prefixes.clear();
    pvars.clear();

    req_template = request_template();
    valid = true;
    join_step = -1;
    fork_step = -1;
};

vector<string>
sparql_parser::get_tokens(istream &is)
{
    vector<string> tokens;
    string t;

    while (is >> t)
        tokens.push_back(t);
    return tokens;
}

bool
sparql_parser::extract_patterns(vector<string> &tokens)
{
    int idx = 0;

    // extract prefixes (e.g., PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>)
    while (tokens.size() > idx && tokens[idx] == "PREFIX") {
        if (tokens.size() < idx + 3) {
            valid = false;
            strerror = "Invalid PREFIX";
            return valid;
        }

        prefixes[tokens[idx + 1]] = tokens[idx + 2];
        idx += 3;
    }

    /// TODO: support more keywords (e.g., PROCEDURE)
    if ((tokens.size() > idx) && (tokens[idx++] != "SELECT")) {
        valid = false;
        strerror = "Invalid keyword";
        return valid;
    }

    /// TODO: support result description (e.g., ?X ?Z)
    while ((tokens.size() > idx) && (tokens[idx++] != "WHERE"));

    if (tokens[idx++] != "{") {
        valid = false;
        strerror = "Invalid bracket";
        return valid;
    }

    vector<string> patterns;
    while (tokens[idx] != "}") {
        if (tokens[idx] == "join")
            join_step = patterns.size() / 4;
        else if (tokens[idx] == "fork")
            fork_step = patterns.size() / 4;
        else
            patterns.push_back(tokens[idx]);
        idx++;
    }

    tokens.swap(patterns);
    return true;
}

void
sparql_parser::replace_prefix(vector<string> &tokens)
{
    for (int i = 0; i < tokens.size(); i++) {
        for (auto iter : prefixes) {
            if (tokens[i].find(iter.first) == 0) {
                string s = iter.second;
                s.insert(s.find("#") + 1, tokens[i], iter.first.size(), string::npos);
                tokens[i] = s;
                break;
            } else if (tokens[i][0] == '%' && tokens[i].find(iter.first) == 1 ) {
                // patent constant with the certain type (batch mode)
                string s = "%" + iter.second;
                s.insert(s.find("#") + 1, tokens[i], iter.first.size() + 1, string::npos);
                tokens[i] = s;
                break;
            }
        }
    }
}

int64_t
sparql_parser::token2id(string &token)
{
    if (token == "") {
        cout << "ERROR: empty string." << endl;
        return INVALID_ID;
    }
    if (token[0] == '?') {  // pattern variable (single mode)
        if (pvars.find(token) == pvars.end()) {
            int64_t id = (- pvars.size()) - 1; // ID starts from -1
            pvars[token] = id;
        }
        return pvars[token];
    } else if (token[0] == '%') {  // patent group (batch mode)
        req_template.ptypes_str.push_back(token.substr(1));
        return PTYPE_PH;
    } else {  // pattern constant (single mode)
        if (str_server->str2id.find(token) == str_server->str2id.end()) {
            cout << "ERROR: unknown token \"" << token << "\"" << endl;
            return INVALID_ID;
        }
        return str_server->str2id[token];
    }
}

void
sparql_parser::dump_cmd_chains(void)
{
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

bool
sparql_parser::do_parse(vector<string> &tokens)
{
    if (!valid) return false;

    if (!extract_patterns(tokens))
        return false;

    replace_prefix(tokens);

    // Wukong uses an internal 4-element format (SPDO) for each pattern
    if (tokens.size() % 4 != 0) {
        cout << "ERROR: invalid token number (" << tokens.size() << ")" << endl;
        return false;
    }

    for (int i = 0; i < tokens.size(); i += 4) {
        // SPO
        string triple[3] = {tokens[i + 0], tokens[i + 1], tokens[i + 2]};

        direction d;
        if (tokens[i + 3] == "." || tokens[i + 3] == "->") {
            d = OUT;
        } else if (tokens[i + 3] == "<-") {
            d = IN;
            swap(triple[0], triple[2]);
        } else {
            cout << "ERROR: invalid seperator (" << tokens[i + 3] << ")" << endl;
            return false;
        }

        req_template.cmd_chains.push_back(token2id(triple[0]));
        req_template.cmd_chains.push_back(token2id(triple[1]));
        req_template.cmd_chains.push_back(d);
        req_template.cmd_chains.push_back(token2id(triple[2]));
    }

    // record positions of pattern group (batch mode)
    for (int i = 0; i < req_template.cmd_chains.size(); i++)
        if (req_template.cmd_chains[i] == PTYPE_PH)
            req_template.ptypes_pos.push_back(i);

    //dump_cmd_chains();
    return true;
}

/**
 * Used by single-mode
 */
bool
sparql_parser::parse(istream &is, request_or_reply &r)
{
    // clear state of parser before a new parsing
    clear();

    // spilt stream into tokens
    vector<string> tokens = get_tokens(is);

    // parse the tokens
    if (!do_parse(tokens))
        return false;

    if (req_template.ptypes_pos.size() != 0) {
        cout << "ERROR: request with PTYPE_PH." << endl;
        return false;
    }

    r = request_or_reply();
    if (join_step >= 0) {
        vector<int64_t> join_pattern;
        join_pattern.push_back(0); // unused
        join_pattern.push_back(0); // unused
        join_pattern.push_back(JOIN);
        join_pattern.push_back(join_step + 1); // because we insert a new cmd in the middle
        req_template.cmd_chains.insert(req_template.cmd_chains.begin() + fork_step * 4,
                                       join_pattern.begin(), join_pattern.end());
    }
    r.cmd_chains = req_template.cmd_chains;
    return true;
}


/**
 * Used by batch-mode
 */
bool
sparql_parser::parse_template(istream &is, request_template &r)
{
    // clear state of parser before a new parsing
    clear();

    vector<string> tokens = get_tokens(is);
    if (!do_parse(tokens))
        return false;

    if (req_template.ptypes_pos.size() == 0) {
        cout << "ERROR: request_template without PTYPE_PH" << endl;
        return false;
    }

    r = req_template;
    return true;
}

bool
sparql_parser::add_type_pattern(string type, request_or_reply &r)
{
    clear();
    r = request_or_reply();

    // add an additonal pattern cmd to collect pattern constants with a certain type
    r.cmd_chains.push_back(str_server->str2id[type]); // type ID
    r.cmd_chains.push_back(global_rdftype_id);  // reserved ID for "rdf:type"
    r.cmd_chains.push_back(IN);
    r.cmd_chains.push_back(-1);
    return true;
}
