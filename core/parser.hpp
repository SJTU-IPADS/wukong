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
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string.hpp>

#include "query.hpp"
#include "type.hpp"
#include "string_server.hpp"

#include "SPARQLParser.hpp"

// utils
#include "assertion.hpp"

using namespace std;

// Read a stream into a string
static string read_input(istream &in) {
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

    // str2id mapping for pattern constants
    // (e.g., <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> 1)
    StringServer *str_server;

    /// SPARQLParser::Element to ssid
    ssid_t transfer_element(const SPARQLParser::Element &e) {
        switch (e.type) {
        case SPARQLParser::Element::Variable:
            return e.id;
        case SPARQLParser::Element::Literal:
        {
            string str = "";
            // string with language tag
            //      Ex. "SuperPatriot"@en
            //      value stored in string server: "SuperPatriot"@en
            //      e.value: SuperPatriot , e.subTypeValue: en
            if (e.subType == SPARQLParser::Element::CustomLanguage)
                str = "\"" + e.value + "\"" + "@" + e.subTypeValue;
            // normal case
            else
                str = "\"" + e.value + "\"";

            if (!str_server->exist(str)) {
                logstream(LOG_ERROR) << "Unknown Literal: " + str << LOG_endl;
                throw WukongException(SYNTAX_ERROR);
            }
            return str_server->str2id(str);
        }
        case SPARQLParser::Element::IRI:
        {
            string str = "<" + e.value + ">"; // IRI
            if (!str_server->exist(str)) {
                logstream(LOG_ERROR) << "Unknown IRI: " + str << LOG_endl;
                throw WukongException(SYNTAX_ERROR);
            }
            return str_server->str2id(str);
        }
        case SPARQLParser::Element::Template:
            return PTYPE_PH;
        case SPARQLParser::Element::Predicate:
            return PREDICATE_ID;
        default:
            throw WukongException(SYNTAX_ERROR);
        }

        throw WukongException(SYNTAX_ERROR);
    }

    /// SPARQLParser::Filter to SPARQLQuery::Filter
    void transfer_filter(SPARQLParser::Filter &src, SPARQLQuery::Filter &dst) {
        dst.type = (SPARQLQuery::Filter::Type)src.type;
        dst.value = src.value;
        dst.valueArg = src.valueArg;
        if (src.arg1 != NULL) {
            dst.arg1 = new SPARQLQuery::Filter();
            transfer_filter(*src.arg1, *dst.arg1);
        }
        if (src.arg2 != NULL) {
            dst.arg2 = new SPARQLQuery::Filter();
            transfer_filter(*src.arg2, *dst.arg2);
        }
        if (src.arg3 != NULL) {
            dst.arg3 = new SPARQLQuery::Filter();
            transfer_filter(*src.arg3, *dst.arg3);
        }
    }

    /// SPARQLParser::PatternGroup to SPARQLQuery::PatternGroup
    void transfer_pg(SPARQLParser::PatternGroup &src, SPARQLQuery::PatternGroup &dst) {
        // Patterns
        for (auto const &p : src.patterns) {
            ssid_t subject = transfer_element(p.subject);
            ssid_t predicate = transfer_element(p.predicate);
            dir_t direction = (dir_t)p.direction;
            ssid_t object = transfer_element(p.object);
            SPARQLQuery::Pattern pattern(subject, predicate, direction, object);

            pattern.pred_type = str_server->pid2type[predicate];
            if ((pattern.pred_type != (char)SID_t) && !Global::enable_vattr) {
                logstream(LOG_ERROR) << "Must enable attribute support"
                                     << LOG_endl;
                ASSERT(false);
            }

            dst.patterns.push_back(pattern);
        }

        // Filters
        for (auto &f : src.filters) {
            dst.filters.push_back(SPARQLQuery::Filter());
            transfer_filter(f, dst.filters.back());
        }

        // Unions
        for (auto &u : src.unions) {
            dst.unions.push_back(SPARQLQuery::PatternGroup());
            transfer_pg(u, dst.unions.back());
        }

        // Optional
        for (auto &o : src.optional) {
            dst.optional.push_back(SPARQLQuery::PatternGroup());
            transfer_pg(o, dst.optional.back());
        }

        /// TODO: support other Grammars in PatternGroup
    }

    void transfer(const SPARQLParser &sp, SPARQLQuery &sq) {
        // required varaibles of SELECT clause
        for (SPARQLParser::projection_iterator iter = sp.projectionBegin();
                iter != sp.projectionEnd();
                iter ++)
            sq.result.required_vars.push_back(*iter);

        // pattern group (patterns, union, filter, optional)
        SPARQLParser::PatternGroup group = sp.getPatterns();
        transfer_pg(group, sq.pattern_group);

        sq.result.nvars = sp.getVariableCount();

        // orders
        for (SPARQLParser::order_iterator iter = sp.orderBegin();
                iter != sp.orderEnd();
                iter ++)
            sq.orders.push_back(SPARQLQuery::Order((*iter).id, (*iter).descending));

        // limit and offset
        sq.limit = sp.getLimit();
        sq.offset = sp.getOffset();

        // distinct
        if ((sp.getProjectionModifier() == SPARQLParser::ProjectionModifier::Modifier_Distinct)
                || (sp.getProjectionModifier() == SPARQLParser::ProjectionModifier::Modifier_Reduced))
            sq.distinct = true;

        // corun optimization (disabled)
        if (sq.corun_enabled = sp.isCorunEnabled()) {
            sq.corun_step = sp.getCorunStep();
            sq.fetch_step = sp.getFetchStep();

            if (!Global::use_rdma) {
                // TODO: corun optimization is not supported w/o RDMA
                logstream(LOG_WARNING) << "RDMA is not enabled, skip corun optimization!" << LOG_endl;
                sq.corun_enabled = false; // skip
            }
        }
    }

    void transfer_template(const SPARQLParser &sp, SPARQLQuery_Template &sqt) {
        // required varaibles of SELECT clause
        for (SPARQLParser::projection_iterator iter = sp.projectionBegin();
                iter != sp.projectionEnd();
                iter ++)
            sqt.required_vars.push_back(*iter);

        // pattern group (patterns)
        // FIXME: union, filter, optional (unsupported now)
        SPARQLParser::PatternGroup group = sp.getPatterns();
        int pos = 0;
        for (auto &p : group.patterns) {
            ssid_t subject = transfer_element(p.subject);
            ssid_t predicate = transfer_element(p.predicate);
            dir_t direction = (dir_t)p.direction;
            ssid_t object = transfer_element(p.object);
            SPARQLQuery::Pattern pattern(subject, predicate, direction, object);

            // template pattern
            if (subject == PTYPE_PH) {
                sqt.ptypes_str.push_back("<" + p.subject.value + ">"); // IRI
                sqt.ptypes_pos.push_back(pos + 0); // subject
            }

            if (object == PTYPE_PH) {
                sqt.ptypes_str.push_back("<" + p.object.value + ">"); // IRI
                sqt.ptypes_pos.push_back(pos + 3); // object
            }

            pattern.pred_type = (char)str_server->pid2type[predicate];
            if ((pattern.pred_type != (char)SID_t) && !Global::enable_vattr) {
                logstream(LOG_ERROR) << "Must enable attribute support" << LOG_endl;
                ASSERT(false);
            }

            sqt.pattern_group.patterns.push_back(pattern);
            pos += 4;
        }

        sqt.nvars = sp.getVariableCount();

        // FIXME: orders (unsupported now)

        // FIXME: limit and offset (unsupported now)

        // FIXME: distinct (unsupported now)

        // FIXME: corun optimization (unsupported now)
    }

public:
    // the stat of query parsing
    std::string strerror;

    Parser(StringServer *_ss): str_server(_ss) { }

    /// a single query
    bool parse(istream &is, SPARQLQuery &sq) {
        string query = read_input(is);
        SPARQLLexer lexer(query);
        SPARQLParser parser(lexer);

        try {
            parser.parse(); //e.g., sparql -f sparql_query/lubm/basic/lubm_q1
            transfer(parser, sq);
        } catch (const SPARQLParser::ParserException &e) {
            logstream(LOG_ERROR) << "Failed to parse a SPARQL query: "
                                 << e.message << LOG_endl;
            return false;
        }

        // check if using custom grammar when planner is on
        if (parser.isUsingCustomGrammar() && Global::enable_planner) {
            logstream(LOG_ERROR) << "Unsupported custom grammar in SPARQL planner!"
                                 << LOG_endl;
            return false;
        }

        logstream(LOG_INFO) << "Parsing a SPARQL query is done." << LOG_endl;
        return true;
    }

    /// a class of queries
    bool parse_template(istream &is, SPARQLQuery_Template &sqt) {
        string query = read_input(is);
        SPARQLLexer lexer(query);
        SPARQLParser parser(lexer);

        try {
            parser.parse();
            transfer_template(parser, sqt);
        } catch (const SPARQLParser::ParserException &e) {
            logstream(LOG_ERROR) << "Failed to parse a SPARQL template: "
                                 << e.message << LOG_endl;
            return false;
        }

        logstream(LOG_INFO) << "Parsing a SPARQL template is done." << LOG_endl;
        return true;
    }
};
