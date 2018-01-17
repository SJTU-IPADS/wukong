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

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/variant.hpp>
#include <vector>

#include "type.hpp"

using namespace std;
using namespace boost::archive;



enum var_type {
    known_var,
    unknown_var,
    const_var
};

// defined as constexpr due to switch-case
constexpr int var_pair(int t1, int t2) { return ((t1 << 4) | t2); }


// EXT = [ TYPE:16 | COL:16 ]
#define NBITS_COL  16
#define NO_RESULT  ((1 << NBITS_COL) - 1)

int col2ext(int col, int t) { return ((t << NBITS_COL) | col); }
int ext2col(int ext) { return (ext & ((1 << NBITS_COL) - 1)); }

class STORECheck {
private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & pid;
        ar & check_ret;
    }

public:
    int pid = -1;    // parent query id

    int check_ret = 0;

    STORECheck() {}
};

class RDFLoad {
private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & pid;
        ar & load_dname;
        ar & load_ret;
        ar & check_dup;
    }

public:
    int pid = -1;    // parent query id

    string load_dname = "";   // the file name used to be inserted
    int load_ret = 0;
    bool check_dup = false;

    RDFLoad() {}

    RDFLoad(string s, bool b) : load_dname(s), check_dup(b) { }
};

class SPARQLQuery {
private:
    void output_result(ostream &stream, int size, String_Server *str_server) {
        for (int i = 0; i < size; i++) {
            stream << i + 1 << ": ";
            for (int c = 0; c < col_num; c++) {
                int id = this->get_row_col(i, c);
                if (str_server->exist(id))
                    stream << str_server->id2str[id] << "\t";
                else
                    stream << id << "\t";
            }
            for (int c = 0; c < this->get_attr_col_num(); c++) {
                attr_t tmp = this->get_attr_row_col(i, c);
                cout << tmp << "\t";
            }
            stream << endl;
        }
    }

    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & id;
        ar & pid;
        ar & tid;
        ar & step;
        ar & col_num;
        ar & row_num;
        ar & attr_col_num;
        ar & blind;
        ar & nvars;
        ar & local_var;
        ar & v2c_map;
        ar & cmd_chains;
        ar & result_table;
        ar & pred_type_chains;
        ar & attr_res_table;
    }

public:
    int id = -1;     // query id
    int pid = -1;    // parqnt query id
    int tid = 0;     // engine thread id (MT)

    // SPARQL query
    int step = 0;
    int col_num = 0;
    int row_num = 0;

    int attr_col_num = 0;

    bool blind = false;

    int nvars = 0; // the number of variables
    ssid_t local_var = 0;   // the local variable

    vector<int> v2c_map; // from variable ID (vid) to column ID

    // ID-format triple patterns (Subject, Predicat, Direction, Object)
    vector<ssid_t> cmd_chains;
    vector<sid_t> result_table; // result table for string IDs

    // ID-format attribute triple patterns (Subject, Attribute, Direction, Value)
    vector<int>  pred_type_chains;
    vector<attr_t> attr_res_table; // result table for others

    SPARQLQuery() { }

    // build a request by existing triple patterns and variables
    SPARQLQuery(vector<ssid_t> cc, int n, vector<int> p)
        : cmd_chains(cc), nvars(n), pred_type_chains(p) {
        v2c_map.resize(n, NO_RESULT);
    }

    void clear_data() { result_table.clear(); attr_res_table.clear(); }

    bool is_finished() { return (step * 4 >= cmd_chains.size()); } // FIXME: it's trick

    bool is_request() { return (id == -1); } // FIXME: it's trick

    bool start_from_index() {
        /*
         * Wukong assumes that its planner will generate a dummy pattern to hint
         * the query should start from a certain index (i.e., predicate or type).
         * For example: ?X __PREDICATE__  ub:undergraduateDegreeFrom
         *
         * NOTE: the graph exploration does not must start from this index,
         * on the contrary, starts from another index would prune bindings MORE efficiently
         * For example, ?X P0 ?Y, ?X P1 ?Z, ...
         *
         * ?X __PREDICATE__ P1 <- // start from index vertex P1
         * ?X P0 ?Y .             // then from ?X's edge with P0
         *
         */
        if (is_tpid(cmd_chains[0])) {
            assert(cmd_chains[1] == PREDICATE_ID || cmd_chains[1] == TYPE_ID);
            return true;
        }
        return false;
    }

    var_type variable_type(ssid_t vid) {
        if (vid >= 0)
            return const_var;
        else if (var2col(vid) == NO_RESULT)
            return unknown_var;
        else
            return known_var;
    }

    // get column id from vid (pattern variable)
    int var2col(ssid_t vid) {
        assert(vid < 0);
        if (v2c_map.size() == 0) // init
            v2c_map.resize(nvars, NO_RESULT);

        int idx = - (vid + 1);
        assert(idx < nvars);

        return ext2col(v2c_map[idx]);
    }

    // add column id to vid (pattern variable)
    void add_var2col(ssid_t vid, int col, int t = SID_t) {
        assert(vid < 0 && col >= 0);
        if (v2c_map.size() == 0) // init
            v2c_map.resize(nvars, NO_RESULT);
        int idx = - (vid + 1);
        assert(idx < nvars);

        assert(v2c_map[idx] == NO_RESULT);
        v2c_map[idx] = col2ext(col, t);
    }

    // result table for string IDs
    void set_col_num(int n) { col_num = n; }

    int get_col_num() { return col_num; }

    int get_row_num() {
        if (col_num == 0) return 0;
        return result_table.size() / col_num;
    }

    sid_t get_row_col(int r, int c) {
        assert(r >= 0 && c >= 0);
        return result_table[col_num * r + c];
    }

    void append_row_to(int r, vector<sid_t> &update) {
        for (int c = 0; c < col_num; c++)
            update.push_back(get_row_col(r, c));
    }

    // result table for others (e.g., integer, float, and double)
    int set_attr_col_num(int n) { attr_col_num = n; }

    int get_attr_col_num() { return  attr_col_num; }

    attr_t get_attr_row_col(int r, int c) {
        return attr_res_table[attr_col_num * r + c];
    }

    void append_attr_row_to(int r, vector<attr_t> &updated_result_table) {
        for (int c = 0; c < attr_col_num; c++)
            updated_result_table.push_back(get_attr_row_col(r, c));
    }

    void print_result(int row2print, String_Server *str_server) {
        cout << "The first " << row2print << " rows of results: " << endl;
        output_result(cout, row2print, str_server);
    }

    void dump_result(string path, int row2print, String_Server *str_server) {
        if (boost::starts_with(path, "hdfs:")) {
            wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
            wukong::hdfs::fstream ofs(hdfs, path, true);
            output_result(ofs, row2print, str_server);
            ofs.close();
        } else {
            ofstream ofs(path);
            if (!ofs.good()) {
                cout << "Can't open/create output file: " << path << endl;
            } else {
                output_result(ofs, row2print, str_server);
                ofs.close();
            }
        }
    }
};

class request_template {

public:
    vector<ssid_t> cmd_chains;

    int nvars;  // the number of variable in triple patterns
    // store the query predicate type
    vector<int> pred_type_chains;

    // no serialize
    vector<int> ptypes_pos; // the locations of random-constants
    vector<string> ptypes_str; // the Types of random-constants
    vector<vector<sid_t>> ptypes_grp; // the candidates for random-constants

    SPARQLQuery instantiate(int seed) {
        for (int i = 0; i < ptypes_pos.size(); i++)
            cmd_chains[ptypes_pos[i]] =
                ptypes_grp[i][seed % ptypes_grp[i].size()];
        return SPARQLQuery(cmd_chains, nvars, pred_type_chains);
    }
};

enum req_type { SPARQL_QUERY, DYNAMIC_LOAD, STORE_CHECK };

class Bundle {
public:
    req_type type;
    string data;

    Bundle() { }

    Bundle(string str) {
        set_type(str.at(0));
        data = str.substr(1);
    }

    Bundle(SPARQLQuery r): type(SPARQL_QUERY) {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);

        oa << r;
        data = ss.str();
    }

    Bundle(RDFLoad r): type(DYNAMIC_LOAD) {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);

        oa << r;
        data = ss.str();
    }

    Bundle(STORECheck r): type(STORE_CHECK) {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);

        oa << r;
        data = ss.str();
    }

    string get_type() {
        switch (type) {
        case SPARQL_QUERY: return "0";
        case DYNAMIC_LOAD: return "1";
        case STORE_CHECK: return "2";
        }
    }

    void set_type(char t) {
        switch (t) {
        case '0': type = SPARQL_QUERY; return;
        case '1': type = DYNAMIC_LOAD; return;
        case '2': type = STORE_CHECK; return;
        }
    }

    SPARQLQuery get_sparql_query() {
        assert(type == SPARQL_QUERY);

        std::stringstream ss;
        ss << data;

        boost::archive::binary_iarchive ia(ss);
        SPARQLQuery result;
        ia >> result;
        return result;
    }

    RDFLoad get_rdf_load() {
        assert(type == DYNAMIC_LOAD);

        std::stringstream ss;
        ss << data;

        boost::archive::binary_iarchive ia(ss);
        RDFLoad result;
        ia >> result;
        return result;
    }

    STORECheck get_store_check() {
        assert(type == STORE_CHECK);

        std::stringstream ss;
        ss << data;

        boost::archive::binary_iarchive ia(ss);
        STORECheck result;
        ia >> result;
        return result;
    }

private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & type;
        ar & data;
    }
};
