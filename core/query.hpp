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

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/split_free.hpp>
#include <vector>

#include "type.hpp"

using namespace std;
using namespace boost::archive;


enum req_type { SPARQL_QUERY, DYNAMIC_LOAD, GSTORE_CHECK };


enum var_type {
    known_var,
    unknown_var,
    const_var
};

// EXT = [ TYPE:16 | COL:16 ]
#define NBITS_COL  16
#define NO_RESULT  ((1 << NBITS_COL) - 1)

int col2ext(int col, int t) { return ((t << NBITS_COL) | col); }
int ext2col(int ext) { return (ext & ((1 << NBITS_COL) - 1)); }

/**
 * SPARQL Query
 */
class SPARQLQuery {
private:
    friend class boost::serialization::access;

public:
    class Pattern {
    private:
        friend class boost::serialization::access;

    public:
        ssid_t subject;
        ssid_t predicate;
        ssid_t object;
        dir_t  direction;
        char  pred_type;

        Pattern() { }

        Pattern(ssid_t subject, ssid_t predicate, dir_t direction, ssid_t object):
            subject(subject), predicate(predicate), object(object), direction(direction) { }

        Pattern(ssid_t subject, ssid_t predicate, ssid_t direction, ssid_t object):
            subject(subject), predicate(predicate), object(object), direction((dir_t)direction) { }

        void print_pattern() { }
    };

    class Filter {
    private:
        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(Archive &ar, const unsigned int version) {
            ar & type;
            ar & arg1;
            ar & arg2;
            ar & arg3;
            ar & value;
            ar & valueArg;
        }

    public:
        enum Type {Or, And, Equal, NotEqual, Less, LessOrEqual, Greater,
                   GreaterOrEqual, Plus, Minus, Mul, Div, Not, UnaryPlus, UnaryMinus,
                   Literal, Variable, IRI, Function, ArgumentList, Builtin_str,
                   Builtin_lang, Builtin_langmatches, Builtin_datatype, Builtin_bound,
                   Builtin_sameterm, Builtin_isiri, Builtin_isblank, Builtin_isliteral,
                   Builtin_regex, Builtin_in
                  };

        Type type;
        Filter *arg1, *arg2, *arg3; /// Input arguments
        std::string value; /// The value (for constants param)
        int valueArg; /// variable ids

        /// Constructor
        Filter() : arg1(0), arg2(0), arg3(0), valueArg(0) { }

        /// Copy-Constructor
        Filter(const Filter &other)
            : type(other.type), arg1(0), arg2(0), arg3(0),
              value(other.value), valueArg(other.valueArg) {
            if (other.arg1)
                arg1 = new Filter(*other.arg1);
            if (other.arg2)
                arg2 = new Filter(*other.arg2);
            if (other.arg3)
                arg3 = new Filter(*other.arg3);
        }

        /// Destructor
        ~Filter() {
            delete arg1;
            delete arg2;
            delete arg3;
        }

        void print_filter() {
            // print info
            logstream(LOG_INFO) << "---------------------filter---------------------------" << LOG_endl;
            logstream(LOG_INFO) << "TYPE: " << this->type << LOG_endl;
            if (this->value != "")
                logstream(LOG_INFO) << "value: " << this->value << LOG_endl;
            logstream(LOG_INFO) << "valueArg: " << this->valueArg << LOG_endl;

            if (arg1 != NULL) arg1->print_filter();
            if (arg2 != NULL) arg2->print_filter();
            if (arg3 != NULL) arg3->print_filter();

            logstream(LOG_INFO) << "------------------------------------------------------" << LOG_endl;
        }
    };

    class PatternGroup {
    private:
        friend class boost::serialization::access;

    public:
        vector<Pattern> patterns;
        vector<Filter> filters;
        vector<PatternGroup> optional;
        vector<PatternGroup> unions;

        void print_group() const {
            logstream(LOG_INFO) << "patterns[" << patterns.size() << "]:" << LOG_endl;
            for (auto const &p : patterns)
                logstream(LOG_INFO) << "\t" << p.subject
                                    << "\t" << p.predicate
                                    << "\t" << p.direction
                                    << "\t" << p.object << LOG_endl;

            logstream(LOG_INFO) << "unions[" << unions.size() << "]:" << LOG_endl;
            for (auto const &g : unions)
                g.print_group();
        }
    };

    class Order {
    private:
        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(Archive &ar, const unsigned int version) {
            ar & id;
            ar & descending;
        }

    public:
        ssid_t id;  /// variable id
        bool descending;    /// desending

        Order() { }

        Order(ssid_t _id, bool _descending)
            : id(_id), descending(_descending) { }
    };

    class Result {
    private:
        friend class boost::serialization::access;

        void output_result(ostream &stream, int size, String_Server *str_server) {
            for (int i = 0; i < size; i++) {
                stream << i + 1 << ": ";
                for (int j = 0; j < col_num; j++) {
                    int id = this->get_row_col(i, j);
                    if (str_server->exist(id))
                        stream << str_server->id2str[id] << "\t";
                    else
                        stream << id << "\t";
                }

                for (int c = 0; c < this->get_attr_col_num(); c++) {
                    attr_t tmp = this->get_attr_row_col(i, c);
                    stream << tmp << "\t";
                }

                stream << endl;
            }
        }

    public:
        int col_num = 0;
        int row_num = 0;
        int attr_col_num = 0;
        bool blind = false;
        int nvars = 0; // the number of variables
        vector<int> v2c_map; // from variable ID (vid) to column ID, index: vid, value: col
        vector<ssid_t> required_vars; // variables selected to return
        vector<sid_t> result_table; // result table for string IDs
        vector<attr_t> attr_res_table; // result table for others

        void clear_data() {
            result_table.clear();
            attr_res_table.clear();
            required_vars.clear();
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

        // TODO unused set get
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

        // insert a blank col to result table without updating col_num and v2c_map
        void insert_blank_col(int col) {
            vector<sid_t> new_table;
            for (int i = 0; i < this->get_row_num(); i++) {
                new_table.insert(new_table.end(),
                                 this->result_table.begin() + (i * this->col_num),
                                 this->result_table.begin() + (i * this->col_num + col));
                new_table.push_back(BLANK_ID);
                new_table.insert(new_table.end(),
                                 this->result_table.begin() + (i * this->col_num + col),
                                 this->result_table.begin() + ((i + 1) * this->col_num));
            }
            this->result_table.swap(new_table);
        }

        void merge_optional(SPARQLQuery::Result &result) {
            vector<int> col_map(this->nvars, -1);  // idx: my_col, value: your_col
            vector<int> common_cols;    // value: the #col of my vids that you also have
            vector<int> new_cols;   // value: my #col of new vids
            int old_col_num = this->col_num;
            for (int i = 0; i < result.v2c_map.size(); i++) {
                ssid_t vid = -1 - i;
                if (this->v2c_map[i] != NO_RESULT && result.v2c_map[i] != NO_RESULT) {
                    col_map[this->var2col(vid)] = result.var2col(vid);
                    common_cols.push_back(this->var2col(vid));
                } else if (this->v2c_map[i] == NO_RESULT && result.v2c_map[i] != NO_RESULT) {
                    this->add_var2col(vid, this->col_num);
                    col_map[this->col_num] = result.var2col(vid);
                    new_cols.push_back(this->col_num);
                    this->col_num++;
                }
            }

            vector<sid_t> new_table;
            for (int i = 0; i < this->row_num; i++) {
                bool printed = false;
                for (int j = 0; j < result.row_num; j++) {
                    // check if common_cols match
                    bool same_common = true;
                    for (auto &common_col : common_cols) {
                        if (result.result_table[j * result.col_num + col_map[common_col]]
                                != this->result_table[i * old_col_num + common_col]) {
                            same_common = false;
                            break;
                        }
                    }

                    if (same_common) {
                        printed = true;
                        new_table.insert(new_table.end(),
                                         this->result_table.begin() + (i * old_col_num),
                                         this->result_table.begin() + ((i + 1) * old_col_num));
                        for (auto &new_col : new_cols)
                            new_table.push_back(result.result_table[j * result.col_num + col_map[new_col]]);
                    }
                }

                if (!printed) {
                    new_table.insert(new_table.end(),
                                     this->result_table.begin() + (i * old_col_num),
                                     this->result_table.begin() + ((i + 1) * old_col_num));
                    for (auto &new_col : new_cols)
                        new_table.push_back(BLANK_ID);
                }
            }
            this->result_table.swap(new_table);
            this->row_num = this->get_row_num();
        }

        void merge_union(SPARQLQuery::Result &result) {
            this->nvars = result.nvars;
            this->v2c_map.resize(this->nvars, NO_RESULT);
            this->blind = result.blind;
            this->row_num += result.row_num;
            this->attr_col_num = result.attr_col_num;
            vector<int> col_map(this->nvars, -1);  // idx: my_col, value: your_col

            for (int i = 0; i < result.v2c_map.size(); i++) {
                ssid_t vid = -1 - i;
                if (this->v2c_map[i] == NO_RESULT && result.v2c_map[i] != NO_RESULT) {
                    this->insert_blank_col(this->col_num);
                    this->add_var2col(vid, this->col_num);
                    col_map[this->col_num] = result.var2col(vid);
                    this->col_num++;
                } else if (this->v2c_map[i] != NO_RESULT && result.v2c_map[i] == NO_RESULT) {
                    col_map[this->var2col(vid)] = -1;
                } else if (this->v2c_map[i] != NO_RESULT && result.v2c_map[i] != NO_RESULT) {
                    col_map[this->var2col(vid)] = result.var2col(vid);
                }
            }

            int new_size = this->col_num * this->row_num;
            this->result_table.reserve(new_size);
            for (int i = 0; i < result.row_num; i++) {
                for (int j = 0; j < this->col_num; j++) {
                    if (col_map[j] == -1)
                        this->result_table.push_back(BLANK_ID);
                    else
                        this->result_table.push_back(result.result_table[i * result.col_num + col_map[j]]);
                }
            }
            new_size = this->attr_res_table.size() + result.attr_res_table.size();
            this->attr_res_table.reserve(new_size);
            this->attr_res_table.insert(this->attr_res_table.end(),
                                        result.attr_res_table.begin(),
                                        result.attr_res_table.end());
        }

        void append_result(SPARQLQuery::Result &result) {
            this->col_num = result.col_num;
            this->blind = result.blind;
            this->row_num += result.row_num;
            this->attr_col_num = result.attr_col_num;
            this->v2c_map = result.v2c_map;

            if (!this->blind) {
                int new_size = this->result_table.size()
                               + result.result_table.size();
                this->result_table.reserve(new_size);
                this->result_table.insert(this->result_table.end(),
                                          result.result_table.begin(),
                                          result.result_table.end());
                new_size = this->attr_res_table.size()
                           + result.attr_res_table.size();
                this->attr_res_table.reserve(new_size);
                this->attr_res_table.insert(this->attr_res_table.end(),
                                            result.attr_res_table.begin(),
                                            result.attr_res_table.end());
            }
        }

        void print_result(int row2print, String_Server *str_server) {
            logstream(LOG_INFO) << "The first " << row2print << " rows of results: " << LOG_endl;
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
                    logstream(LOG_INFO) << "Can't open/create output file: " << path << LOG_endl;
                } else {
                    output_result(ofs, row2print, str_server);
                    ofs.close();
                }
            }
        }
    };

    int id = -1;     // query id
    int pid = -1;    // parqnt query id
    int tid = 0;     // engine thread id (MT)

    // SPARQL query
    int step = 0;
    bool corun_enabled = false;
    int corun_step = 0;
    int fetch_step = 0;
    int limit = -1;
    unsigned offset = 0;
    bool distinct = false;
    bool optional_dispatched = true;

    ssid_t local_var = 0;   // the local variable

    bool force_dispatch = false;
    int mt_factor = 0;
    int priority = 0;

    // ID-format triple patterns (Subject, Predicat, Direction, Object)
    PatternGroup pattern_group;
    vector<Order> orders;
    Result result;

    SPARQLQuery() { }

    // build a request by existing triple patterns and variables
    SPARQLQuery(PatternGroup g, int n) : pattern_group(g) {
        result.nvars = n;
        result.v2c_map.resize(n, NO_RESULT);
    }

    Pattern & get_current_pattern() {
        assert(this->step < pattern_group.patterns.size());
        return pattern_group.patterns[this->step];
    }

    Pattern & get_pattern(int step) {
        assert(step < pattern_group.patterns.size());
        return pattern_group.patterns[step];
    }

    // clear query when send back result to achieve better performance
    void clear_data() {
        orders.clear();
        // the first pattern indicating if this query is starting from index. It can't be removed.
        pattern_group.patterns.erase(pattern_group.patterns.begin() + 1,
                                     pattern_group.patterns.end());
        pattern_group.filters.clear();
        pattern_group.optional.clear();
        pattern_group.unions.clear();

        if (result.blind)
            result.clear_data(); // avoid take back all the results
    }

    bool is_union() { return pattern_group.unions.size() > 0; }

    bool is_optional() { return pattern_group.optional.size() > 0; }

    bool is_finished() { return (step >= pattern_group.patterns.size()); } // FIXME: it's trick

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
        if (pattern_group.patterns.size() == 0) return false;
        else if (is_tpid(pattern_group.patterns[0].subject)) {
            assert(pattern_group.patterns[0].predicate == PREDICATE_ID
                   || pattern_group.patterns[0].predicate == TYPE_ID);
            return true;
        }
        return false;
    }

    void print_sparql_query() {
        logstream(LOG_INFO) << "SPARQL-Query"
                            << "[ ID=" << id << " | PID=" << pid << " | TID=" << tid << " ]"
                            << LOG_endl;
        pattern_group.print_group();
        /// TODO: print more fields
        logstream(LOG_INFO) << LOG_endl;
    }
};


/**
 * SPARQL query template
 */
class request_template {
private:
    // no serialize

public:
    SPARQLQuery::PatternGroup pattern_group;

    int nvars;  // the number of variable in triple patterns

    vector<string> ptypes_str; // the Types of random-constants
    vector<int> ptypes_pos; // the locations of random-constants

    vector<vector<sid_t>> ptypes_grp; // the candidates for random-constants

    SPARQLQuery instantiate(int seed) {
        for (int i = 0; i < ptypes_pos.size(); i++) {
            int pos = ptypes_pos[i];
            switch (pos % 4) {
            case 0:
                pattern_group.patterns[pos / 4].subject =
                    ptypes_grp[i][seed % ptypes_grp[i].size()];
                break;
            case 1:
                pattern_group.patterns[pos / 4].predicate =
                    ptypes_grp[i][seed % ptypes_grp[i].size()];
                break;
            case 3:
                pattern_group.patterns[pos / 4].object =
                    ptypes_grp[i][seed % ptypes_grp[i].size()];
                break;
            }
        }

        return SPARQLQuery(pattern_group, nvars);
    }
};


/**
 * GStore consistency checker
 */
class GStoreCheck {
private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & pid;
        ar & check_ret;
        ar & index_check;
        ar & normal_check;
    }

public:
    int pid = -1;    // parent query id

    int check_ret = 0;
    bool index_check = false;
    bool normal_check = false;

    GStoreCheck() { }

    GStoreCheck(bool i, bool n) : index_check(i), normal_check(n) { }
};

/**
 * RDF data loader
 */
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

    RDFLoad() { }

    RDFLoad(string s, bool b) : load_dname(s), check_dup(b) { }
};

namespace boost {
namespace serialization {
char occupied = 0;
char empty = 1;

template<class Archive>
void save(Archive &ar, const SPARQLQuery::Pattern &t, unsigned int version) {
    ar << t.subject;
    ar << t.predicate;
    ar << t.object;
    ar << t.direction;
    ar << t.pred_type;
}

template<class Archive>
void load(Archive &ar, SPARQLQuery::Pattern &t, unsigned int version) {
    ar >> t.subject;
    ar >> t.predicate;
    ar >> t.object;
    ar >> t.direction;
    ar >> t.pred_type;
}

template<class Archive>
void save(Archive &ar, const SPARQLQuery::PatternGroup &t, unsigned int version) {
    ar << t.patterns;
    if (t.filters.size() > 0) {
        ar << occupied;
        ar << t.filters;
    } else ar << empty;

    if (t.optional.size() > 0) {
        ar << occupied;
        ar << t.optional;
    } else ar << empty;

    if (t.unions.size() > 0) {
        ar << occupied;
        ar << t.unions;
    } else ar << empty;
}

template<class Archive>
void load(Archive &ar, SPARQLQuery::PatternGroup &t, unsigned int version) {
    char temp = 2;
    ar >> t.patterns;
    ar >> temp;
    if (temp == occupied) {
        ar >> t.filters;
        temp = 2;
    }
    ar >> temp;
    if (temp == occupied) {
        ar >> t.optional;
        temp = 2;
    }
    ar >> temp;
    if (temp == occupied) {
        ar >> t.unions;
        temp = 2;
    }
}

template<class Archive>
void save(Archive &ar, const SPARQLQuery::Result &t, unsigned int version) {
    ar << t.col_num;
    ar << t.row_num;
    ar << t.attr_col_num;
    ar << t.blind;
    ar << t.nvars;
    ar << t.v2c_map;
    if (!t.blind) ar << t.required_vars;
    // attr_res_table must be empty if result_table is empty
    if (t.result_table.size() > 0) {
        ar << occupied;
        ar << t.result_table;
        ar << t.attr_res_table;
    } else {
        ar << empty;
    }
}

template<class Archive>
void load(Archive & ar, SPARQLQuery::Result &t, unsigned int version) {
    char temp = 2;
    ar >> t.col_num;
    ar >> t.row_num;
    ar >> t.attr_col_num;
    ar >> t.blind;
    ar >> t.nvars;
    ar >> t.v2c_map;
    if (!t.blind) ar >> t.required_vars;
    ar >> temp;
    if (temp == occupied) {
        ar >> t.result_table;
        ar >> t.attr_res_table;
    }
}

template<class Archive>
void save(Archive & ar, const SPARQLQuery &t, unsigned int version) {
    ar << t.id;
    ar << t.pid;
    ar << t.tid;
    ar << t.limit;
    ar << t.offset;
    ar << t.distinct;
    ar << t.optional_dispatched;
    ar << t.step;
    ar << t.corun_step;
    ar << t.fetch_step;
    ar << t.local_var;
    ar << t.force_dispatch;
    ar << t.mt_factor;
    ar << t.priority;
    ar << t.pattern_group;
    if (t.orders.size() > 0) {
        ar << occupied;
        ar << t.orders;
    } else {
        ar << empty;
    }
    ar << t.result;
}

template<class Archive>
void load(Archive & ar, SPARQLQuery &t, unsigned int version) {
    char temp = 2;
    ar >> t.id;
    ar >> t.pid;
    ar >> t.tid;
    ar >> t.limit;
    ar >> t.offset;
    ar >> t.distinct;
    ar >> t.optional_dispatched;
    ar >> t.step;
    ar >> t.corun_step;
    ar >> t.fetch_step;
    ar >> t.local_var;
    ar >> t.force_dispatch;
    ar >> t.mt_factor;
    ar >> t.priority;
    ar >> t.pattern_group;
    ar >> temp;
    if (temp == occupied) ar >> t.orders;
    ar >> t.result;
}

}
}

BOOST_SERIALIZATION_SPLIT_FREE(SPARQLQuery::Pattern);
BOOST_SERIALIZATION_SPLIT_FREE(SPARQLQuery::PatternGroup);
BOOST_SERIALIZATION_SPLIT_FREE(SPARQLQuery::Result);
BOOST_SERIALIZATION_SPLIT_FREE(SPARQLQuery);

// remove class information at the cost of losing auto versioning,
// which is useless currently because wukong use boost serialization to transmit data
// between endpoints running the same code.
BOOST_CLASS_IMPLEMENTATION(SPARQLQuery::Pattern, boost::serialization::object_serializable);
BOOST_CLASS_IMPLEMENTATION(SPARQLQuery::PatternGroup, boost::serialization::object_serializable);
BOOST_CLASS_IMPLEMENTATION(SPARQLQuery::Filter, boost::serialization::object_serializable);
BOOST_CLASS_IMPLEMENTATION(SPARQLQuery::Order, boost::serialization::object_serializable);
BOOST_CLASS_IMPLEMENTATION(SPARQLQuery::Result, boost::serialization::object_serializable);
BOOST_CLASS_IMPLEMENTATION(SPARQLQuery, boost::serialization::object_serializable);
BOOST_CLASS_IMPLEMENTATION(GStoreCheck, boost::serialization::object_serializable);
BOOST_CLASS_IMPLEMENTATION(RDFLoad, boost::serialization::object_serializable);

// remove object tracking information at the cost of that multiple identical objects
// may be created when an archive is loaded.
// current query data structure does not contain two identical object reference
// with the same pointer
BOOST_CLASS_TRACKING(SPARQLQuery::Pattern, boost::serialization::track_never);
BOOST_CLASS_TRACKING(SPARQLQuery::Filter, boost::serialization::track_never);
BOOST_CLASS_TRACKING(SPARQLQuery::PatternGroup, boost::serialization::track_never);
BOOST_CLASS_TRACKING(SPARQLQuery::Order, boost::serialization::track_never);
BOOST_CLASS_TRACKING(SPARQLQuery::Result, boost::serialization::track_never);
BOOST_CLASS_TRACKING(SPARQLQuery, boost::serialization::track_never);
BOOST_CLASS_TRACKING(GStoreCheck, boost::serialization::track_never);
BOOST_CLASS_TRACKING(RDFLoad, boost::serialization::track_never);

/**
 * Bundle to be sent by network, with data type labeled
 * Note this class does not use boost serialization
 */
class Bundle {
public:
    req_type type;
    string data;

    Bundle() { }

    Bundle(string str) {
        set_type(str.at(0));
        data = str.substr(1);
    }

    Bundle(SPARQLQuery &r): type(SPARQL_QUERY) {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);

        oa << r;
        data = ss.str();
    }

    Bundle(RDFLoad &r): type(DYNAMIC_LOAD) {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);

        oa << r;
        data = ss.str();
    }

    Bundle(GStoreCheck r): type(GSTORE_CHECK) {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);

        oa << r;
        data = ss.str();
    }

    string get_type() {
        switch (type) {
        case SPARQL_QUERY: return "0";
        case DYNAMIC_LOAD: return "1";
        case GSTORE_CHECK: return "2";
        }
    }

    void set_type(char t) {
        switch (t) {
        case '0': type = SPARQL_QUERY; return;
        case '1': type = DYNAMIC_LOAD; return;
        case '2': type = GSTORE_CHECK; return;
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

    GStoreCheck get_gstore_check() {
        assert(type == GSTORE_CHECK);

        std::stringstream ss;
        ss << data;

        boost::archive::binary_iarchive ia(ss);
        GStoreCheck result;
        ia >> result;
        return result;
    }
};
