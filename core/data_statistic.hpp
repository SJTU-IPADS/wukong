#pragma once

#include <unordered_map>
#include <unordered_set>
#include <boost/mpi.hpp>
#include <boost/functional/hash.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <tbb/concurrent_hash_map.h>

#include "global.hpp"

#include "store/gstore.hpp"
#include "comm/tcp_adaptor.hpp"

using namespace std;

struct type_t {
    bool data_type;   //true for type_composition, false for index_composition
    std::unordered_set<int> composition;

    void set_type_composition(std::unordered_set<int> c) {
        data_type = true;
        this->composition = c;
    }

    void set_index_composition(std::unordered_set<int> c) {
        data_type = false;
        this->composition = c;
    };

    bool operator == (const type_t &other) const {
        if (data_type != other.data_type) return false;
        return this->composition == other.composition;
    }

    bool equal(const type_t &other) const {
        if (data_type != other.data_type) return false;
        return this->composition == other.composition;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & data_type;
        ar & composition;
    }
};

struct type_t_hasher {
    size_t operator()( const type_t& type ) const {
        return hash(type);
    }

    // for tbb hashcompare
    size_t hash( const type_t& type ) const {
        size_t res = 17;
        for (auto it = type.composition.cbegin(); it != type.composition.cend(); ++it)
            res += *it + 17;
        return res;
    }

    // for tbb hashcompare
    bool equal(const type_t& type1, const type_t& type2) const {
        return type1.equal(type2);
    }
};

struct ty_count {
    ssid_t ty;
    int count;
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & ty;
        ar & count;
    }
};

struct type_stat {
    unordered_map<ssid_t, vector<ty_count>> pstype;
    unordered_map<ssid_t, vector<ty_count>> potype;
    unordered_map<pair<ssid_t, ssid_t>, vector<ty_count>, boost::hash<pair<int, int>>> fine_type;

    // pair<subject, predicate> means subject predicate -> ?
    // pair<predicate, object> means ? predicate -> object
    int get_pstype_count(ssid_t predicate, ssid_t type) {
        vector<ty_count> &types = pstype[predicate];
        for (size_t i = 0; i < types.size(); i++)
            if (types[i].ty == type)
                return types[i].count;
        return 0;
    }

    int get_potype_count(ssid_t predicate, ssid_t type) {
        vector<ty_count> &types = potype[predicate];
        for (size_t i = 0; i < types.size(); i++)
            if (types[i].ty == type)
                return types[i].count;
        return 0;
    }

    int insert_stype(ssid_t predicate, ssid_t type, int count) {
        vector<ty_count> &types = pstype[predicate];
        for (size_t i = 0; i < types.size(); i++) {
            if (types[i].ty == type) {
                types[i].count += count;
                return 0;
            }
        }

        ty_count newty;
        newty.ty = type;
        newty.count = count;
        types.push_back(newty);
        return 1;
    }

    int insert_otype(ssid_t predicate, ssid_t type, int count) {
        vector<ty_count> &types = potype[predicate];
        for (size_t i = 0; i < types.size(); i++) {
            if (types[i].ty == type) {
                types[i].count += count;
                return 0;
            }
        }

        ty_count newty;
        newty.ty = type;
        newty.count = count;
        types.push_back(newty);
        return 1;
    }

    int insert_finetype(ssid_t first, ssid_t second, ssid_t type, int count) {
        vector<ty_count> &types = fine_type[make_pair(first, second)];
        for (size_t i = 0; i < types.size(); i++) {
            if (types[i].ty == type) {
                types[i].count += count;
                return 0;
            }
        }

        ty_count newty;
        newty.ty = type;
        newty.count = count;
        types.push_back(newty);
        return 1;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & pstype;
        ar & potype;
        ar & fine_type;
    }
};

typedef tbb::concurrent_unordered_set<ssid_t> tbb_set;
typedef tbb::concurrent_hash_map<type_t, ssid_t, type_t_hasher> tbb_map;

class data_statistic {
private:
    // after the master server get whole statistics,
    // this method is used to send it to all machines.
    void send_stat_to_all_machines(TCP_Adaptor *tcp_ad) {
        if (sid == 0) {
            // master server sends statistics
            std::stringstream ss;
            boost::archive::binary_oarchive my_oa(ss);
            my_oa << global_tyscount
                  << global_tystat
                  << global_type2int
                  << global_single2complex;

            for (int i = 1; i < global_num_servers; i++)
                tcp_ad->send(i, 0, ss.str());

        } else {
            // every slave server recieves statistics
            std::string str;
            str = tcp_ad->recv(0);
            std::stringstream ss;
            ss << str;
            boost::archive::binary_iarchive ia(ss);
            ia >> global_tyscount
               >> global_tystat
               >> global_type2int
               >> global_single2complex;
        }
    }

    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & local_tyscount;
        ar & local_tystat;
        ar & local_int2type;
        ar & local_type2int;
    }

public:
    const double TYPE_REMOVE_RATE = 0.1;

    unordered_map<ssid_t, int> local_tyscount;
    unordered_map<ssid_t, int> global_tyscount;

    type_stat local_tystat;
    type_stat global_tystat;

    // use negative numbers to represent complex types
    // (type_composition and index_composition)
    unordered_map<ssid_t, type_t> local_int2type;
    unordered_map<type_t, ssid_t, type_t_hasher> local_type2int;
    unordered_map<ssid_t, type_t> global_int2type;  //not used in planner.hpp currently
    unordered_map<type_t, ssid_t, type_t_hasher> global_type2int;

    // single type may be contained by several multitype
    unordered_map<ssid_t, unordered_set<ssid_t>> global_single2complex;

    unordered_set<ssid_t> global_useful_type;

    int sid;

    data_statistic(int _sid) : sid(_sid) { }

    data_statistic() { }

    // for debug usage
    void show_stat_info() {
        int number_of_notype = 0;
        int number_of_multitype = 0;
        int number_of_singletype = 0;
        for (auto const &token : global_type2int) {
            if (token.first.data_type)
                number_of_multitype ++;
            else
                number_of_notype ++;
        }
        for (auto const &token : global_tyscount) {
            if (token.first > 0)
                number_of_singletype++;
        }

        cout << "number_of_multitype: " << number_of_multitype << endl;
        cout << "number_of_notype: " << number_of_notype << endl;
        cout << "number_of_singletype: " << number_of_singletype << endl;

        const int NUMBER = 10;
        int temp[NUMBER];
        int temp2[NUMBER];
        for (int i = 0; i < NUMBER; i ++) {
            temp[i] = 0;
            temp2[i] = 0;
        }
        for (auto const &token : global_tyscount) {
            if (token.second <= NUMBER) {
                temp[token.second - 1] ++;
                if (token.first > 0) {
                    temp2[token.second - 1] ++;
                }
            }
        }

        cout << "useless type number: " << endl;
        for (int i = 0; i < NUMBER; i ++) {
            cout << temp[i] << "\t" << temp2[i] << endl;
        }
    }

    // reduce number of types to speed up planning procedure
    // sacrifice accuracy in change for speed
    // MODIFICATIONS:
    // global_tyscount: useful_type count
    // global_type2int: all type to its type_No
    // global_single2complex: single to useful_multipletype
    void merge_type() {

        //show_stat_info();

        uint64_t total_number = 0;
        map<int, ssid_t> tys;
        int minimum_count = 0;
        for (auto const &token : global_tyscount) {
            total_number += token.second;
            if (tys.find(token.second) == tys.end())
                tys[token.second] = 1;
            else
                tys[token.second] ++;
        }

        uint64_t sum = 0;

        for (auto const &token : tys) {
            sum += token.first * token.second;
            if (sum >= total_number * TYPE_REMOVE_RATE) {
                minimum_count = token.first;
                break;
            }
        }

        //cout << "minimum_count: " << minimum_count << endl;

        tbb_map new_type2int;
        // type of which has too few vertices (among notype & multitype)
        unordered_set<ssid_t> global_useless_type;
        for (auto const &token : global_tyscount) {
            //global_useless_type.insert(token.first);

            // generated type && vertices of this type less than threshold
            if (token.first < 0 && token.second < minimum_count)
                global_useless_type.insert(token.first);
            else
                global_useful_type.insert(token.first);
        }

# if 0
        // TODO: Using similarity may have better performance for some queries. This strategy may be useful in future.
        auto similarity = [&](type_t t1, type_t t2) -> int{
            if (t1.data_type != t2.data_type) return 0;
            for (auto &token : t2.composition) {
                if (t1.composition.find(token) == t1.composition.end())
                    return 0;
            }
            return t2.composition.size();
        };

        // get useful type2int
        #pragma omp parallel
        for (auto const &token : global_useless_type) {
            #pragma omp single nowait
            {
                type_t useless_type_No = global_int2type[token];
                // take it as the closest useful type or 0-type
                ssid_t result = 0;
                int max_similarity = 0;
                for (auto const &token2 : global_useful_type) {
                    type_t useful_type = global_int2type[token2];
                    int sim = similarity(useless_type_No, useful_type);
                    if (sim > 0 && sim > max_similarity) {
                        result = token2;
                    }
                }
                tbb_map::accessor a;
                new_type2int.insert(a, useless_type_No);
                a->second = result;
            }
        }
# endif

        // useful type2int
        unordered_map<type_t, ssid_t, type_t_hasher> type2int_new;
//      for(auto const &token: new_type2int){
//          type2int_new[token.first] = token.second;
//      }

        // set all useless types to 0-type
        for (auto const &token : global_useless_type) {
            type2int_new[global_int2type[token]] = 0;
        }
        for (auto const &token : global_useful_type) {
            type2int_new[global_int2type[token]] = token;
        }

        // update global_tyscount
        unordered_map<ssid_t, int> tyscount;
        for (auto const &token : global_useful_type) {
            tyscount[token] = global_tyscount[token];
        }
        for (auto const &token : type2int_new) {
            if (tyscount.find(token.second) != tyscount.end()) {
                tyscount[token.second] += global_tyscount[global_type2int[token.first]];
            }
            else {
                tyscount[token.second] = global_tyscount[global_type2int[token.first]];
            }
        }
        global_tyscount.swap(tyscount);

        // add global_single2complex info
        for (auto const &type_No : global_useful_type) {
            type_t type = global_int2type[type_No];
            if (type.data_type) {
                for (auto const &single_type : type.composition) {
                    if (global_single2complex.find(single_type) != global_single2complex.end()) {
                        global_single2complex[single_type].insert(type_No);
                    } else {
                        unordered_set<ssid_t> multi_type_set;
                        // set will automatically ensure no duplicated element exist
                        multi_type_set.insert(type_No);
                        global_single2complex[single_type] = multi_type_set;
                    }
                }
            }
        }

        // update global_type2int
        global_type2int.swap(type2int_new);

    }

    void gather_stat(TCP_Adaptor *tcp_ad) {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);
        oa << (*this);
        tcp_ad->send(0, 0, ss.str());

        if (sid == 0) {
            vector<data_statistic> all_gather;
            // complex type have different corresponding number on different machine
            // assume type < 0 here
            auto type_transform = [&](ssid_t type_No, data_statistic & stat) -> ssid_t{

                type_t complex_type;
                if (stat.local_int2type.find(type_No) != stat.local_int2type.end())
                    complex_type = stat.local_int2type[type_No];
                else
                    logstream(LOG_ERROR) << "type_No: " << type_No << " is not in local_int2type" << LOG_endl;

                if (global_type2int.find(complex_type) != global_type2int.end())
                    return global_type2int[complex_type];
                else {
                    logstream(LOG_ERROR) << "type not found" << LOG_endl;
                    return 0;
                }
            };

            // receive from all proxies
            for (int i = 0; i < global_num_servers; i++) {
                std::string str;
                str = tcp_ad->recv(0);
                data_statistic tmp_data;
                std::stringstream s;
                s << str;
                boost::archive::binary_iarchive ia(s);
                ia >> tmp_data;
                all_gather.push_back(tmp_data);
            }

            // register all types in global_tyscount
            for (int i = 0; i < all_gather.size(); i++) {
                for (auto const & token : all_gather[i].local_tyscount) {
                    ssid_t raw_type_No = token.first;
                    int number = token.second;
                    ssid_t new_type_No = raw_type_No;
                    if (raw_type_No < 0) {
                        type_t complex_type;

                        if (all_gather[i].local_int2type.find(raw_type_No) != all_gather[i].local_int2type.end())
                            complex_type = all_gather[i].local_int2type[raw_type_No];
                        else
                            logstream(LOG_ERROR) << "type: " << raw_type_No << " is not in local_int2type" << LOG_endl;

                        if (global_type2int.find(complex_type) == global_type2int.end()) {
                            ssid_t number = global_type2int.size();
                            number ++;
                            number = -number;
                            global_type2int[complex_type] = number;
                            global_int2type[number] = complex_type;
                            new_type_No = number;
                        }
                        else
                            new_type_No = global_type2int[complex_type];
                    }

                    if (global_tyscount.find(new_type_No) == global_tyscount.end())
                        global_tyscount[new_type_No] = number;
                    else
                        global_tyscount[new_type_No] += number;

                }
            }

            // merge
            if (global_tyscount.size() > 100)
                merge_type();
            else {
                // add global_single2complex info
                for (auto const &token : global_tyscount) {
                    ssid_t type_No = token.first;
                    if (type_No >= 0) continue;
                    type_t type = global_int2type[type_No];
                    if (type.data_type) {
                        for (auto const &single_type : type.composition) {
                            if (global_single2complex.find(single_type) != global_single2complex.end()) {
                                global_single2complex[single_type].insert(type_No);
                            } else {
                                unordered_set<ssid_t> multi_type_set;
                                // set will automatically ensure no duplicated element exist
                                multi_type_set.insert(type_No);
                                global_single2complex[single_type] = multi_type_set;
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < all_gather.size(); i++) {

                for (unordered_map<ssid_t, vector<ty_count>>::iterator it = all_gather[i].local_tystat.pstype.begin();
                        it != all_gather[i].local_tystat.pstype.end(); it++ ) {
                    ssid_t key = it->first;
                    vector<ty_count>& types = it->second;
                    for (size_t k = 0; k < types.size(); k++)
                        global_tystat.insert_stype(key,
                                                   //0,
                                                   types[k].ty < 0 ? type_transform(types[k].ty, all_gather[i]) : types[k].ty,
                                                   types[k].count);
                }

                for (unordered_map<ssid_t, vector<ty_count>>::iterator it = all_gather[i].local_tystat.potype.begin();
                        it != all_gather[i].local_tystat.potype.end(); it++ ) {
                    ssid_t key = it->first;
                    vector<ty_count>& types = it->second;
                    for (size_t k = 0; k < types.size(); k++)
                        global_tystat.insert_otype(key,
                                                   //0,
                                                   types[k].ty < 0 ? type_transform(types[k].ty, all_gather[i]) : types[k].ty,
                                                   types[k].count);
                }

                for (unordered_map<pair<ssid_t, ssid_t>, vector<ty_count>, boost::hash<pair<int, int>>>::iterator
                        it = all_gather[i].local_tystat.fine_type.begin();
                        it != all_gather[i].local_tystat.fine_type.end(); it++ ) {
                    pair<ssid_t, ssid_t> key = it->first;
                    vector<ty_count>& types = it->second;
                    for (size_t k = 0; k < types.size(); k++)
                        global_tystat.insert_finetype(
                            key.first < 0 ? type_transform(key.first, all_gather[i]) : key.first,
                            key.second < 0 ? type_transform(key.second, all_gather[i]) : key.second,
                            //(key.first < 0 || key.first > (1 << 17)) ? 0 : key.first ,
                            //(key.second < 0 || key.second > (1 << 17)) ? 0 : key.second ,
                            types[k].ty < 0 ? type_transform(types[k].ty, all_gather[i]) : types[k].ty,
                            //0,
                            types[k].count);
                }
            }

            // clear useless type in global_type2int
            unordered_map<type_t, ssid_t, type_t_hasher> type2int;
            for (auto const &token : global_useful_type) {
                type2int[global_int2type[token]] = token;
            }
            global_type2int.swap(type2int);

            logstream(LOG_INFO) << "global_tyscount size: " << global_tyscount.size() << LOG_endl;
            logstream(LOG_INFO) << "global_tystat.pstype.size: " << global_tystat.pstype.size() << LOG_endl;
            logstream(LOG_INFO) << "global_tystat.potype.size: " << global_tystat.potype.size() << LOG_endl;
            logstream(LOG_INFO) << "global_tystat.fine_type.size: " << global_tystat.fine_type.size() << LOG_endl;
            logstream(LOG_INFO) << "global_tyscount[0]: " << global_tyscount[0] << LOG_endl;
        }

        send_stat_to_all_machines(tcp_ad);

        logstream(LOG_INFO) << "#" << sid << ": load stats of DGraph is finished." << LOG_endl;

    }

    void load_stat_from_file(string fname, TCP_Adaptor *tcp_ad) {
        uint64_t t1 = timer::get_usec();

        // master server loads statistics and dispatchs them to all slave servers
        if (sid == 0) {
            ifstream file(fname.c_str());
            if (!file.good()) {
                logstream(LOG_WARNING) << "statistics file "  << fname
                                       << " does not exsit, pleanse check the fname"
                                       << " and use load-stat to mannually set it"  << LOG_endl;

                /// FIXME: HANG bug if master return here
                return;
            }

            ifstream ifs(fname);
            boost::archive::binary_iarchive ia(ifs);
            ia >> global_tyscount;
            ia >> global_tystat;
            ia >> global_type2int;
            ia >> global_single2complex;
            ifs.close();
        }

        send_stat_to_all_machines(tcp_ad);

        uint64_t t2 = timer::get_usec();
        logstream(LOG_INFO) << (t2 - t1) / 1000 << " ms for loading statistics"
                            << " at server " << sid << LOG_endl;

    }

    void store_stat_to_file(string fname) {
        // data only cached on master server
        if (sid != 0) return;

        // avoid saving when it already exsits
        ifstream file(fname.c_str());
        if (!file.good()) {
            try {
                ofstream ofs(fname);
                boost::archive::binary_oarchive oa(ofs);
                oa << global_tyscount;
                oa << global_tystat;
                oa << global_type2int;
                oa << global_single2complex;
                ofs.close();
            } catch (exception& e) {
                logstream(LOG_ERROR) << "store statistics unsuccessfully: " << e.what() << LOG_endl;
                return;
            }

            logstream(LOG_INFO) << "store statistics to file "
                                << fname << " is finished." << LOG_endl;
        }
    }

    ssid_t get_simple_type(type_t &type) {
        auto iter = local_type2int.find(type);

        if (iter == local_type2int.end()) {
            ssid_t number = local_type2int.size();
            number ++;
            number = -number;
            local_type2int[type] = number;
            local_int2type[number] = type;
            return number;
        } else {
            return iter->second;
        }
    }

    // prepare data for planner
    void generate_statistic(GStore *gstore) {

        // find if the same raw type have similar predicates
        // unordered_map<ssid_t, unordered_set<type_t,type_t_hasher>> rawType_to_predicates;
        // unordered_map<type_t, int, type_t_hasher> each_predicate_number;

#ifndef VERSATILE
        logstream(LOG_ERROR) << "please turn off generate_statistics in config "
                             << "and use stat file cache instead"
                             << " OR "
                             << "turn on VERSATILE option in CMakefiles to generate_statistic." << LOG_endl;
        exit(-1);
#endif

        unordered_map<ssid_t, int> &tyscount = local_tyscount;
        type_stat &ty_stat = local_tystat;
        // for complex type vertex numbering
        unordered_set<ssid_t> record_set;

        //use index_composition as type of no_type
        auto generate_no_type = [&](ssid_t id) -> ssid_t {
            type_t type;
            uint64_t psize1 = 0;
            unordered_set<int> index_composition;

            edge_t *res1 = gstore->get_edges(0, id, PREDICATE_ID, OUT, psize1);
            for (uint64_t k = 0; k < psize1; k++) {
                ssid_t pre = res1[k].val;
                index_composition.insert(pre);
            }

            uint64_t psize2 = 0;
            edge_t *res2 = gstore->get_edges(0, id, PREDICATE_ID, IN, psize2);
            for (uint64_t k = 0; k < psize2; k++) {
                ssid_t pre = res2[k].val;
                index_composition.insert(-pre);
            }

            type.set_index_composition(index_composition);
            // TODO: there should be no following situation according to comments
            // on gstore layout, but actually it happends 25 times and will not affect
            // the correctness of optimizer
            // if(index_composition.size() == 0){
            //     cout << "empty index, may be type" << endl;
            // }
            return get_simple_type(type);
        };

        //use type_composition as type of no_type
        auto generate_multi_type = [&](edge_t *res, uint64_t type_sz) -> ssid_t {
            type_t type;
            unordered_set<int> type_composition;
            for (int i = 0; i < type_sz; i ++)
                type_composition.insert(res[i].val);

            type.set_type_composition(type_composition);
            return get_simple_type(type);
        };

        // return success or not, because one id can only be recorded once
        auto insert_no_type_count = [&](ssid_t id, ssid_t type) -> bool{
            if (record_set.count(id) > 0) {
                return false;
            } else{
                record_set.insert(id);

                if (tyscount.find(type) == tyscount.end())
                    tyscount[type] = 1;
                else
                    tyscount[type]++;
                return true;
            }
        };

        int percent_number = 1;
        for (uint64_t bucket_id = 0; bucket_id < gstore->num_buckets + gstore->num_buckets_ext; bucket_id++) {
            // print progress percent info
            if (bucket_id * 1.0 / (gstore->num_buckets + gstore->num_buckets_ext) > percent_number * 1.0 / 10) {
                logstream(LOG_INFO) << "#" << sid << ": already generate statistics " << percent_number << "0%" << LOG_endl;
                percent_number ++;
            }

            uint64_t slot_id = bucket_id * gstore->ASSOCIATIVITY;
            for (int i = 0; i < gstore->ASSOCIATIVITY - 1; i++, slot_id++) {
                // skip empty slot
                if (gstore->vertices[slot_id].key.is_empty()) continue;

                sid_t vid = gstore->vertices[slot_id].key.vid;
                sid_t pid = gstore->vertices[slot_id].key.pid;

                uint64_t sz = gstore->vertices[slot_id].ptr.size;
                uint64_t off = gstore->vertices[slot_id].ptr.off;
                if (vid == PREDICATE_ID || pid == PREDICATE_ID)
                    continue; // skip for index vertex

                if (gstore->vertices[slot_id].key.dir == IN) {
                    // for type derivation
                    // get types of values found by key (Subjects)
                    vector<ssid_t> res_type;
                    for (uint64_t k = 0; k < sz; k++) {
                        ssid_t sbid = gstore->edges[off + k].val;
                        uint64_t type_sz = 0;
                        edge_t *res = gstore->get_edges(0, sbid, TYPE_ID, OUT, type_sz);
                        if (type_sz > 1) {
                            ssid_t type = generate_multi_type(res, type_sz);
                            res_type.push_back(type);
                        } else if (type_sz == 0) {
                            ssid_t type = generate_no_type(sbid);
                            res_type.push_back(type);
                        } else if (type_sz == 1) {
                            res_type.push_back(res[0].val);
                        } else {
                            assert(false);
                        }
                    }

                    // type for objects
                    // get type of vid (Object)
                    uint64_t type_sz = 0;
                    edge_t *res = gstore->get_edges_local(0, vid, TYPE_ID, OUT, type_sz);
                    ssid_t type;
                    if (type_sz > 1) {
                        type = generate_multi_type(res, type_sz);
                    } else {
                        if (type_sz == 0) {
                            type = generate_no_type(vid);
                            insert_no_type_count(vid, type);
                        } else {
                            type = res[0].val;
                        }
                    }

                    ty_stat.insert_otype(pid, type, 1);
                    for (int j = 0; j < res_type.size(); j++)
                        ty_stat.insert_finetype(pid, type, res_type[j], 1);
                } else {
                    // no_type only need to be counted in one direction (using OUT)
                    // get types of values found by key (Objects)
                    vector<ssid_t> res_type;
                    for (uint64_t k = 0; k < sz; k++) {
                        ssid_t obid = gstore->edges[off + k].val;
                        uint64_t type_sz = 0;
                        edge_t *res = gstore->get_edges(0, obid, TYPE_ID, OUT, type_sz);

                        if (type_sz > 1) {
                            ssid_t type = generate_multi_type(res, type_sz);
                            res_type.push_back(type);
                        } else if (type_sz == 0) {
                            // in this situation, obid may be some TYPE
                            if (pid != 1) {
                                ssid_t type = generate_no_type(obid);
                                res_type.push_back(type);
                            }
                        } else if (type_sz == 1) {
                            res_type.push_back(res[0].val);
                        } else {
                            assert(false);
                        }
                    }

                    // type for subjects
                    // get type of vid (Subject)
                    uint64_t type_sz = 0;
                    edge_t *res = gstore->get_edges_local(0, vid, TYPE_ID, OUT, type_sz);
                    ssid_t type;
                    if (type_sz > 1) {
                        type = generate_multi_type(res, type_sz);
                    } else {
                        if (type_sz == 0) {
                            type = generate_no_type(vid);
                            insert_no_type_count(vid, type);
                        } else {
                            type = res[0].val;
                        }
                    }

                    ty_stat.insert_stype(pid, type, 1);
                    for (int j = 0; j < res_type.size(); j++)
                        ty_stat.insert_finetype(type, pid, res_type[j], 1);

                    // count type predicate
                    if (pid == TYPE_ID) {
                        // multi-type
                        if (sz > 1) {
                            type_t complex_type;
                            unordered_set<int> type_composition;
                            for (int i = 0; i < sz; i ++)
                                type_composition.insert(gstore->edges[off + i].val);

                            complex_type.set_type_composition(type_composition);
                            ssid_t type_number = get_simple_type(complex_type);

                            if (tyscount.find(type_number) == tyscount.end())
                                tyscount[type_number] = 1;
                            else
                                tyscount[type_number]++;
                        } else if (sz == 1) { // single type
                            sid_t obid = gstore->edges[off].val;

                            if (tyscount.find(obid) == tyscount.end())
                                tyscount[obid] = 1;
                            else
                                tyscount[obid]++;
                        }
                    }
                }
            }
        }

        logstream(LOG_INFO) << "server#" << sid << ": generating stats is finished." << endl;
    }
};
