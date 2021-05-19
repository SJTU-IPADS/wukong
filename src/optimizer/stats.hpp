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
#define DEFAULT_TYPE 0

#include <unordered_map>
#include <unordered_set>
#include <omp.h>

#include <boost/mpi.hpp>
#include <boost/algorithm/string.hpp>

#include <tbb/concurrent_unordered_set.h>

#include "core/common/global.hpp"

#include "core/network/tcp_adaptor.hpp"

#include "optimizer/stats_type.hpp"

namespace wukong {

class Stats {
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

            for (int i = 1; i < Global::num_servers; i++)
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

    std::unordered_map<ssid_t, int> local_tyscount;
    std::unordered_map<ssid_t, int> global_tyscount;

    type_stat local_tystat;
    type_stat global_tystat;

    // use negative numbers to represent complex types
    // (type_composition and index_composition)
    std::unordered_map<ssid_t, type_t> local_int2type;
    std::unordered_map<type_t, ssid_t, type_t_hasher> local_type2int;
    std::unordered_map<ssid_t, type_t> global_int2type;  //not used in planner.hpp currently
    std::unordered_map<type_t, ssid_t, type_t_hasher> global_type2int;

    // single type may be contained by several multitype
    std::unordered_map<ssid_t, std::unordered_set<ssid_t>> global_single2complex;

    std::unordered_set<ssid_t> global_useful_type;

    int sid;

    Stats(int sid) : sid(sid) { }

    Stats() { }

    // reduce number of types to speed up planning procedure
    // sacrifice accuracy in change for speed
    // MODIFICATIONS:
    // global_tyscount: useful_type count
    // global_type2int: all type to its type_No
    // global_single2complex: single to useful_multipletype
    // return: number of types removed in merge operation
    void merge_type() {
        uint64_t total_number = 0;
        std::map<int, ssid_t> tys;
        int minimum_count = 0;
        for (auto const &token : global_tyscount) {
            total_number += token.second;
            if (tys.find(token.second) == tys.end())
                tys[token.second] = 1;
            else
                tys[token.second]++;
        }

        uint64_t sum = 0;

        for (auto const &token : tys) {
            sum += token.first * token.second;
            if (sum >= total_number * TYPE_REMOVE_RATE) {
                minimum_count = token.first;
                break;
            }
        }

        // std::cout << "minimum_count: " << minimum_count << std::endl;

        tbb_map new_type2int;
        // type of which has too few vertices (among notype & multitype)
        std::unordered_set<ssid_t> global_useless_type;
        for (auto const &token : global_tyscount) {
            // global_useless_type.insert(token.first);

            // generated type && vertices of this type less than threshold
            if (token.first < 0 && token.second < minimum_count)
                global_useless_type.insert(token.first);
            else
                global_useful_type.insert(token.first);
        }

        // useful type2int
        std::unordered_map<type_t, ssid_t, type_t_hasher> type2int_new;
        /**
         * remove type of which has too few vertices (among notype & multitype)
         * Use 'ratio of min tyscount to max tyscount' to judge
         * whether a type is rare.
         * Other methods can't limit types number finely grained
         * because of uneven tyscount's density.
         */

        for (auto const &token : global_tyscount) {
            // generated type && vertices of this type less than threshold
            if (token.first < 0 && token.second < minimum_count)
                global_useless_type.insert(token.first);
            else
                global_useful_type.insert(token.first);
        }

        for (auto const &token : global_useless_type) {
            type2int_new[global_int2type[token]] = DEFAULT_TYPE;
        }

        for (auto const &token : global_useful_type) {
            type2int_new[global_int2type[token]] = token;
        }

        // update global_tyscount
        std::unordered_map<ssid_t, int> tyscount;
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
                        std::unordered_set<ssid_t> multi_type_set;
                        // set will automatically ensure no duplicated element exist
                        multi_type_set.insert(type_No);
                        global_single2complex[single_type] = multi_type_set;
                    }
                }
            }
        }

        // update global_type2int
        global_type2int.swap(type2int_new);
        return ;
    }

    void gather_stat(TCP_Adaptor *tcp_ad) {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);
        oa << (*this);
        tcp_ad->send(0, 0, ss.str());

        if (sid == 0) {
            std::vector<Stats> all_gather;
            // complex type have different corresponding number on different machine
            // assume type < 0 here
            auto type_transform = [&](ssid_t type_No, Stats & stat) -> ssid_t{

                type_t complex_type;
                if (stat.local_int2type.find(type_No) != stat.local_int2type.end())
                    complex_type = stat.local_int2type[type_No];
                else
                    logstream(LOG_ERROR) << "type_No: " << type_No << " is not in local_int2type" << LOG_endl;

                if (global_type2int.find(complex_type) != global_type2int.end())
                    return global_type2int[complex_type];
                else {
                    logstream(LOG_ERROR) << "type not found, size " << complex_type.composition.size() << LOG_endl;
                    return 0;
                }
            };

            // receive from all proxies
            for (int i = 0; i < Global::num_servers; i++) {
                std::string str;
                str = tcp_ad->recv(0);
                Stats tmp_data;
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
                                std::unordered_set<ssid_t> multi_type_set;
                                // set will automatically ensure no duplicated element exist
                                multi_type_set.insert(type_No);
                                global_single2complex[single_type] = multi_type_set;
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < all_gather.size(); i++) {

                for (std::unordered_map<ssid_t, std::vector<ty_count>>::iterator it = all_gather[i].local_tystat.pstype.begin();
                        it != all_gather[i].local_tystat.pstype.end(); it++ ) {
                    ssid_t key = it->first;
                    std::vector<ty_count>& types = it->second;
                    for (size_t k = 0; k < types.size(); k++){
                        if(types[k].ty <= DEFAULT_TYPE){
                            ssid_t new_type = type_transform(types[k].ty, all_gather[i]);
                            if(new_type!=DEFAULT_TYPE)
                                global_tystat.insert_stype(key, new_type, types[k].count);
                            else{
                                global_tystat.insert_stype(key, DEFAULT_TYPE, types[k].count);
                            }
                        }
                        else
                            global_tystat.insert_stype(key, types[k].ty, types[k].count);
                    }
                        
                }

                for (std::unordered_map<ssid_t, std::vector<ty_count>>::iterator it = all_gather[i].local_tystat.potype.begin();
                        it != all_gather[i].local_tystat.potype.end(); it++ ) {
                    ssid_t key = it->first;
                    std::vector<ty_count>& types = it->second;
                    for (size_t k = 0; k < types.size(); k++)
                        if (types[k].ty <= DEFAULT_TYPE){
                            ssid_t new_type = type_transform(types[k].ty, all_gather[i]);
                            if(new_type!=DEFAULT_TYPE)
                                global_tystat.insert_otype(key, new_type, types[k].count);
                            else
                                global_tystat.insert_otype(key, DEFAULT_TYPE, types[k].count);
                        }
                        else
                            global_tystat.insert_otype(key, types[k].ty, types[k].count);
                }

                for (std::unordered_map<std::pair<ssid_t, ssid_t>, std::vector<ty_count>, boost::hash<std::pair<int, int>>>::iterator
                        it = all_gather[i].local_tystat.fine_type.begin();
                        it != all_gather[i].local_tystat.fine_type.end(); it++ ) {
                    std::pair<ssid_t, ssid_t> key = it->first;
                    std::vector<ty_count>& types = it->second;
                    for (size_t k = 0; k < types.size(); k++){
                        double temp_count = types[k].count;
                        ssid_t new_type1;
                        ssid_t new_type2 = types[k].ty;
                        if(types[k].ty < 0){
                            new_type2 = type_transform(types[k].ty, all_gather[i]);
                        }
                        if(key.first < 0)  {
                            new_type1 = type_transform(key.first, all_gather[i]);
                            global_tystat.insert_finetype(new_type1, key.second, new_type2, types[k].count);
                        }
                        else if (key.second < 0) {
                            new_type1 = type_transform(key.second, all_gather[i]);
                            global_tystat.insert_finetype(key.first, new_type1, new_type2, types[k].count);
                        }
                        else {
                            global_tystat.insert_finetype(key.first, key.second, new_type2, types[k].count);
                        }
                    }
                }
            }

            // clear useless type in global_type2int
            std::unordered_map<type_t, ssid_t, type_t_hasher> type2int;
            for (auto const &token : global_useful_type) {
                type2int[global_int2type[token]] = token;
            }
            global_type2int.swap(type2int);

            logstream(LOG_INFO) << "[Stat] global_tyscount size: " << global_tyscount.size() << LOG_endl;
            logstream(LOG_INFO) << "[Stat] global_tystat.pstype.size: " << global_tystat.pstype.size() << LOG_endl;
            logstream(LOG_INFO) << "[Stat] global_tystat.potype.size: " << global_tystat.potype.size() << LOG_endl;
            logstream(LOG_INFO) << "[Stat] global_tystat.fine_type.size: " << global_tystat.fine_type.size() << LOG_endl;
            logstream(LOG_INFO) << "[Stat] global_tyscount[0]: " << global_tyscount[0] << LOG_endl;
        }

        send_stat_to_all_machines(tcp_ad);

        logstream(LOG_INFO) << "[Stats] #" << sid << ": load stats of DGraph is finished." << LOG_endl;

    }

    void load_stat_from_file(std::string fname, TCP_Adaptor *tcp_ad) {
        uint64_t t1 = timer::get_usec();

        // master server loads statistics and dispatchs them to all slave servers
        if (sid == 0) {
            std::ifstream file(fname.c_str());
            if (!file.good()) {
                logstream(LOG_WARNING) << "statistics file "  << fname
                                       << " does not exsit, pleanse check the fname"
                                       << " and use load-stat to mannually set it"  << LOG_endl;

                /// FIXME: HANG bug if master return here
                return;
            }

            std::ifstream ifs(fname);
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

    void store_stat_to_file(std::string fname) {
        // data only cached on master server
        if (sid != 0) return;

        // avoid saving when it already exsits
        std::ifstream file(fname.c_str());
        if (!file.good()) {
            try {
                std::ofstream ofs(fname);
                boost::archive::binary_oarchive oa(ofs);
                oa << global_tyscount;
                oa << global_tystat;
                oa << global_type2int;
                oa << global_single2complex;
                ofs.close();
            } catch (std::exception& e) {
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
            number++;
            number = -number;
            local_type2int[type] = number;
            local_int2type[number] = type;
            return number;
        } else {
            return iter->second;
        }
    }

    // prepare data for planner
    void generate_statistics(DGraph *graph) {
        logstream(LOG_INFO) << "[Stats] #" << sid << ": begin to generate statistics..." << LOG_endl;
        // find if the same raw type have similar predicates
        // unordered_map<ssid_t, unordered_set<type_t,type_t_hasher>> rawType_to_predicates;
        // unordered_map<type_t, int, type_t_hasher> each_predicate_number;

#ifndef VERSATILE
        logstream(LOG_ERROR) << "please turn off global_generate_statistics in config file"
                             << " and use stat file cache instead"
                             << " OR "
                             << "turn on VERSATILE option in CMakefiles to generate statistics."
                             << LOG_endl;
        exit(-1);
#endif

        if(!Global::use_rdma){
            logstream(LOG_ERROR) << "Currently, wukong doesn't support to generate statistics without RDMA." << LOG_endl;
            logstream(LOG_ERROR) << "Please turn off global_generate_statistics in config file"
                             << " and use stat file cache instead"
                             << " OR "
                             << "turn on global_use_rdma in config file to generate statistics."
                             << LOG_endl;
            return;
        }

        std::unordered_map<ssid_t, int> &tyscount = local_tyscount;
        type_stat &ty_stat = local_tystat;
        // for complex type vertex numbering
        std::unordered_set<ssid_t> record_set;

        //use index_composition as type of no_type
        auto generate_no_type = [&](int tid, ssid_t id) -> ssid_t {
            type_t type;
            uint64_t psize1 = 0;
            std::unordered_set<int> index_composition;

            edge_t * res1 = graph->get_triples(tid, id, PREDICATE_ID, OUT, psize1);
            for (uint64_t k = 0; k < psize1; k++) {
                ssid_t pre = res1[k].val;
                index_composition.insert(pre);
            }

            uint64_t psize2 = 0;
            edge_t * res2 = graph->get_triples(tid, id, PREDICATE_ID, IN, psize2);
            for (uint64_t k = 0; k < psize2; k++) {
                ssid_t pre = res2[k].val;
                index_composition.insert(-pre);
            }

            type.set_index_composition(index_composition);
            // TODO: there should be no following situation according to comments
            // on gstore layout, but actually it happends 25 times and will not affect
            // the correctness of optimizer
            // if(index_composition.size() == 0){
            //     std::cout << "empty index, may be type" << std::endl;
            // }
            ssid_t result;
            #pragma omp critical (int_type_mapping)
            result = get_simple_type(type);
            return result;
        };

        //use type_composition as type of no_type
        auto generate_multi_type = [&](edge_t *res, uint64_t type_sz) -> ssid_t {
            type_t type;
            std::unordered_set<int> type_composition;
            for (int i = 0; i < type_sz; i ++)
                type_composition.insert(res[i].val);

            type.set_type_composition(type_composition);
            ssid_t result;
            #pragma omp critical (int_type_mapping)
            result = get_simple_type(type);
            return result;
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

        for(sid_t pid : graph->get_edge_predicates()){
            if(pid == TYPE_ID) continue;
            for(int dir = 0; dir < 2; dir++){
                uint64_t vertex_sz = 0;
                edge_t *vertices = graph->get_index(0, pid, dir==0?IN:OUT, vertex_sz);
                #pragma omp parallel for num_threads(Global::num_engines)
                for(int i = 0; i < vertex_sz; i++){
                    int tid = omp_get_thread_num();
                    int vid = vertices[i].val;
                    uint64_t sz;
                    edge_t *neighbors = graph->get_triples(tid, vid, pid, dir==0?OUT:IN, sz);
                    // no_type only need to be counted in one direction (using OUT)
                    // get types of values found by key
                    std::vector<ssid_t> res_type;
                    for (uint64_t k = 0; k < sz; k++) {
                        ssid_t neighbor_id = neighbors[k].val;
                        uint64_t type_sz = 0;
                        edge_t *res = graph->get_triples(tid, neighbor_id, TYPE_ID, OUT, type_sz);
                        if (type_sz > 1) {
                            ssid_t type = generate_multi_type(res, type_sz);
                            res_type.push_back(type);
                        } else if (type_sz == 0) {
                            ssid_t type = generate_no_type(tid, neighbor_id);
                            res_type.push_back(type);
                        } else if (type_sz == 1) {
                            res_type.push_back(res[0].val);
                        }
                    }

                    // type for subjects/objects
                    // get type of vid
                    uint64_t type_sz = 0;
                    edge_t *res = graph->get_triples(tid, vid, TYPE_ID, OUT, type_sz);
                    ssid_t type;
                    if (type_sz > 1) {
                        type = generate_multi_type(res, type_sz);
                    } else {
                        if (type_sz == 0) {
                            type = generate_no_type(tid, vid);
                            #pragma omp critical (local_tyscount)
                            insert_no_type_count(vid, type);
                        } else {
                            type = res[0].val;
                        }
                    }
                    #pragma omp critical (local_tystat)
                    {
                    if(dir == 0){
                        ty_stat.insert_stype(pid, type, 1);
                        for (int j = 0; j < res_type.size(); j++)
                            ty_stat.insert_finetype(type, pid, res_type[j], 1);
                    }
                    else {
                        ty_stat.insert_otype(pid, type, 1);
                        for (int j = 0; j < res_type.size(); j++)
                            ty_stat.insert_finetype(pid, type, res_type[j], 1);
                    }
                    }
                }
            }
        }

        tbb::concurrent_unordered_set<sid_t> vids;
        for(sid_t tid : graph->get_type_predicates()){
            uint64_t vertex_sz = 0;
            edge_t *vertices = graph->get_index(0, tid, IN, vertex_sz);
            #pragma omp parallel for num_threads(Global::num_engines)
            for(int i = 0; i < vertex_sz; i++){
                sid_t vid = vertices[i].val;
                if(vids.count(vid)){
                    continue;
                }else{
                    vids.insert(vid);
                }
                // type for subjects
                // get type of vid (Subject)
                uint64_t type_sz = 0;
                edge_t *res = graph->get_triples(tid, vid, TYPE_ID, OUT, type_sz);
                ssid_t type;
                // multi-type
                if (type_sz > 1) {
                    type = generate_multi_type(res, type_sz);
                } else { // single type
                    assert(type_sz != 0);
                    type = res[0].val;
                }
                // count type predicate
                #pragma omp critical (local_tyscount)
                {
                if (tyscount.find(type) == tyscount.end())
                    tyscount[type] = 1;
                else
                    tyscount[type]++;
                }
                #pragma omp critical (local_tystat)
                {
                ty_stat.insert_stype(TYPE_ID, type, 1);
                // for (int j = 0; j < res_type.size(); j++)
                //     ty_stat.insert_finetype(type, TYPE_ID, res_type[j], 1);
                }
            }
        }

        logstream(LOG_INFO) << "[Stats] #" << sid << ": generating stats is finished." << LOG_endl;
    }

    /**
     * find popular stype, prediate, otype in finetype map
     */
    int get_popular_pattern(std::vector <ssid_t> & stype, std::vector <ssid_t> &p, std::vector <ssid_t> &otype, int size_max=10) {
        for (std::unordered_map<std::pair<ssid_t, ssid_t>, std::vector<ty_count>,
                           boost::hash<std::pair<int, int>>>::iterator it =
                 global_tystat.fine_type.begin();
             it != global_tystat.fine_type.end(); it++) {
            std::pair<ssid_t, ssid_t> key = it->first;
            if (key.first <= 0 || key.second <= 0) continue;
            std::vector<ty_count> &types = it->second;
            for (size_t k = 0; k < types.size(); k++) {
                if (types[k].ty <= 0)
                    continue;
                int temp_count = types[k].count;
                if (temp_count < 30000) {
                    // fine type is <stype, p>
                    if (global_tystat.get_pstype_count(key.second, key.first) > 0){
                        stype.push_back(key.first);
                        p.push_back(key.second);
                        otype.push_back(types[k].ty);
                    } else {  // fine type is <p, otype>
                        stype.push_back(types[k].ty);
                        p.push_back(key.first);
                        otype.push_back(key.second);
                    }
                    if (stype.size() >= size_max) return size_max;
                    break;
                }
            }
        }
        return stype.size();
    }

    /**
     * find three-node circle in stats, use specific algorithm
     */
    int get_circle_pattern(std::vector<ssid_t> &result, int& clockwise_count, int size_max = 100) {
        std::unordered_map<ssid_t, std::vector<ssid_t>> type_out;
        std::unordered_map<ssid_t, std::vector<ssid_t>> type_in;

        for (std::unordered_map<ssid_t, std::vector<ty_count>>::iterator it = global_tystat.potype.begin();
             it != global_tystat.potype.end(); it++) {
            ssid_t p = it->first;
            std::vector<ty_count> &types = it->second;
            for (size_t k = 0; k < types.size(); k++) {
                if (types[k].ty <= 0)
                    continue;
                type_in[types[k].ty].push_back(p);
            }
        }
        for (std::unordered_map<ssid_t, std::vector<ty_count>>::iterator it = global_tystat.pstype.begin();
             it != global_tystat.pstype.end(); it++) {
            ssid_t p = it->first;
            std::vector<ty_count> &types = it->second;
            for (size_t k = 0; k < types.size(); k++) {
                if (types[k].ty <= 0)
                    continue;
                type_out[types[k].ty].push_back(p);
            }
        }
        clockwise_count = 0;

        ssid_t t1, t2, t3;
        ssid_t p1, p2, p3;
        for (std::unordered_map<ssid_t, std::vector<ssid_t>>::iterator it1 = type_out.begin();
             it1 != type_out.end(); it1++) {
            t1 = it1->first;
            if(type_in.find(t1)==type_in.end())
                continue;
            std::vector<ssid_t> ps1 = it1->second;
            for(int i = 0;i < ps1.size();i++){
                p1 = ps1[i];
                if(global_tystat.get_pstype_count(p1, t1) > 50000) continue;

                // t1 -- p1 -> t2 -- p2 -> t3 -- p3 > t1
                std::vector<ty_count> ts2 = global_tystat.fine_type[std::make_pair(t1, p1)];
                for (int j = 0; j < ts2.size(); j++) {
                    t2 = ts2[j].ty;
                    if (t2==t1 || t2 <= 0||type_out.find(t2) == type_out.end()) continue;
                    std::vector<ssid_t> ps2 = type_out[t2];
                    for (int k = 0; k < ps2.size(); k++) {
                        p2 = ps2[k];
                        if (p2 == p1) continue;
                        std::vector<ty_count> ts3 = global_tystat.fine_type[std::make_pair(t2, p2)];
                        for (int x = 0; x < ts3.size(); x++){
                            t3 = ts3[x].ty;
                            if (t3 <= 0||type_out.find(t3) == type_out.end()) continue;
                            std::vector<ssid_t> ps3 = type_out[t3];
                            for (int y = 0; y < ps3.size(); y++) {
                                p3 = ps3[y];
                                if(global_tystat.fine_type.find(std::make_pair(p3, t1))!=global_tystat.fine_type.end()){
                                    result.push_back(t1);result.push_back(t2);result.push_back(t3);
                                    result.push_back(p1);result.push_back(p2);result.push_back(p3);
                                    clockwise_count++;
                                    // logstream(LOG_INFO) << clockwise_count << LOG_endl;
                                    if(clockwise_count >= size_max) return result.size()/6;
                                }
                            }
                        }
                    }
                }

                // t1 -- p1 -> t2 <- p2 -- t3 -- p3 > t1
                for (int j = 0; j < ts2.size(); j++) {
                    t2 = ts2[j].ty;
                    if (t2==t1 || t2 <= 0||type_in.find(t2) == type_in.end()) continue;

                    std::vector<ssid_t> ps2 = type_in[t2];
                    for (int k = 0; k < ps2.size(); k++) {
                        p2 = ps2[k];
                        if(p2==p1)  continue;
                        std::vector<ty_count> ts3 = global_tystat.fine_type[std::make_pair(p2, t2)];
                        for (int x = 0; x < ts3.size(); x++){
                            t3 = ts3[x].ty;
                            if (t3 <= 0||type_out.find(t3) == type_out.end()) continue;
                            std::vector<ssid_t> ps3 = type_out[t3];
                            for (int y = 0; y < ps3.size(); y++) {
                                p3 = ps3[y];
                                if(global_tystat.fine_type.find(std::make_pair(p3, t1))!=global_tystat.fine_type.end()){
                                    result.push_back(t1);result.push_back(t2);result.push_back(t3);
                                    result.push_back(p1);result.push_back(p2);result.push_back(p3);
                                    // logstream(LOG_INFO) << result.size()/6 << LOG_endl;
                                    if (result.size()/6 >= size_max) return result.size()/6;
                                }
                            }
                        }
                    }
                }
            }
        }
        return result.size() / 6;
    }
};

} // namespace wukong