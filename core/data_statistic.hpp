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

#include "tcp_adaptor.hpp"
#include "config.hpp"

using namespace std;

struct type_t{
    bool data_type;   //true for type_composition, false for index_composition
    std::unordered_set<int> composition;

    void set_type_composition(std::unordered_set<int> composition){
        data_type = true;
        this->composition = composition;
    }

    void set_index_composition(std::unordered_set<int> composition){
        data_type = false;
        this->composition = composition;
    };
    bool operator==(const type_t &other) const{
        if(data_type != other.data_type) return false;
        return this->composition == other.composition;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & data_type;
        ar & composition;
    }
};

struct type_t_hasher{
    size_t operator()( const type_t& type ) const{
        size_t res = 17;
        for ( auto it = type.composition.cbegin(); it != type.composition.cend(); ++it )
            res = (res + *it) << 1;
        return res;
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
        vector<ty_count>& types = pstype[predicate];
        for (size_t i = 0; i < types.size(); i++) {
            if (types[i].ty == type) {
                return types[i].count;
            }
        }
        return 0;
    }
    int get_potype_count(ssid_t predicate, ssid_t type) {
        vector<ty_count>& types = potype[predicate];
        for (size_t i = 0; i < types.size(); i++) {
            if (types[i].ty == type) {
                return types[i].count;
            }
        }
        return 0;
    }
    int insert_stype(ssid_t predicate, ssid_t type, int count) {
        vector<ty_count>& types = pstype[predicate];
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
        vector<ty_count>& types = potype[predicate];
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
    int insert_finetype(ssid_t first, ssid_t second, ssid_t type, int count){
        vector<ty_count>& types = fine_type[make_pair(first, second)];
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

class data_statistic {
private:
    // after the master server get whole statistics, this method is used to send it to all machines.
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
    unordered_map<ssid_t, int> local_tyscount;
    unordered_map<ssid_t, int> global_tyscount;

    type_stat local_tystat;
    type_stat global_tystat;

    // use negative numbers to represent complex types (type_composition and index_composition)
    unordered_map<ssid_t, type_t> local_int2type;    
    unordered_map<type_t, ssid_t, type_t_hasher> local_type2int;
    unordered_map<type_t, ssid_t, type_t_hasher> global_type2int;

    // single type may be contained by several multitype
    unordered_map<ssid_t, unordered_set<ssid_t>> global_single2complex;    

    int sid;

    data_statistic(int _sid) : sid(_sid) { }

    data_statistic() { }

    void gather_stat(TCP_Adaptor *tcp_ad) {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);
        oa << (*this);
        tcp_ad->send(0, 0, ss.str());

        if (sid == 0) {
            vector<data_statistic> all_gather;
            unordered_map<ssid_t, type_t> global_int2type;    

            // complex type have different corresponding number on different machine
            // assume type < 0 here
            auto type_transform = [&](ssid_t type, data_statistic &stat) -> ssid_t{
                // type may not occur in local_int2type

                type_t complex_type;
                if(local_int2type.find(type) != stat.local_int2type.end()){
                    complex_type = stat.local_int2type[type];
                }
                else{
                    logstream(LOG_ERROR) << "type: " << type << " is not in local_int2type" << LOG_endl;
                }
                if(global_type2int.find(complex_type) != global_type2int.end()){
                    return global_type2int[complex_type];
                }
                else{
                    ssid_t number = global_type2int.size();
                    number ++;
                    number = -number;
                    global_type2int[complex_type] = number;
                    global_int2type[number] = complex_type;

                    // debug
                    // cout << "number: " << number << endl;
                    // cout << "data_type: " << complex_type.data_type << endl;
                    // cout << "size: " << complex_type.composition.size() << endl;
                    // for ( auto it = complex_type.composition.cbegin(); it != complex_type.composition.cend(); ++it )
                    //     cout << *it << " ";
                    // cout << endl;

                    // add single2complex index
                    if(complex_type.data_type){
                        for ( auto iter = complex_type.composition.cbegin(); iter != complex_type.composition.cend(); ++iter ){
                            if(global_single2complex.find(*iter) != global_single2complex.end())
                                global_single2complex[*iter].insert(number);
                            else{
                                unordered_set<ssid_t> multi_type_set;
                                // set will automatically ensure no duplicated element exist
                                multi_type_set.insert(number);
                                global_single2complex[*iter] = multi_type_set;
                            }
                        }
                    }
                    return number;
                }
            };
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

            for (int i = 0; i < all_gather.size(); i++) {

                //for type predicate
                for (unordered_map<ssid_t, int>::iterator it = all_gather[i].local_tyscount.begin();
                        it != all_gather[i].local_tyscount.end(); it++ ) {
                    ssid_t key = it->first;
                    int number = it->second;
                    if(key < 0) key = type_transform(key, all_gather[i]);
                    if (global_tyscount.find(key) == global_tyscount.end()) {
                        global_tyscount[key] = number;
                    } else {
                        global_tyscount[key] += number;
                    }
                }

                for (unordered_map<ssid_t, vector<ty_count>>::iterator it = all_gather[i].local_tystat.pstype.begin();
                        it != all_gather[i].local_tystat.pstype.end(); it++ ) {
                    ssid_t key = it->first;
                    vector<ty_count>& types = it->second;
                    for (size_t k = 0; k < types.size(); k++)
                        global_tystat.insert_stype(key, 
                                                    types[k].ty<0 ? type_transform(types[k].ty, all_gather[i]) : types[k].ty,
                                                    types[k].count);
                }
                for (unordered_map<ssid_t, vector<ty_count>>::iterator it = all_gather[i].local_tystat.potype.begin();
                        it != all_gather[i].local_tystat.potype.end(); it++ ) {
                    ssid_t key = it->first;
                    vector<ty_count>& types = it->second;
                    for (size_t k = 0; k < types.size(); k++)
                        global_tystat.insert_otype(key, 
                                                    types[k].ty<0 ? type_transform(types[k].ty, all_gather[i]) : types[k].ty,
                                                    types[k].count);
                }
                for (unordered_map<pair<ssid_t, ssid_t>, vector<ty_count>, boost::hash<pair<int, int> > >::iterator 
                        it = all_gather[i].local_tystat.fine_type.begin();
                        it != all_gather[i].local_tystat.fine_type.end(); it++ ) {
                    pair<ssid_t, ssid_t> key = it->first;
                    vector<ty_count>& types = it->second;
                    for (size_t k = 0; k < types.size(); k++)
                        global_tystat.insert_finetype(key.first<0 ? type_transform(key.first, all_gather[i]) : key.first,
                                                      key.second<0 ? type_transform(key.second, all_gather[i]) : key.second,
                                                      types[k].ty<0 ? type_transform(types[k].ty, all_gather[i]) : types[k].ty,
                                                      types[k].count);
                }
            }

            logstream(LOG_INFO) << "global_tyscount size: " << global_tyscount.size() << LOG_endl;
            logstream(LOG_INFO) << "global_tystat.pstype.size: " << global_tystat.pstype.size() << LOG_endl;
            logstream(LOG_INFO) << "global_tystat.potype.size: " << global_tystat.potype.size() << LOG_endl;
            logstream(LOG_INFO) << "global_tystat.fine_type.size: " << global_tystat.fine_type.size() << LOG_endl;

            //debug single2complex
            // for ( auto iter = global_single2complex.cbegin(); iter != global_single2complex.cend(); ++iter ){
            //     cout << iter->first ;
            //     const unordered_set<ssid_t> set = iter->second;
            //     for(auto it = set.cbegin(); it != set.cend(); ++it){
            //         cout << " " << *it;
            //     }
            //     cout << endl;
            // }

            //debug tyscount
            // cout << "local................." << endl;
            // for ( auto iter = local_tyscount.cbegin(); iter != local_tyscount.cend(); ++iter ){
            //     cout << iter->first << ": " << iter -> second << endl;;
            // }

            // cout << "global................." << endl;
            // for ( auto iter = global_tyscount.cbegin(); iter != global_tyscount.cend(); ++iter ){
            //     cout << iter->first << ": " << iter -> second << endl;;
            // }
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
        	try{
                ofstream ofs(fname);
                boost::archive::binary_oarchive oa(ofs);
                oa << global_tyscount;
                oa << global_tystat;
                oa << global_type2int;
                oa << global_single2complex;
                ofs.close();
        	}
        	catch(exception& e){
                logstream(LOG_ERROR) << "store statistics unsuccessfully: " << e.what() << LOG_endl;
                return;
        	}

            logstream(LOG_INFO) << "store statistics to file "
                                << fname << " is finished." << LOG_endl;
        }
    }

    ssid_t get_simple_type(type_t &type){
        auto iter = local_type2int.find(type);

        if(iter == local_type2int.end()){
            ssid_t number = local_type2int.size();
            number ++;
            number = -number;
            local_type2int[type] = number;
            local_int2type[number] = type;
            return number;
        }
        else{
            return iter->second;
        }
    }
};

