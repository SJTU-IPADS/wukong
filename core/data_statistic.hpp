#pragma once

#include <boost/mpi.hpp>
#include <boost/functional/hash.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/unordered_map.hpp>

#include "tcp_adaptor.hpp"
#include "config.hpp"

using namespace std;

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
            my_oa << global_pscount
                  << global_tyscount
                  << global_tystat;

            for (int i = 1; i < global_num_servers; i++)
                tcp_ad->send(i, 0, ss.str());
        } else {
            // every slave server recieves statistics
            std::string str;
            str = tcp_ad->recv(0);
            std::stringstream ss;
            ss << str;
            boost::archive::binary_iarchive ia(ss);
            ia >> global_pscount
               >> global_tyscount
               >> global_tystat;
        }
    }

    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & predicate_to_subject;
        ar & type_to_subject;
        ar & local_tystat;
    }

public:
    unordered_map<ssid_t, int> predicate_to_subject;
    unordered_map<ssid_t, int> type_to_subject;

    unordered_map<ssid_t, int> global_pscount;
    unordered_map<ssid_t, int> global_tyscount;

    type_stat local_tystat;
    type_stat global_tystat;

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

                for (unordered_map<ssid_t, int>::iterator it = all_gather[i].predicate_to_subject.begin();
                        it != all_gather[i].predicate_to_subject.end(); it++ ) {
                    ssid_t key = it->first;
                    int subject = it->second;
                    if (global_pscount.find(key) == global_pscount.end()) {
                        global_pscount[key] = subject;
                    } else {
                        global_pscount[key] += subject;
                    }
                }

                //for type predicate
                for (unordered_map<ssid_t, int>::iterator it = all_gather[i].type_to_subject.begin();
                        it != all_gather[i].type_to_subject.end(); it++ ) {
                    ssid_t key = it->first;
                    int subject = it->second;
                    if (global_tyscount.find(key) == global_tyscount.end()) {
                        global_tyscount[key] = subject;
                    } else {
                        global_tyscount[key] += subject;
                    }
                }

                for (unordered_map<ssid_t, vector<ty_count>>::iterator it = all_gather[i].local_tystat.pstype.begin();
                        it != all_gather[i].local_tystat.pstype.end(); it++ ) {
                    ssid_t key = it->first;
                    vector<ty_count>& types = it->second;
                    for (size_t k = 0; k < types.size(); k++)
                        global_tystat.insert_stype(key, types[k].ty, types[k].count);
                }
                for (unordered_map<ssid_t, vector<ty_count>>::iterator it = all_gather[i].local_tystat.potype.begin();
                        it != all_gather[i].local_tystat.potype.end(); it++ ) {
                    ssid_t key = it->first;
                    vector<ty_count>& types = it->second;
                    for (size_t k = 0; k < types.size(); k++)
                        global_tystat.insert_otype(key, types[k].ty, types[k].count);
                }
                for (unordered_map<pair<ssid_t, ssid_t>, vector<ty_count>, boost::hash<pair<int, int> > >::iterator 
                        it = all_gather[i].local_tystat.fine_type.begin();
                        it != all_gather[i].local_tystat.fine_type.end(); it++ ) {
                    pair<ssid_t, ssid_t> key = it->first;
                    vector<ty_count>& types = it->second;
                    for (size_t k = 0; k < types.size(); k++)
                        global_tystat.insert_finetype(key.first, key.second, types[k].ty, types[k].count);
                }
            }

            logstream(LOG_INFO) << "global_pscount size: " << global_pscount.size() << LOG_endl;
            logstream(LOG_INFO) << "global_tyscount size: " << global_tyscount.size() << LOG_endl;
            logstream(LOG_INFO) << "global_tystat.pstype.size: " << global_tystat.pstype.size() << LOG_endl;
            logstream(LOG_INFO) << "global_tystat.potype.size: " << global_tystat.potype.size() << LOG_endl;
            logstream(LOG_INFO) << "global_tystat.fine_type.size: " << global_tystat.fine_type.size() << LOG_endl;

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
            ia >> global_pscount;
            ia >> global_tyscount;
            ia >> global_tystat;
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
            ofstream ofs(fname);
            boost::archive::binary_oarchive oa(ofs);
            oa << global_pscount;
            oa << global_tyscount;
            oa << global_tystat;
            ofs.close();

            logstream(LOG_INFO) << "store statistics to file "
                                << fname << " is finished." << LOG_endl;
        }
    }

};

