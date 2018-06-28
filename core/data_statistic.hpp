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

struct four_num {
    int out_out;
    int out_in;
    int in_in;
    int in_out;
    four_num(): out_out(0), out_in(0), in_in(0), in_out(0) {}
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & out_out;
        ar & out_in;
        ar & in_in;
        ar & in_out;
    }
};

struct direct_p {
    ssid_t dir;
    ssid_t p;
    direct_p(): dir(-1), p(-1) {}
    direct_p(ssid_t x, ssid_t y): dir(x), p(y) {}
};

class data_statistic {
private:
    // after the master server get whole statistics, this method is used to send it to all machines.
    void send_stat_to_all_machines(TCP_Adaptor *tcp_ad) {
        if (sid == 0) {
            // master server sends statistics
            std::stringstream ss;
            boost::archive::binary_oarchive my_oa(ss);
            my_oa << global_ptcount
                  << global_pscount
                  << global_pocount
                  << global_ppcount
                  << global_tyscount;

            for (int i = 1; i < global_num_servers; i++)
                tcp_ad->send(i, 0, ss.str());
        } else {
            // every slave server recieves statistics
            std::string str;
            str = tcp_ad->recv(0);
            std::stringstream ss;
            ss << str;
            boost::archive::binary_iarchive ia(ss);
            ia >> global_ptcount
               >> global_pscount
               >> global_pocount
               >> global_ppcount
               >> global_tyscount;
        }
    }

    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & predicate_to_triple;
        ar & predicate_to_subject;
        ar & predicate_to_object;
        ar & type_to_subject;
        ar & correlation;
    }

public:
    unordered_map<ssid_t, int> predicate_to_triple;
    unordered_map<ssid_t, int> predicate_to_subject;
    unordered_map<ssid_t, int> predicate_to_object;
    unordered_map<ssid_t, int> type_to_subject;
    unordered_map<pair<ssid_t, ssid_t>, four_num, boost::hash<pair<int, int>>> correlation;
    unordered_map<ssid_t, vector<direct_p> > id_to_predicate;

    unordered_map<ssid_t, int> global_ptcount;
    unordered_map<ssid_t, int> global_pscount;
    unordered_map<ssid_t, int> global_pocount;
    unordered_map<ssid_t, int> global_tyscount;
    unordered_map<pair<ssid_t, ssid_t>, four_num, boost::hash<pair<int, int>>> global_ppcount;

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
                for (unordered_map<ssid_t, int>::iterator it = all_gather[i].predicate_to_triple.begin();
                        it != all_gather[i].predicate_to_triple.end(); it++ ) {
                    ssid_t key = it->first;
                    int triple = it->second;
                    if (global_ptcount.find(key) == global_ptcount.end()) {
                        global_ptcount[key] = triple;
                    } else {
                        global_ptcount[key] += triple;
                    }
                }

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

                for (unordered_map<ssid_t, int>::iterator it = all_gather[i].predicate_to_object.begin();
                        it != all_gather[i].predicate_to_object.end(); it++ ) {
                    ssid_t key = it->first;
                    int object = it->second;
                    if (global_pocount.find(key) == global_pocount.end()) {
                        global_pocount[key] = object;
                    } else {
                        global_pocount[key] += object;
                    }
                }

                for (unordered_map<pair<ssid_t, ssid_t>, four_num, boost::hash<pair<int, int>>>::iterator it = all_gather[i].correlation.begin();
                        it != all_gather[i].correlation.end(); it++ ) {
                    pair<ssid_t, ssid_t> key = it->first;
                    four_num value = it->second;
                    if (global_ppcount.find(it->first) == global_ppcount.end()) {
                        global_ppcount[key] = value;
                    } else {
                        global_ppcount[key].out_out += value.out_out;
                        global_ppcount[key].out_in += value.out_in;
                        global_ppcount[key].in_in += value.in_in;
                        global_ppcount[key].in_out += value.in_out;
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

            }

            logstream(LOG_INFO) << "global_ptcount size: " << global_ptcount.size() << LOG_endl;
            logstream(LOG_INFO) << "global_pscount size: " << global_pscount.size() << LOG_endl;
            logstream(LOG_INFO) << "global_pocount size: " << global_pocount.size() << LOG_endl;
            logstream(LOG_INFO) << "global_ppcount size: " << global_ppcount.size() << LOG_endl;
            logstream(LOG_INFO) << "global_tyscount size: " << global_tyscount.size() << LOG_endl;

            // for type predicate
            global_pocount[1] = global_tyscount.size();
            int triple = 0;
            for (unordered_map<ssid_t, int>::iterator it = global_tyscount.begin();
                    it != global_tyscount.end(); it++ ) {
                triple += it->second;
            }
            global_ptcount[1] = triple;

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
            ia >> global_ptcount;
            ia >> global_pscount;
            ia >> global_pocount;
            ia >> global_tyscount;
            ia >> global_ppcount;
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
            oa << global_ptcount;
            oa << global_pscount;
            oa << global_pocount;
            oa << global_tyscount;
            oa << global_ppcount;
            ofs.close();

            logstream(LOG_INFO) << "store statistics to file "
                                << fname << " is finished." << LOG_endl;
        }
    }

};

