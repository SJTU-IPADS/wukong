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

    TCP_Adaptor* tcp_adaptor;
    boost::mpi::communicator* world;

    data_statistic(TCP_Adaptor* _tcp_adaptor, boost::mpi::communicator* _world)
        : tcp_adaptor(_tcp_adaptor), world(_world) {

    }

    data_statistic() {

    }

    void gather_data() {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);
        oa << (*this);
        tcp_adaptor->send(0, 0, ss.str());

        if (world->rank() == 0) {
            vector<data_statistic> all_gather;
            for (int i = 0; i < world->size(); i++) {
                std::string str;
                str = tcp_adaptor->recv(0);
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

                for (unordered_map<pair<ssid_t, ssid_t>, four_num, boost::hash<pair<int, int> > >::iterator it = all_gather[i].correlation.begin();
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

        if (world->rank() == 0) {
            std::stringstream my_ss;
            boost::archive::binary_oarchive my_oa(my_ss);
            my_oa << global_ptcount
                  << global_pscount
                  << global_pocount
                  << global_ppcount
                  << global_tyscount;
            for (int i = 1; i < world->size(); i++) {
                tcp_adaptor->send(i, 0, my_ss.str());
            }
        }

        if (world->rank() != 0) {
            std::string str;
            str = tcp_adaptor->recv(0);
            std::stringstream s;
            s << str;
            boost::archive::binary_iarchive ia(s);
            ia >> global_ptcount
               >> global_pscount
               >> global_pocount
               >> global_ppcount
               >> global_tyscount;
        }

        logstream(LOG_INFO) << "#" << world->rank() << ": gathering stats of DGraph is finished." << LOG_endl;

    }

    void gather_data_from_file() {
        // data only cached on master server
        if (world->rank() != 0) return;

        vector<string> strs;
        boost::split(strs,global_input_folder,boost::is_any_of("/"));
        string filename = strs[strs.size() - 2] + ".statfile";

        // exit if file does not exist
        ifstream file(filename.c_str());
        if(!file.good()){
	        logstream(LOG_FATAL) << "statistics file "  << filename << " is missing, pleanse set global_use_statistics_cache = 0"  << LOG_endl;
			exit(EXIT_FAILURE);
        }

        ifstream ifs(filename);
        boost::archive::binary_iarchive ia(ifs);
        ia >> global_ptcount;
        ia >> global_pscount;
        ia >> global_pocount;
        ia >> global_tyscount;
        ia >> global_ppcount;
        ifs.close();
	    logstream(LOG_INFO) << "loading statistics from file "  << filename << " is finished."  << LOG_endl;

    }

    void write_data_to_file(){
        // data only cached on master server        
        if (world->rank() != 0) return;

        vector<string> strs;
        boost::split(strs,global_input_folder,boost::is_any_of("/"));
        string filename = strs[strs.size() - 2] + ".statfile";

        // avoid saving when it already exsits
        ifstream file(filename.c_str());
        if(!file.good()){
            ofstream ofs(filename);
            boost::archive::binary_oarchive oa(ofs);
            oa << global_ptcount;
            oa << global_pscount;
            oa << global_pocount;
            oa << global_tyscount;
            oa << global_ppcount;
            ofs.close();
            logstream(LOG_INFO) << "saving statistics data to file "    << filename	<< " is finished."  << LOG_endl;
        }

        if(!global_use_statistics_cache)
	        logstream(LOG_INFO) << "if you want to use the cached statistics data with dataset unchanged, set global_use_statistics_cache = 1 "	<< LOG_endl;        

    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & predicate_to_triple;
        ar & predicate_to_subject;
        ar & predicate_to_object;
        ar & type_to_subject;
        ar & correlation;
    }

};

