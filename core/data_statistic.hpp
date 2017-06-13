#pragma once

#include <boost/mpi.hpp>
#include <boost/functional/hash.hpp>

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
    int dir;
    int p;
    direct_p(): dir(-1), p(-1) {}
    direct_p(int x, int y): dir(x), p(y) {}
};

class data_statistic {
public:
    unordered_map<int, int> predicate_to_triple;
    unordered_map<int, int> predicate_to_subject;
    unordered_map<int, int> predicate_to_object;
    unordered_map<int, int> type_to_subject;
    unordered_map<pair<int, int>, four_num, boost::hash<pair<int, int> > > correlation;
    unordered_map<int, vector<direct_p> > id_to_predicate;

    unordered_map<int, int> global_ptcount;
    unordered_map<int, int> global_pscount;
    unordered_map<int, int> global_pocount;
    unordered_map<int, int> global_tyscount;
    unordered_map<pair<int, int>, four_num, boost::hash<pair<int, int> > > global_ppcount;

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
                for (unordered_map<int, int>::iterator it = all_gather[i].predicate_to_triple.begin();
                        it != all_gather[i].predicate_to_triple.end(); it++ ) {
                    int key = it->first;
                    int triple = it->second;
                    if (global_ptcount.find(key) == global_ptcount.end()) {
                        global_ptcount[key] = triple;
                    } else {
                        global_ptcount[key] += triple;
                    }
                }
                for (unordered_map<int, int>::iterator it = all_gather[i].predicate_to_subject.begin();
                        it != all_gather[i].predicate_to_subject.end(); it++ ) {
                    int key = it->first;
                    int subject = it->second;
                    if (global_pscount.find(key) == global_pscount.end()) {
                        global_pscount[key] = subject;
                    } else {
                        global_pscount[key] += subject;
                    }
                }
                for (unordered_map<int, int>::iterator it = all_gather[i].predicate_to_object.begin();
                        it != all_gather[i].predicate_to_object.end(); it++ ) {
                    int key = it->first;
                    int object = it->second;
                    if (global_pocount.find(key) == global_pocount.end()) {
                        global_pocount[key] = object;
                    } else {
                        global_pocount[key] += object;
                    }
                }
                for (unordered_map<pair<int, int>, four_num, boost::hash<pair<int, int> > >::iterator it = all_gather[i].correlation.begin();
                        it != all_gather[i].correlation.end(); it++ ) {
                    pair<int, int> key = it->first;
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
                for (unordered_map<int, int>::iterator it = all_gather[i].type_to_subject.begin();
                        it != all_gather[i].type_to_subject.end(); it++ ) {
                    int key = it->first;
                    int subject = it->second;
                    if (global_tyscount.find(key) == global_tyscount.end()) {
                        global_tyscount[key] = subject;
                    } else {
                        global_tyscount[key] += subject;
                    }
                }

            }

            cout << "global_ptcount size: " << global_ptcount.size() << std::endl;
            cout << "global_pscount size: " << global_pscount.size() << std::endl;
            cout << "global_pocount size: " << global_pocount.size() << std::endl;
            cout << "global_ppcount size: " << global_ppcount.size() << std::endl;
            cout << "global_tyscount size: " << global_tyscount.size() << std::endl;

            // for type predicate
            global_pocount[1] = global_tyscount.size();
            int triple = 0;
            for (unordered_map<int, int>::iterator it = global_tyscount.begin();
                    it != global_tyscount.end(); it++ ) {
                triple += it->second;
            }
            global_ptcount[1] = triple;

        }

        cout << "INFO#" << world->rank() << ": gathering stats of DGraph is finished." << endl;

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

