#pragma once

#include <iostream>
#include "utils.h"
#include <boost/serialization/unordered_map.hpp>

using namespace std;
struct req_stats{
    int query_type;
    uint64_t start_time;
    uint64_t end_time;
    template <typename Archive>
	void serialize(Archive &ar, const unsigned int version) {
        ar & query_type;
        ar & start_time;
        ar & end_time;
	}
};
class batch_logger{
    uint64_t init_time;
    unordered_map<int,req_stats> stats_map;
public:
    void init(){
        init_time=timer::get_usec();
    }
    void start_record(int reqid,int query_type){
        stats_map[reqid].query_type=query_type;
        stats_map[reqid].start_time=timer::get_usec() - init_time;
	}
    void end_record(int reqid){
        stats_map[reqid].end_time=timer::get_usec() - init_time;
    }
    void finish(){

    }
    void merge(batch_logger& other){
        for(auto iter: other.stats_map){
            stats_map[iter.first]=iter.second;
        }
    }
    void print(){
        vector<int> throughput_vec;
        int print_interval=200; //100ms
        cout<<"Throughput (Kops)"<<endl;
        for(auto iter: stats_map){
            int idx=iter.second.start_time/(1000*print_interval);
            if(throughput_vec.size()<=idx){
                throughput_vec.resize(idx+1);
            }
            throughput_vec[idx]++;
        }
        for(int i=0;i<throughput_vec.size();i++){
            cout<<throughput_vec[i]*1.0/print_interval<<" ";
            if(i%5==4 || i==throughput_vec.size()-1){
                cout<<endl;
            }
        }

    }
    template <typename Archive>
	void serialize(Archive &ar, const unsigned int version) {
        ar & stats_map;
	}
};
