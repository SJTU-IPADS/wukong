#pragma once
#include <stdint.h> //uint64_t
#include <iostream>
#include "timer.h"
#include "request.h"
using namespace std;

class latency_logger{
	vector<uint64_t> send_time_vec;
	vector<uint64_t> recv_time_vec;
	uint64_t start_time;
	uint64_t stop_time;
	uint64_t accum_latency;
	uint64_t min_latency;
	uint64_t max_latency;
public:
	void start(){
		send_time_vec.clear();
		recv_time_vec.clear();
		start_time=timer::get_usec();
		accum_latency=0;
		min_latency=-1;
		max_latency=0;
	}
	void stop(){
		stop_time=timer::get_usec();
	}
	void record(uint64_t send_time,uint64_t recv_time){
		send_time_vec.push_back(send_time);
		recv_time_vec.push_back(recv_time);
		accum_latency+=recv_time-send_time;
		min_latency=min(recv_time-send_time,min_latency);
		max_latency=max(recv_time-send_time,max_latency);
	}
	void print(){
		if(send_time_vec.size()==0){
			cout<<"No any record"<<endl;
			return ;
		}
		cout<<"Finish batch in "<<(stop_time-start_time)/1000.0<<" ms"<<endl;
		cout<<"Avg\tmin\tmax\tlatency us"<<endl;
		cout<<accum_latency/send_time_vec.size()<<"\t"<<min_latency<<"\t"<<max_latency<<endl;
		cout<<"Throughput "<<send_time_vec.size()*1000.0/(stop_time-start_time)<<" Kops"<<endl;
	}

};
