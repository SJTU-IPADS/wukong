#pragma once
#include <stdint.h> //uint64_t
#include <iostream>
#include "timer.h"
#include "string"
using namespace std;

class profile{
public:
	//use to calculate average latency
	uint64_t sum_latency;
	uint64_t count_latency;


	//used to count average neighbor number
	uint64_t current_req;
	uint64_t split_req;
	uint64_t non_split_req;
	uint64_t neighbor_num;

	//used to count average msg size
	uint64_t min_msg;
	uint64_t max_msg;
	uint64_t count_msg;
	uint64_t sum_msg;

	timer t;
	profile(){
		sum_latency=0;
		count_latency=0;

		current_req=0;
		split_req=0;
		non_split_req=0;
		neighbor_num=0;

		min_msg=1000000000;
		max_msg=0;
		sum_msg=0;
		count_msg=0;

		t.reset();
	}
	void record_and_report_latency(uint64_t size){
		count_latency++;
		sum_latency+=size;
		if(count_latency%10000==9999){
			cout<<"average latency:"<<sum_latency/count_latency << " us"<<endl;
			//count_latency=0;
			//sum_latency=0;
		}
	}

	void record_msgsize(uint64_t size){
		if(size > max_msg)
			max_msg=size;
		if(size < min_msg)
			min_msg=size;
		sum_msg=sum_msg+size;
		count_msg++;
	}
	void report_msgsize(){
		current_req++;
		if(current_req==10000){
			current_req=0;
			timer t2;
			cout<<"average neighbor:"<<neighbor_num*1.0/(split_req+non_split_req)
				<<"\t"<<"split-rate:"<<split_req*1.0/(split_req+non_split_req)
				<<"\t"<<"msgsize=["<<min_msg<<","<<max_msg<<"]("<<sum_msg*1.0/count_msg<<")"
				<<"\t"<<"total_msg="<<sum_msg/(1024*1024)<<" MB"
				<<"\t"<<t2.diff(t)<<" ms"<<endl;
			t.reset();
		}
	}
};