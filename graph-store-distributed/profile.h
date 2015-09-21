#pragma once
#include <stdint.h> //uint64_t
#include <iostream>
#include "timer.h"
#include "string"
using namespace std;

class profile{
public:
	uint64_t current_req;
	uint64_t split_req;
	uint64_t non_split_req;
	uint64_t neighbor_num;
	uint64_t min_msg;
	uint64_t max_msg;
	uint64_t count_msg;
	uint64_t sum_msg;

	timer t;
	profile(){
		min_msg=1000000000;
		max_msg=0;
		sum_msg=0;
		count_msg=0;

		current_req=0;
		split_req=0;
		non_split_req=0;
		neighbor_num=0;
		t.reset();
	}
	void record(uint64_t size){
		if(size > max_msg)
			max_msg=size;
		if(size < min_msg)
			min_msg=size;
		sum_msg=sum_msg+size;
		count_msg++;
	}
	void report(){
		current_req++;
		if(current_req==10000){
			current_req=0;
			timer t2;
			cout<<"average neighbor:"<<neighbor_num*1.0/(split_req+non_split_req)
				<<"\t"<<"split-rate:"<<split_req*1.0/(split_req+non_split_req)
				<<"\t"<<"msgsize=["<<min_msg<<","<<max_msg<<"]("<<sum_msg*1.0/count_msg<<")"
				<<"\t"<<t2.diff(t)<<" ms"<<endl;
			t.reset();
		}
	}
};