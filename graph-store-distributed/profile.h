#pragma once
#include <stdint.h> //uint64_t
#include <iostream>
#include "timer.h"
using namespace std;

class profile{
public:
	uint64_t current_req;
	uint64_t split_req;
	uint64_t non_split_req;
	uint64_t neighbor_num;
	timer t;
	profile(){
		current_req=0;
		split_req=0;
		non_split_req=0;
		neighbor_num=0;
		t.reset();
	}
	void report(){
		current_req++;
		if(current_req==10000){
			current_req=0;
			timer t2;
			cout<<"average neighbor:"<<neighbor_num*1.0/(split_req+non_split_req)
				<<"\t"<<"split-rate:"<<split_req*1.0/(split_req+non_split_req)
				<<"\t"<<t2.diff(t)<<" ms"<<endl;
			t.reset();
		}
	}
};