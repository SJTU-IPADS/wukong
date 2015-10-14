#pragma once
#include <stdint.h> //uint64_t
#include <iostream>
#include "timer.h"
#include "string"
#include "request.h"
using namespace std;

class profile{
public:
	//use to draw the average shape of result
	vector<uint64_t>  shape_vec;
	uint64_t shape_count;

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

	
	double throughput;
	int throughput_count;

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

		throughput=0;
		throughput_count=-5;
		t.reset();
	}
	void record_and_report_shape(request& r){
		if(r.result_paths.size()>shape_vec.size()){
			shape_vec.resize(r.result_paths.size());
		}
		for(int i=0;i<r.result_paths.size();i++){
			shape_vec[i]+=r.result_paths[i].size();
		}
		shape_count++;
		if(shape_count%1000==999){
			cout<<"shape:";
			for(int i=0;i<shape_vec.size();i++){
				cout<<shape_vec[i]*1.0/shape_count<<" ";
			}
			cout<<endl;
			shape_count=0;
			shape_vec.clear();
		}
	}
	void record_and_report_latency(uint64_t size){
		count_latency++;
		sum_latency+=size;
		if(count_latency%1000==999){
			cout<<"average latency:"<<sum_latency/count_latency << " us"<<endl;
			count_latency=0;
			sum_latency=0;
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
		int interval=1000;//*10;
		current_req++;
		if(current_req==interval){
			
			current_req=0;
			timer t2;
			throughput_count++;
			if(throughput_count>0){
				throughput+=interval*1.0/t2.diff(t);
				//cout<<"avg throughput:"<<throughput/throughput_count<<" K ops"<<endl;
			
			cout<<"avg neighbor:"<<neighbor_num*1.0/(split_req+non_split_req)
				<<"\t"<<"split:"<<split_req*1.0/(split_req+non_split_req)
				<<"\t"<<"msgsize=["<<min_msg<<","<<max_msg<<"]("<<sum_msg*1.0/count_msg<<")"
				//<<"\t"<<"total_msg="<<sum_msg/(1024*1024)<<" MB"
				//<<"\t"<<interval*1.0/t2.diff(t)<<" K ops"<<endl;
				<<"\t"<<"avg throughput:"<<throughput/throughput_count<<" K ops"<<endl;
			}	

			t.reset();
		}
	}
};