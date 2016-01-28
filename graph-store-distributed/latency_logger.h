#pragma once
#include <stdint.h> //uint64_t
#include <iostream>
#include "timer.h"
#include "request.h"
using namespace std;

class latency_logger{
	vector<uint64_t> send_time_vec;
	vector<uint64_t> recv_time_vec;
	vector<int> query_type_vec;
	uint64_t start_time;
	uint64_t stop_time;
	uint64_t accum_latency;
	uint64_t min_latency;
	uint64_t max_latency;
public:
	void reserve(int total_request){
		send_time_vec.reserve(total_request);
		recv_time_vec.reserve(total_request);
		query_type_vec.reserve(total_request);
	}
	
	void merge(latency_logger& r){
		for(int i=0;i<r.send_time_vec.size();i++){
			send_time_vec.push_back(r.send_time_vec[i]);
			recv_time_vec.push_back(r.recv_time_vec[i]);
			query_type_vec.push_back(r.query_type_vec[i]);
		}
	}
	void clear(){
		//send_time_vec.clear();
		//recv_time_vec.clear();
		//query_type_vec.clear();
	}
	void start(){
		//clear();
		start_time=timer::get_usec();
		accum_latency=0;
		min_latency=-1;
		max_latency=0;
	}
	void stop(){
		stop_time=timer::get_usec();
	}
	void record(uint64_t send_time,uint64_t recv_time,int query_type=0){
		//start_time of different machine may be different.
		//so we use the delta
		send_time_vec.push_back(send_time-start_time);
		recv_time_vec.push_back(recv_time-start_time);
		query_type_vec.push_back(query_type);
		accum_latency+=recv_time-send_time;
		min_latency=min(recv_time-send_time,min_latency);
		max_latency=max(recv_time-send_time,max_latency);
	}
	void print(){
		if(send_time_vec.size()==0){
			cout<<"No any record"<<endl;
			return ;
		}
		uint64_t end_time=0;
		for(int i=0;i<recv_time_vec.size();i++){
			end_time=max(end_time,recv_time_vec[i]);
		}
		cout<<"Finish batch in "<<end_time/1000.0<<" ms"<<endl;
		cout<<"Throughput "<<recv_time_vec.size()*1000.0/end_time<<" Kops"<<endl;
/*
		////print CDF
		vector<vector<int> >cdf_data;
		for(int i=0;i<query_type_vec.size();i++){
			if(query_type_vec[i] >= cdf_data.size()){
				cdf_data.resize(query_type_vec[i]+1);
			}
			cdf_data[query_type_vec[i]].push_back(recv_time_vec[i]- send_time_vec[i]);
		}
		int cdf_pirnt_rate=100;
		for(int i=0;i<cdf_data.size();i++){
			sort(cdf_data[i].begin(),cdf_data[i].end());
			cout<<"query "<<i<<endl;
			int count=0;
			for(int j=0;j<cdf_data[i].size();j++){
				if((j+1)%(cdf_data[i].size()/cdf_pirnt_rate)==0 ){
					cout<<cdf_data[i][j]<<"\t";
					count++;
					if(count%5==0){
						cout<<endl;
					}
				}
			}
		}
*/
		///print throughput-time graph
		int time_pirnt_ms=100; //print every 100 ms
		vector<int> count_vec;
		count_vec.resize(end_time/(time_pirnt_ms*1000)+1);
		for(int i=0;i<recv_time_vec.size();i++){
			int idx=recv_time_vec[i]/(time_pirnt_ms*1000);
			count_vec[idx]++;
		}
		for(int i=0;i<count_vec.size();i++){
			cout<<count_vec[i]*1.0/time_pirnt_ms<<"\t";
			if((i+1)%5==0 || i==count_vec.size()-1){
				cout<<endl;
			}
		}


		//cout<<"Avg latency us"<<endl;
		// cout<<"Finish batch in "<<(stop_time-start_time)/1000.0<<" ms"<<endl;
		// cout<<"Avg\tmin\tmax\tlatency us"<<endl;
		// cout<<accum_latency/send_time_vec.size()<<"\t"<<min_latency<<"\t"<<max_latency<<endl;
		// cout<<"Throughput "<<send_time_vec.size()*1000.0/(stop_time-start_time)<<" Kops"<<endl;
	}
	template <typename Archive>
	void serialize(Archive &ar, const unsigned int version) { 
		ar & send_time_vec;
		ar & recv_time_vec; 
		ar & query_type_vec; 
	}

};

void SendLog(thread_cfg* cfg,int r_mid,int r_tid,latency_logger& r){
    std::stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << r;
    //cfg->node->Send(r_mid,r_tid,ss.str());
    cfg->rdma->rbfSend(cfg->t_id,r_mid, r_tid, ss.str().c_str(),ss.str().size());  
}

latency_logger RecvLog(thread_cfg* cfg){
    std::string str;
    //str=cfg->node->Recv();
    str=cfg->rdma->rbfRecv(cfg->t_id);

    std::stringstream s;
    s << str;
    boost::archive::binary_iarchive ia(s);
    latency_logger r;
    ia >> r;
    return r;
}
