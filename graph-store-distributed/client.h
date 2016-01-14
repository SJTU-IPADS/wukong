#pragma once

#include <string>  
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <assert.h> 
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#include "request.h"
#include "ontology.h"
#include "timer.h"
#include "network_node.h"
#include "message_wrap.h"
#include "thread_cfg.h"
#include "index_server.h"
#include "ingress.h"

class client{
	thread_cfg* cfg;
	profile latency_profile;
	int first_target;
public:
	index_server* is;
	request req;
	client(index_server* _is,thread_cfg* _cfg):is(_is),cfg(_cfg){
		first_target=0;
	}
	bool parse_cmd_vector(vector<string>& str_vec){
		int i=0;
		while(i<str_vec.size()){
			if(str_vec[i]=="lookup"){
				lookup(str_vec[i+1]);
				i+=2;
			} else if(str_vec[i]=="neighbors"){
				neighbors(str_vec[i+1],str_vec[i+2]);
				i+=3;
			} else if(str_vec[i]=="get_attr"){
				// get_attr(str_vec[i+1]);
				// i+=2;
				get_attr(str_vec[i+1],str_vec[i+2]);
				i+=3;
			} else if(str_vec[i]=="subclass_of"){
				subclass_of(str_vec[i+1]);
				i+=2;
			} else if(str_vec[i]=="filter"){
				filter(str_vec[i+1],str_vec[i+2],str_vec[i+3]);
				i+=4;
			} else if(str_vec[i]=="type_index"){
				type_index(str_vec[i+1],str_vec[i+2]);
				i+=3;
			} else if(str_vec[i]=="predict_index"){
				predict_index(str_vec[i+1],str_vec[i+2],str_vec[i+3]);
				i+=4;
			} else if(str_vec[i]=="join"){
				join();
				i+=1;
			} else if(str_vec[i]=="triangle"){
				vector<string> type_vec;
				vector<string> dir_vec;
				vector<string> predict_vec;
				type_vec.resize(3);
				dir_vec.resize(3);
				predict_vec.resize(3);
				i++;
				string parallel_count=str_vec[i];
				i++;
				for(int k=0;k<3;k++){
					type_vec[k]=str_vec[i];
					dir_vec[k]=str_vec[i+1];
					predict_vec[k]=str_vec[i+2];
					i+=3;
				}
				triangle(parallel_count,type_vec,dir_vec,predict_vec);
			} else if(str_vec[i]=="execute"){
				return true;
			} else {
				return false;
			}
		}
		return false;
	}
	client& lookup(string subject){
		assert(is->subject_to_id.find(subject)!=is->subject_to_id.end());
		first_target=ingress::vid2mid(is->subject_to_id[subject] , (cfg->m_num));
		req.clear();
		vector<int> vec_dataid;
		vec_dataid.push_back(is->subject_to_id[subject]);
		req.result_table.push_back(vec_dataid);
		return *this;
	}
	client& lookup_id(int id){
		first_target=ingress::vid2mid(id,cfg->m_num);
		req.clear();
		vector<int> vec_dataid;
		vec_dataid.push_back(id);
		req.result_table.push_back(vec_dataid);
		return *this;
	}

	client& predict_index(string parallel_count,string predict,string dir){
		if(!global_use_index_table)
			assert(false);
		req.clear();
		req.parallel_total=atoi(parallel_count.c_str());
		req.cmd_chains.push_back(cmd_predict_index);
		req.cmd_chains.push_back(is->predict_to_id[predict]);
		if(dir =="in" ){
			req.cmd_chains.push_back(para_in);
		} else if (dir =="out" ){
			req.cmd_chains.push_back(para_out);
		} else {
			req.cmd_chains.push_back(para_all);
		}
		return *this;
	}
	client& type_index(string parallel_count,string type){
		req.clear();
		req.parallel_total=atoi(parallel_count.c_str());
		req.cmd_chains.push_back(cmd_type_index);
		req.cmd_chains.push_back(is->subject_to_id[type]);
		return *this;
	}
	client& neighbors(string dir,string predict){
		req.cmd_chains.push_back(cmd_neighbors);
		if(dir =="in" ){
			req.cmd_chains.push_back(para_in);
		} else if (dir =="out" ){
			req.cmd_chains.push_back(para_out);
		} else {
			req.cmd_chains.push_back(para_all);
		}
		req.cmd_chains.push_back(is->predict_to_id[predict]);
		return *this;
	}
	client& filter(string dir,string predict,string target){
		req.cmd_chains.push_back(cmd_filter);
		if(dir =="in" ){
			req.cmd_chains.push_back(para_in);
		} else if (dir =="out" ){
			req.cmd_chains.push_back(para_out);
		} else {
			req.cmd_chains.push_back(para_all);
		}
		req.cmd_chains.push_back(is->predict_to_id[predict]);
		req.cmd_chains.push_back(atoi(target.c_str()));
		return *this;
	}
	client& join(){
		req.cmd_chains.push_back(cmd_join);
		return *this;
	}
	//This interface should be refined! 
	client& triangle(string parallel_count,vector<string> type_vec,
						vector<string> dir_vec,vector<string> predict_vec){
		req.clear();
		req.parallel_total=atoi(parallel_count.c_str());
		req.cmd_chains.push_back(cmd_triangle);
		for(int i=0;i<3;i++){
			req.cmd_chains.push_back(is->subject_to_id[type_vec[i]]);
			if(dir_vec[i] =="in" ){
				req.cmd_chains.push_back(para_in);
			} else if (dir_vec[i] =="out" ){
				req.cmd_chains.push_back(para_out);
			} else {
				assert(false);
				req.cmd_chains.push_back(para_all);
			}
			req.cmd_chains.push_back(is->predict_to_id[predict_vec[i]]);
		}
		return *this;
	}
	client& get_attr(string dir,string predict){
		req.cmd_chains.push_back(cmd_get_attr);
		if(dir =="in" ){
			req.cmd_chains.push_back(para_in);
		} else if (dir =="out" ){
			req.cmd_chains.push_back(para_out);
		} 
		req.cmd_chains.push_back(is->predict_to_id[predict]);
		return *this;
	}

	client& subclass_of(string target){
		assert(is->subject_to_id.find(target)!=is->subject_to_id.end());
		req.cmd_chains.push_back(cmd_subclass_of);
		req.cmd_chains.push_back(is->subject_to_id[target]);
		return *this;
	}
	client& execute(){
		// reverse cmd_chains
		// so we can easily pop the cmd and do recursive operation
		reverse(req.cmd_chains.begin(),req.cmd_chains.end()); 	
		req.req_id=-1;
		req.parent_id=cfg->get_inc_id();
		SendReq(cfg,first_target, cfg->client_num+rand()%cfg->server_num, req);
		req=RecvReq(cfg);
		req.cmd_chains.clear();
		return *this;
	}

	void Send(){
		reverse(req.cmd_chains.begin(),req.cmd_chains.end()); 	
		req.req_id=-1;
		req.parent_id=cfg->get_inc_id();
		command cmd=(command)req.cmd_chains.back();
		if(cmd == cmd_type_index|| cmd == cmd_predict_index || cmd==cmd_triangle){
			int chain_sz=req.cmd_chains.size();
			for(int i=0;i<cfg->m_num;i++){
				for(int j=0;j<req.parallel_total;j++){
					req.parallel_id=j;
					SendReq(cfg,i, cfg->client_num+j, req);
				}
			}
		} else {
			SendReq(cfg,first_target, cfg->client_num+rand()%cfg->server_num, req);			
		}
	}
	request Recv(){
		request reply;
		command cmd=(command)req.cmd_chains.back();
		if(cmd == cmd_type_index|| cmd == cmd_predict_index || cmd==cmd_triangle){
			int chain_sz=req.cmd_chains.size();
			reply=RecvReq(cfg);
			for(int i=1;i<cfg->m_num* req.parallel_total;i++){
				request tmp=RecvReq(cfg);
				reply.final_row_number+=tmp.final_row_number;
				// for(int j=0;j<tmp.row_num();j++){
				// 	tmp.append_row_to(reply.result_table,j);
				// }
			}
		} else {
			reply=RecvReq(cfg);
		}
		return reply;
	}
};