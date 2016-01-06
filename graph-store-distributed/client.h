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

/*
	client& lookup(string subject){
		first_target=ingress::vid2mid(is->subject_to_id[subject] , (cfg->m_num));
		req.clear();
		path_node node(is->subject_to_id[subject],-1);
		vector<path_node> vec;
		vec.push_back(node);
		req.result_paths.push_back(vec);
		return *this;
	}
	client& lookup_id(int id){
		first_target=ingress::vid2mid(id,cfg->m_num);
		req.clear();
		path_node node(id,-1);
		vector<path_node> vec;
		vec.push_back(node);
		req.result_paths.push_back(vec);
		return *this;
	}
*/
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

	client& predict_index(string predict,string dir){
		if(!global_use_predict_index)
			assert(false);
		req.clear();
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
	client& type_index(string type){
		req.clear();
		req.cmd_chains.push_back(cmd_type_index);
		req.cmd_chains.push_back(is->subject_to_id[type]);
		return *this;
	}
	client& get_subtype(string target){
		//not supported anymore
		assert(false);
		req.clear();
		int target_id=is->subject_to_id[target];
		req.cmd_chains.push_back(cmd_get_subtype);
		req.cmd_chains.push_back(target_id);
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
	//This interface should be refined! 
	client& triangle(vector<string> type_vec,vector<string> dir_vec,vector<string> predict_vec){
		req.clear();
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
	client& get_attr(string predict){
		req.cmd_chains.push_back(cmd_get_attr);
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
		if(cmd==cmd_triangle|| cmd == cmd_predict_index || cmd == cmd_type_index){
			for(int i=0;i<cfg->m_num;i++){
				if(global_interactive ){
					//only one core working
					SendReq(cfg,i, cfg->client_num, req);
				} else {
					SendReq(cfg,i, cfg->client_num+rand()%cfg->server_num, req);
				}
			}
		} else {
			if(global_interactive ){
				SendReq(cfg,first_target, cfg->client_num, req);
			} else {
				SendReq(cfg,first_target, cfg->client_num+rand()%cfg->server_num, req);
			}
		}

	}
	request Recv(){
		command cmd=(command)req.cmd_chains.back();
		if(cmd==cmd_triangle|| cmd == cmd_predict_index || cmd == cmd_type_index){
			req=RecvReq(cfg);
			for(int i=1;i<cfg->m_num;i++){
				request tmp=RecvReq(cfg);
				for(int j=0;j<tmp.row_num();j++){
					tmp.append_row_to(req.result_table,j);
				}
			}
		} else {
			req=RecvReq(cfg);
		}
		if(cfg->m_id==0){
			//latency_profile.record_and_report_latency(timer::get_usec()-req.timestamp);
			//latency_profile.record_and_report_shape(req);
		}
		return req;
	}
};