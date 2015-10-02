#pragma once
#include "graph.h"
#include "request.h"
#include "request_queue.h"
#include "network_node.h"
#include "message_wrap.h"
#include "profile.h"
//traverser will remember all the paths just like
//traverser_keeppath in single machine

class traverser{
	graph& g;
	concurrent_request_queue& concurrent_req_queue;
	request_queue req_queue;
	thread_cfg* cfg;
	profile split_profile;
	int req_id;
	int get_id(){
		int result=req_id;
		req_id+=cfg->m_num* cfg->server_num;
		return result;
	}
	vector<path_node> do_neighbors(request& r){
		vector<path_node> vec;
		r.cmd_chains.pop_back();
		int dir=r.cmd_chains.back();
		r.cmd_chains.pop_back();
		int	predict_id=r.cmd_chains.back();
		r.cmd_chains.pop_back();
		int path_len=r.result_paths.size();
		for (int i=0;i< r.result_paths[path_len-1].size();i++){
			int prev_id=r.result_paths[path_len-1][i].id;
			vertex vdata=g.kstore.getVertex(prev_id);
			if(dir ==para_in || dir == para_all){
				edge_row* edge_ptr=g.kstore.getEdgeArray(vdata.in_edge_ptr);
				for(uint64_t k=0;k<vdata.in_degree;k++){
					if(predict_id==edge_ptr[k].predict){
						vec.push_back(path_node(edge_ptr[k].vid,i));
					}					
				}
			}
			if(dir ==para_out || dir == para_all){
				edge_row* edge_ptr=g.kstore.getEdgeArray(vdata.out_edge_ptr);
				for(uint64_t k=0;k<vdata.out_degree;k++){
					if(predict_id==edge_ptr[k].predict){
						vec.push_back(path_node(edge_ptr[k].vid,i));
					}					
				}
			}
		}
		return vec;
	}
	void do_subclass_of(request& r){
		//int predict_id=g.predict_to_id["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"];
		int predict_id=0;//

		r.cmd_chains.pop_back();
		int target_id=r.cmd_chains.back();
		r.cmd_chains.pop_back();

		int path_len=r.result_paths.size();
		vector<path_node>& prev_vec = r.result_paths[path_len-1];
		vector<path_node> new_vec;
		for (int i=0;i<prev_vec.size();i++){
			int prev_id=prev_vec[i].id;	
			vertex vdata=g.kstore.getVertex(prev_id);
			edge_row* edge_ptr=g.kstore.getEdgeArray(vdata.out_edge_ptr);
			for(uint64_t k=0;k<vdata.out_degree;k++){
				if(predict_id==edge_ptr[k].predict && 
							g.ontology_table.is_subtype_of(edge_ptr[k].vid,target_id)){
					new_vec.push_back(prev_vec[i]);
					break;
				}				
			}
		}
		r.result_paths[path_len-1]=new_vec;
	}
	vector<path_node> do_get_subtype(request& r){
		r.cmd_chains.pop_back();
		int parent_type_id=r.cmd_chains.back();
		r.cmd_chains.pop_back();
		unordered_set<int> ids = g.ontology_table.get_all_subtype(parent_type_id);
		vector<path_node> vec;
		for(auto id: ids){
			vec.push_back(path_node(id,-1));
		}
		return vec;
	}

	void merge_reqs(vector<request>& sub_reqs,request& r){
		if(sub_reqs.size()>1){
			//iterate on all sub_reqs
			for(int i=1;i<sub_reqs.size();i++){
				//reversely iterate on all column 
				for(int column=sub_reqs[i].result_paths.size()-1;column>=0;column--){
					for(auto node:sub_reqs[i].result_paths[column]){
						if(column>0){
							node.prev=node.prev + sub_reqs[0].result_paths[column-1].size();
						}
						sub_reqs[0].result_paths[column].push_back(node);
					}
				}
			}
		}
		for(int i=0;i<sub_reqs[0].result_paths.size();i++){
			r.result_paths.push_back(sub_reqs[0].result_paths[i]);
		}
	}
	vector<request> split_request(vector<path_node>& vec,request& r){
		vector<request> sub_reqs;
		int num_sub_request=cfg->m_num;
		sub_reqs.resize(num_sub_request);
		for(int i=0;i<sub_reqs.size();i++){
			sub_reqs[i].parent_id=r.req_id;
			sub_reqs[i].cmd_chains=r.cmd_chains;
			sub_reqs[i].result_paths.push_back(vector<path_node>());
		}
		for(int i=0;i<vec.size();i++){
			//int machine = i % num_sub_request;
			int machine = vec[i].id % num_sub_request;
			sub_reqs[machine].result_paths[0].push_back(vec[i]);
		}
		return sub_reqs;
	}
public:
	traverser(graph& gg,concurrent_request_queue& crq,thread_cfg* _cfg)
			:g(gg),concurrent_req_queue(crq),cfg(_cfg){
		req_id=cfg->m_id+cfg->t_id*cfg->m_num;
	}
	void handle_request(request& r){
		if(r.cmd_chains.size()==0)
			return;
		vector<path_node> vec;
		if(r.cmd_chains.back() == cmd_subclass_of){
			// subclass_of is a filter operation
			// it just remove some of the output
			do_subclass_of(r);
			handle_request(r);
			return ;
		} else if(r.cmd_chains.back() == cmd_neighbors){
			vec=do_neighbors(r);
		} else if(r.cmd_chains.back() == cmd_get_subtype){
			assert(r.path_length()==0);
			vec=do_get_subtype(r);
		} else{
			assert(false);
		}
		//trying to execute using one-side RDMA here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		// if(r.cmd_chains.size()!=0){
		// 	split_profile.neighbor_num+=vec.size();
		// 	if(vec.size()<cfg->m_num*10){
		// 		split_profile.split_req++;
		// 	} else {
		// 		split_profile.non_split_req++;
		// 	}
		// }

		//if(false){
		//if(r.cmd_chains.size()!=0 && vec.size()<cfg->m_num*10){
		while(r.cmd_chains.size()!=0 ){
			split_profile.neighbor_num+=vec.size();
			if(vec.size()<cfg->m_num*10){
				split_profile.split_req++;
			} else {
				split_profile.non_split_req++;
			}
			if(vec.size()>=cfg->m_num*10)
				break;

			int dir=para_out;			
			int cmd_type=r.cmd_chains.back();
			r.cmd_chains.pop_back();
			if(cmd_type==cmd_neighbors){
				dir=r.cmd_chains.back();
				r.cmd_chains.pop_back();
			}
			int target_id=r.cmd_chains.back();
			r.cmd_chains.pop_back();
			vector<path_node> new_vec;
			if(cmd_type == cmd_subclass_of){
				for(int i=0;i<vec.size();i++){
					vector<edge_row> edges=g.kstore.readGlobal(cfg->t_id,vec[i].id,dir);
					for(int k=0;k<edges.size();k++){
						if(0==edges[k].predict && 
							g.ontology_table.is_subtype_of(edges[k].vid,target_id)){
							new_vec.push_back(vec[i]);
							break;
						}	
					}
				}
				vec=new_vec; //replace old vec since subclass is a filter operation
			} else if(cmd_type == cmd_neighbors){
				for(int i=0;i<vec.size();i++){
					vector<edge_row> edges=g.kstore.readGlobal(cfg->t_id,vec[i].id,dir);
					for(int k=0;k<edges.size();k++){
						if(target_id==edges[k].predict){
							new_vec.push_back(path_node(edges[k].vid,i));
						}
					}
				}
				r.result_paths.push_back(vec);
				vec=new_vec;
			} else {
				assert(false);
			}
		}
		//end of trying to execute using one-side RDMA here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		
		if(r.cmd_chains.size()==0){
			// end here
			r.result_paths.push_back(vec);
			return ;
		} else {
			//recursive execute 
			r.blocking=true;
			vector<request> sub_reqs=split_request(vec,r);
			req_queue.put_req(r,sub_reqs.size());
			//concurrent_req_queue.put_req(r,sub_reqs.size());
			for(int i=0;i<sub_reqs.size();i++){
				int traverser_id=cfg->t_id;
				SendReq(cfg,i ,traverser_id, sub_reqs[i],&split_profile);
			}
			//merge_reqs(sub_reqs,r);
		}	
	} 
	void run(){	
		while(true){
			//request r=node->RecvReq();
			request r=RecvReq(cfg);
			//node->SendReq(r.parent_id + cfg->m_num ,0, r);
			//continue;

			if(r.req_id==-1){ //it means r is a request and shoule be executed
				//r.req_id=get_id();
				r.req_id=cfg->get_inc_id();

				handle_request(r);
				if(!r.blocking){
					if(cfg->is_client(r.parent_id)){
						//less print 
						if(cfg->m_id==0 && cfg->t_id==cfg->client_num)
						split_profile.report_msgsize();
					}
					SendReq(cfg,cfg->mid_of(r.parent_id) ,cfg->tid_of(r.parent_id), r,&split_profile);
				}
			} else {
				//if(concurrent_req_queue.put_reply(r)){
				if(req_queue.put_reply(r)){
					if(cfg->is_client(r.parent_id)){
						//less print 
						if(cfg->m_id==0 && cfg->t_id==cfg->client_num)
						split_profile.report_msgsize();
					}
					SendReq(cfg,cfg->mid_of(r.parent_id) ,cfg->tid_of(r.parent_id), r,&split_profile);
				}
			}
		}

	}
};