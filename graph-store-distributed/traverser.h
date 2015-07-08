#pragma once
#include "graph.h"
#include "request.h"
#include "request_queue.h"

//traverser will remember all the paths just like
//traverser_keeppath in single machine

class traverser{
	graph& g;
	boost::mpi::communicator& world;
	request_queue req_queue;
	int req_id;
	int get_id(){
		int result=req_id;
		req_id+=world.size();
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
			if(dir ==para_in || dir == para_all){
				for (auto row : g.vertex_table[prev_id].in_edges){
					if(predict_id==row.predict){
						vec.push_back(path_node(row.vid,i));
					}
				}
			}
			if(dir ==para_out || dir == para_all){
				for (auto row : g.vertex_table[prev_id].out_edges){
					if(predict_id==row.predict){
						vec.push_back(path_node(row.vid,i));
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
			for (auto row : g.vertex_table[prev_id].out_edges){
				if(predict_id==row.predict && 
							g.ontology_table.is_subtype_of(row.vid,target_id)){
					new_vec.push_back(prev_vec[i]);
				}
			}
		}
		r.result_paths[path_len-1]=new_vec;
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
		int num_sub_request=world.size()-1;
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
	traverser(boost::mpi::communicator& para_world,graph& gg):world(para_world),g(gg){
		req_id=world.rank();
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
		}

		if(r.cmd_chains.size()==0){
			// end here
			r.result_paths.push_back(vec);
			return ;
		} else {
			//recursive execute 
			r.blocking=true;
			vector<request> sub_reqs=split_request(vec,r);
			for(int i=0;i<sub_reqs.size();i++){
				//handle_request(sub_reqs[i]);
				world.send(i, 1, sub_reqs[i]);
			}
			req_queue.put_req(r,sub_reqs.size());
			//merge_reqs(sub_reqs,r);
		}	
	} 
	void run(){
		while(true){
			request r;
			boost::mpi::status s =world.recv(boost::mpi::any_source, 1, r);
			cout<< world.rank()<<" recv a request from " <<s.source()<<endl;
			if(r.req_id==-1){
				r.req_id=get_id();
				handle_request(r);
				if(!r.blocking){
					cout<< "success execute, result= " <<  r.path_num() <<endl;
					world.send(r.parent_id % world.size() , 1, r);
				}
			} else {
				if(req_queue.put_reply(r))
					world.send(r.parent_id % world.size() , 1, r);
			}
		}

	}
};