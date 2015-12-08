#pragma once
#include "graph.h"
#include "request.h"
#include "request_queue.h"
#include "network_node.h"
#include "message_wrap.h"
#include "profile.h"
#include "global_cfg.h"
#include "ingress.h"
#include <set>
//#include <unordered_set>
#include <boost/unordered_set.hpp>
#include <boost/container/set.hpp>
#include <map>
//traverser will remember all the paths just like
//traverser_keeppath in single machine

class traverser{
	graph& g;
	concurrent_request_queue& concurrent_req_queue;
	request_queue req_queue;
	thread_cfg* cfg;
	profile split_profile;
	vector<request> msg_fast_path;
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
			int edge_num=0;
			edge_row* edge_ptr=g.kstore.readLocal_predict(cfg->t_id, prev_id,dir,predict_id,&edge_num);
			for(int k=0;k<edge_num;k++){
				if(predict_id==edge_ptr[k].predict ){
					vec.push_back(path_node(edge_ptr[k].vid,i));
				}	
			}
		}
		return vec;
	}
	vector<path_node> do_triangle(request& r){
		//vertex_set triangle d1 p1 d2 p2 d3 p3
		// find all matching 
		// v0 belongs to vertex_set
		// v0 d1,p1 vsrc
		// vsrc d2,p2 vdst
		// vdst d3,p3 v0
		vector<path_node> vec1;
		vector<path_node> vec2;
		r.cmd_chains.pop_back();
		int type_src_id=r.cmd_chains.back();r.cmd_chains.pop_back();
		int type_dst_id=r.cmd_chains.back();r.cmd_chains.pop_back();
		int dir1=r.cmd_chains.back();r.cmd_chains.pop_back();
		int	pre1=r.cmd_chains.back();r.cmd_chains.pop_back();	
		int dir2=r.cmd_chains.back();r.cmd_chains.pop_back();
		int	pre2=r.cmd_chains.back();r.cmd_chains.pop_back();	
		int dir3=r.cmd_chains.back();r.cmd_chains.pop_back();
		int	pre3=r.cmd_chains.back();r.cmd_chains.pop_back();	
		//triangle == neighbor+neighbor+filter
		r.cmd_chains.push_back(pre1);
		r.cmd_chains.push_back(dir1);
		r.cmd_chains.push_back(cmd_neighbors);
		
		uint64_t t1=timer::get_usec();

		vec1=do_neighbors(r);

		uint64_t t2=timer::get_usec();
		pthread_spinlock_t triangle_lock;
		pthread_spin_init(&triangle_lock,0);

		map<uint64_t,bool> filter_src;
		typedef std::pair<uint64_t,uint64_t> v_pair;
		struct hash_vpair{
			size_t operator()(const v_pair &x) const{
				return hash<uint64_t>()(x.first) ^ hash<uint64_t>()(x.second);
			}
		};
		//set<v_pair> filter_edge;
		//unordered_set<v_pair,hash_vpair> filter_edge;
		boost::unordered_set<v_pair,hash_vpair> filter_edge;
		//boost::container::set<v_pair> filter_edge;
		for(uint64_t i=0;i<vec1.size();i++){
			int edge_num=0;
			edge_row* edge_ptr;
			if(filter_src.find(vec1[i].id)!=filter_src.end()){
				continue;
			}
			edge_ptr=g.kstore.readGlobal_predict(cfg->t_id,
											vec1[i].id,para_out,global_rdftype_id,&edge_num);
			bool found=false;
			for(int k=0;k<edge_num;k++){
				if(global_rdftype_id==edge_ptr[k].predict && 
						g.ontology_table.is_subtype_of(edge_ptr[k].vid,type_src_id)){
					found=true;
					filter_src[vec1[i].id]=true;
					break;
				}
			}
			if(!found){
				filter_src[vec1[i].id]=false;
				continue;
			}
			edge_num=0;
			edge_ptr=g.kstore.readGlobal_predict(cfg->t_id,
											vec1[i].id,dir2,pre2,&edge_num);
			for(int k=0;k<edge_num;k++){
				if(pre2==edge_ptr[k].predict ){
					filter_edge.insert(v_pair(vec1[i].id,edge_ptr[k].vid));
				}	
			}
		}
		filter_edge.rehash(filter_edge.size()*2);
		cout<<"filter_edge.size()="<<filter_edge.size()<<endl;
		uint64_t t3=timer::get_usec();
		int path_len=r.result_paths.size();
		vector<path_node>& prev_vec = r.result_paths[path_len-1];
		//#pragma omp parallel for num_threads(8)
		#pragma omp parallel for num_threads(8)
		for(uint64_t i=0;i<vec1.size();i++){
			uint64_t prev_index=vec1[i].prev;
			uint64_t prev_id=prev_vec[prev_index].id;
			int edge_num=0;
			edge_row* edges=g.kstore.readLocal_predict(cfg->t_id, prev_id,reverse_dir(dir3),pre3,&edge_num);
			for(int k=0;k<edge_num;k++){
				if(pre3==edges[k].predict ){
					//check (vec1[i].id, edges[k].vid)
					v_pair e= v_pair(vec1[i].id,edges[k].vid);
					for(int i=0;i<10;i++)
						hash_vpair()(e);
					if(filter_edge.find(e)!=filter_edge.end()){
						pthread_spin_lock(&triangle_lock);
						vec2.push_back(path_node(edges[k].vid,i));
						pthread_spin_unlock(&triangle_lock);
					}
				}	
			}
		}
		uint64_t t4=timer::get_usec();
		r.result_paths.push_back(vec1);
		vector<path_node> new_vec2;
		map<uint64_t,bool> filter_dst;
		for(uint64_t i=0;i<vec2.size();i++){
			int edge_num=0;
			edge_row* edge_ptr;
			if(filter_dst.find(vec2[i].id)==filter_dst.end()){
				filter_dst[vec2[i].id]=false;
				edge_ptr=g.kstore.readGlobal_predict(cfg->t_id,
											vec2[i].id,para_out,global_rdftype_id,&edge_num);
				for(int k=0;k<edge_num;k++){
					if(global_rdftype_id==edge_ptr[k].predict && 
							g.ontology_table.is_subtype_of(edge_ptr[k].vid,type_dst_id)){
						filter_dst[vec2[i].id]=true;
						break;
					}
				}
			}
			if(filter_dst[vec2[i].id])
				new_vec2.push_back(vec2[i]);
		}

		uint64_t t5=timer::get_usec();
		// cout<<"[triangle]: do_neighbors "<<(t2-t1)/1000.0<<"ms "<<endl;
		// cout<<"[triangle]: filter_edge "<<(t3-t2)/1000.0<<"ms "<<endl;
		// cout<<"[triangle]: construct vec2 "<<(t4-t3)/1000.0<<"ms "<<endl;
		// cout<<"[triangle]: filter vec2 "<<(t5-t4)/1000.0<<"ms "<<endl;
		return new_vec2;
	}
	void do_subclass_of(request& r){
		int predict_id=global_rdftype_id;

		r.cmd_chains.pop_back();
		int target_id=r.cmd_chains.back();
		r.cmd_chains.pop_back();

		int path_len=r.result_paths.size();
		vector<path_node>& prev_vec = r.result_paths[path_len-1];
		vector<path_node> new_vec;
		for (int i=0;i<prev_vec.size();i++){
			int prev_id=prev_vec[i].id;	
			int edge_num=0;
			edge_row* edge_ptr=g.kstore.readLocal_predict(cfg->t_id, prev_id,para_out,predict_id,&edge_num);
			for(int k=0;k<edge_num;k++){
				if(predict_id==edge_ptr[k].predict && 
							g.ontology_table.is_subtype_of(edge_ptr[k].vid,target_id)){
					new_vec.push_back(prev_vec[i]);
				}	
			}
		}
		r.result_paths[path_len-1]=new_vec;
	}
	void do_get_attr(request& r,int dir=para_out){
		r.cmd_chains.pop_back();
		int predict_id=r.cmd_chains.back();
		r.cmd_chains.pop_back();

		int path_len=r.result_paths.size();
		vector<path_node>& prev_vec = r.result_paths[path_len-1];
		vector<path_node> new_vec_attr;
		vector<path_node> new_vec_id;
		for (int i=0;i<prev_vec.size();i++){
			int prev_id=prev_vec[i].id;	
			int edge_num=0;
			edge_row* edge_ptr=g.kstore.readLocal_predict(cfg->t_id, prev_id,dir,predict_id,&edge_num);
			for(int k=0;k<edge_num;k++){
				if(predict_id==edge_ptr[k].predict ){
					new_vec_attr.push_back(path_node(edge_ptr[k].vid,i));
					new_vec_id.push_back(path_node(prev_id ,new_vec_attr.size()-1));
					break;
				}	
			}
		}
		r.result_paths.push_back(new_vec_attr);
		r.result_paths.push_back(new_vec_id);
		//r.result_paths[path_len-1]=new_vec;
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
			int machine = ingress::vid2mid(vec[i].id , num_sub_request);
			sub_reqs[machine].result_paths[0].push_back(vec[i]);
		}
		return sub_reqs;
	}
	vector<vector<request> > split_request_mt(vector<path_node>& vec,request& r){
		vector<vector<request> > sub_reqs;
		sub_reqs.resize(cfg->m_num);
		for(int i=0;i<sub_reqs.size();i++){
			sub_reqs[i].resize(cfg->server_num);
			for(int j=0;j<sub_reqs[i].size();j++){
				sub_reqs[i][j].parent_id=r.req_id;
				sub_reqs[i][j].cmd_chains=r.cmd_chains;
				sub_reqs[i][j].result_paths.push_back(vector<path_node>());
			}
		}
		unsigned int seed=cfg->m_id*cfg->t_num+cfg->t_id;
		for(int i=0;i<vec.size();i++){
			int machine = ingress::vid2mid(vec[i].id, cfg->m_num);
			int thread  = rand_r(&seed) % cfg->server_num;
			sub_reqs[machine][thread].result_paths[0].push_back(vec[i]);
		}
		return sub_reqs;
	}
public:
	traverser(graph& gg,concurrent_request_queue& crq,thread_cfg* _cfg)
			:g(gg),concurrent_req_queue(crq),cfg(_cfg){
		req_id=cfg->m_id+cfg->t_id*cfg->m_num;
	}
	void try_rdma_execute(request& r,vector<path_node>& vec){
		while(r.cmd_chains.size()!=0 ){

			if(vec.size()>global_tuning_threshold){
				vec.resize(global_tuning_threshold);
			}
		

			split_profile.neighbor_num+=vec.size();
			if(vec.size()<global_rdma_threshold){
				split_profile.split_req++;
			} else {
				split_profile.non_split_req++;
			}
			if(vec.size()>=global_rdma_threshold)
				break;

			int dir=para_out;			
			int cmd_type=r.cmd_chains.back();
			if(cmd_type==cmd_triangle){
				//not supported now.
				return ;
			}
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
					int edge_num=0;
					//edge_row* edges=g.kstore.readGlobal(cfg->t_id,vec[i].id,dir,&edge_num);
					edge_row* edges=g.kstore.readGlobal_predict(cfg->t_id,
											vec[i].id,dir,global_rdftype_id,&edge_num);
					for(int k=0;k<edge_num;k++){
						if(global_rdftype_id==edges[k].predict && 
							g.ontology_table.is_subtype_of(edges[k].vid,target_id)){
							new_vec.push_back(vec[i]);
							break;
						}	
					}
				}
				vec=new_vec; //replace old vec since subclass is a filter operation
			} else if(cmd_type == cmd_get_attr){
				vector<path_node> new_vec_attr;
				vector<path_node> new_vec_id;
				for(int i=0;i<vec.size();i++){
					int edge_num=0;
					//edge_row* edges=g.kstore.readGlobal(cfg->t_id,vec[i].id,dir,&edge_num);
					edge_row* edges=g.kstore.readGlobal_predict(cfg->t_id,
											vec[i].id,dir,target_id,&edge_num);
					for(int k=0;k<edge_num;k++){
						if(target_id==edges[k].predict) {
							new_vec_attr.push_back(path_node(edges[k].vid,i));
							new_vec_id.push_back(path_node(vec[i].id,new_vec_attr.size()-1));
							break;
						}	
					}
				}
				r.result_paths.push_back(vec);
				r.result_paths.push_back(new_vec_attr);
				vec=new_vec_id;
			} else if(cmd_type == cmd_neighbors){
				for(int i=0;i<vec.size();i++){
					int edge_num=0;
					//edge_row* edges=g.kstore.readGlobal(cfg->t_id,vec[i].id,dir,&edge_num);
					edge_row* edges=g.kstore.readGlobal_predict(cfg->t_id,
											vec[i].id,dir,target_id,&edge_num);
					for(int k=0;k<edge_num;k++){
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
		} else if(r.cmd_chains.back() == cmd_get_attr){
			// get_attr is similar to subclass_of
			// it just remove some of the output
			do_get_attr(r);
			handle_request(r);
			return ;
		} else if(r.cmd_chains.back() == cmd_neighbors){
			vec=do_neighbors(r);
		} else if(r.cmd_chains.back() == cmd_triangle){
			vec=do_triangle(r);
		} else if(r.cmd_chains.back() == cmd_get_subtype){
			assert(r.path_length()==0);
			vec=do_get_subtype(r);
		} else{
			assert(false);
		}
		
		//Tuning the threshold; should be remove because it will throw away part of result
		//trying to execute using one-side RDMA here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		if(global_use_rdma){
			try_rdma_execute(r,vec);
		}
		//end of trying to execute using one-side RDMA here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		
		if(r.cmd_chains.size()==0){
			// end here
			r.result_paths.push_back(vec);
			return ;
		} else {
			//recursive execute 
			r.blocking=true;
			
			if(global_use_multithread ){
			//if(global_use_multithread && vec.size() >= global_rdma_threshold * cfg->t_num){
				vector<vector<request> > sub_reqs=split_request_mt(vec,r);
				req_queue.put_req(r, cfg->m_num * cfg->server_num);
				for(int i=0;i<sub_reqs.size();i++){
					for(int j=0;j<sub_reqs[i].size();j++){
						//int traverser_id=cfg->t_id;
						int traverser_id=cfg->client_num+j;
						SendReq(cfg,i ,traverser_id, sub_reqs[i][j],&split_profile);
					}
				}
			} else {
				vector<request> sub_reqs=split_request(vec,r);
				req_queue.put_req(r,sub_reqs.size());
				//uint64_t t1=timer::get_usec();
				for(int i=0;i<sub_reqs.size();i++){
					int traverser_id=cfg->t_id;	
					if(i == cfg->m_id ){
						msg_fast_path.push_back(sub_reqs[i]);
					} else {
						SendReq(cfg,i ,traverser_id, sub_reqs[i],&split_profile);
					}
				}
				//uint64_t t2=timer::get_usec();
				//cout<<t2-t1<<" usec in sending messages!!!!!!!!!!!!!!!!!!"<<endl;
			}
			
			//merge_reqs(sub_reqs,r);
		}	
	} 
	void run(){	
		//uint64_t t1;//timer::get_usec();;
		while(true){
			request r;
			if(msg_fast_path.size()>0){
				r=msg_fast_path.back();
				msg_fast_path.pop_back();
			}
			else {
				r=RecvReq(cfg);
			}
			if(r.req_id==-1){ //it means r is a request and shoule be executed
				//r.req_id=get_id();
				r.req_id=cfg->get_inc_id();
				if(cfg->is_client(r.parent_id)){
					//t1=timer::get_usec();
					//r.timestamp=timer::get_usec();
				}
				handle_request(r);
				if(!r.blocking){
					if(cfg->is_client(r.parent_id)){
						//less print 
						//if(cfg->m_id==0 && cfg->t_id==cfg->client_num)
						//	split_profile.report_msgsize();
						//uint64_t timestamp=timer::get_usec();
						//split_profile.record_and_report_latency(timestamp-r.timestamp);
						//if(global_interactive)
						//	cout<<"without send back to user :"<<timestamp-r.timestamp<<endl;
						//r.timestamp=timestamp-r.timestamp;
											
						if(global_clear_final_result){
							r.result_paths.clear();
						}
					}
					if(cfg->mid_of(r.parent_id)== cfg->m_id && cfg->tid_of(r.parent_id)==cfg->t_id){
						msg_fast_path.push_back(r);
					} else {
						SendReq(cfg,cfg->mid_of(r.parent_id) ,cfg->tid_of(r.parent_id), r,&split_profile);
					}
				}
			} else {
				//if(concurrent_req_queue.put_reply(r)){
				if(req_queue.put_reply(r)){
					if(cfg->is_client(r.parent_id)){
						//less print 
						//if(cfg->m_id==0 && cfg->t_id==cfg->client_num)
						//	split_profile.report_msgsize();
						//uint64_t timestamp=timer::get_usec();
						//split_profile.record_and_report_latency(timestamp-r.timestamp);
						//if(global_interactive)
						//	cout<<"without send back to user :"<<timestamp-r.timestamp<<endl;
						//r.timestamp=timestamp-r.timestamp;
						
						if(global_clear_final_result){
							r.result_paths.clear();
						}
					}
					if(cfg->mid_of(r.parent_id)== cfg->m_id && cfg->tid_of(r.parent_id)==cfg->t_id){
						msg_fast_path.push_back(r);
					} else {
						SendReq(cfg,cfg->mid_of(r.parent_id) ,cfg->tid_of(r.parent_id), r,&split_profile);
					}
				}
			}
		}

	}
};