#pragma once
#include "graph.h"
#include "request.h"
#include "request_queue.h"
#include "network_node.h"
#include "message_wrap.h"
#include "profile.h"
#include "global_cfg.h"
#include "ingress.h"
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include "simple_filter.h"
//traverser will remember all the paths just like
//traverser_keeppath in single machine

class traverser{
	graph& g;
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

	void do_neighbors(request& r){
		r.cmd_chains.pop_back();
		int dir=r.cmd_chains.back();
		r.cmd_chains.pop_back();
		int	predict_id=r.cmd_chains.back();
		r.cmd_chains.pop_back();

		vector<vector<int> >updated_result_table;
		updated_result_table.resize(r.column_num()+1);
		
		for (int i=0;i<r.row_num();i++){
			int prev_id=r.last_column(i);
			int edge_num=0;
			edge_row* edge_ptr;
			//edge_ptr=g.kstore.readLocal_predict(cfg->t_id, prev_id,dir,predict_id,&edge_num); 
			edge_ptr=g.kstore.readGlobal_predict(cfg->t_id, prev_id,dir,predict_id,&edge_num); 
			for(int k=0;k<edge_num;k++){
				if(predict_id==edge_ptr[k].predict ){
					r.append_row_to(updated_result_table,i);
					updated_result_table[r.column_num()].push_back(edge_ptr[k].vid);
				}	
			}
		}
		r.result_table.swap(updated_result_table);
	}

	void do_subclass_of(request& r){
		int predict_id=global_rdftype_id;
		r.cmd_chains.pop_back();
		int target_id=r.cmd_chains.back();
		r.cmd_chains.pop_back();

		vector<vector<int> >updated_result_table;
		updated_result_table.resize(r.column_num());

		for (int i=0;i<r.row_num();i++){
			int prev_id=r.last_column(i);	
			int edge_num=0;
			edge_row* edge_ptr=g.kstore.readGlobal_predict(cfg->t_id, prev_id,para_out,predict_id,&edge_num);
			for(int k=0;k<edge_num;k++){
				if(predict_id==edge_ptr[k].predict && 
							g.ontology_table.is_subtype_of(edge_ptr[k].vid,target_id)){
					r.append_row_to(updated_result_table,i);
				}	
			}
		}
		r.result_table.swap(updated_result_table);
	}
	void do_get_attr(request& r,int dir=para_out){
		r.cmd_chains.pop_back();
		int predict_id=r.cmd_chains.back();
		r.cmd_chains.pop_back();

		vector<vector<int> >updated_result_table;
		updated_result_table.resize(r.column_num()+1);
		
		for (int i=0;i<r.row_num();i++){
			int prev_id=r.last_column(i);
			int edge_num=0;
			//edge_row* edge_ptr=g.kstore.readLocal_predict(cfg->t_id, prev_id,dir,predict_id,&edge_num);
			edge_row* edge_ptr=g.kstore.readGlobal_predict(cfg->t_id, prev_id,dir,predict_id,&edge_num);
			for(int k=0;k<edge_num;k++){
				if(predict_id==edge_ptr[k].predict ){
					r.append_row_to(updated_result_table,i);
					updated_result_table[r.column_num()].push_back(edge_ptr[k].vid);
				}	
			}
		}
		r.result_table.swap(updated_result_table);
		r.result_table[r.column_num()-1].swap(r.result_table[r.column_num()-2]);
	}
	void do_predict_index(request& r){
		r.cmd_chains.pop_back();
		int predict_id=r.cmd_chains.back();
		r.cmd_chains.pop_back();
		int dir=r.cmd_chains.back();
		r.cmd_chains.pop_back();
		const boost::unordered_set<uint64_t>& ids=g.kstore.get_predict_index(predict_id,dir);
		r.result_table.resize(1);
		for(auto id: ids){
			r.result_table[0].push_back(id);
		}
		return ;
	}
	vector<request> split_request(request& r){
		vector<request> sub_reqs;
		int num_sub_request=cfg->m_num;
		sub_reqs.resize(num_sub_request);
		for(int i=0;i<sub_reqs.size();i++){
			sub_reqs[i].parent_id=r.req_id;
			sub_reqs[i].cmd_chains=r.cmd_chains;
			sub_reqs[i].result_table.resize(r.column_num());
		}
		for(int i=0;i<r.row_num();i++){
			int machine = ingress::vid2mid(r.last_column(i), num_sub_request);
			r.append_row_to(sub_reqs[machine].result_table,i);
		}
		r.result_table.clear();
		return sub_reqs;
	}

	void do_triangle(request& r){
		//vertex_set triangle d1 p1 d2 p2 d3 p3
		// find all matching 
		// v0 belongs to vertex_set
		// v0 d1,p1 vsrc
		// vsrc d2,p2 vdst
		// vdst d3,p3 v0
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
		cout<<"[triangle]: number of r.row_num() = "<<r.row_num()<<endl;
		
		uint64_t t1=timer::get_usec();

		do_neighbors(r);

		cout<<"[triangle]: number of r.row_num() after do_neighbors = "<<r.row_num()<<endl;

		uint64_t t2=timer::get_usec();
		pthread_spinlock_t triangle_lock;
		pthread_spin_init(&triangle_lock,0);

		//boost::unordered_set<uint64_t> filter_src;
		boost::unordered_map<uint64_t,bool> filter_src;
		boost::unordered_map<uint64_t,bool> filter_dst;

		simple_filter filter_edge;
		for(uint64_t i=0;i<r.row_num();i++){
			int edge_num=0;
			edge_row* edge_ptr;
			if(filter_src.find(r.last_column(i))!=filter_src.end()){
				continue;
			}
			filter_src[r.last_column(i)]=true;					

			edge_ptr=g.kstore.readGlobal_predict(cfg->t_id,
										r.last_column(i),para_out,global_rdftype_id,&edge_num);
			bool found=false;
			for(int k=0;k<edge_num;k++){
				if(global_rdftype_id==edge_ptr[k].predict && 
						g.ontology_table.is_subtype_of(edge_ptr[k].vid,type_src_id)){
					found=true;
					filter_src[r.last_column(i)]=true;
					break;
				}
			}
			if(!found){
				filter_src[r.last_column(i)]=false;
				continue;
			}

			edge_num=0;
			edge_ptr=g.kstore.readGlobal_predict(cfg->t_id,
											r.last_column(i),dir2,pre2,&edge_num);
			for(int k=0;k<edge_num;k++){
				if(pre2==edge_ptr[k].predict ){
					filter_edge.insert(r.last_column(i),edge_ptr[k].vid);
				}	
			}
		}
		cout<<"[triangle]: filter_edge.vec_id.size() = "<<filter_edge.vec_id.size()<<endl;

		filter_edge.rehash();
		uint64_t t3=timer::get_usec();
		
		vector<vector<int> >updated_result_table;
		updated_result_table.resize(r.column_num()+1);

		vector<int>& prev_vec = r.result_table[r.column_num()-2];
		//#pragma omp parallel for num_threads(8)
		for(uint64_t i=0;i<r.row_num();i++){
			uint64_t prev_id=prev_vec[i];
			int edge_num=0;
			edge_row* edges=g.kstore.readLocal_predict(cfg->t_id,prev_id,reverse_dir(dir3),pre3,&edge_num);
			for(int k=0;k<edge_num;k++){
				if(pre3==edges[k].predict ){
					if(filter_edge.contain(r.last_column(i),edges[k].vid)) {
						pthread_spin_lock(&triangle_lock);
						r.append_row_to(updated_result_table,i);
						updated_result_table[r.column_num()].push_back(edges[k].vid);
						pthread_spin_unlock(&triangle_lock);
					}
				}	
			}
		}
		r.result_table.swap(updated_result_table);
		uint64_t t4=timer::get_usec();
		r.cmd_chains.push_back(type_dst_id);
		r.cmd_chains.push_back(cmd_subclass_of);
		do_subclass_of(r);
		cout<<"[triangle]: number of r.row_num() = "<<r.row_num()<<endl;
		
		uint64_t t5=timer::get_usec();
		cout<<"[triangle]: do_neighbors "<<(t2-t1)/1000.0<<"ms "<<endl;
		cout<<"[triangle]: filter_edge "<<(t3-t2)/1000.0<<"ms "<<endl;
		cout<<"[triangle]: construct triples "<<(t4-t3)/1000.0<<"ms "<<endl;
		cout<<"[triangle]: do_subclass_of "<<(t5-t4)/1000.0<<"ms "<<endl;
		return ;
	}

	void try_rdma_execute(request& r){
		while(r.cmd_chains.size()!=0 ){	
			split_profile.neighbor_num+=r.row_num();
			if(r.row_num()<global_rdma_threshold){
				split_profile.split_req++;
			} else {
				split_profile.non_split_req++;
			}
			if(r.row_num()>=global_rdma_threshold){
				break;
			}

			int dir=para_out;			
			int cmd_type=r.cmd_chains.back();
			if(cmd_type==cmd_triangle){
				//not supported now.
				return ;
			}
			if(r.cmd_chains.back() == cmd_subclass_of){
				do_subclass_of(r);
			} else if(r.cmd_chains.back() == cmd_get_attr){
				do_get_attr(r);
			} else if(r.cmd_chains.back() == cmd_neighbors){
				do_neighbors(r);
			} 
		}
	}
public:
	traverser(graph& gg,thread_cfg* _cfg)
			:g(gg),cfg(_cfg){
		req_id=cfg->m_id+cfg->t_id*cfg->m_num;
	}
	
	void handle_request(request& r){
		if(r.cmd_chains.size()==0)
			return;
		if(r.cmd_chains.back() == cmd_subclass_of){
			// subclass_of is a filter operation
			// it just remove some of the output
			do_subclass_of(r);
			handle_request(r);
			return ;
		} else if(r.cmd_chains.back() == cmd_get_attr){
			do_get_attr(r);
			handle_request(r);
			return ;
		} else if(r.cmd_chains.back() == cmd_triangle){
			do_triangle(r);
		} else if(r.cmd_chains.back() == cmd_neighbors){
			do_neighbors(r);
		} else if(r.cmd_chains.back() == cmd_predict_index){
			assert(r.column_num()==0);
			do_predict_index(r);
		} else{
			assert(false);
		}
		
		//Tuning the threshold; should be remove because it will throw away part of result
		//trying to execute using one-side RDMA here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		if(global_use_rdma){
			try_rdma_execute(r);
		}
		//end of trying to execute using one-side RDMA here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		
		if(r.cmd_chains.size()==0){
			// end here
			return ;
		} else {
			//recursive execute 
			r.blocking=true;
			
			if(global_use_multithread ){
				assert(false);
			} else {
				vector<request> sub_reqs=split_request(r);
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
			}	
		}	
	} 
	void run(){	
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
				r.req_id=cfg->get_inc_id();
				handle_request(r);
				if(!r.blocking){
					if(cfg->is_client(r.parent_id)){
						if(global_clear_final_result){
							r.result_table.clear();
						}
					}
					if(cfg->mid_of(r.parent_id)== cfg->m_id && cfg->tid_of(r.parent_id)==cfg->t_id){
						msg_fast_path.push_back(r);
					} else {
						SendReq(cfg,cfg->mid_of(r.parent_id) ,cfg->tid_of(r.parent_id), r,&split_profile);
					}
				}
			} else {
				if(req_queue.put_reply(r)){
					if(cfg->is_client(r.parent_id)){
						if(global_clear_final_result){
							r.result_table.clear();
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