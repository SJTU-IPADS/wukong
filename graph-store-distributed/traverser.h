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
			edge_v2* edge_ptr;
			//edge_ptr=g.kstore.readLocal_predict(cfg->t_id, prev_id,dir,predict_id,&edge_num); 
			edge_ptr=g.kstore.readGlobal_predict(cfg->t_id, prev_id,dir,predict_id,&edge_num); 
			for(int k=0;k<edge_num;k++){
				r.append_row_to(updated_result_table,i);
				updated_result_table[r.column_num()].push_back(edge_ptr[k].val);
					
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
			edge_v2* edge_ptr=g.kstore.readGlobal_predict(cfg->t_id, prev_id,para_out,predict_id,&edge_num);
			for(int k=0;k<edge_num;k++){
				if(g.ontology_table.is_subtype_of(edge_ptr[k].val,target_id)){
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
			edge_v2* edge_ptr=g.kstore.readGlobal_predict(cfg->t_id, prev_id,dir,predict_id,&edge_num);
			for(int k=0;k<edge_num;k++){
				r.append_row_to(updated_result_table,i);
				updated_result_table[r.column_num()].push_back(edge_ptr[k].val);	
			}
		}
		r.result_table.swap(updated_result_table);
		r.result_table[r.column_num()-1].swap(r.result_table[r.column_num()-2]);
	}
	void do_filter(request& r){
		r.cmd_chains.pop_back();
		int dir=r.cmd_chains.back();
		r.cmd_chains.pop_back();
		int	predict_id=r.cmd_chains.back();
		r.cmd_chains.pop_back();
		int target_column=r.cmd_chains.back();
		r.cmd_chains.pop_back();
		
		vector<vector<int> >updated_result_table;
		updated_result_table.resize(r.column_num());

		for (int i=0;i<r.row_num();i++){
			int prev_id=r.last_column(i);
			int target_id=r.get(i,target_column);	
			int edge_num=0;
			edge_v2* edge_ptr=g.kstore.readGlobal_predict(cfg->t_id, prev_id,para_out,predict_id,&edge_num);
			for(int k=0;k<edge_num;k++){
				if(edge_ptr[k].val == target_id){
					r.append_row_to(updated_result_table,i);
					break;
				}
			}
		}
		r.result_table.swap(updated_result_table);
	}
	// void async_do_subclass_of(request& r){
	// 	int predict_id=global_rdftype_id;
	// 	r.cmd_chains.pop_back();
	// 	int target_id=r.cmd_chains.back();
	// 	r.cmd_chains.pop_back();

	// 	vector<vector<int> >updated_result_table;
	// 	updated_result_table.resize(r.column_num());
	// 	vector<edge_row*> edge_ptr_vec;
	// 	vector<int> size_vec;
	// 	g.kstore.batch_readGlobal_predict(cfg->t_id,r.result_table[r.column_num()-1],
	// 												para_out,predict_id,edge_ptr_vec,size_vec);
	// 	for (int i=0;i<r.row_num();i++){
	// 		int prev_id=r.last_column(i);	
	// 		edge_row* edge_ptr=edge_ptr_vec[i];
	// 		int edge_num=size_vec[i];
	// 		for(int k=0;k<edge_num;k++){
	// 			if(predict_id==edge_ptr[k].predict && 
	// 						g.ontology_table.is_subtype_of(edge_ptr[k].vid,target_id)){
	// 				r.append_row_to(updated_result_table,i);
	// 			}	
	// 		}
	// 	}
		
	// 	r.result_table.swap(updated_result_table);
	// }
	// void async_do_neighbors(request& r){
	// 	r.cmd_chains.pop_back();
	// 	int dir=r.cmd_chains.back();
	// 	r.cmd_chains.pop_back();
	// 	int	predict_id=r.cmd_chains.back();
	// 	r.cmd_chains.pop_back();

	// 	vector<vector<int> >updated_result_table;
	// 	updated_result_table.resize(r.column_num()+1);
		
	// 	vector<edge_row*> edge_ptr_vec;
	// 	vector<int> size_vec;
	// 	g.kstore.batch_readGlobal_predict(cfg->t_id,r.result_table[r.column_num()-1],
	// 												dir,predict_id,edge_ptr_vec,size_vec);

	// 	for (int i=0;i<r.row_num();i++){
	// 		int prev_id=r.last_column(i);	
	// 		edge_row* edge_ptr=edge_ptr_vec[i];
	// 		int edge_num=size_vec[i];
	// 		for(int k=0;k<edge_num;k++){
	// 			if(predict_id==edge_ptr[k].predict){
	// 				r.append_row_to(updated_result_table,i);
	// 				updated_result_table[r.column_num()].push_back(edge_ptr[k].vid);
	// 			}	
	// 		}
	// 	}
	// 	r.result_table.swap(updated_result_table);
	// }
	// void async_do_get_attr(request& r,int dir=para_out){
	// 	r.cmd_chains.pop_back();
	// 	int predict_id=r.cmd_chains.back();
	// 	r.cmd_chains.pop_back();

	// 	vector<vector<int> >updated_result_table;
	// 	updated_result_table.resize(r.column_num()+1);

	// 	vector<edge_row*> edge_ptr_vec;
	// 	vector<int> size_vec;
	// 	g.kstore.batch_readGlobal_predict(cfg->t_id,r.result_table[r.column_num()-1],
	// 												dir,predict_id,edge_ptr_vec,size_vec);
	// 	for (int i=0;i<r.row_num();i++){
	// 		int prev_id=r.last_column(i);	
	// 		edge_row* edge_ptr=edge_ptr_vec[i];
	// 		int edge_num=size_vec[i];
	// 		for(int k=0;k<edge_num;k++){
	// 			if(predict_id==edge_ptr[k].predict){
	// 				r.append_row_to(updated_result_table,i);
	// 				updated_result_table[r.column_num()].push_back(edge_ptr[k].vid);
	// 			}	
	// 		}
	// 	}
		
	// 	r.result_table.swap(updated_result_table);
	// 	r.result_table[r.column_num()-1].swap(r.result_table[r.column_num()-2]);
	// }
	
	void do_predict_index(request& r){
		assert(false);
		// r.cmd_chains.pop_back();
		// int predict_id=r.cmd_chains.back();
		// r.cmd_chains.pop_back();
		// int dir=r.cmd_chains.back();
		// r.cmd_chains.pop_back();
		// const boost::unordered_set<uint64_t>& ids=g.kstore.get_predict_index(predict_id,dir);
		// r.result_table.resize(1);
		// for(auto id: ids){
		// 	r.result_table[0].push_back(id);
		// }
		return ;
	}
	void do_type_index(request& r){
		r.cmd_chains.pop_back();
		int type_id=r.cmd_chains.back();
		r.cmd_chains.pop_back();
		if(global_use_index_table){
			vector<uint64_t>& ids=g.kstore.get_vector(
					g.kstore.type_table,type_id);
			vector<vector<int> >updated_result_table;
			updated_result_table.resize(1);
			int start_id=cfg->t_id-cfg->client_num;
			for(int i=start_id;i<ids.size();i+=cfg->server_num){
				//int tid = cfg->client_num+ingress::hash(id) % cfg->server_num ;
				//if(tid==cfg->t_id){
				updated_result_table[0].push_back(ids[i]);
				//}
			}
			r.result_table.swap(updated_result_table);
			return ;
		}
		{//use rdma to read data
			vector<vector<int> >updated_result_table;
			updated_result_table.resize(1);
			int edge_num=0;
			edge_v2* edge_ptr;
			edge_ptr=g.kstore.readGlobal_predict(cfg->t_id,
					type_id,para_in,global_rdftype_id,&edge_num);
			for(int k=0;k<edge_num;k++){
				int mid = ingress::vid2mid(edge_ptr[k].val, cfg->m_num);
				int tid = cfg->client_num+ingress::hash(edge_ptr[k].val) % cfg->server_num ;
				if(mid==cfg->m_id && tid==cfg->t_id){
					updated_result_table[0].push_back(edge_ptr[k].val);
				}
			}
			r.result_table.swap(updated_result_table);
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
	vector<request> split_request_mt(request& r){
		vector<request> sub_reqs;
		int num_sub_request=cfg->m_num * cfg->server_num ;
		sub_reqs.resize(num_sub_request);
		for(int i=0;i<sub_reqs.size();i++){
			sub_reqs[i].parent_id=r.req_id;
			sub_reqs[i].cmd_chains=r.cmd_chains;
			sub_reqs[i].result_table.resize(r.column_num());
		}
		for(int i=0;i<r.row_num();i++){
			int machine = ingress::vid2mid(r.last_column(i), cfg->m_num);
			int tid = ingress::hash(r.last_column(i)) % cfg->server_num ;
			r.append_row_to(sub_reqs[machine*cfg->server_num+tid].result_table,i);
		}
		r.result_table.clear();
		return sub_reqs;
	}
	void do_triangle(request& r){
		r.cmd_chains.pop_back();
		//find all matching 
		//type_0 d0 p0
		//type_1 d1 p1
		//type_2 d2 p2
		vector<int> v_type;
		vector<int> v_dir;
		vector<int> v_predict;
		for(int i=0;i<3;i++){
			v_type.push_back(r.cmd_chains.back());
			r.cmd_chains.pop_back();
			v_dir.push_back(r.cmd_chains.back());
			r.cmd_chains.pop_back();
			v_predict.push_back(r.cmd_chains.back());
			r.cmd_chains.pop_back();
		}

		//uint64_t num_parallel_thread=global_num_server;
		uint64_t num_parallel_thread=1;
		uint64_t t1=timer::get_usec();

		//step 1 : find all type_0. type_0 is local
		r.cmd_chains.push_back(v_type[0]);
		r.cmd_chains.push_back(cmd_type_index);
		do_type_index(r);
		uint64_t t1_5=timer::get_usec();
		
		//step 2 : find all type_0,type_1
		r.cmd_chains.push_back(v_predict[0]);
		r.cmd_chains.push_back(v_dir[0]);
		r.cmd_chains.push_back(cmd_neighbors);		
		do_neighbors(r);

		uint64_t t2=timer::get_usec();
		
		//step 3 : find all type_1,type_2 and create a simple_filter
		//pthread_spinlock_t triangle_lock;
		//pthread_spin_init(&triangle_lock,0);
		vector<boost::unordered_map<uint64_t,bool> > type_filter;
		type_filter.resize(3);
		simple_filter edge_filter;
		vector<vector<v_pair> >pair_vec;
		pair_vec.resize(num_parallel_thread);
		//type_filter[1].reserve(r.row_num());
		int count_type1=0;
		//#pragma omp parallel for num_threads(num_parallel_thread)
		for(uint64_t i=0;i<r.row_num();i++){
			int working_tid = omp_get_thread_num();
			
			int edge_num=0;
			edge_v2* edge_ptr;
			//pthread_spin_lock(&triangle_lock);
			if(type_filter[1].find(r.last_column(i))!=type_filter[1].end()){
				//	pthread_spin_unlock(&triangle_lock);
				continue;
			}
			count_type1++;
			type_filter[1][r.last_column(i)]=true;					
			//pthread_spin_unlock(&triangle_lock);

			// check whether it's type_1 or not
			//if(num_parallel_thread==1){
				edge_ptr=g.kstore.readGlobal_predict(cfg->t_id,
										r.last_column(i),para_out,global_rdftype_id,&edge_num);
			// } else {
			// 	edge_ptr=g.kstore.readGlobal_predict(1+working_tid,
			// 							r.last_column(i),para_out,global_rdftype_id,&edge_num);
			// }

			bool found=false;
			for(int k=0;k<edge_num;k++){
				if(g.ontology_table.is_subtype_of(edge_ptr[k].val,v_type[1])){
					found=true;
					//pthread_spin_lock(&triangle_lock);
					type_filter[1][r.last_column(i)]=true;
					//pthread_spin_unlock(&triangle_lock);
					break;
				}
			}
			if(!found){
				//pthread_spin_lock(&triangle_lock);
				type_filter[1][r.last_column(i)]=false;
				//pthread_spin_unlock(&triangle_lock);
				continue;
			}
			// fetch and insert to edge_filter
			edge_num=0;
			//if(num_parallel_thread==1){
				edge_ptr=g.kstore.readGlobal_predict(cfg->t_id,
											r.last_column(i),v_dir[1],v_predict[1],&edge_num);
			// } else {
			// 	edge_ptr=g.kstore.readGlobal_predict(1+working_tid,
			// 								r.last_column(i),v_dir[1],v_predict[1],&edge_num);
			// }
			for(int k=0;k<edge_num;k++){
				pair_vec[working_tid].push_back(v_pair(r.last_column(i),edge_ptr[k].val));
					//edge_filter.insert(r.last_column(i),edge_ptr[k].vid);
					
			}
		}

		uint64_t t3=timer::get_usec();

		int count=0;
		for(int j=0;j<pair_vec.size();j++){
			count+=pair_vec[j].size();
		}
		edge_filter.tbb_set.rehash(2*count);
		//#pragma omp parallel for num_threads(num_parallel_thread)
		for(uint64_t i=0;i<num_parallel_thread;i++){
			for(int j=0;j<pair_vec[i].size();j++){
				tbb_hashtable::accessor a; 
				edge_filter.tbb_set.insert( a, pair_vec[i][j]);
				a->second = true;
				//edge_filter.insert(pair_vec[i][j]);
			}
		}
		//step 4 : rehash edge_filter
		//edge_filter.rehash();

		uint64_t t4=timer::get_usec();
		
		//step 5 : append type_2 to existing type_0,type_1 pair
		{
			vector<vector<int> >updated_result_table;
			updated_result_table.resize(r.column_num()+1);

			vector<int>& prev_vec = r.result_table[r.column_num()-2];
			//#pragma omp parallel for num_threads(num_parallel_thread)
			for(uint64_t i=0;i<r.row_num();i++){
				uint64_t prev_id=prev_vec[i];
				int edge_num=0;
				edge_v2* edges=g.kstore.readLocal_predict(cfg->t_id,
							prev_id,reverse_dir(v_dir[2]),v_predict[2],&edge_num);
				for(int k=0;k<edge_num;k++){
					
					if(edge_filter.contain(r.last_column(i),edges[k].val)) {
						//pthread_spin_lock(&triangle_lock);
						r.append_row_to(updated_result_table,i);
						updated_result_table[r.column_num()].push_back(edges[k].val);
						//pthread_spin_unlock(&triangle_lock);
					}
						
				}
			}
			r.result_table.swap(updated_result_table);	
		}
		uint64_t t5=timer::get_usec();
		
		//setp 6: do final filter on type_2
		r.cmd_chains.push_back(v_type[2]);
		r.cmd_chains.push_back(cmd_subclass_of);
		do_subclass_of(r);

		uint64_t t6=timer::get_usec();
		
		if(global_verbose && cfg->t_id ==cfg->client_num){
			cout<<cfg->m_id <<"[triangle]: result size= "<<r.row_num()<<endl;
			cout<<cfg->m_id <<" [triangle]: find  all 0,1  "<<(t2-t1)/1000.0<<"ms "<<endl;
			cout<<cfg->m_id <<" [triangle]: fetch all 1,2  "<<(t3-t2)/1000.0<<"ms "<<endl;
			cout<<cfg->m_id <<" [triangle]: edge   filter  "<<(t4-t3)/1000.0<<"ms "<<endl;
			cout<<cfg->m_id <<" [triangle]: make all 0,1,2 "<<(t5-t4)/1000.0<<"ms "<<endl;
			cout<<cfg->m_id <<" [triangle]: final filter   "<<(t6-t5)/1000.0<<"ms "<<endl;
		}
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
				//async_do_subclass_of(r);
				do_subclass_of(r);
			} else if(r.cmd_chains.back() == cmd_get_attr){
				//async_do_get_attr(r);
				do_get_attr(r);
			} else if(r.cmd_chains.back() == cmd_neighbors){
				//async_do_neighbors(r);
				do_neighbors(r);
			} else if(r.cmd_chains.back() == cmd_filter){
				do_filter(r);
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
		} else if(r.cmd_chains.back() == cmd_filter){
			do_filter(r);
		} else if(r.cmd_chains.back() == cmd_predict_index){
			assert(r.column_num()==0);
			do_predict_index(r);
		} else if(r.cmd_chains.back() == cmd_type_index){
			assert(r.column_num()==0);
			do_type_index(r);
			handle_request(r);
			return ;
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
			
			if(global_use_multithread &&  r.row_num()>=global_rdma_threshold*100){
				//cout<<"size of r.row_num() is too large, use multi-thread "<<endl;
				vector<request> sub_reqs=split_request_mt(r);
				req_queue.put_req(r,sub_reqs.size());
				for(int i=0;i<sub_reqs.size();i++){
					//i=machine*cfg->server_num+tid
					int m_id= i / cfg->server_num;
					int traverser_id= cfg->client_num + i % cfg->server_num;
					if(m_id == cfg->m_id && traverser_id==cfg->t_id){
						msg_fast_path.push_back(sub_reqs[i]);
					} else {
						SendReq(cfg,m_id ,traverser_id, sub_reqs[i],&split_profile);
					}
				}
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
				uint64_t t1=timer::get_usec();
				r.req_id=cfg->get_inc_id();
				handle_request(r);
				if(!r.blocking){
					if(cfg->is_client(r.parent_id)){
						if(global_clear_final_result){
							r.result_table.clear();
						}
						uint64_t t2=timer::get_usec();
						//cout<<"request finished in "<<(t2-t1)<<" us"<<endl;
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