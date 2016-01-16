#pragma once
#include "graph.h"
#include "request.h"
#include "blocking_queue.h"
#include "network_node.h"
#include "message_wrap.h"
#include "profile.h"
#include "global_cfg.h"
#include "ingress.h"
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include "simple_filter.h"
struct per_thread_resource{
	bool need_help;
	char padding[63];
	pthread_spinlock_t internal_lock;
	blocking_queue req_queue;
	vector<request> msg_fast_path;
	per_thread_resource(){
		need_help=false;
		pthread_spin_init(&internal_lock,0);
	}
	void lock(){pthread_spin_lock(&internal_lock);}
	void unlock(){pthread_spin_unlock(&internal_lock);}
};
class traverser{
	graph& g;
	thread_cfg* cfg;
	thread_cfg* cfg_array;

	profile split_profile;
	per_thread_resource * res_array;
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
	void do_get_attr(request& r){
		r.cmd_chains.pop_back();
		int dir=r.cmd_chains.back();
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
		assert(global_use_index_table);
		r.cmd_chains.pop_back();
		int predict_id=r.cmd_chains.back();
		r.cmd_chains.pop_back();
		int dir=r.cmd_chains.back();
		r.cmd_chains.pop_back();
		vector<vector<int> >updated_result_table;
		updated_result_table.resize(1);
		int start_id=r.parallel_id;
		if(dir==para_in){
			vector<uint64_t>& ids=g.kstore.get_vector(
					g.kstore.src_predict_table,predict_id);
			for(int i=start_id;i<ids.size();i+=r.parallel_total){
				updated_result_table[0].push_back(ids[i]);
			}
		} else {
			vector<uint64_t>& ids=g.kstore.get_vector(
					g.kstore.dst_predict_table,predict_id);
			for(int i=start_id;i<ids.size();i+=r.parallel_total){
				updated_result_table[0].push_back(ids[i]);
			}
		}
		r.result_table.swap(updated_result_table);
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
			int start_id=r.parallel_id;
			for(int i=start_id;i<ids.size();i+=r.parallel_total){
				updated_result_table[0].push_back(ids[i]);
			}
			r.result_table.swap(updated_result_table);
			return ;
		} else {//use rdma to read data
			vector<vector<int> >updated_result_table;
			updated_result_table.resize(1);
			int edge_num=0;
			edge_v2* edge_ptr;
			edge_ptr=g.kstore.readGlobal_predict(cfg->t_id,
					type_id,para_in,global_rdftype_id,&edge_num);
			for(int k=0;k<edge_num;k++){
				int mid = ingress::vid2mid(edge_ptr[k].val, cfg->m_num);
				if(mid==cfg->m_id && r.parallel_id ==  k % r.parallel_total){
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
		// (m0,t0),(m1,t0),(m2,t0)...
		vector<request> sub_reqs;
		//int threads_per_machine=min(cfg->server_num,r.parallel_total);
		int threads_per_machine=cfg->server_num;
		int num_sub_request=cfg->m_num * threads_per_machine ;
		sub_reqs.resize(num_sub_request);
		for(int i=0;i<sub_reqs.size();i++){
			sub_reqs[i].parallel_total=r.parallel_total;
			sub_reqs[i].parent_id=r.req_id;
			sub_reqs[i].cmd_chains=r.cmd_chains;
			sub_reqs[i].result_table.resize(r.column_num());
		}
		for(int i=0;i<r.row_num();i++){
			int machine = ingress::vid2mid(r.last_column(i), cfg->m_num);
			int tid = ingress::hash(r.last_column(i)) % threads_per_machine ;
			r.append_row_to(sub_reqs[tid*cfg->m_num+machine].result_table,i);
			//r.append_row_to(sub_reqs[machine*cfg->server_num+tid].result_table,i);
		}
		r.result_table.clear();
		return sub_reqs;
	}
	vector<request> split_request_join(request& r){
		vector<request> sub_reqs;
		int num_sub_request=cfg->m_num;
		sub_reqs.resize(num_sub_request);
		for(int i=0;i<sub_reqs.size();i++){
			sub_reqs[i].parent_id=r.req_id;
			sub_reqs[i].parallel_total=r.parallel_total;
			sub_reqs[i].cmd_chains=r.cmd_chains;
			sub_reqs[i].cmd_chains.pop_back();
			sub_reqs[i].result_table.resize(1);
		}
		boost::unordered_set<int> remove_dup_set;
		for(int i=0;i<r.row_num();i++){
			if(remove_dup_set.find(r.last_column(i))!=remove_dup_set.end()){
				continue;
			}
			remove_dup_set.insert(r.last_column(i));
			int machine = ingress::vid2mid(r.last_column(i), num_sub_request);
			sub_reqs[machine].result_table[0].push_back(r.last_column(i));
		}
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
		vector<boost::unordered_map<uint64_t,bool> > type_filter;
		type_filter.resize(3);
		simple_filter edge_filter;
		vector<vector<v_pair> >pair_vec;
		pair_vec.resize(num_parallel_thread);
		int count_type1=0;
		for(uint64_t i=0;i<r.row_num();i++){
			int working_tid = omp_get_thread_num();
			
			int edge_num=0;
			edge_v2* edge_ptr;
			if(type_filter[1].find(r.last_column(i))!=type_filter[1].end()){
				continue;
			}
			count_type1++;
			type_filter[1][r.last_column(i)]=true;					
				edge_ptr=g.kstore.readGlobal_predict(cfg->t_id,
										r.last_column(i),para_out,global_rdftype_id,&edge_num);
			bool found=false;
			for(int k=0;k<edge_num;k++){
				if(g.ontology_table.is_subtype_of(edge_ptr[k].val,v_type[1])){
					found=true;
					type_filter[1][r.last_column(i)]=true;
					break;
				}
			}
			if(!found){
				type_filter[1][r.last_column(i)]=false;
				continue;
			}
			// fetch and insert to edge_filter
			edge_num=0;
				edge_ptr=g.kstore.readGlobal_predict(cfg->t_id,
											r.last_column(i),v_dir[1],v_predict[1],&edge_num);
			for(int k=0;k<edge_num;k++){
				pair_vec[working_tid].push_back(v_pair(r.last_column(i),edge_ptr[k].val));		
			}
		}

		if(global_verbose && cfg->t_id ==cfg->client_num){
			cout<<cfg->m_id <<" [triangle]: count_type1  "<<count_type1<<endl;
		}

		uint64_t t3=timer::get_usec();

		int count=0;
		for(int j=0;j<pair_vec.size();j++){
			count+=pair_vec[j].size();
		}
		edge_filter.tbb_set.rehash(2*count);
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
	void traverser_SendReq(int r_mid,int r_tid,request& r){
		if(r_mid == cfg->m_id && r_tid==cfg->t_id){
			if(global_enable_workstealing){
				//fast_path may conflict with the stealing thread
				res_array[cfg->t_id].lock();
			}
			res_array[cfg->t_id].msg_fast_path.push_back(r);
			if(global_enable_workstealing){
				res_array[cfg->t_id].unlock();
			}
		} else {
			SendReq(cfg,r_mid,r_tid, r,&split_profile);
		}
	}
public:
	traverser(graph& gg,per_thread_resource* _res_array, thread_cfg* _cfg,thread_cfg* _cfg_array)
			:g(gg),res_array(_res_array),cfg(_cfg),cfg_array(_cfg_array){
		req_id=cfg->m_id+cfg->t_id*cfg->m_num;
	}
	
	void handle_request(request& r){
		if(r.cmd_chains.size()==0){
			if(global_clear_final_result){
				r.final_row_number=r.row_num();
				r.clear_data();
			}
			return ;
		}
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
			handle_request(r);
			return ;
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
			if(global_clear_final_result){
				// cout<<"clear,("<<cfg->m_id<<","<<cfg->t_id
				// 	<<")  r.row_num()=" <<r.row_num()<<endl;
				r.final_row_number=r.row_num();
				r.clear_data();
			}
			return ;
		} else {
			//recursive execute 
			r.blocking=true;
			if(r.cmd_chains.back() == cmd_join){
				vector<request> sub_reqs=split_request_join(r);
				res_array[cfg->t_id].req_queue.put_req(r,sub_reqs.size());
				
				for(int i=0;i<sub_reqs.size();i++){
					//i=tid*cfg->m_num+machine
					int m_id= i % cfg->m_num;
					int traverser_id=cfg->t_id;	
					//int traverser_id= cfg->client_num + i / cfg->m_num;
					traverser_SendReq(m_id,traverser_id,sub_reqs[i]);
				}
			}
			else if(global_use_multithread &&  r.row_num()>=global_rdma_threshold*100){
				//cout<<"size of r.row_num()= "<< r.row_num()<< " is too large, use multi-thread "<<endl;
				vector<request> sub_reqs=split_request_mt(r);
				res_array[cfg->t_id].req_queue.put_req(r,sub_reqs.size());
				for(int i=0;i<sub_reqs.size();i++){
					//i=tid*cfg->m_num+machine
					int m_id= i % cfg->m_num;
					int traverser_id= cfg->client_num + i / cfg->m_num;
					traverser_SendReq(m_id,traverser_id,sub_reqs[i]);
				}
			} else {
				vector<request> sub_reqs=split_request(r);
				res_array[cfg->t_id].req_queue.put_req(r,sub_reqs.size());
				for(int i=0;i<sub_reqs.size();i++){
					int traverser_id=cfg->t_id;	
					traverser_SendReq(i,traverser_id,sub_reqs[i]);
				}
			}	
		}	
	} 
	void run(){	
		while(true){
			request r;
			bool steal=false;
			if(global_enable_workstealing){
				res_array[cfg->t_id].lock();
				while(true){
					if(res_array[cfg->t_id].msg_fast_path.size()>0){
						r=res_array[cfg->t_id].msg_fast_path.back();
						res_array[cfg->t_id].msg_fast_path.pop_back();
						res_array[cfg->t_id].unlock();
						break;
					} 
					bool success=TryRecvReq(cfg,r);
					if(success){
						res_array[cfg->t_id].unlock();
						break;
					}
					if(cfg->t_id>8 && res_array[cfg->t_id].need_help==false
									&& res_array[cfg->t_id-8].need_help==true){
						//we are going to steal from other thread
						res_array[cfg->t_id-8].lock();
						success=TryRecvReq(&cfg_array[cfg->t_id-8],r);
						if(success ){
							if(cfg->is_client(r.parent_id)|| r.parallel_total==0){
								//we will steal
								//1. client request 
								//2. sub-request or sub-reply of smaller request
								steal=true;
							} else {
								success=false;
								res_array[cfg->t_id-8].msg_fast_path.push_back(r);

							}
						}
						res_array[cfg->t_id-8].unlock();
						if(success){
							res_array[cfg->t_id].unlock();
							break;
						}
					}
				}



			} else { //just simply Recv
				if(res_array[cfg->t_id].msg_fast_path.size()>0){
					r=res_array[cfg->t_id].msg_fast_path.back();
					res_array[cfg->t_id].msg_fast_path.pop_back();
				}
				else {
					r=RecvReq(cfg);
				}
			}

			if(r.req_id==-1){ //it means r is a request and shoule be executed
				command cmd=(command)r.cmd_chains.back();
				if(cmd == cmd_type_index|| cmd == cmd_predict_index || cmd==cmd_triangle){
					res_array[cfg->t_id].need_help=true;
					//cfg->rdma->set_need_help(cfg->t_id,true);
				}
				r.req_id=cfg->get_inc_id();
				handle_request(r);
				if(!r.blocking){
					if(cfg->is_client(r.parent_id)){
						if(global_clear_final_result){
							// cout<<"clear,("<<cfg->m_id<<","<<cfg->t_id
							// 	<<")  r.row_num()=" <<r.row_num()<<endl;
							r.clear_data();
						}
						res_array[cfg->t_id].need_help=false;
						//cfg->rdma->set_need_help(cfg->t_id,false);
					}
					traverser_SendReq(cfg->mid_of(r.parent_id),cfg->tid_of(r.parent_id),r);
				} else {
					//else this request is put into waiting queue
					//so we doesn't need to do any-thing
				}
			} else {
				bool success;
				if(steal){
					success=res_array[cfg->t_id-8].req_queue.put_reply(r);
				} else {
					success=res_array[cfg->t_id].req_queue.put_reply(r);
				}
				if(success){
					if(cfg->is_client(r.parent_id)){
						if(global_clear_final_result){
							r.clear_data();
						}
						res_array[cfg->t_id].need_help=false;
						//cfg->rdma->set_need_help(cfg->t_id,false);
					}
					traverser_SendReq(cfg->mid_of(r.parent_id),cfg->tid_of(r.parent_id),r);
				}
			}
		}

	}
};