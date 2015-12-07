#pragma once
#include <stdint.h> //uint64_t
#include <vector>
#include "rdma_resource.h"
#include "request.h" //para_in, para_out, para_all 
#include "global_cfg.h"
#include "ingress.h"

#include <iostream>
#include <pthread.h>
struct edge_row{
	edge_row(uint64_t p,uint64_t v){
		predict=p;
		vid=v;
	}
	edge_row(){
		predict=-1;
		vid=-1;
	}
	uint64_t predict;
	uint64_t vid;
	bool operator < (edge_row const& _A) const {  
			if(predict < _A.predict)  
				return true;  
			if(predict == _A.predict) 
				return vid< _A.vid;  
			return false;  
	}
};
struct vertex_row{
	vector<edge_row> in_edges;
	vector<edge_row> out_edges;
};

struct vertex{
	uint64_t id;
	uint64_t in_degree;
	uint64_t out_degree;
	uint64_t in_edge_ptr;
	uint64_t out_edge_ptr;

	void print(){
		std::cout<<"("<<id<<","<<in_degree<<","<<out_degree
				<<","<<in_edge_ptr<<","<<out_edge_ptr<<")"<<std::endl;
	}
	vertex():id(-1){
	}
};

uint64_t hash64(uint64_t key) 
{ 
	key = (~key) + (key << 21); // key = (key << 21) - key - 1; 
	key = key ^ (key >> 24); 
	key = (key + (key << 3)) + (key << 8); // key * 265 
	key = key ^ (key >> 14); 
	key = (key + (key << 2)) + (key << 4); // key * 21 
	key = key ^ (key >> 28); 
	key = key + (key << 31); 
	return key; 
}

class loc_cache{
	struct loc_cache_entry{
		pthread_spinlock_t lock;
		vertex v;
	};
	loc_cache_entry * limited_cache;
	uint64_t cache_num;
	uint64_t p_num; 
public:
	loc_cache(uint64_t num,uint64_t partition_num){
		cache_num=num;
		p_num=partition_num;
		limited_cache = new loc_cache_entry[cache_num];
		for(uint64_t i=0;i<cache_num;i++){
			pthread_spin_init(&limited_cache[i].lock,0);
		}
	}
	vertex loc_cache_lookup(uint64_t id){
		uint64_t i=hash64(id)%cache_num;
		vertex v;
		pthread_spin_lock(&limited_cache[i].lock);
		if(limited_cache[i].v.id==id)
			v=limited_cache[i].v;
		pthread_spin_unlock(&limited_cache[i].lock);
		return v;
	}
	void loc_cache_insert(vertex v){
		uint64_t i=hash64(v.id)%cache_num;
		pthread_spin_lock(&limited_cache[i].lock);
		limited_cache[i].v=v;
		pthread_spin_unlock(&limited_cache[i].lock);
	}
};


class klist_store{
	// key to edge-lists
	vertex* vertex_addr;
	edge_row* edge_addr;
	RdmaResource* rdma;
	
	uint64_t max_edge_ptr;
	
	uint64_t v_num;
	uint64_t p_num;
	uint64_t p_id;
	loc_cache* location_cache; 
	pthread_spinlock_t allocation_lock;
public:

	uint64_t new_edge_ptr;
	klist_store(){
		pthread_spin_init(&allocation_lock,0);
	};
	vertex getVertex_local(uint64_t id){
		int num_to_try=1000;
		uint64_t vertex_ptr=ingress::hash(id)% v_num;
		while(num_to_try>0){
			if(vertex_addr[vertex_ptr].id==id){
				return vertex_addr[vertex_ptr];	
			}
			num_to_try--;
			vertex_ptr=(vertex_ptr+1)% v_num;
		}
		assert(false);
		//assert(vertex_addr[id/p_num].id==id);
		//return vertex_addr[id/p_num];
	}
	vertex getVertex_remote(int tid,uint64_t id){
		char *local_buffer = rdma->GetMsgAddr(tid);
		int num_to_try=1000;
		uint64_t vertex_ptr=ingress::hash(id)% v_num;
		while(num_to_try>0){
			uint64_t start_addr=sizeof(vertex) * vertex_ptr;
			uint64_t read_length=sizeof(vertex);
			rdma->RdmaRead(tid,ingress::vid2mid(id,p_num),(char *)local_buffer,read_length,start_addr);
			vertex v=*((vertex*)local_buffer);
			if(v.id==id){
				return v;	
			}
			num_to_try--;
			vertex_ptr=(vertex_ptr+1)% v_num;
		}
		assert(false);
		//assert(vertex_addr[id/p_num].id==id);
		//return vertex_addr[id/p_num];
	}
	edge_row* getEdgeArray(uint64_t edgeptr){
		return &(edge_addr[edgeptr]);
	}
	uint64_t getEdgeOffset(uint64_t edgeptr){
		return v_num*sizeof(vertex)+sizeof(edge_row)*edgeptr;
	}
	edge_row* readGlobal_predict(int tid,uint64_t id,int direction,int predict,int* size){
		int edge_num=0;
		edge_row* edge_ptr=readGlobal(tid,id,direction,&edge_num);
		int i=0;
		while(i<edge_num){
			assert(edge_ptr[i].predict==-1);
			if(edge_ptr[i+1].predict<predict){
				i=i+1+edge_ptr[i].vid;
			} else if(edge_ptr[i+1].predict==predict){
				*size=edge_ptr[i].vid;
				return &(edge_ptr[i+1]);
			} else {
				*size=0;
				return NULL;
			}
		}
		return edge_ptr;
	}
	edge_row* readLocal_predict(int tid,uint64_t id,int direction,int predict,int* size){
		assert(ingress::vid2mid(id,p_num) ==p_id);
		return readGlobal_predict(tid,id,direction,predict,size); 
	}
	edge_row* readLocal(int tid,uint64_t id,int direction,int* size){
		assert(ingress::vid2mid(id,p_num) ==p_id);
		vertex v=getVertex_local(id);
		if(direction == para_in){
			edge_row* edge_ptr=getEdgeArray(v.in_edge_ptr);
			*size=v.in_degree;
			return edge_ptr;
		}
		if(direction == para_out){
			edge_row* edge_ptr=getEdgeArray(v.out_edge_ptr);
			*size=v.out_degree;
			return edge_ptr;
		}
		if(direction == para_all){
			cout<<"not support para_all now"<<endl;
			assert(false);
		}
		return NULL;
	}
	edge_row* readGlobal(int tid,uint64_t id,int direction,int* size){
		if( ingress::vid2mid(id,p_num) ==p_id){
			return readLocal(tid,id,direction,size);
		} else {
			//read vertex data first
			char *local_buffer = rdma->GetMsgAddr(tid);
			uint64_t start_addr=sizeof(vertex)*(id/p_num);
			uint64_t read_length=sizeof(vertex);
			vertex v;
			if(global_use_loc_cache){
				 v=location_cache->loc_cache_lookup(id);
				 if(v.id!=id){
				 	v=getVertex_remote(tid,id);
				 	location_cache->loc_cache_insert(v);
				 }				
			} else {
				v=getVertex_remote(tid,id);
			}
			//read edge data
			*size=0;
			if(direction == para_all){
				cout<<"not support para_all now"<<endl;
			}
			if(direction == para_in ){
				start_addr=getEdgeOffset(v.in_edge_ptr);
				read_length=sizeof(edge_row)*v.in_degree;
				rdma->RdmaRead(tid,ingress::vid2mid(id,p_num),(char *)local_buffer,read_length,start_addr);
				*size=*size+v.in_degree;
			}
			if(direction == para_out ){
				start_addr=getEdgeOffset(v.out_edge_ptr);
				read_length=sizeof(edge_row)*v.out_degree;
				rdma->RdmaRead(tid,ingress::vid2mid(id,p_num),
									local_buffer+(*size)*sizeof(edge_row),read_length,start_addr);
				*size=*size+v.out_degree;
			}
			edge_row* edge_ptr=(edge_row*)local_buffer;
			return edge_ptr;
		}
	}
	void init(RdmaResource* _rdma,uint64_t vertex_num,uint64_t partition_num,uint64_t partition_id){
		rdma=_rdma;
		v_num=vertex_num;
		p_num=partition_num;
		p_id=partition_id;
		vertex_addr=(vertex*)(rdma->get_buffer());
		edge_addr=(edge_row*)(rdma->get_buffer()+v_num*sizeof(vertex));
		
		new_edge_ptr=0;
		max_edge_ptr=(rdma->get_memorystore_size()-v_num*sizeof(vertex))/sizeof(edge_row);
		
		for(uint64_t i=0;i<v_num;i++){
			vertex_addr[i].id=-1;
		}
		if(global_use_loc_cache){
			location_cache=new loc_cache(100000,p_num);
		}
	}

	uint64_t atomic_alloc_edges(uint64_t num_edge){
		uint64_t curr_edge_ptr;
		pthread_spin_lock(&allocation_lock);
		curr_edge_ptr=new_edge_ptr;
		new_edge_ptr+=num_edge;
		pthread_spin_unlock(&allocation_lock);
		if(new_edge_ptr>=max_edge_ptr){
			cout<<"atomic_alloc_edges out of memory !!!! "<<endl;
			assert(false);
		}
		return curr_edge_ptr;
	}
	void insert_at(uint64_t id,vertex_row& v,uint64_t curr_edge_ptr){
		uint64_t vertex_ptr;
		if(curr_edge_ptr+v.in_edges.size()+v.out_edges.size() >=max_edge_ptr)
			assert(false);
		int num_to_try=1000;
		pthread_spin_lock(&allocation_lock);
		vertex_ptr=ingress::hash(id)% v_num;
		while(num_to_try>0){
			if(vertex_addr[vertex_ptr].id==-1){
				vertex_addr[vertex_ptr].id=id;
				break;
			}
			num_to_try--;
			vertex_ptr=(vertex_ptr+1)% v_num;
		}
		if(num_to_try==0){
			cout<<"fail to alloc for vertex "<<id<<endl;
			assert(false);
		}
		pthread_spin_unlock(&allocation_lock);
		vertex_addr[vertex_ptr].id=id;
		vertex_addr[vertex_ptr].in_degree=v.in_edges.size();
		vertex_addr[vertex_ptr].out_degree=v.out_edges.size();

		vertex_addr[vertex_ptr].in_edge_ptr=curr_edge_ptr;
		for(uint64_t i=0;i<v.in_edges.size();i++){
			edge_addr[curr_edge_ptr+i]=v.in_edges[i];
		}
		curr_edge_ptr+=vertex_addr[vertex_ptr].in_degree;
		vertex_addr[vertex_ptr].out_edge_ptr=curr_edge_ptr;
		for(uint64_t i=0;i<v.out_edges.size();i++){
			edge_addr[curr_edge_ptr+i]=v.out_edges[i];
		}
		curr_edge_ptr+=vertex_addr[vertex_ptr].out_degree;

	}
	void calculate_edge_cut(){
		uint64_t local_num=0;
		uint64_t remote_num=0;
		for(int i=0;i<v_num;i++){
			if(vertex_addr[i].id!=-1){
				uint64_t degree;
				uint64_t edge_ptr;
				degree  =vertex_addr[i].in_degree;
				edge_ptr=vertex_addr[i].in_edge_ptr;
				for(uint64_t j=0;j<degree;j++){
					if(edge_addr[edge_ptr+j].predict== -1)
						continue;
					if(edge_addr[edge_ptr+j].predict== global_rdftype_id)
						continue;
					if(ingress::vid2mid(edge_addr[edge_ptr+j].vid,p_num)==p_id){
						local_num++;
					} else {
						remote_num++;
					}
				}
				degree  =vertex_addr[i].out_degree;
				edge_ptr=vertex_addr[i].out_edge_ptr;
				for(uint64_t j=0;j<degree;j++){
					if(edge_addr[edge_ptr+j].predict== -1)
						continue;
					if(edge_addr[edge_ptr+j].predict== global_rdftype_id)
						continue;
					if(ingress::vid2mid(edge_addr[edge_ptr+j].vid,p_num)==p_id){
						local_num++;
					} else {
						remote_num++;
					}
				}
			}
		}
		cout<<"edge cut rate: "	<<remote_num<<"/"<<(local_num+remote_num)<<"="
								<<remote_num*1.0/(local_num+remote_num)<<endl;
	}
};