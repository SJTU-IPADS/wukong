#pragma once
#include <stdint.h> //uint64_t
#include <vector>
#include "rdma_resource.h"
#include "request.h" //para_in, para_out, para_all 
#include "global_cfg.h"
#include <iostream>
struct edge_row{
	edge_row(uint64_t p,uint64_t v){
		predict=p;
		vid=v;
	}
	uint64_t predict;
	uint64_t vid;
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
public:
	uint64_t new_edge_ptr;
	klist_store(){};
	vertex getVertex(uint64_t id){
		assert(vertex_addr[id/p_num].id==id);
		return vertex_addr[id/p_num];
	}
	edge_row* getEdgeArray(uint64_t edgeptr){
		return &(edge_addr[edgeptr]);
	}
	uint64_t getEdgeOffset(uint64_t edgeptr){
		return v_num*sizeof(vertex)+sizeof(edge_row)*edgeptr;
	}
	// vector<edge_row> readGlobal(int tid,uint64_t id,int direction){
	// 	vector<edge_row> result;
	// 	if(id%p_num ==p_id){
	// 		vertex v=getVertex(id);
	// 		if(direction == para_in || direction == para_all){
	// 			edge_row* edge_ptr=getEdgeArray(v.in_edge_ptr);
	// 			for(int i=0;i<v.in_degree;i++)
	// 				result.push_back(edge_ptr[i]);
	// 		}
	// 		if(direction == para_out || direction == para_all){
	// 			edge_row* edge_ptr=getEdgeArray(v.out_edge_ptr);
	// 			for(int i=0;i<v.out_degree;i++)
	// 				result.push_back(edge_ptr[i]);
	// 		}
	// 	} else {
	// 		//read vertex data first
	// 		uint64_t *local_buffer = (uint64_t *)rdma->GetMsgAddr(tid);
	// 		uint64_t start_addr=sizeof(vertex)*(id/p_num);
	// 		uint64_t read_length=sizeof(vertex);
	// 		rdma->RdmaRead(tid,id%p_num,(char *)local_buffer,read_length,start_addr);
	// 		vertex v=*((vertex*)local_buffer);
	// 		//read edge data
	// 		if(direction == para_in || direction == para_all){
	// 			start_addr=getEdgeOffset(v.in_edge_ptr);
	// 			read_length=sizeof(edge_row)*v.in_degree;
	// 			rdma->RdmaRead(tid,id%p_num,(char *)local_buffer,read_length,start_addr);
	// 			edge_row* edge_ptr=(edge_row*)local_buffer;
	// 			for(int i=0;i<v.in_degree;i++)
	// 				result.push_back(edge_ptr[i]);
	// 		}
	// 		if(direction == para_out || direction == para_all){
	// 			start_addr=getEdgeOffset(v.out_edge_ptr);
	// 			read_length=sizeof(edge_row)*v.out_degree;
	// 			rdma->RdmaRead(tid,id%p_num,(char *)local_buffer,read_length,start_addr);
	// 			edge_row* edge_ptr=(edge_row*)local_buffer;
	// 			for(int i=0;i<v.out_degree;i++)
	// 				result.push_back(edge_ptr[i]);
	// 		}
	// 	}
	// 	return result;
	// }
	edge_row* readGlobal(int tid,uint64_t id,int direction,int* size){
		if(id%p_num ==p_id){
			vertex v=getVertex(id);
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
				char *local_buffer = rdma->GetMsgAddr(tid);
				edge_row* edge_ptr;
				edge_ptr=getEdgeArray(v.in_edge_ptr);
				memcpy ((void *) local_buffer, (void *) edge_ptr, sizeof(edge_row)*v.in_degree);
				edge_ptr=getEdgeArray(v.out_edge_ptr);
				memcpy ((void *) (local_buffer+sizeof(edge_row)*v.in_degree), 
											(void *) edge_ptr, sizeof(edge_row)*v.out_degree);
				edge_ptr=(edge_row*)local_buffer;
				*size=v.in_degree+v.out_degree;
				return edge_ptr;
			}
		} else {
			//read vertex data first
			char *local_buffer = rdma->GetMsgAddr(tid);
			uint64_t start_addr=sizeof(vertex)*(id/p_num);
			uint64_t read_length=sizeof(vertex);
			vertex v;
			if(global_use_loc_cache){
				 v=location_cache->loc_cache_lookup(id);
				 if(v.id!=id){
				 	rdma->RdmaRead(tid,id%p_num,(char *)local_buffer,read_length,start_addr);
					v=*((vertex*)local_buffer);
					location_cache->loc_cache_insert(v);
				 }				
			} else {
				rdma->RdmaRead(tid,id%p_num,(char *)local_buffer,read_length,start_addr);
				v=*((vertex*)local_buffer);
			}
			//rdma->RdmaRead(tid,id%p_num,(char *)local_buffer,read_length,start_addr);
			//vertex v=*((vertex*)local_buffer);
			//read edge data
			*size=0;
			if(direction == para_in || direction == para_all){
				start_addr=getEdgeOffset(v.in_edge_ptr);
				read_length=sizeof(edge_row)*v.in_degree;
				rdma->RdmaRead(tid,id%p_num,(char *)local_buffer,read_length,start_addr);
				*size=*size+v.in_degree;
			}
			if(direction == para_out || direction == para_all){
				start_addr=getEdgeOffset(v.out_edge_ptr);
				read_length=sizeof(edge_row)*v.out_degree;
				rdma->RdmaRead(tid,id%p_num,local_buffer+(*size)*sizeof(edge_row),read_length,start_addr);
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
		max_edge_ptr=(rdma->get_size()-v_num*sizeof(vertex))/sizeof(edge_row);
		
		for(uint64_t i=0;i<v_num;i++){
			vertex_addr[i].id=-1;
		}
		if(global_use_loc_cache){
			location_cache=new loc_cache(100000,p_num);
		}
	}
	void insert(uint64_t id,vertex_row& v){
		if(new_edge_ptr+v.in_edges.size()+v.out_edges.size() >=max_edge_ptr)
			assert(false);
		if(vertex_addr[id/p_num].id!=-1){
			cout<<"conflict!!!! "<<vertex_addr[id/p_num].id<<"  "<<id<<endl;
			exit(0);
		}
		assert(vertex_addr[id/p_num].id==-1);
		vertex_addr[id/p_num].id=id;
		vertex_addr[id/p_num].in_degree=v.in_edges.size();
		vertex_addr[id/p_num].out_degree=v.out_edges.size();

		vertex_addr[id/p_num].in_edge_ptr=new_edge_ptr;
		for(uint64_t i=0;i<v.in_edges.size();i++){
			edge_addr[new_edge_ptr+i]=v.in_edges[i];
		}
		new_edge_ptr+=vertex_addr[id/p_num].in_degree;

		vertex_addr[id/p_num].out_edge_ptr=new_edge_ptr;
		for(uint64_t i=0;i<v.out_edges.size();i++){
			edge_addr[new_edge_ptr+i]=v.out_edges[i];
		}
		new_edge_ptr+=vertex_addr[id/p_num].out_degree;

	}
};