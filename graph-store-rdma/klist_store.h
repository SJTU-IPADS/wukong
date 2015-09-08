#pragma once
#include <stdint.h> //uint64_t
struct edge_row{
	edge_row(int p,int v){
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
};

class klist_store{
	// key to edge-lists
	vertex* vertex_addr;
	edge_row* edge_addr;

	
	uint64_t max_edge_ptr;
	
	uint64_t v_num;
	uint64_t p_num;
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
	void init(char* start_addr,uint64_t vertex_num,uint64_t partition_num,uint64_t store_max_size){
		v_num=vertex_num;
		p_num=partition_num;
		vertex_addr=(vertex*)start_addr;
		edge_addr=(edge_row*)(start_addr+v_num*sizeof(vertex));
		
		new_edge_ptr=0;
		max_edge_ptr=(store_max_size-v_num*sizeof(vertex))/sizeof(edge_row);
		
		for(uint64_t i=0;i<v_num;i++){
			vertex_addr[i].id=-1;
		}
	}
	void insert(uint64_t id,vertex_row& v){
		if(new_edge_ptr+v.in_edges.size()+v.out_edges.size() >=max_edge_ptr)
			assert(false);
		if(vertex_addr[id/p_num].id!=-1){
			cout<<"conflict!!!! "<<vertex_addr[id/p_num].id<<"  "<<id<<endl;
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