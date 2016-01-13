#pragma once
#include <stdint.h> //uint64_t
#include <vector>
#include "rdma_resource.h"
#include "request.h" //para_in, para_out, para_all 
#include "global_cfg.h"
#include "ingress.h"

#include <iostream>
#include <pthread.h>
#include <boost/unordered_set.hpp>
#include <tbb/concurrent_hash_map.h>

struct edge_triple{
	uint64_t s;
	uint64_t p;
	uint64_t o;
	edge_triple(uint64_t _s,uint64_t _p, uint64_t _o):
		s(_s),p(_p),o(_o){
	}
	edge_triple():
		s(-1),p(-1),o(-1){
	}
};
struct edge_sort_by_spo
{
    inline bool operator() (const edge_triple& struct1, const edge_triple& struct2)
    {
        if(struct1.s < struct2.s){  
			return true;  
		} else if(struct1.s == struct2.s){
			if(struct1.p < struct2.p){
				return true; 
			} else if(struct1.p == struct2.p && struct1.o < struct2.o){
				return true;
			}
		}
		//otherwise 
		return false;  
	}
};
struct edge_sort_by_ops
{
    inline bool operator() (const edge_triple& struct1, const edge_triple& struct2)
    {
        if(struct1.o < struct2.o){  
			return true;  
		} else if(struct1.o == struct2.o){
			if(struct1.p < struct2.p){
				return true; 
			} else if(struct1.p == struct2.p && struct1.s < struct2.s){
				return true;
			}
		}
		//otherwise 
		return false;  
	}
};

struct klist_key{
	uint64_t dir:1;
	uint64_t predict:15;
	uint64_t id:48;
	klist_key():dir(0),predict(0),id(0){
		dir-=1;
		predict-=1;
		id-=1;
	}
	void print(){
		cout<<"("<<id<<","
				<<dir<<","
				<<predict<<")"<<endl;
	}
	uint64_t hash(){
		uint64_t r=0;
		r+=dir;
		r<<=15;
		r+=predict;
		r<<=48;
		r+=id;
		return ingress::hash(r);
	}
	klist_key(uint64_t i,uint64_t d,uint64_t p):id(i),dir(d),predict(p){
	}
	bool operator==(const klist_key& another_key){
		if(dir!=another_key.dir){
			return false;
		}
		if(predict!=another_key.predict){
			return false;
		}
		if(id!=another_key.id){
			return false;
		}
		return true;
	}
	bool operator!=(const klist_key& another_key){
		return !(operator==(another_key));
	}
};
struct klist_val{
	uint64_t size:24;
	uint64_t ptr:40;
	klist_val():size(0),ptr(0){
		size-=1;
		ptr-=1;
	}
	klist_val(uint64_t s,uint64_t p):size(s),ptr(p){
	}
	bool operator==(const klist_val& another_val){
		if(size!=another_val.size){
			return false;
		}
		if(ptr!=another_val.ptr){
			return false;
		}
		return true;
	}
	bool operator!=(const klist_val& another_val){
		return !(operator==(another_val));
	}
};
struct vertex_v2{
	klist_key key;
	klist_val val;
};
struct edge_v2{
	uint64_t val;
};


class klist_store{
	// key to edge-lists
	vertex_v2* vertex_addr;
	edge_v2* edge_addr;
	RdmaResource* rdma;
	
	
	uint64_t v_num;
	uint64_t p_num;
	uint64_t p_id;
	pthread_spinlock_t allocation_lock;
	static const int num_locks=1024;
	pthread_spinlock_t fine_grain_locks[num_locks];

	static const int indirect_ratio=5; // 	1/5 of buckets are used as indirect buckets
	static const int cluster_size=4;   //	each bucket has 4 slot
	uint64_t header_num;
	uint64_t indirect_num;
public:
	
	uint64_t used_indirect_num;
	uint64_t max_edge_ptr;
	uint64_t new_edge_ptr;
	klist_store(){
		pthread_spin_init(&allocation_lock,0);
		for(int i=0;i<num_locks;i++){
			pthread_spin_init(&fine_grain_locks[i],0);
		}
	};
	void init(RdmaResource* _rdma,uint64_t vertex_num,uint64_t partition_num,uint64_t partition_id){
		rdma=_rdma;
		p_num=partition_num;
		p_id=partition_id;
		
		v_num=vertex_num;
		header_num	=(v_num/cluster_size)/indirect_ratio*(indirect_ratio-1);
		indirect_num=(v_num/cluster_size)/indirect_ratio;

		vertex_addr	=(vertex_v2*)(rdma->get_buffer());
		edge_addr	=(edge_v2*)(rdma->get_buffer()+v_num*sizeof(vertex_v2));

		used_indirect_num=0;
		new_edge_ptr=0;
		max_edge_ptr=(rdma->get_memorystore_size()-v_num*sizeof(vertex_v2))/sizeof(edge_v2);
		
		#pragma omp parallel for num_threads(20)
		for(uint64_t i=0;i<v_num;i++){
			vertex_addr[i].key=klist_key();
		}
		if(global_use_loc_cache){
			//TODO please add cache related code back
			assert(false);
			//location_cache=new loc_cache(100000,p_num);
		}
	}
	void print_memory_usage(){
		cout<<"graph-store use "<<used_indirect_num <<" / "
			<<indirect_num 	<<" indirect_num"<<endl;
		cout<<"graph-store use "<<v_num*sizeof(vertex_v2) / 1048576<<" MB for vertex data"<<endl;
		cout<<"graph-store use "<<new_edge_ptr*sizeof(edge_v2)/1048576<<"/"
								<<max_edge_ptr*sizeof(edge_v2)/1048576<<" MB for edge data"<<endl;
	}
	edge_v2* getEdgeArray(uint64_t edgeptr){
		return &(edge_addr[edgeptr]);
	}
	uint64_t getEdgeOffset(uint64_t edgeptr){
		return v_num*sizeof(vertex_v2)+sizeof(edge_v2)*edgeptr;
	}

	
	edge_v2* readGlobal_predict(int tid,uint64_t id,int direction,int predict,int* size){
		if( ingress::vid2mid(id,p_num) ==p_id){
			return readLocal_predict(tid,id,direction,predict,size);
		}
		klist_key key=klist_key(id,direction,predict);
		vertex_v2 v=getKV_remote(tid,key);
		if(v.key==klist_key()){
			*size=0;
			return NULL;
		}
		char *local_buffer = rdma->GetMsgAddr(tid);
		uint64_t start_addr=getEdgeOffset(v.val.ptr);
		uint64_t read_length=sizeof(edge_v2)*v.val.size;
		rdma->RdmaRead(tid,ingress::vid2mid(id,p_num),(char *)local_buffer,read_length,start_addr);
		edge_v2* result_ptr=(edge_v2*)local_buffer;
		*size=v.val.size;
		return result_ptr;
	}
	edge_v2* readLocal_predict(int tid,uint64_t id,int direction,int predict,int* size){
		assert(ingress::vid2mid(id,p_num) ==p_id);
		klist_key key=klist_key(id,direction,predict);
		vertex_v2 v=getKV_local(key);
		if(v.key==klist_key()){
			*size=0;
			return NULL;
		}
		*size=v.val.size;
		uint64_t ptr=v.val.ptr;
		return getEdgeArray(ptr);
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


	vertex_v2 getKV_local(klist_key key){
		uint64_t bucket_id=key.hash()%header_num;
		while(true){
			for(uint64_t i=0;i<cluster_size;i++){
				uint64_t slot_id=bucket_id*cluster_size+i;
				if(i<cluster_size-1){
					//data part
					if(vertex_addr[slot_id].key==key){
						//we found it
						return vertex_addr[slot_id];
					}
				} else { 
					if(vertex_addr[slot_id].key!=klist_key()){
						//next pointer
						bucket_id=vertex_addr[slot_id].key.id;
						//break from for loop, will go to next bucket
						break;
					} else {
						return vertex_v2();
					}
				}
			} 
		}
	}
	vertex_v2 getKV_remote(int tid,klist_key key){
		char *local_buffer = rdma->GetMsgAddr(tid);
		uint64_t bucket_id=key.hash()%header_num;
		while(true){
			uint64_t start_addr=sizeof(vertex_v2) * bucket_id *cluster_size;
			uint64_t read_length=sizeof(vertex_v2) * cluster_size;
			rdma->RdmaRead(tid,ingress::vid2mid(key.id,p_num),(char *)local_buffer,read_length,start_addr);
			vertex_v2* ptr=(vertex_v2*)local_buffer;
			for(uint64_t i=0;i<cluster_size;i++){
				if(i<cluster_size-1){
					if(ptr[i].key==key){
						//we found it
						return ptr[i];
					}
				} else {
					if(ptr[i].key!=klist_key()){
						//next pointer
						bucket_id=ptr[i].key.id;
						//break from for loop, will go to next bucket
						break;
					} else {
						return vertex_v2();
					}
				}
			}
		}
	}
	uint64_t insertKey(klist_key key){
		uint64_t vertex_ptr;
		uint64_t bucket_id=key.hash()%header_num;
		uint64_t lock_id=bucket_id% num_locks;
		uint64_t slot_id=0;
		bool found=false;
		pthread_spin_lock(&fine_grain_locks[lock_id]);
		//last slot is used as next pointer
		while(!found){
			for(uint64_t i=0;i<cluster_size-1;i++){
				slot_id=bucket_id*cluster_size+i;
				if(vertex_addr[slot_id].key==key){
					cout<<"inserting duplicate key" <<endl;
					vertex_addr[slot_id].key.print();
					key.print();
					assert(false);
				}
				if(vertex_addr[slot_id].key==klist_key()){
					vertex_addr[slot_id].key=key;
					found=true;
					break;
				}
			}
			if(found){
				break;
			} else {
				slot_id=bucket_id*cluster_size+cluster_size-1;
				if(vertex_addr[slot_id].key!=klist_key()){
					bucket_id=vertex_addr[slot_id].key.id;
					//continue and jump to next bucket
					continue;
				} else {
					pthread_spin_lock(&allocation_lock);
					if(used_indirect_num>=indirect_num){
						assert(false);
					}
					vertex_addr[slot_id].key.id=header_num+used_indirect_num;
					used_indirect_num++;
					pthread_spin_unlock(&allocation_lock);
					bucket_id=vertex_addr[slot_id].key.id;
					slot_id=bucket_id*cluster_size+0;
					vertex_addr[slot_id].key=key;
					//break the while loop since we successfully insert
					break;
				}
			}
		}
		pthread_spin_unlock(&fine_grain_locks[lock_id]);
		assert(vertex_addr[slot_id].key==key);
		return slot_id;
	}
	void batch_insert(vector<edge_triple>& vec_spo,
						vector<edge_triple>& vec_ops,uint64_t curr_edge_ptr){
		uint64_t start;
		start=0;
		while(start<vec_spo.size()){
			uint64_t end=start+1;
			while(end<vec_spo.size() 
					&& vec_spo[start].s==vec_spo[end].s
					&& vec_spo[start].p==vec_spo[end].p){
				end++;
			}
			klist_key key= klist_key(vec_spo[start].s,para_out,vec_spo[start].p);
			uint64_t vertex_ptr=insertKey(key);
			klist_val val= klist_val(end-start,curr_edge_ptr);
			vertex_addr[vertex_ptr].val=val;
			for(uint64_t i=start;i<end;i++){
				edge_addr[curr_edge_ptr].val=vec_spo[i].o;
				curr_edge_ptr++;
			}
			start=end;
		}
		start=0;
		while(start<vec_ops.size()){
			uint64_t end=start+1;
			while(end<vec_ops.size() 
					&& vec_ops[start].o==vec_ops[end].o
					&& vec_ops[start].p==vec_ops[end].p){
				end++;
			}
			klist_key key= klist_key(vec_ops[start].o,para_in,vec_ops[start].p);
			uint64_t vertex_ptr=insertKey(key);
			klist_val val= klist_val(end-start,curr_edge_ptr);
			vertex_addr[vertex_ptr].val=val;
			for(uint64_t i=start;i<end;i++){
				edge_addr[curr_edge_ptr].val=vec_ops[i].s;
				curr_edge_ptr++;
			}
			start=end;
		}
	}


	typedef tbb::concurrent_hash_map<uint64_t,vector<uint64_t> > tbb_vector_table;
	tbb_vector_table type_table;
	tbb_vector_table src_predict_table;
	tbb_vector_table dst_predict_table;
	vector<uint64_t> empty;

	vector<uint64_t>& get_vector(tbb_vector_table& table,uint64_t index_id){
		tbb_vector_table::accessor a;
		if (!table.find(a,index_id)){
			cout<<"[warning] index_table "<< index_id << "not found"<<endl; 
			return empty;
		}
		return a->second;
	}
	void insert_vector(tbb_vector_table& table,uint64_t index_id,uint64_t value_id){
		tbb_vector_table::accessor a; 
		table.insert(a,index_id); 
		a->second.push_back(value_id);
	}

	void init_index_table(){
		#pragma omp parallel for num_threads(20)
		for(int x=0;x<header_num+indirect_num;x++){
			for(int y=0;y<cluster_size-1;y++){
				uint64_t i=x*cluster_size+y;
				if(vertex_addr[i].key==klist_key()){
					//empty slot, skip it
					continue;
				}
				uint64_t vid=vertex_addr[i].key.id;
				uint64_t p=vertex_addr[i].key.predict;
				if(vertex_addr[i].key.dir==para_in){
					if(p==global_rdftype_id){
						//it means vid is a type vertex
						//we just skip it
						continue;
					} else {
						//this edge is in-direction, so vid is the dst of predict
						insert_vector(dst_predict_table,p,vid);
					}
				} else {
					if(p==global_rdftype_id){
						uint64_t degree=vertex_addr[i].val.size;
						uint64_t edge_ptr=vertex_addr[i].val.ptr;
						for(uint64_t j=0;j<degree;j++){
							//src may belongs to multiple types
							insert_vector(type_table,edge_addr[edge_ptr+j].val,vid);
						}
					} else {
						insert_vector(src_predict_table,p,vid);
					}
				}
			}
		}
	cout<<"sizeof type_table = "<<type_table.size()<<endl;
	cout<<"sizeof src_predict_table = "<<src_predict_table.size()<<endl;
	cout<<"sizeof dst_predict_table = "<<dst_predict_table.size()<<endl;
	}

};