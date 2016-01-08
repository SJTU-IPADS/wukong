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

struct key_type{
	uint64_t dir:1;
	uint64_t predict:15;
	uint64_t key:48;
	key_type():dir(0),predict(0),key(0){
		dir-=1;
		predict-=1;
		key-=1;
	}
	key_type(uint64_t d,uint64_t p,uint64_t k):dir(d),predict(p),key(k){
	}
};
struct val_type{
	uint64_t size:24;
	uint64_t ptr:40;
	val_type():size(0),ptr(0){
		size-=1;
		ptr-=1;
	}
	val_type(uint64_t s,uint64_t p):size(s),ptr(p){
	}
};
// struct vertex_v2{
// 	key_t key;
// 	val_t val;
// };

struct vertex_v2{
	//struct of key
	const static int bits_dir=1;
	const static int bits_predict=15;
	const static int bits_id=48;
	//struct of val
	const static int bits_size=24;
	const static int bits_ptr=40;

	uint64_t key;
	uint64_t val;	
	vertex_v2():key(-1),val(-1){
	}
	static uint64_t make_key(uint64_t id,uint64_t dir,uint64_t predict){
		//	1 bit for dir
		//	15 bit for predict
		//	48 bit for id
		assert(dir<= (1<<bits_dir)-1 );
		assert(predict<=(1<<bits_predict)-1);
		uint64_t r=0;
		r+=dir;
		r<<=bits_predict;
		r+=predict;
		r<<=bits_id;
		r+=id;
		return r;
	}
	static uint64_t make_val(uint64_t size,uint64_t ptr){
		uint64_t one=1;
		assert(size<= (one<<bits_size)-1 );
		assert(ptr<=(one<<bits_ptr)-1);
		uint64_t r=0;
		r+=size;
		r<<=bits_ptr;
		r+=ptr;
		return r;
	}
	uint64_t get_dir(){
		uint64_t r=key>>(bits_id+bits_predict);
		return r;
	}
	uint64_t get_predict(){
		uint64_t one=1;
		uint64_t r=key>>bits_id;
		uint64_t mask=(one<<bits_predict)-1;
		return r&mask;
	}
	uint64_t get_id(){
		uint64_t one=1;
		uint64_t mask=(one<<bits_id)-1;
		return key&mask;
	}
	

	uint64_t get_size(){
		return val>>bits_ptr;
	}
	uint64_t get_ptr(){
		uint64_t one=1;
		uint64_t mask=(one<<bits_ptr)-1;
		return val&mask;
	}
	
};


struct edge_v2{
	uint64_t val;
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

	
public:
	uint64_t num_try;
	
	uint64_t used_v_num;
	uint64_t used_indirect_num;
	uint64_t max_edge_ptr;
	uint64_t new_edge_ptr;
	klist_store(){
		pthread_spin_init(&allocation_lock,0);
		for(int i=0;i<num_locks;i++){
			pthread_spin_init(&fine_grain_locks[i],0);
		}
	};
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
		uint64_t key=vertex_v2::make_key(id,direction,predict);
		vertex_v2 v=getKV_remote(tid,key);
		if(v.key==-1){
			*size=0;
			return NULL;
		}
		char *local_buffer = rdma->GetMsgAddr(tid);
		uint64_t start_addr=getEdgeOffset(v.get_ptr());
		uint64_t read_length=sizeof(edge_v2)*v.get_size();
		rdma->RdmaRead(tid,ingress::vid2mid(id,p_num),(char *)local_buffer,read_length,start_addr);
		edge_v2* result_ptr=(edge_v2*)local_buffer;
		*size=v.get_size();
		return result_ptr;
	}
	edge_v2* readLocal_predict(int tid,uint64_t id,int direction,int predict,int* size){
		assert(ingress::vid2mid(id,p_num) ==p_id);
		uint64_t key=vertex_v2::make_key(id,direction,predict);
		vertex_v2 v=getKV_local(key);
		if(v.key==-1){
			*size=0;
			return NULL;
		}
		*size=v.get_size();
		uint64_t ptr=v.get_ptr();
		return getEdgeArray(ptr);
	}

	void init(RdmaResource* _rdma,uint64_t vertex_num,uint64_t partition_num,uint64_t partition_id){
		rdma=_rdma;
		v_num=vertex_num;
		used_v_num=0;
		used_indirect_num=0;
		num_try=0;
		p_num=partition_num;
		p_id=partition_id;
		vertex_addr=(vertex_v2*)(rdma->get_buffer());
		edge_addr=(edge_v2*)(rdma->get_buffer()+v_num*sizeof(vertex_v2));
		
		new_edge_ptr=0;
		max_edge_ptr=(rdma->get_memorystore_size()-v_num*sizeof(vertex_v2))/sizeof(edge_v2);
		
		for(uint64_t i=0;i<v_num;i++){
			vertex_addr[i].key=-1;
		}
		if(global_use_loc_cache){
			//location_cache=new loc_cache(100000,p_num);
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


	vertex_v2 getKV_local(uint64_t id){
		uint64_t header_num=(v_num/4)/5*4;
		uint64_t indirect_num=(v_num/4)/5*1;
		uint64_t bucket_id=ingress::hash(id)%header_num;
		while(true){
			for(uint64_t i=0;i<3;i++){
				if(vertex_addr[bucket_id*4+i].key==id){
					//we found it
					return vertex_addr[bucket_id*4+i];
				}
			}
			if(vertex_addr[bucket_id*4+3].key!=-1){
				//next pointer
				bucket_id=vertex_addr[bucket_id*4+3].key;
				continue;
			} else {
				//we didn't found it!
				return vertex_v2();
			}
		}
	}
	vertex_v2 getKV_remote(int tid,uint64_t id){
		char *local_buffer = rdma->GetMsgAddr(tid);
		uint64_t header_num=(v_num/4)/5*4;
		uint64_t indirect_num=(v_num/4)/5*1;
		uint64_t bucket_id=ingress::hash(id)%header_num;
		while(true){
			uint64_t start_addr=sizeof(vertex_v2) * bucket_id *4;
			uint64_t read_length=sizeof(vertex_v2) * 4;
			rdma->RdmaRead(tid,ingress::vid2mid(id,p_num),(char *)local_buffer,read_length,start_addr);
			vertex_v2* ptr=(vertex_v2*)local_buffer;
			for(uint64_t i=0;i<3;i++){
				if(ptr[i].key==id){
					//we found it
					return ptr[i];
				}
			}
			if(ptr[3].key!=-1){
				//next pointer
				bucket_id=ptr[3].key;
				continue;
			} else {
				//we didn't found it!
				return vertex_v2();
			}
		}
	}
	uint64_t insertKey(uint64_t id){
		uint64_t vertex_ptr;
		//4-associate
		uint64_t header_num=(v_num/4)/5*4;
		uint64_t indirect_num=(v_num/4)/5*1;
		uint64_t bucket_id=ingress::hash(id)%header_num;
		uint64_t lock_id=bucket_id% num_locks;
		int slot_id=0;
		bool found=false;
		pthread_spin_lock(&fine_grain_locks[lock_id]);
		//last slot is used as next pointer
		while(!found){
			for(uint64_t i=0;i<3;i++){
				if(vertex_addr[bucket_id*4+i].key==id){
					cout<<"inserting duplicate key= " <<vertex_addr[bucket_id*4+i].key<<endl;
					assert(false);
					found=true;
					slot_id=i;
					break;
				}
				if(vertex_addr[bucket_id*4+i].key==-1){
					vertex_addr[bucket_id*4+i].key=id;
					found=true;
					slot_id=i;
					break;
				}
			}
			if(found){
				break;
			} else if(vertex_addr[bucket_id*4+3].key!=-1){
				//next pointer
				bucket_id=vertex_addr[bucket_id*4+3].key;
				continue;
			} else {
				//need alloc
				pthread_spin_lock(&allocation_lock);
				if(used_indirect_num>=indirect_num){
					assert(false);
				}
				vertex_addr[bucket_id*4+3].key=header_num+used_indirect_num;
				used_indirect_num++;
				pthread_spin_unlock(&allocation_lock);
				bucket_id=vertex_addr[bucket_id*4+3].key;
				vertex_addr[bucket_id*4+0].key=id;
				slot_id=0;
				break;
			}
		}
		pthread_spin_unlock(&fine_grain_locks[lock_id]);
		vertex_ptr = bucket_id*4+slot_id;
		assert(vertex_addr[vertex_ptr].key==id);
		return vertex_ptr;
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
			uint64_t key= vertex_v2::make_key(vec_spo[start].s,para_out,vec_spo[start].p);
			uint64_t vertex_ptr=insertKey(key);
			uint64_t val= vertex_v2::make_val(end-start,curr_edge_ptr);
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
			uint64_t key= vertex_v2::make_key(vec_ops[start].o,para_in,vec_ops[start].p);
			uint64_t vertex_ptr=insertKey(key);
			uint64_t val= vertex_v2::make_val(end-start,curr_edge_ptr);
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
	vector<uint64_t>& get_vector(tbb_vector_table& table,uint64_t index_id){
		tbb_vector_table::accessor a;
		if (!table.find(a,index_id)){
			assert(false);
		}
		return a->second;
	}
	void insert_vector(tbb_vector_table& table,uint64_t index_id,uint64_t value_id){
		tbb_vector_table::accessor a; 
		table.insert(a,index_id); 
		a->second.push_back(value_id);
	}


	void init_index_table(){
		int count=0;
		//4-associate, 3 data and 1 next
		uint64_t header_num=(v_num/4)/5*4;
		uint64_t indirect_num=(v_num/4)/5*1;
		#pragma omp parallel for num_threads(20)
		for(int x=0;x<header_num+indirect_num;x++){
			for(int y=0;y<3;y++){
				uint64_t i=x*4+y;
				if(vertex_addr[i].key!=-1){
					if(vertex_addr[i].get_dir()==para_in){
						continue;
					}
					uint64_t degree=vertex_addr[i].get_size();
					uint64_t edge_ptr=vertex_addr[i].get_ptr();
					for(uint64_t j=0;j<degree;j++){
						uint64_t s=vertex_addr[i].get_id();
						uint64_t p=vertex_addr[i].get_predict();
						uint64_t o=edge_addr[edge_ptr+j].val;
						if(p==-1){
							continue;
						} else if(p==global_rdftype_id){
							insert_vector(type_table,o,s);
						} else {
							//TODO
						}
					}
				}
			}
		}
	}

};