#pragma once
#include "utils.h"
struct edge_triple{
	uint64_t s;
	uint64_t p;
	uint64_t o;
	edge_triple(uint64_t _s,uint64_t _p, uint64_t _o): s(_s),p(_p),o(_o){

	}
	edge_triple(): s(-1),p(-1),o(-1){

	}
};
struct edge_sort_by_spo {
    inline bool operator() (const edge_triple& struct1, const edge_triple& struct2) {
        if(struct1.s < struct2.s){
			return true;
		} else if(struct1.s == struct2.s) {
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
struct edge_sort_by_ops {
    inline bool operator() (const edge_triple& struct1, const edge_triple& struct2) {
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

const int nbit_predict=15;
const int nbit_id=48;
static inline bool is_index_vertex(int id){
	return id< (1<<nbit_predict);
}
struct local_key{
	uint64_t dir:1;
	uint64_t predict: nbit_predict;
	uint64_t id: nbit_id;
	local_key():dir(0),predict(0),id(0){
		dir-=1;
		predict-=1;
		id-=1;
	}
	// void print(){
	// 	cout<<"("<<id<<","<<dir<<","<<predict<<")"<<endl;
	// }
	uint64_t hash(){
		uint64_t r=0;
		r+=dir;
		r<<=nbit_predict;
		r+=predict;
		r<<=nbit_id;
		r+=id;
		return mymath::hash(r);
	}
	local_key(uint64_t i,uint64_t d,uint64_t p):id(i),dir(d),predict(p){

	}
	bool operator==(const local_key& another_key){
		if(dir==another_key.dir
            && predict==another_key.predict
            && id==another_key.id){
			    return true;
		}
		return false;
	}
	bool operator!=(const local_key& another_key){
		return !(operator==(another_key));
	}
};

struct local_val{
	uint64_t size:24;
	uint64_t ptr:40;
	local_val():size(0),ptr(0){
		size-=1;
		ptr-=1;
	}
	local_val(uint64_t s,uint64_t p):size(s),ptr(p){

	}
	bool operator==(const local_val& another_val){
		if(size==another_val.size
            &&  ptr==another_val.ptr){
			return true;
		}
		return false;
	}
	bool operator!=(const local_val& another_val){
		return !(operator==(another_val));
	}
};

struct vertex{
	local_key key;
	local_val val;
};
struct edge{
	uint64_t val;
};
enum direction{
	direction_in,
	direction_out,
	join_cmd
};
