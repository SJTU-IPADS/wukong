#pragma once

#include <vector>
#include <boost/unordered_set.hpp>
#include <tbb/concurrent_hash_map.h>
#include "timer.h"
typedef std::pair<int,int> v_pair;
struct hash_vpair{
	size_t operator()(const v_pair &x) const{
		return hash<int>()(x.first) ^ hash<int>()(x.second);
	}
};
size_t hash1(const v_pair &x){
	size_t r=x.first;
	r=r<<32;
	r+=x.second;
	return hash<size_t>()(r);
}
size_t hash2(const v_pair &x){
	size_t r=x.second;
	r=r<<32;
	r+=x.first;
	return hash<size_t>()(r);
}

struct TbbHashCompare {
    static size_t hash( const v_pair& x ) {
    	return hash1(x);
    }
    //! True if strings are equal
    static bool equal( const v_pair& x , const v_pair& y) {
        return x.first==y.first && x.second==y.second;
    }
};
typedef tbb::concurrent_hash_map<v_pair,bool,TbbHashCompare> tbb_hashtable;

// size_t hash1(const v_pair &x){
// 	return hash<int>()(x.first^x.second);
// }
// size_t hash2(const v_pair &x){
// 	return hash<int>()(x.first+x.second);
// }
// size_t hash1(const v_pair &x){
// 	return hash<int>()(x.first)^hash<int>()(x.second);
// }
// size_t hash2(const v_pair &x){
// 	return hash<int>()(x.first)+hash<int>()(x.second);
// }
//#define USE_BOOST_SET

#define USE_TBB

class simple_filter{
public:
	vector<v_pair> vec_id;
	vector<v_pair> data;
	tbb_hashtable tbb_set;
// #ifdef USE_BOOST_SET
// 	boost::unordered_set<v_pair> pair_set;
// #endif 

	int count;
	int accum_count;
	int max_level;
	simple_filter(){
		count=0;
		max_level=0;
		accum_count=0;
	}
	void recursive_cuckoo_insert(v_pair vp,int level,int prev_bucket){
		if(level>max_level){
			max_level=level;
		}
		if(level==5000){
			cout<<"Too many evict"<<endl;
			cout<<"inserting "<<count<<"/"<<vec_id.size()<<endl;
			assert(false);
		}
		int bucket1=hash1(vp)%(data.size()/4);
		int bucket2=hash2(vp)%(data.size()/4);
		int slot_array[4];
		for(int i=0;i<4;i++){
			if(bucket1!=prev_bucket){
				slot_array[i]=bucket1*4+i;
				prev_bucket=bucket1;
			} else {
				slot_array[i]=bucket2*4+i;
				prev_bucket=bucket2;
			}
		}
		for(int i=0;i<4;i++){
			if(data[slot_array[i]]==v_pair(-1,-1)){
				data[slot_array[i]]=vp;
				accum_count+=(level+1);
				return ;
			}
		}
		int kick_slot=rand()%4;
		v_pair kick_vp=data[slot_array[kick_slot]];
		data[slot_array[kick_slot]]=vp;
		recursive_cuckoo_insert(kick_vp,level+1,prev_bucket);
	}
	void cuckoo_insert(v_pair vp,int level){
		if(level>max_level){
			max_level=level;
		}
		if(level==0){
		}
		if(level==50000){
			cout<<"Too many evict"<<endl;
			cout<<"inserting "<<count<<"/"<<vec_id.size()<<endl;
			assert(false);
		}
		int slot_array[8];
		
		int bucket1=hash1(vp)%(data.size()/4);
		for(int i=0;i<4;i++){
			slot_array[i]=bucket1*4+i;
		}
		for(int i=0;i<4;i++){
			if(data[slot_array[i]]==v_pair(-1,-1)){
				data[slot_array[i]]=vp;
				accum_count+=(level+1);
				return ;
			}
		}
		
		int bucket2=hash2(vp)%(data.size()/4);
		for(int i=4;i<8;i++){
			slot_array[i]=bucket2*4+(i-4);
		}
		for(int i=4;i<8;i++){
			if(data[slot_array[i]]==v_pair(-1,-1)){
				data[slot_array[i]]=vp;
				accum_count+=(level+1);
				return ;
			}
		}
		int kick_slot=rand()%8;
		v_pair kick_vp=data[slot_array[kick_slot]];
		data[slot_array[kick_slot]]=vp;
		cuckoo_insert(kick_vp,level+1);
		// if(kick_slot<4){
		// 	recursive_cuckoo_insert(kick_vp,level+1,bucket1);
		// } else {
		// 	recursive_cuckoo_insert(kick_vp,level+1,bucket2);
		// }
	}


public:
	void insert(int id1,int id2){
#ifdef USE_TBB
		tbb_hashtable::accessor a; 
		tbb_set.insert( a, v_pair(id1,id2));
		a->second = true;
#else
		vec_id.push_back(v_pair(id1,id2));
#endif 
	}

	void rehash(){
#ifdef USE_TBB
		///tbb
		// tbb_set.rehash(vec_id.size()*2);
		// #pragma omp parallel for num_threads(global_num_server)
		// for(int i=0;i<vec_id.size();i++){
		// 	tbb_hashtable::accessor a; 
		// 	tbb_set.insert( a, vec_id[i]);
		// 	a->second = true;
		// }
		// vec_id.clear();

		// pair_set.reserve(vec_id.size()*1.5);
		// for(int i=0;i<vec_id.size();i++){
		// 	pair_set.insert(vec_id[i]);
		// }
		// vec_id.clear();
#else

		uint64_t t1=timer::get_usec();
		
		data.resize((1+vec_id.size()/2)*16);
		for(int i=0;i<data.size();i++){
			data[i]=v_pair(-1,-1);
		}
		for(int i=0;i<vec_id.size();i++){
			cuckoo_insert(vec_id[i],0);
			count++;
		}

		
		vec_id.clear();
		if(global_verbose){
			cout<<"max_level "<<max_level<<endl;
			if(count>0){
				cout<<"average_level "<<accum_count*1.0/count<<endl;
			}
			cout<<"cuckoo "<<(t2-t1)/1000.0<<"ms "<<endl;
		}
#endif 
	}

	bool contain(int id1,int id2){
#ifdef USE_TBB
		tbb_hashtable::const_accessor a; 
		return tbb_set.find(a, v_pair(id1,id2));
		// v_pair target=v_pair(id1,id2);
		// if(pair_set.find(target)!=pair_set.end())
		// 	return true;
		// return false;
#else
		v_pair vp=v_pair(id1,id2);
		int slot_array[4];
		int bucket1=hash1(vp)%(data.size()/4);
		for(int i=0;i<4;i++){
			slot_array[i]=bucket1*4+i;
		}
		for(int i=0;i<4;i++){
			if(data[slot_array[i]]==vp){
				return true;
			}
		}
		int bucket2=hash2(vp)%(data.size()/4);
		for(int i=0;i<4;i++){
			slot_array[i]=bucket2*4+i;
		}
		for(int i=0;i<4;i++){
			if(data[slot_array[i]]==vp){
				return true;
			}
		}
		return false;
#endif		
	}
};