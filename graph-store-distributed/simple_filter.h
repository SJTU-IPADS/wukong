#pragma once

#include <vector>
#include <boost/unordered_set.hpp>
typedef std::pair<uint64_t,uint64_t> v_pair;
struct hash_vpair{
	size_t operator()(const v_pair &x) const{
		return hash<uint64_t>()(x.first) ^ hash<uint64_t>()(x.second);
	}
};
size_t hash1(const v_pair &x){
	return hash<uint64_t>()(x.first^x.second);
}
size_t hash2(const v_pair &x){
	return hash<uint64_t>()(x.first+x.second);
}
#define USE_BOOST_SET


class simple_filter{
public:
	vector<v_pair> vec_id;
	vector<v_pair> data;
	boost::unordered_set<v_pair> pair_set;
	void cuckoo_insert(v_pair vp,uint64_t level){
		if(level==500){
			cout<<"Too many evict"<<endl;
			assert(false);
		}
		uint64_t bucket1=hash1(vp)%(data.size()/4);
		uint64_t bucket2=hash2(vp)%(data.size()/4);
		uint64_t slot_array[8];
		for(int i=0;i<4;i++){
			slot_array[i]=bucket1*4+i;
		}
		for(int i=4;i<8;i++){
			slot_array[i]=bucket2*4+(i-4);
		}
		for(int i=0;i<8;i++){
			if(data[slot_array[i]]==v_pair(-1,-1)){
				data[slot_array[i]]=vp;
				return ;
			}
		}
		int kick_slot=rand()%8;
		v_pair kick_vp=data[slot_array[kick_slot]];
		data[slot_array[kick_slot]]=vp;
		cuckoo_insert(kick_vp,level+1);
	}
public:
	void insert(uint64_t id1,uint64_t id2){
		vec_id.push_back(v_pair(id1,id2));
	}

	void rehash(){
#ifdef USE_BOOST_SET
		for(uint64_t i=0;i<vec_id.size();i++){
			pair_set.insert(vec_id[i]);
		}
		vec_id.clear();
#else

		data.resize(vec_id.size()/2*4);
		for(uint64_t i=0;i<data.size();i++){
			data[i]=v_pair(-1,-1);
		}
		for(uint64_t i=0;i<vec_id.size();i++){
			cuckoo_insert(vec_id[i],0);
		}
#endif 
	}

	bool contain(uint64_t id1,uint64_t id2){
#ifdef USE_BOOST_SET
		v_pair target=v_pair(id1,id2);
		if(pair_set.find(target)!=pair_set.end())
			return true;
		return false;
#else
		v_pair vp=v_pair(id1,id2);
		uint64_t bucket1=hash1(vp)%(data.size()/4);
		uint64_t bucket2=hash2(vp)%(data.size()/4);
		uint64_t slot_array[8];
		for(int i=0;i<4;i++){
			slot_array[i]=bucket1*4+i;
		}
		for(int i=4;i<8;i++){
			slot_array[i]=bucket2*4+(i-4);
		}
		for(int i=0;i<8;i++){
			if(data[slot_array[i]]==vp){
				return true;
			}
		}
		return false;
#endif		
	}
};