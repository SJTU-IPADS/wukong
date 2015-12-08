#pragma once

#include <vector>
//#include <boost/unordered_set.hpp>
typedef std::pair<uint64_t,uint64_t> v_pair;
struct hash_vpair{
	size_t operator()(const v_pair &x) const{
		return hash<uint64_t>()(x.first) ^ hash<uint64_t>()(x.second);
	}
};
class simple_filter{
	vector<v_pair> vec_id;
	vector<v_pair> data;
public:
	void insert(uint64_t id1,uint64_t id2){
		vec_id.push_back(v_pair(id1,id2));
	}
	void rehash(){
		data.resize(vec_id.size()*2);
		for(uint64_t i=0;i<data.size();i++){
			data[i]=v_pair(-1,-1);
		}
		for(uint64_t i=0;i<vec_id.size();i++){
			uint64_t hash= hash_vpair()(vec_id[i]);
			hash=hash%data.size();
			while(data[hash]!=v_pair(-1,-1)){
				hash=(hash+1)%data.size();
			}
			data[hash]=vec_id[i];
		}
		vec_id.clear();
	}
	bool contain(uint64_t id1,uint64_t id2){
		v_pair target=v_pair(id1,id2);
		uint64_t hash= hash_vpair()(target);
		hash=hash%data.size();
		while(data[hash]!=v_pair(-1,-1)){
			if(data[hash]==target)
				return true;
			hash=(hash+1)%data.size();
		}
		return false;
	}
};