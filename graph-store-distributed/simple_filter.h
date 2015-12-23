#pragma once

#include <vector>
#include <boost/unordered_set.hpp>
typedef std::pair<int,int> v_pair;
struct hash_vpair{
	size_t operator()(const v_pair &x) const{
		return hash<int>()(x.first) ^ hash<int>()(x.second);
	}
};
size_t hash1(const v_pair &x){
	return hash<int>()(x.first^x.second);
}
size_t hash2(const v_pair &x){
	return hash<int>()(x.first+x.second);
}
#define USE_BOOST_SET


class simple_filter{
public:
	vector<v_pair> vec_id;
	vector<v_pair> data;
#ifdef USE_BOOST_SET
	boost::unordered_set<v_pair> pair_set;
#endif 
	int count;
	int accum_count;
	int max_level;
	simple_filter(){
		count=0;
		max_level=0;
		accum_count=0;
	}
	void cuckoo_insert(v_pair vp,int level){
		if(level>max_level){
			max_level=level;
		}
		if(level==0){
		}
		if(level==500){
			cout<<"Too many evict"<<endl;
			cout<<"inserting "<<count<<"/"<<vec_id.size()<<endl;
			assert(false);
		}
		int bucket1=hash1(vp)%(data.size()/4);
		int bucket2=hash2(vp)%(data.size()/4);
		int slot_array[8];
		for(int i=0;i<4;i++){
			slot_array[i]=bucket1*4+i;
		}
		for(int i=4;i<8;i++){
			slot_array[i]=bucket2*4+(i-4);
		}
		for(int i=0;i<8;i++){
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
	}
public:
	void insert(int id1,int id2){
		vec_id.push_back(v_pair(id1,id2));
	}

	void rehash(){
#ifdef USE_BOOST_SET
		for(int i=0;i<vec_id.size();i++){
			pair_set.insert(vec_id[i]);
		}
		vec_id.clear();
#else

		data.resize((vec_id.size()/2)*8);
		for(int i=0;i<data.size();i++){
			data[i]=v_pair(-1,-1);
		}
		for(int i=0;i<vec_id.size();i++){
			cuckoo_insert(vec_id[i],0);
			count++;
		}
		vec_id.clear();
		cout<<"max_level "<<max_level<<endl;
		cout<<"average_level "<<accum_count*1.0/count<<endl;
#endif 
	}

	bool contain(int id1,int id2){
#ifdef USE_BOOST_SET
		v_pair target=v_pair(id1,id2);
		if(pair_set.find(target)!=pair_set.end())
			return true;
		return false;
#else
		v_pair vp=v_pair(id1,id2);
		int bucket1=hash1(vp)%(data.size()/4);
		int bucket2=hash2(vp)%(data.size()/4);
		int slot_array[8];
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