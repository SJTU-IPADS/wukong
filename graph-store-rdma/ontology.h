#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;


class ontology{
public:
	unordered_map<int,vector<int> > id_to_children;
	unordered_map<int,int> id_to_parent;
	void insert(int child,int parent){
		id_to_parent[child]=parent;
		if(id_to_parent.find(parent)==id_to_parent.end())
			id_to_parent[parent]=-1;

		if(id_to_children.find(child)==id_to_children.end())
			id_to_children[child]=vector<int>();
		id_to_children[parent].push_back(child);
		
	}
	void insert_type(int type){
		if(id_to_parent.find(type)==id_to_parent.end())
			id_to_parent[type]=-1;
		if(id_to_children.find(type)==id_to_children.end())
			id_to_children[type]=vector<int>();
	}
	bool is_subtype_of(int child,int parent){
		while(true){
			if(child==parent)
				return true;
			child=id_to_parent[child];
			if(child==-1)
				return false;
		}
	}

	unordered_set<int>  get_all_subtype(int id){
		unordered_set<int>  result;
		result.insert(id);
		for(auto child : id_to_children[id]){
			unordered_set<int> subtype_of_child=get_all_subtype(child);
			for(auto item : subtype_of_child){
				result.insert(item);
			}
		}
		return result;
	}	
};
