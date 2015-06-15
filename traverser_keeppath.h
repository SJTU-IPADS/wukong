#pragma once

#include "graph.h"
struct path_node{
	path_node(int _id,int _prev):id(_id),prev(_prev){
	}
	int id;
	int prev;
};
class traverser_keeppath{
	
	graph& g;
	
public:
	vector<vector<path_node> >paths;
	traverser_keeppath(graph& gg):g(gg){
	} 
	traverser_keeppath& print_count(){
		cout<<paths[paths.size()-1].size()<<endl;
		return *this;
	}
	traverser_keeppath& lookup(string subject){
		paths.clear();
		vector<path_node> vec;
		vec.push_back(path_node(g.subject_to_id[subject],-1));
		paths.push_back(vec);
		return *this;
	}
	traverser_keeppath& LoadNeighbors(string dir,string predict){
		vector<path_node> vec;
		int	predict_id=g.predict_to_id[predict];

		vector<path_node>& prev_vec=paths[paths.size()-1];
		for (int i=0;i<prev_vec.size();i++){
			int prev_id=prev_vec[i].id;
			if(dir =="in" || dir == "all"){
				for (auto row : g.vertex_array[prev_id].in_edges){
					if(predict_id==row.predict){
						vec.push_back(path_node(row.vid,prev_id));
					}
				}
			}
			if(dir =="out" || dir == "all"){
				for (auto row : g.vertex_array[prev_id].out_edges){
					if(predict_id==row.predict){
						vec.push_back(path_node(row.vid,prev_id));
					}
				}
			}
		}
		paths.push_back(vec);
		return *this;
	}
	traverser_keeppath& get_all_subtype(string target){
		
		int target_id=g.subject_to_id[target];
		unordered_set<int> ids = g.ontology_array.get_all_subtype(target_id);
		paths.clear();
		vector<path_node> vec;
		for(auto id: ids){
			vec.push_back(path_node(id,-1));
		}
		paths.push_back(vec);
		return *this;
	}
	traverser_keeppath& is_subclass_of(string target){

		int predict_id=g.predict_to_id["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"];
		int target_id=g.subject_to_id[target];

		vector<path_node>& prev_vec=paths[paths.size()-1];
		vector<path_node> vec;
		for (int i=0;i<prev_vec.size();i++){
			int prev_id=prev_vec[i].id;	
			int prev_of_prev=prev_vec[i].prev;		
			for (auto row : g.vertex_array[prev_id].out_edges){
				if(predict_id==row.predict && 
							g.ontology_array.is_subtype_of(row.vid,target_id)){
					vec.push_back(path_node(prev_id,prev_of_prev));
				}
			}
		}
		//remove all useless ids
		paths[paths.size()-1]=vec;
		return *this;
	}

	traverser_keeppath& print_property(){
		//don't print now
		return *this;
	}


};