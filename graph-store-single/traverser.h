#pragma once

#include "graph.h"
class traverser{
	
	graph& g;
	
public:
	unordered_set<int> ids;
	traverser(graph& gg):g(gg){
	} 
	traverser& print_count(){
		cout<<ids.size()<<endl;
		return *this;
	}
	int count(){
		return ids.size();
	}
	traverser& lookup(string subject){
		unordered_set<int> empty;
		ids=empty;
		ids.insert(g.subject_to_id[subject]);
		return *this;
	}
	traverser& LoadNeighbors(string dir,string predict){
		unordered_set<int> neighbors;
		int	predict_id=g.predict_to_id[predict];

		for (auto i : ids){
			if(dir =="in" || dir == "all"){
				for (auto row : g.vertex_array[i].in_edges){
					if(predict_id==row.predict){
						neighbors.insert(row.vid);
					}
				}
			}
			if(dir =="out" || dir == "all"){
				for (auto row : g.vertex_array[i].out_edges){
					if(predict_id==row.predict){
						neighbors.insert(row.vid);
					}
				}
			}
		}
		ids=neighbors;
		return *this;
	}
	traverser& get_all_subtype(string target){
		int target_id=g.subject_to_id[target];
		ids=g.ontology_array.get_all_subtype(target_id);
		return *this;
	}
	traverser& is_subclass_of(string target){
		unordered_set<int> left_ids;

		int predict_id=g.predict_to_id["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"];
		int target_id=g.subject_to_id[target];

		for (auto i : ids){
			for (auto row : g.vertex_array[i].out_edges){
				if(predict_id==row.predict && 
							g.ontology_array.is_subtype_of(row.vid,target_id)){
					left_ids.insert(i);
					break;
				}
			}
		}
		ids=left_ids;
		return *this;
	}
	// traverser& filter(string dir,string predict,string target){
	// 	unordered_set<int> left_ids;

	// 	int predict_id=g.predict_to_id[predict];
	// 	int vid=g.subject_to_id[target];

	// 	for (auto i : ids){
	// 		if(dir =="in" || dir == "all"){
	// 			for (auto row : g.vertex_array[i].in_edges){
	// 				if(predict_id==row.predict && vid==row.vid){
	// 					left_ids.insert(i);
	// 					break;
	// 				}
	// 			}
	// 		}
	// 		if(dir =="out" || dir == "all"){
	// 			for (auto row : g.vertex_array[i].out_edges){
	// 				if(predict_id==row.predict && vid==row.vid){
	// 					left_ids.insert(i);
	// 					break;
	// 				}
	// 			}
	// 		}
	// 	}
	// 	ids=left_ids;
	// 	return *this;
	// }

	traverser& print_property(){
		for (auto i : ids){
			cout<<g.id_to_subject[i]<<endl;
			for (auto j : g.vertex_array[i].propertys){
				cout<<"\t"<<g.id_to_predict[j.property]<<"\t"<<j.value<<endl;
			}
			for (auto row : g.vertex_array[i].out_edges){
				cout<<"\t"<<g.id_to_predict[row.predict]<<"\t"<<g.id_to_subject[row.vid]<<endl;
			}
		}
		return *this;
	}


};