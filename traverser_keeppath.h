#pragma once

#include "graph.h"
struct path_node{
	path_node(int _id,int _prev):id(_id),prev(_prev){
	}
	int id;
	int prev;
};
struct path_node_less_than
{
    inline bool operator() (const path_node& struct1, const path_node& struct2)
    {
        return (struct1.id < struct2.id);
    }
};
class traverser_keeppath{

	graph& g;
	path_node get_node(int row,int column){
		assert(row<paths[paths.size()-1].size());
		assert(column<paths.size());
		int current_column=paths.size()-1;
		while(column<current_column){
			row=paths[current_column][row].prev;
			current_column--;
		}
		return paths[column][row];
	}
	vector<vector<path_node> >paths;
public:
	traverser_keeppath(graph& gg):g(gg){
	} 
	traverser_keeppath( const traverser_keeppath& other ):g(other.g),paths(other.paths){
	}
	traverser_keeppath& sort(){
		vector<path_node>& vec_to_sort=paths[paths.size()-1];
		std::sort(vec_to_sort.begin(), vec_to_sort.end(), path_node_less_than());
		return *this;
	}
	int get_path_length(){
		return paths.size();
	}
	int get_path_num(){
		return paths[paths.size()-1].size();
	}
	traverser_keeppath& merge(traverser_keeppath& other, int split_length){
		sort();
		other.sort();
		int my_row=0;
		int other_start=0;
		vector<vector<path_node> >new_path;
		for(int i=split_length;i<other.get_path_length();i++){
			new_path.push_back(vector<path_node>());
		}
		while(my_row<get_path_num()){
			path_node my_path_begin=get_node(my_row,split_length-1);
			path_node my_path_end  =get_node(my_row,get_path_length()-1);
			int other_row=other_start;
			while(other_row<other.get_path_num()){
				path_node other_path_begin=other.get_node(other_row,split_length-1);
				path_node other_path_end  =other.get_node(other_row,other.get_path_length()-1);
				if(other_path_end.id<my_path_end.id){
					//skip this other.row forever 
					other_row++;
					other_start++;
					continue;
				} else if(other_path_end.id == my_path_end.id){
					if(other_path_begin.id  == my_path_begin.id){
						// this row match
						path_node node=other.get_node(other_row,split_length);
						node.prev=my_row;
						new_path[0].push_back(node);
						for(int column=split_length+1;column<other.get_path_length();column++){
							path_node node=other.get_node(other_row,column);
							node.prev=new_path[0].size()-1;
							new_path[column-split_length].push_back(node);
						}
						other_row++;
					} else {
						// begin of path doesn't match
						other_row++;
					}
				} else {
					// we already check all possible other.rows for my_row
					break;
				}
			}
			my_row++;
		}
		for(int i=0;i<new_path.size();i++){
			paths.push_back(new_path[i]);
		}
		return *this;
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
						vec.push_back(path_node(row.vid,i));
					}
				}
			}
			if(dir =="out" || dir == "all"){
				for (auto row : g.vertex_array[prev_id].out_edges){
					if(predict_id==row.predict){
						vec.push_back(path_node(row.vid,i));
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
			for (auto row : g.vertex_array[prev_id].out_edges){
				if(predict_id==row.predict && 
							g.ontology_array.is_subtype_of(row.vid,target_id)){
					vec.push_back(prev_vec[i]);
				}
			}
		}
		//instead of append vec to path
		//we just replace the last vec
		paths[paths.size()-1]=vec;
		return *this;
	}

	traverser_keeppath& print_property(){
		//don't print now
		for(int i=0;i<paths[paths.size()-1].size();i++){
			for(int column=0;column<paths.size();column++){
				if(i>=paths[column].size()){
					cout<<"\t";
				} else {
					cout<<paths[column][i].id<<","
						<<paths[column][i].prev<<"\t";
				}
			}
			cout<<endl;
		}

		return *this;
	}


};