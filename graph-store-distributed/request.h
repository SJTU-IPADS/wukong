
#pragma once

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#include <vector>
using namespace boost::archive;
using std::vector;

enum command{
	cmd_subclass_of,
	cmd_neighbors,
	cmd_get_subtype
};
enum parameter{
	para_in,
	para_out,
	para_all
};

struct path_node{
	path_node():id(-1),prev(-1){
	}
	path_node(int _id,int _prev):id(_id),prev(_prev){
	}
	int id;
	int prev;	
	template <typename Archive> 
	void serialize(Archive &ar, const unsigned int version) { 
		ar & id; 
		ar & prev; 
	}
};
struct path_node_less_than
{
    inline bool operator() (const path_node& struct1, const path_node& struct2)
    {
        return (struct1.id < struct2.id);
    }
};
struct request{
	int req_id;
	int parent_id;
	bool blocking;
	vector<int> cmd_chains;
	vector<vector<path_node> >result_paths;
	request(){
		req_id=-1;
		parent_id=-1;
		blocking=false;
	}
	void clear(){
		cmd_chains.clear();
		result_paths.clear();
	}
	int path_length(){
		return result_paths.size();
	}
	int path_num(){
		int path_len=result_paths.size();
		if(path_len==0)
			return 0;
		return result_paths[path_len-1].size();
	}
	vector<path_node> * last_level(){
		int path_len=result_paths.size();
		if(path_len==0)
			return NULL;
		return &result_paths[path_len-1];
	}
	path_node get_node(int row,int column){
		assert(row<path_num());
		assert(column<path_length());
		int current_column=path_length()-1;
		while(column<current_column){
			row=result_paths[current_column][row].prev;
			current_column--;
		}
		return result_paths[column][row];
	}
	void sort(){
		int path_len=result_paths.size();
		if(path_len==0)
			return ;
		vector<path_node>& vec_to_sort=result_paths[path_len-1];
		std::sort(vec_to_sort.begin(), vec_to_sort.end(), path_node_less_than());
		return ;
	}
	void merge(request& other, int split_length){
		sort();
		other.sort();
		int my_row=0;
		int other_start=0;
		vector<vector<path_node> >new_path;
		for(int i=split_length;i<other.path_length();i++){
			new_path.push_back(vector<path_node>());
		}
		while(my_row<path_num()){
			path_node my_path_begin=get_node(my_row,split_length-1);
			path_node my_path_end  =get_node(my_row,path_length()-1);
			int other_row=other_start;
			while(other_row<other.path_num()){
				path_node other_path_begin=other.get_node(other_row,split_length-1);
				path_node other_path_end  =other.get_node(other_row,other.path_length()-1);
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
						for(int column=split_length+1;column<other.path_length();column++){
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
			result_paths.push_back(new_path[i]);
		}
		return ;
	}

	template <typename Archive>
	void serialize(Archive &ar, const unsigned int version) { 
		ar & req_id; 
		ar & parent_id; 
		ar & cmd_chains; 
		ar & result_paths; 
	}
};
