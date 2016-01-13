
#pragma once

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/unordered_set.hpp>
#include <vector>
using namespace boost::archive;
using std::vector;

enum command{
	cmd_subclass_of,
	cmd_neighbors,
	cmd_triangle,
	cmd_get_attr,
	cmd_get_subtype,
	cmd_predict_index,
	cmd_type_index,
	cmd_filter,
	cmd_join
};
enum parameter{
	para_in,
	para_out,
	para_all
};
enum direction{
	direction_forward,
	direction_reverse
};
int reverse_dir(int dir){
	if(dir==para_in)
		return para_out;
	if(dir==para_out)
		return para_in;
	return para_all;
}

struct request{
	uint64_t timestamp;
	int req_id;
	int parent_id;
	bool blocking;
	vector<int> cmd_chains;
	vector<vector<int> >result_table;
	int parallel_total;
	int parallel_id;
	request(){
		req_id=-1;
		parent_id=-1;
		blocking=false;
		parallel_total=1000;
		parallel_id=0;
	}
	void clear(){
		cmd_chains.clear();
		result_table.clear();
	}
	void clear_data(){
		// boost::unordered_set<int> final_set;
		// for(int i=0;i<row_num();i++){
		// 	final_set.insert(last_column(i));
		// }
		//cout<<"size = "<<row_num()<< " non-dup = "<<final_set.size()<<endl;
		for(int i=0;i<result_table.size();i++){
			result_table[i].clear();
		}		
	}
	int column_num(){
		return result_table.size();
	}
	int row_num(){
		int path_len=result_table.size();
		if(path_len==0)
			return 0;
		return result_table[path_len-1].size();
	}
	vector<int> * last_level(){
		int path_len=result_table.size();
		if(path_len==0)
			return NULL;
		return &result_table[path_len-1];
	}
	int get(int row,int column){
		assert(row<row_num());
		assert(column<column_num());
		return result_table[column][row];
	}
	int last_column(int row){
		assert(row<row_num());
		assert(0<column_num());
		return result_table[column_num()-1][row];
	}
	void append_row_to(vector<vector<int> >& target_table,int row){
		assert(column_num()<= target_table.size());
		for(int i=0;i<column_num();i++){
			target_table[i].push_back(result_table[i][row]);
		}
	}
	template <typename Archive>
	void serialize(Archive &ar, const unsigned int version) { 
		ar & timestamp;
		ar & req_id; 
		ar & parent_id; 
		ar & cmd_chains; 
		ar & result_table; 
		ar & parallel_total; 
		ar & parallel_id; 
	}
};
