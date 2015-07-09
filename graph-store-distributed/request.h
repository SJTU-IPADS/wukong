
#pragma once

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
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
	template <typename Archive>
	void serialize(Archive &ar, const unsigned int version) { 
		ar & req_id; 
		ar & parent_id; 
		ar & cmd_chains; 
		ar & result_paths; 
	}
};

struct path_node_less_than
{
    inline bool operator() (const path_node& struct1, const path_node& struct2)
    {
        return (struct1.id < struct2.id);
    }
};
