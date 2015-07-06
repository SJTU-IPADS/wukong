#pragma once

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <vector>
using namespace boost::archive;
using std::vector;

enum command{
	cmd_subclass_of,
	cmd_neighbors
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
	vector<int> cmd_chains;
	vector<vector<path_node> >result_paths;
	void clear(){
		cmd_chains.clear();
		result_paths.clear();
	}
	template <typename Archive>
	void serialize(Archive &ar, const unsigned int version) { 
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
