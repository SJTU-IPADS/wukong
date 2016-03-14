#pragma once
#include "graph_basic_types.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#include <vector>
using namespace std;
using namespace boost::archive;

enum var_type{
	known_var,
	unknown_var,
    const_var
};
//defined as constexpr
//because it's used in switch-case
constexpr int var_pair(int type1,int type2){
    return ((type1<<4) | type2);
};
struct request_template{
	vector<string> place_holder_str;	// no serialize
	vector<int> place_holder_position;	// no serialize
	vector<vector<int>*  > place_holder_vecptr;	// no serialize
	vector<int> cmd_chains;
};
struct request_or_reply{
	int first_target;// no serialize

	int id;
	int parent_id;
    int step;
    int col_num;
	bool silent;
	int silent_row_num;
	int local_var;
    vector<int> cmd_chains; //n*(start,p,direction,end)
    vector<int> result_table;

	int mt_total_thread;
	int mt_current_thread;

    request_or_reply(){
		first_target=-1;
        parent_id=-1;
		id=-1;
        step=0;
        col_num=0;
		silent=false;
		silent_row_num=0;
		local_var=0;
		mt_total_thread=1;
		mt_current_thread=0;

    }
    template <typename Archive>
	void serialize(Archive &ar, const unsigned int version) {
		ar & id;
		ar & parent_id;
        ar & step;
        ar & col_num;
		ar & silent;
		ar & silent_row_num;
		ar & local_var;
        ar & cmd_chains;
        ar & result_table;
		ar & mt_total_thread;
		ar & mt_current_thread;
	}
	void clear_data(){
		result_table.clear();
	}
	bool is_finished(){
		return step*4>=cmd_chains.size();
	}
	bool is_request(){
		return id==-1;
	}
	bool use_index_vertex(){
		return cmd_chains[2]==pindex_in ||
			cmd_chains[2]==pindex_out ||
			cmd_chains[2]==tindex_in ;
	}
    var_type variable_type(int v){
        if(v>=0){
            return const_var;
        }
        if((-v)>column_num()){
            return unknown_var;
        } else {
            return known_var;
        }
    };

    int var2column(int v){
        return (-v-1);
    }
    void set_column_num(int n){
        col_num = n;
    }
    int column_num(){
        return col_num;
    };
    int row_num(){
        if(col_num==0){
            return 0;
        }
        return result_table.size()/col_num;
    }
    int get_row_column(int r,int c){
        return result_table[col_num*r+c];
    }
    void append_row_to(int r,vector<int>& updated_result_table){
        for(int c=0;c<column_num();c++){
            updated_result_table.push_back(get_row_column(r,c));
        }
    };
};
