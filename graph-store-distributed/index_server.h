#pragma once

#include <string>  
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <assert.h> 
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#include "request.h"
#include "ontology.h"
#include "timer.h"
#include "network_node.h"
using std::string;

class index_server{
	unordered_map<string,int> subject_to_id;
	unordered_map<string,int> predict_to_id;
	vector<string> id_to_subject;
	vector<string> id_to_predict;
	ontology ontology_table;
	Network_Node* node;
	void load_ontology(string filename){
		cout<<"index_server loading "<<filename<<endl;
		ifstream file(filename.c_str());
		int child,parent;
		while(file>>child>>parent){
			ontology_table.insert_type(child);
			if(parent !=-1){
				ontology_table.insert_type(parent);
				ontology_table.insert(child,parent);
			}
		}
		file.close();
	}
	void load_index(string filename,unordered_map<string,int>& str2id,
					vector<string>& id2str){
		cout<<"index_server loading "<<filename<<endl;
		ifstream file(filename.c_str());
		string str;
		while(file>>str){
			str2id[str]=id2str.size();
			id2str.push_back(str);
		}
		file.close();
	}
	boost::mpi::communicator& world;
	int req_id;
	int get_id(){
		int result=req_id;
		req_id+=world.size();
		return result;
	}
	int first_target;
public:
	request req;
	index_server(boost::mpi::communicator& para_world,char* dir_name,int id):world(para_world){
		first_target=0;
		req_id=world.rank();
		node=new Network_Node(world.rank(),id);
		struct dirent *ptr;    
		DIR *dir;
		dir=opendir(dir_name);
		while((ptr=readdir(dir))!=NULL){
			if(ptr->d_name[0] == '.')
				continue;
			string fname(ptr->d_name);
			string complete_fname=string(dir_name)+"/"+fname;
			string index_prefix="index";
			if(fname == "index_ontology"){
				load_ontology(complete_fname);
			} else if(fname == "index_subject"){
				load_index(complete_fname,subject_to_id,id_to_subject);
			} else if(fname == "index_predict"){
				load_index(complete_fname,predict_to_id,id_to_predict);
			} else{
				continue;
			}
		}
	}
	index_server& lookup(string subject){
		first_target=subject_to_id[subject]%(world.size());
		req.clear();
		path_node node(subject_to_id[subject],-1);
		vector<path_node> vec;
		vec.push_back(node);
		req.result_paths.push_back(vec);
		return *this;
	}
	index_server& get_subtype(string target){
		req.clear();
		int target_id=subject_to_id[target];
		req.cmd_chains.push_back(cmd_get_subtype);
		req.cmd_chains.push_back(target_id);
		return *this;
	}
	index_server& neighbors(string dir,string predict){
		req.cmd_chains.push_back(cmd_neighbors);
		if(dir =="in" ){
			req.cmd_chains.push_back(para_in);
		} else if (dir =="out" ){
			req.cmd_chains.push_back(para_out);
		} else {
			req.cmd_chains.push_back(para_all);
		}
		req.cmd_chains.push_back(predict_to_id[predict]);
		return *this;
	}
	index_server& subclass_of(string target){
		req.cmd_chains.push_back(cmd_subclass_of);
		req.cmd_chains.push_back(subject_to_id[target]);
		return *this;
	}
	index_server& execute(){
		// reverse cmd_chains
		// so we can easily pop the cmd and do recursive operation
		reverse(req.cmd_chains.begin(),req.cmd_chains.end()); 	
		req.req_id=-1;
		//req.parent_id=get_id();
		req.parent_id=world.rank()-world.size();
		node->SendReq(first_target, 1, req);
		req=node->RecvReq();
		req.cmd_chains.clear();
		return *this;
	}
	index_server& print_count(){
		int path_len=req.result_paths.size();
		cout<<req.result_paths[path_len-1].size()<<endl;
		return *this;
	}

};