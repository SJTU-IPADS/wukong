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
#include "message_wrap.h"
#include "thread_cfg.h"

using std::string;

class index_server{
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
		uint64_t t1=timer::get_usec();
	    
		cout<<"index_server loading "<<filename<<endl;
		ifstream file(filename.c_str());
		string str;
		while(file>>str){
			str2id[str]=id2str.size();
			id2str.push_back(str);
		}
		file.close();
		uint64_t t2=timer::get_usec();
		cout<<"loading "<<filename<<" in "<<(t2-t1)/1000.0/1000.0<<"s ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;

	}
	void load_minimal_index(string filename,unordered_map<string,int>& str2id,
					vector<string>& id2str){
		uint64_t t1=timer::get_usec();
	    
		cout<<"index_server loading "<<filename<<endl;
		ifstream file(filename.c_str());
		string str;
		int id;
		while(file>>str>>id){
			str2id[str]=id;
		}
		file.close();
		uint64_t t2=timer::get_usec();
		cout<<"loading "<<filename<<" in "<<(t2-t1)/1000.0/1000.0<<"s ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;

	}
	
	
	void print_tree(int id,int level){
		for(int i=0;i<level;i++)
			cout<<"\t";
		//cout<<id_to_subject[id]<<endl;
		for(auto child : ontology_table.id_to_children[id]){
			print_tree(child,level+1);
		}
	}
	void print_ontology_tree(){
		cout<<"#############  print tree #############"<<endl;
		for (auto row : ontology_table.id_to_parent){
			if(row.second==-1)
				print_tree(row.first,0);
		}
	}

public:
	unordered_map<string,int> subject_to_id;
	unordered_map<string,int> predict_to_id;
	vector<string> id_to_subject;
	vector<string> id_to_predict;
	ontology ontology_table;
	request req;
	index_server(const char* dir_name){
		
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
				if(!global_load_minimal_index)
					load_index(complete_fname,subject_to_id,id_to_subject);
			} else if(fname == "minimal_index_subject"){
				if(global_load_minimal_index)
					load_minimal_index(complete_fname,subject_to_id,id_to_subject);
			}else if(fname == "index_predict"){
				load_index(complete_fname,predict_to_id,id_to_predict);
			} else{
				continue;
			}
		}
	}
};