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
#include "timer.h"
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>

#include "ontology.h"
#include "klist_store.h"
#include "rdma_resource.h"

using namespace std;

class graph{
	boost::mpi::communicator& world;
	RdmaResource* rdma;
public:

	unordered_map<uint64_t,vertex_row> vertex_table;
	klist_store kstore;
	ontology ontology_table;
	uint64_t in_edges;
	uint64_t out_edges;
	void load_ontology(string filename){
		cout<<"loading "<<filename<<endl;
		ifstream file(filename.c_str());
		uint64_t child,parent;
		while(file>>child>>parent){
			if(vertex_table.find(child)==vertex_table.end()){
				if(child%(world.size())==world.rank())
					vertex_table[child]=vertex_row();
				ontology_table.insert_type(child);				
			}
			if(parent!=-1 && vertex_table.find(parent)==vertex_table.end()){
				if(parent%(world.size())==world.rank())
					vertex_table[parent]=vertex_row();
				ontology_table.insert_type(parent);	
			} 
			if(parent !=-1){
				ontology_table.insert(child,parent);
			}
		}
		file.close();
	}
	void load_data(string filename){
		cout<<"loading "<<filename<<endl;
		ifstream file(filename.c_str());
		uint64_t s,p,o;
		while(file>>s>>p>>o){
			if(s%(world.size())==world.rank()){
				vertex_table[s].out_edges.push_back(edge_row(p,o));
				in_edges++;
			}
			if(o%(world.size())==world.rank()){
				vertex_table[o].in_edges.push_back(edge_row(p,s));
				out_edges++;
			}
		}
		file.close();
	}
	void print_graph_info(){
		cout<<world.rank()<<" has "<<vertex_table.size()<<" vertex"<<endl;
		cout<<world.rank()<<" has "<<in_edges<<" in_edges"<<endl;
	}
	graph(boost::mpi::communicator& para_world,RdmaResource* _rdma,const char* dir_name)
			:world(para_world),rdma(_rdma){
		in_edges=0;
		out_edges=0;
		struct dirent *ptr;    
	    DIR *dir;
	    dir=opendir(dir_name);
	    vector<string> filenames;
	    while((ptr=readdir(dir))!=NULL){
	        if(ptr->d_name[0] == '.')
	            continue;
	        string fname(ptr->d_name);
	        string index_prefix="index";
	        string data_prefix="id";
			if(equal(index_prefix.begin(), index_prefix.end(), fname.begin())){
				// we only need to load index_ontology
				string ontology_prefix="index_ontology";
				if(equal(ontology_prefix.begin(), ontology_prefix.end(), fname.begin())){
					load_ontology(string(dir_name)+"/"+fname);
				} else {
					continue;
				}
			} else if(equal(data_prefix.begin(), data_prefix.end(), fname.begin())){
				filenames.push_back(string(dir_name)+"/"+fname);
			} else{
				cout<<"What's this file:"<<fname<<endl;
				//assert(false);
			}
	    }
	    for(int i=0;i<filenames.size();i++){
	    	load_data(filenames[i]);
	    }
	    print_graph_info();

	    // uint64_t store_max_size=1024*1024*1024;
	    // store_max_size=store_max_size*2;
	    // char* start_addr=(char*)malloc(store_max_size);
	    // kstore.init(start_addr,1000000,world.size(),store_max_size);
	    uint64_t max_v_num=1000000*20;
	    kstore.init(rdma,max_v_num,world.size(),world.rank());
	    unordered_map<uint64_t,vertex_row>::iterator iter;
		for(iter=vertex_table.begin();iter!=vertex_table.end();iter++){
			kstore.insert(iter->first,iter->second);
		}
		cout<<"graph-store use "<<max_v_num*sizeof(vertex) / 1024 / 1024<<" MB for vertex data"<<endl;
		cout<<"graph-store use "<<kstore.new_edge_ptr * sizeof(edge_row) / 1024 / 1024<<" MB for edge data"<<endl;
	}
};


