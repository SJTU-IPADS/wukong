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
#include "omp.h"
using namespace std;

class graph{
	boost::mpi::communicator& world;
	RdmaResource* rdma;
public:
	static const int num_vertex_table=100;
	//unordered_map<uint64_t,vertex_row> vertex_table;
	unordered_map<uint64_t,vertex_row> vertex_table[num_vertex_table];
	pthread_spinlock_t vertex_table_lock[num_vertex_table];
	klist_store kstore;
	ontology ontology_table;
	uint64_t in_edges;
	uint64_t out_edges;
	void load_ontology(string filename){
		cout<<"loading "<<filename<<endl;
		ifstream file(filename.c_str());
		uint64_t child,parent;
		while(file>>child>>parent){
			int vt_id=0;
			//vt_id=(child/world.size())%num_vertex_table;
			if(vertex_table[vt_id].find(child)==vertex_table[vt_id].end()){
				if(child%(world.size())==world.rank())
					vertex_table[vt_id][child]=vertex_row();
				ontology_table.insert_type(child);				
			}
			//vt_id=(parent/world.size())%num_vertex_table;
			if(parent!=-1 && vertex_table[vt_id].find(parent)==vertex_table[vt_id].end()){
				if(parent%(world.size())==world.rank())
					vertex_table[vt_id][parent]=vertex_row();
				ontology_table.insert_type(parent);	
			} 
			if(parent !=-1){
				ontology_table.insert(child,parent);
			}
		}
		vertex_table[0].clear();
		file.close();
	}
	void load_data(string filename){
		//cout<<"loading "<<filename<<endl;
		ifstream file(filename.c_str());
		uint64_t s,p,o;
		while(file>>s>>p>>o){
			int vt_id;
			if(s%(world.size())==world.rank()){
				vt_id=(s/world.size())%num_vertex_table;
				pthread_spin_lock(&vertex_table_lock[vt_id]);
				vertex_table[vt_id][s].out_edges.push_back(edge_row(p,o));
				pthread_spin_unlock(&vertex_table_lock[vt_id]);
			}
			if(o%(world.size())==world.rank()){
				vt_id=(o/world.size())%num_vertex_table;
				pthread_spin_lock(&vertex_table_lock[vt_id]);
				vertex_table[vt_id][o].in_edges.push_back(edge_row(p,s));
				pthread_spin_unlock(&vertex_table_lock[vt_id]);
			}
		}
		file.close();
	}
	void load_convert_data(int i,string s_file,string o_file){
		ifstream file1(s_file.c_str());
		ifstream file2(o_file.c_str());
		uint64_t s,p,o;
		uint64_t count=0;
		while(file1>>s>>p>>o){
			vertex_table[i][s].out_edges.push_back(edge_row(p,o));
			count++;
		}
		while(file2>>s>>p>>o){
			vertex_table[i][o].in_edges.push_back(edge_row(p,s));
			count++;
		}
		pthread_spin_lock(&vertex_table_lock[0]);
		uint64_t curr_edge_ptr=kstore.alloc_edges(count);
		pthread_spin_unlock(&vertex_table_lock[0]);
		unordered_map<uint64_t,vertex_row>::iterator iter;
		for(iter=vertex_table[i].begin();iter!=vertex_table[i].end();iter++){
			if(iter->second.out_edges.size()==0 && iter->second.in_edges.size()==0){
				continue;
			}
			kstore.insert_at(iter->first,iter->second,curr_edge_ptr);
			curr_edge_ptr+=iter->second.out_edges.size();
			curr_edge_ptr+=iter->second.in_edges.size();
		}
		cout<<"finished loading "<<s_file<<endl;
	}

	void load_convert_s_data(int i,string filename){
		//cout<<"loading "<<filename<<endl;
		ifstream file(filename.c_str());
		uint64_t s,p,o;
		while(file>>s>>p>>o){
			int vt_id=(s/world.size())%num_vertex_table;
			pthread_spin_lock(&vertex_table_lock[vt_id]);
			vertex_table[vt_id][s].out_edges.push_back(edge_row(p,o));
			pthread_spin_unlock(&vertex_table_lock[vt_id]);
		}	
	}
	void load_convert_o_data(int i,string filename){
		//cout<<"loading "<<filename<<endl;
		ifstream file(filename.c_str());
		uint64_t s,p,o;
		while(file>>s>>p>>o){
			int vt_id=(o/world.size())%num_vertex_table;
			pthread_spin_lock(&vertex_table_lock[vt_id]);
			vertex_table[vt_id][o].in_edges.push_back(edge_row(p,s));
			pthread_spin_unlock(&vertex_table_lock[vt_id]);
		}
	}
		
	void print_graph_info(){
		//cout<<world.rank()<<" has "<<vertex_table.size()<<" vertex"<<endl;
		//cout<<world.rank()<<" has "<<in_edges<<" in_edges"<<endl;
		cout<<world.rank()<<" finished "<<endl;
	}
	graph(boost::mpi::communicator& para_world,RdmaResource* _rdma,const char* dir_name)
			:world(para_world),rdma(_rdma){
		in_edges=0;
		out_edges=0;
		struct dirent *ptr;    
	    DIR *dir;
	    dir=opendir(dir_name);
	    vector<string> filenames;
	    vector<string> convert_s_files;
	    vector<string> convert_o_files;
	    for(int i=0;i<num_vertex_table;i++){
			pthread_spin_init(&vertex_table_lock[i],0);
		}
	    while((ptr=readdir(dir))!=NULL){
	        if(ptr->d_name[0] == '.')
	            continue;
	        string fname(ptr->d_name);
	        string index_prefix="index";
	        string data_prefix="id";
			string convert_s_prefix="s_";
			string convert_o_prefix="o_";
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
			} else if(equal(convert_s_prefix.begin(), convert_s_prefix.end(), fname.begin())){
				//filename example : s_1
				if(atoi(fname.c_str()+2)%world.size()==world.rank()){
					convert_s_files.push_back(string(dir_name)+"/"+fname);
				}
			} else if(equal(convert_o_prefix.begin(), convert_o_prefix.end(), fname.begin())){
				//filename example : o_1
				if(atoi(fname.c_str()+2)%world.size()==world.rank()){
					convert_o_files.push_back(string(dir_name)+"/"+fname);
				}
			} else{
				//cout<<"What's this file:"<<fname<<endl;
				//assert(false);
			}
	    }
	    uint64_t max_v_num=1000000*80;//80;
	    kstore.init(rdma,max_v_num,world.size(),world.rank());
	    
	    uint64_t t1=timer::get_usec();
		if(!global_load_convert_format){
			#pragma omp parallel for num_threads(10)
			for(int i=0;i<filenames.size();i++){
		    	load_data(filenames[i]);
		    }
			uint64_t t2=timer::get_usec();
		    //sleep(1000);
		    print_graph_info();
		    
		    for(int i=0;i<num_vertex_table;i++){
			    unordered_map<uint64_t,vertex_row>::iterator iter;
			    for(iter=vertex_table[i].begin();iter!=vertex_table[i].end();iter++){
					kstore.insert(iter->first,iter->second);
				}
				vertex_table[i].clear();
			}
			
			uint64_t t3=timer::get_usec();
			cout<<"loading files in "<<(t2-t1)/1000.0/1000.0<<"s ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
		    cout<<"init rdma store in "<<(t3-t2)/1000.0/1000.0<<"s ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
		} else {
			if(convert_s_files.size()==0){
				cout<<"convert files not found!!!!!!!!!"<<endl;
				exit(0);
			}
			sort(convert_s_files.begin(),convert_s_files.end());
			sort(convert_o_files.begin(),convert_o_files.end());
			
			#pragma omp parallel for num_threads(5)
			for(int i=0;i<convert_s_files.size();i++){
		    	load_convert_data(i,convert_s_files[i],convert_o_files[i]);
		    }

			// #pragma omp parallel for num_threads(10)
			// for(int i=0;i<convert_s_files.size();i++){
			// 	load_convert_s_data(i,convert_s_files[i]);
			// }
			// #pragma omp parallel for num_threads(10)
			// for(int i=0;i<convert_o_files.size();i++){
			// 	load_convert_o_data(i,convert_o_files[i]);
			// }

		    uint64_t t2=timer::get_usec();
			cout<<"machine "<<world.rank()<<" load and init in "<<(t2-t1)/1000.0/1000.0<<"s ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
		}
	    
	    
		cout<<"graph-store use "<<max_v_num*sizeof(vertex) / 1024 / 1024<<" MB for vertex data"<<endl;
		cout<<"graph-store use "<<kstore.new_edge_ptr * sizeof(edge_row) / 1024 / 1024<<" MB for edge data"<<endl;
	
	}
};


