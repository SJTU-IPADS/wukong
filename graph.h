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
using namespace std;
struct edge_row{
	edge_row(int p,int v){
		predict=p;
		vid=v;
	}
	int predict;
	int vid;
};
struct vertex_row{
	vertex_row(int p,string v){
		property=p;
		value=v;
	}
	int property;
	string value;
};
struct vertex{
	vertex():type(-1){};
	int type;
	vector<vertex_row> propertys;
	vector<edge_row> in_edges;
	vector<edge_row> out_edges;
};

class ontology{
public:
	unordered_map<int,vector<int> > id_to_children;
	unordered_map<int,int> id_to_parent;
	void insert(int child,int parent){
		id_to_parent[child]=parent;
		if(id_to_parent.find(parent)==id_to_parent.end())
			id_to_parent[parent]=-1;

		if(id_to_children.find(child)==id_to_children.end())
			id_to_children[child]=vector<int>();
		id_to_children[parent].push_back(child);
		
	}
	void insert_type(int type){
		if(id_to_parent.find(type)==id_to_parent.end())
			id_to_parent[type]=-1;
		if(id_to_children.find(type)==id_to_children.end())
			id_to_children[type]=vector<int>();
	}
	bool is_subtype_of(int child,int parent){
		while(true){
			if(child==parent)
				return true;
			child=id_to_parent[child];
			if(child==-1)
				return false;
		}
	}

	unordered_set<int>  get_all_subtype(int id){
		unordered_set<int>  result;
		result.insert(id);
		for(auto child : id_to_children[id]){
			unordered_set<int> subtype_of_child=get_all_subtype(child);
			for(auto item : subtype_of_child){
				result.insert(item);
			}
		}
		return result;
	}	
};
class graph{
public:

	unordered_map<string,int> subject_to_id;
	unordered_map<string,int> predict_to_id;
	vector<string> id_to_subject;
	vector<string> id_to_predict;
	// main array
	vector<vertex> vertex_array;
	ontology ontology_array;
	void print_tree(int id,int level){
		for(int i=0;i<level;i++)
			cout<<"\t";
		cout<<id_to_subject[id]<<endl;
		for(auto child : ontology_array.id_to_children[id]){
			print_tree(child,level+1);
		}
	}
	void print_ontology_tree(){
		cout<<"#############  print tree #############"<<endl;
		for (auto row : ontology_array.id_to_parent){
			if(row.second==-1)
				print_tree(row.first,0);
		}
	}
	void insert_subclass(string child,string parent){
		int id[2];
		if(subject_to_id.find(child)==subject_to_id.end()){
			int size =subject_to_id.size();
			subject_to_id[child]=size;
			id_to_subject.push_back(child);
			vertex_array.push_back(vertex());
		}
		if(subject_to_id.find(parent)==subject_to_id.end()){
			int size =subject_to_id.size();
			subject_to_id[parent]=size;
			id_to_subject.push_back(parent);
			vertex_array.push_back(vertex());
		}
		id[0]=subject_to_id[child];
		id[1]=subject_to_id[parent];
		ontology_array.insert(id[0],id[1]);
	}
	graph(char* input){
		struct dirent *ptr;    
	    DIR *dir;
	    dir=opendir(input);
	    printf("files:\n");
	    
	    timer t1;
	    while((ptr=readdir(dir))!=NULL)
	    {
	        if(ptr->d_name[0] == '.')
	            continue;
	        string filename=string(input)+"/"+string(ptr->d_name);
	        printf("loading %s ...\n",ptr->d_name);
	        ifstream file(filename.c_str());
	        while(!file.eof()){
				// S P O .
				string subject;
				string predict;
				string object;
				string useless_dot;
				int id[3];
				file>>subject>>predict>>object>>useless_dot;
				//replace prefix
				string prefix="<http://swat.cse.lehigh.edu/onto/univ-bench.owl";
				if(equal(prefix.begin(), prefix.end(), subject.begin()))
					subject="<ub"+subject.substr(prefix.size());
				if(equal(prefix.begin(), prefix.end(), predict.begin()))
					predict="<ub"+predict.substr(prefix.size());
				if(equal(prefix.begin(), prefix.end(), object.begin()))
					object="<ub"+object.substr(prefix.size());

				if(subject_to_id.find(subject)==subject_to_id.end()){
					int size =subject_to_id.size();
					subject_to_id[subject]=size;
					id_to_subject.push_back(subject);
					vertex_array.push_back(vertex());
				}
				if(predict_to_id.find(predict)==predict_to_id.end()){
					int size =predict_to_id.size();
					predict_to_id[predict]=size;
					id_to_predict.push_back(predict);
				}
				bool object_is_vertex=true;
				if(object[0]=='"'){
					object_is_vertex=false;
				}
				if(subject_to_id.find(object)==subject_to_id.end() ){
					int size =subject_to_id.size();
					subject_to_id[object]=size;
					id_to_subject.push_back(object);
					vertex_array.push_back(vertex());
				}
				id[0]=subject_to_id[subject];
				id[1]=predict_to_id[predict];
				id[2]=subject_to_id[object];
				if(object[0]=='"'){
					vertex_array[id[0]].propertys.push_back(vertex_row(id[1],object));
				}else {
					vertex_array[id[0]].out_edges.push_back(edge_row(id[1],id[2]));
					vertex_array[id[2]].in_edges.push_back(edge_row(id[1],id[0]));
				}
				if(predict=="<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"){
					vertex_array[id[0]].type=id[2];
					ontology_array.insert_type(id[2]);
				}
			}        
	    }
	    closedir(dir);
	    timer t2;

	    int count=0;
	    for (int i=0;i<vertex_array.size();i++){
	    	if(vertex_array[i].type==-1){
	    		count++;
	    		//cout<<id_to_subject[i]<<endl;
				//for (auto j : g.vertex_array[i].propertys){
				//	cout<<"\t"<<g.id_to_predict[j.property]<<"\t"<<j.value<<endl;
				//}
	    	}
	    }
	    printf("%d vertex in %d doesn't have type\n", count,vertex_array.size());
	    printf("Loading in %d ms\n", t2.diff(t1));
	    printf("sizeof vertex_array is %d\n", vertex_array.size());
	    printf("List of propertys or predict\n", t2.diff(t1));
	    for(int i=0;i<id_to_predict.size();i++){
	    	cout<<"\t"<<id_to_predict[i]<<endl;
	    }
	}
};
