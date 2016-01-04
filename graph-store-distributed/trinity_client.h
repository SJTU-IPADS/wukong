#pragma once
#include "thread_cfg.h"
#include "index_server.h"
#include "ingress.h"
#include "message_wrap.h"
#include <vector>
#include <map>
#include <string>
using namespace std;

class trinity_client{
	thread_cfg* cfg;
public:
	vector<string> cmd_string;
	vector<int> cmd_int;
	index_server* is;
	int num_val;
	trinity_client(index_server* _is,thread_cfg* _cfg):is(_is),cfg(_cfg){
	}
	void parse_cmd(){
		num_val=0;
		cmd_int.clear();
		assert(cmd_string.size()%4==0);
		map<string,int> variable_map;
		for(int i=0;i<cmd_string.size();i++){
			assert(cmd_string[i]!="");
			cout<<cmd_string[i]<<endl;
			int id=0;
			if(i%4==0){
				if(cmd_string[i]=="forward"){
					id=direction_forward;
				} else {
					id=direction_reverse;
				}
			}
			if(i%4==1 || i%4==3){
				if(cmd_string[i][0]=='?'){
					//this is a variable , transfer it to a negative integer
					if(variable_map.find(cmd_string[i])==variable_map.end()){
						id=-1-variable_map.size();
						variable_map[cmd_string[i]]=id;
						num_val++;
					}
					id=variable_map[cmd_string[i]];
				} else {
					id=is->subject_to_id[cmd_string[i]];
				}
			} 
			if(i%4==2){
				id=is->predict_to_id[cmd_string[i]];
			}
			cmd_int.push_back(id);
		}
		//transfer to following format
		//variable dir predict variable
		int idx=0;
		while(idx<cmd_int.size()){
			int start;
			int end;
			int dir;
			int predict=cmd_int[idx+2];
			if(cmd_int[idx]==direction_forward){
				start=cmd_int[idx+1];
				dir=para_out;
				end=cmd_int[idx+3];
			} else {
				end=cmd_int[idx+1];
				dir=para_in;
				start=cmd_int[idx+3];
			}
			cmd_int[idx]=start;
			cmd_int[idx+1]=dir;
			cmd_int[idx+2]=predict;
			cmd_int[idx+3]=end;
			idx+=4;
		}
		cmd_string.clear();
	}
	void Send(){
		parse_cmd();
		SendVector(cfg,0, cfg->client_num, cmd_int);
		//SendReq(cfg,first_target, cfg->client_num+rand()%cfg->server_num, req);
	}
	void join(vector<int>& old_table,vector<int>& match_table,int start_val,int end_val){
		//new_table = old_table join match_table
		vector<int> new_table;
		for(int i=0;i<old_table.size()/num_val;i++){
			//row in old_table is old_table[i*num_val],...[i*num_val+num_val-1]
			for(int j=0;j<match_table.size()/2;j++){
				//row in match_table is match_table[j*2],match_table[j*2+1]
				assert(start_val<0 || end_val<0);
				//otherwise what we want to match?
				bool matched=true;
				if(start_val<0){
					int pos=-1-start_val;
					if(old_table[i*num_val+pos]>=0 && old_table[i*num_val+pos]!=match_table[j*2]){
						matched=false;
					}
				}
				if(end_val<0){
					int pos=-1-end_val;
					if(old_table[i*num_val+pos]>=0 && old_table[i*num_val+pos]!=match_table[j*2+1]){
						matched=false;
					}
				}
				if(matched){
					vector<int> new_row;
					for(int k=0;k<num_val;k++){
						new_row.push_back(old_table[i*num_val+k]);
					}
					if(start_val<0){
						int pos=-1-start_val;
						new_row[pos]=match_table[j*2];
					}
					if(end_val<0){
						int pos=-1-end_val;
						new_row[pos]=match_table[j*2+1];
					}
					for(int k=0;k<num_val;k++){
						new_table.push_back(new_row[k]);
					}
				}
			}
		}
		old_table.swap(new_table);
	}
	void Recv(){
		vector<vector<int> > result=RecvTables(cfg);
		//vector<int> result=RecvVector(cfg);
		for(int i=0;i<result.size();i++){
			cout<<result[i].size()/2<<endl;
		}
		
		vector<int> init_table;
		init_table.resize(num_val,-1);
		// -1 means the val are not matched yet.
		// so it can match with anything
		for(int i=0;i<result.size();i++){
			join(init_table,result[i],cmd_int[i*4],cmd_int[i*4+3]);
		}
		cout<<"Final result size = "<<init_table.size()/num_val<<endl;
	}
	//SendStr(thread_cfg* cfg,int r_mid,int r_tid,std::string& str);
};