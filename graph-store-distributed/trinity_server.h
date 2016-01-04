#pragma once
#include "graph.h"
#include "global_cfg.h"
#include "message_wrap.h"
#include <set>
#include <vector>


struct bingding{
	set<int> val_set;
	bool exist;
	bingding(){
		exist=false;
	}
};
class trinity_server{
	graph& g;
	thread_cfg* cfg;

	vector<vector<int> > table_vec;
	vector<bingding> binding_vec;
public:

	trinity_server(graph& gg,thread_cfg* _cfg)
			:g(gg),cfg(_cfg){
	}

	void match_single_patten(int patten_num,int start,int dir,int predict_id,int end){
		cout<<"matching "<<start<<" "<<dir<<" "<<predict_id<<" "<<end<<endl;
		bingding* start_binding;
		if(start>=0){
			start_binding=new bingding();
			start_binding->val_set.insert(start);
			start_binding->exist=true;
		} else {
			start_binding=&binding_vec[-start];
		}
		bingding* end_binding;
		if(end>=0){
			end_binding=new bingding();
			end_binding->val_set.insert(end);
			end_binding->exist=true;
		} else {
			end_binding=&binding_vec[-end];
		}
		{
			assert(start_binding->exist);
			for(auto i : start_binding->val_set){
				int edge_num=0;
				edge_row* edge_ptr;
				edge_ptr=g.kstore.readLocal_predict(cfg->t_id, i,dir,predict_id,&edge_num); 
				for(int k=0;k<edge_num;k++){
					if(predict_id==edge_ptr[k].predict ){
						if(end_binding->exist){
							if(end_binding->val_set.find(edge_ptr[k].vid)!=end_binding->val_set.end()){
									table_vec[patten_num].push_back(i);
									table_vec[patten_num].push_back(edge_ptr[k].vid);
							}
						} else {
							//end is a free variable
							table_vec[patten_num].push_back(i);
							table_vec[patten_num].push_back(edge_ptr[k].vid);
							end_binding->val_set.insert(edge_ptr[k].vid);
						}
					}
				}
			}
		}

		if(start>=0){
			delete start_binding;
		}
		end_binding->exist=true;
		if(end>=0){
			delete end_binding;
		}
	}
	void run(){	
		while(true){
			vector<int> cmd_int;
			cmd_int=RecvVector(cfg);
			table_vec.clear();
			binding_vec.clear();
			table_vec.resize(cmd_int.size()/4);
			binding_vec.resize(cmd_int.size()/4+1);
			int idx=0;
			while(idx<cmd_int.size()){
				match_single_patten(idx/4,cmd_int[idx],cmd_int[idx+1],cmd_int[idx+2],cmd_int[idx+3]);
				idx+=4;
			}
			// for(int i=0;i<table_vec.size();i++){
			// 	cout<<"size="<<table_vec[i].size()/2<<endl;
			// 	for(int j=0;j<table_vec[i].size();j++){
			// 		cout<<table_vec[i][j]<<" ";
			// 		if(j%2==1){
			// 			cout<<endl;
			// 		}
			// 	}
			// }
			//send message back
			//SendVector(cfg,0,0,cmd_int);
			SendTables(cfg,0,0,table_vec);
		}
	}
};