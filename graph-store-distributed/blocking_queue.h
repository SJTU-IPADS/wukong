#pragma once

#include "request.h"
#include <vector>
#include <unordered_map>
#include <pthread.h>
#include "global_cfg.h"
#include <boost/unordered_map.hpp>

struct item{
	int count;
	request req;
	vector<request> sub_reqs;
};
class blocking_queue{
	unordered_map<int,item> req_queue;
	pthread_spinlock_t internal_lock;
	void merge_reqs(vector<request>& sub_reqs,request& r){
		if(r.cmd_chains.back()==cmd_join){

			//cout<<"join "<<r.row_num()<<" start"<<endl;
			boost::unordered_map<int,pair<int,int> > id_2_row_start;
			for(int i=0;i<sub_reqs.size();i++){
				for(int j=0;j<sub_reqs[i].row_num();j++){
					int id = sub_reqs[i].result_table[0][j];
					if(j==0 || id!=sub_reqs[i].result_table[0][j-1]){
						id_2_row_start[id]=pair<int,int>(i,j);
					}
				}
			}
			//cout<<"create unordered_map "<<r.row_num()<<endl;
			vector<vector<int> >updated_result_table;
			int new_size=0;
			updated_result_table.resize(r.column_num()+sub_reqs[0].column_num()-1);
			for(int i=0;i<r.row_num();i++){
				if(id_2_row_start.find(r.last_column(i)) ==id_2_row_start.end()){
					continue;
				}
				int table_id=id_2_row_start[r.last_column(i)].first;
				int start_row=id_2_row_start[r.last_column(i)].second;
				int current_row=start_row;
				while(current_row<sub_reqs[table_id].row_num()  && sub_reqs[table_id].result_table[0][start_row] 
													== sub_reqs[table_id].result_table[0][current_row]){
					// r.append_row_to(updated_result_table,i);
					// for(int col=1;col<sub_reqs[table_id].column_num();col++){
					// 	updated_result_table[r.column_num()+col-1].
					// 			push_back(sub_reqs[table_id].result_table[col][current_row]);
					// }
					current_row++;
					new_size++;
				}
			}
			//cout<<"create new updated_result_table "<<r.row_num()<<endl;
			//cout<<"r from"<<r.row_num()<<" to"<<updated_result_table[0].size()<<endl;
			cout<<"r from"<<r.row_num()<<" to"<<new_size<<endl;
			
			return ;
		}
		if(sub_reqs.size()>0)
			r.result_table.resize(sub_reqs[0].column_num());
		r.final_row_number=0;
		for(int i=0;i<sub_reqs.size();i++){
			for(int j=0;j<sub_reqs[i].row_num();j++){
				sub_reqs[i].append_row_to(r.result_table,j);	
			}
			r.final_row_number+=sub_reqs[i].final_row_number;
		}
	}
public:
	blocking_queue(){
		pthread_spin_init(&internal_lock,0);
	}
	void put_req(request& req,int count){	
		if(global_enable_workstealing){
			pthread_spin_lock(&internal_lock);
		}			
		item data;
		data.count=count;
		data.req=req;
		req_queue[req.req_id]=data;		
		if(global_enable_workstealing){
			pthread_spin_unlock(&internal_lock);
		}			
	}
	bool put_reply(request& req){
		if(global_enable_workstealing){
			pthread_spin_lock(&internal_lock);
		}	
		bool ret=false;;				
		//if we get all replies , we return true
		int id=req.parent_id;
		req_queue[id].count--;
		req_queue[id].sub_reqs.push_back(req);

		if(req_queue[id].count==0){
			req=req_queue[id].req;
			merge_reqs(req_queue[id].sub_reqs,req);		
			req_queue.erase(id);
			ret=true;
		}
		if(global_enable_workstealing){
			pthread_spin_unlock(&internal_lock);
		}	
		return ret;
	}	
};
