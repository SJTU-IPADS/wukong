#pragma once

#include "request.h"
#include <vector>
#include <unordered_map>
#include <pthread.h>

struct item{
	int count;
	request req;
	vector<request> sub_reqs;
};
class request_queue{
	unordered_map<int,item> req_queue;
	void merge_reqs(vector<request>& sub_reqs,request& r){
		if(sub_reqs.size()>0)
			r.result_table.resize(sub_reqs[0].column_num());
		for(int i=0;i<sub_reqs.size();i++){
			for(int j=0;j<sub_reqs[i].row_num();j++){
				sub_reqs[i].append_row_to(r.result_table,j);
			}
		}
	}
public:
	void put_req(request& req,int count){				
		item data;
		data.count=count;
		data.req=req;
		req_queue[req.req_id]=data;		
	}
	bool put_reply(request& req){
		
		//if we get all replies , we return true
		int id=req.parent_id;
		req_queue[id].count--;
		req_queue[id].sub_reqs.push_back(req);

		if(req_queue[id].count==0){
			req=req_queue[id].req;
			merge_reqs(req_queue[id].sub_reqs,req);		
			req_queue.erase(id);
			return true;
		}
		return false;
	}	
};
