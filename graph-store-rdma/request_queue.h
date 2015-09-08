#pragma once

#include "request.h"
#include <vector>
#include <unordered_map>
struct item{
	int count;
	request req;
	vector<request> sub_reqs;
};
class request_queue{
	unordered_map<int,item> req_queue;
	void merge_reqs(vector<request>& sub_reqs,request& r){
		if(sub_reqs.size()>1){
			//iterate on all sub_reqs
			for(int i=1;i<sub_reqs.size();i++){
				//reversely iterate on all column 
				for(int column=sub_reqs[i].result_paths.size()-1;column>=0;column--){
					for(auto node:sub_reqs[i].result_paths[column]){
						if(column>0){
							node.prev=node.prev + sub_reqs[0].result_paths[column-1].size();
						}
						sub_reqs[0].result_paths[column].push_back(node);
					}
				}
			}
		}
		for(int i=0;i<sub_reqs[0].result_paths.size();i++){
			r.result_paths.push_back(sub_reqs[0].result_paths[i]);
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
			return true;
		}
		return false;
	}
	
};