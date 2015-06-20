#pragma once

#include "graph.h"
struct path_node{
	path_node(int _id,int _prev):id(_id),prev(_prev){
	}
	int id;
	int prev;
};
struct path_node_less_than
{
    inline bool operator() (const path_node& struct1, const path_node& struct2)
    {
        return (struct1.id < struct2.id);
    }
};
struct request{
	vector<int> cmd_chains;
	vector<vector<path_node> >result_paths;
	void clear(){
		cmd_chains.clear();
		result_paths.clear();
	}
};



class traverser_keeppath{

	graph& g;
	request req;
	enum command{
		cmd_subclass_of,
		cmd_neighbors
	};
	enum parameter{
		para_in,
		para_out,
		para_all
	};

	void do_execute(request* r){
		if(r->cmd_chains.size()==0)
			return;
		vector<path_node> vec;
		if(r->cmd_chains.back() == cmd_subclass_of){
			// subclass_of is a filter operation
			// it just remove some of the output
			do_subclass_of(r);
			do_execute(r);
			return ;
		} else if(r->cmd_chains.back() == cmd_neighbors){
			vec=do_neighbors(r);
		}

		if(r->cmd_chains.size()==0){
			// end here
			r->result_paths.push_back(vec);
			return ;
		} else {
			//recursive execute 
			vector<request> sub_reqs=split_request(vec,r);
			for(int i=0;i<sub_reqs.size();i++){
				do_execute(&sub_reqs[i]);
			}
			merge_reqs(sub_reqs,r);
		}
		return ;
	}
	void merge_reqs(vector<request>& sub_reqs,request* r){
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
			r->result_paths.push_back(sub_reqs[0].result_paths[i]);
		}
	}
	vector<request> split_request(vector<path_node>& vec,request* r){
		vector<request> sub_reqs;
		int num_sub_request=2;
		sub_reqs.resize(num_sub_request);
		for(int i=0;i<sub_reqs.size();i++){
			sub_reqs[i].cmd_chains=r->cmd_chains;
			sub_reqs[i].result_paths.push_back(vector<path_node>());
		}
		for(int i=0;i<vec.size();i++){
			//int machine = i % num_sub_request;
			int machine = vec[i].id % num_sub_request;
			sub_reqs[machine].result_paths[0].push_back(vec[i]);
		}
		return sub_reqs;
	}

	vector<path_node> do_neighbors(request* r){
		vector<path_node> vec;
		r->cmd_chains.pop_back();
		int dir=r->cmd_chains.back();
		r->cmd_chains.pop_back();
		int	predict_id=r->cmd_chains.back();
		r->cmd_chains.pop_back();
		int path_len=r->result_paths.size();
		for (int i=0;i< r->result_paths[path_len-1].size();i++){
			int prev_id=r->result_paths[path_len-1][i].id;
			if(dir ==para_in || dir == para_all){
				for (auto row : g.vertex_array[prev_id].in_edges){
					if(predict_id==row.predict){
						vec.push_back(path_node(row.vid,i));
					}
				}
			}
			if(dir ==para_out || dir == para_all){
				for (auto row : g.vertex_array[prev_id].out_edges){
					if(predict_id==row.predict){
						vec.push_back(path_node(row.vid,i));
					}
				}
			}
		}
		return vec;
	}
	void do_subclass_of(request* r){
		int predict_id=g.predict_to_id["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"];

		r->cmd_chains.pop_back();
		int target_id=r->cmd_chains.back();
		r->cmd_chains.pop_back();

		int path_len=r->result_paths.size();
		vector<path_node>& prev_vec = r->result_paths[path_len-1];
		vector<path_node> new_vec;
		for (int i=0;i<prev_vec.size();i++){
			int prev_id=prev_vec[i].id;	
			for (auto row : g.vertex_array[prev_id].out_edges){
				if(predict_id==row.predict && 
							g.ontology_array.is_subtype_of(row.vid,target_id)){
					new_vec.push_back(prev_vec[i]);
				}
			}
		}
		r->result_paths[path_len-1]=new_vec;
	}

public:
	traverser_keeppath(graph& gg):g(gg){
	} 
	traverser_keeppath( const traverser_keeppath& other ):g(other.g),req(other.req){
	}
	traverser_keeppath& lookup(string subject){
		req.clear();
		path_node node(g.subject_to_id[subject],-1);
		vector<path_node> vec;
		vec.push_back(node);
		req.result_paths.push_back(vec);
		return *this;
	}
	traverser_keeppath& get_subtype(string target){
		req.clear();
		int target_id=g.subject_to_id[target];
		unordered_set<int> ids = g.ontology_array.get_all_subtype(target_id);
		vector<path_node> vec;
		for(auto id: ids){
			vec.push_back(path_node(id,-1));
		}
		req.result_paths.push_back(vec);
		return *this;
	}
	traverser_keeppath& neighbors(string dir,string predict){
		req.cmd_chains.push_back(cmd_neighbors);
		if(dir =="in" ){
			req.cmd_chains.push_back(para_in);
		} else if (dir =="out" ){
			req.cmd_chains.push_back(para_out);
		} else {
			req.cmd_chains.push_back(para_all);
		}
		req.cmd_chains.push_back(g.predict_to_id[predict]);
		return *this;
	}
	traverser_keeppath& subclass_of(string target){
		req.cmd_chains.push_back(cmd_subclass_of);
		req.cmd_chains.push_back(g.subject_to_id[target]);
		return *this;
	}
	traverser_keeppath& sort(){
		int path_len=req.result_paths.size();
		vector<path_node>& vec_to_sort=req.result_paths[path_len-1];
		std::sort(vec_to_sort.begin(), vec_to_sort.end(), path_node_less_than());
		return *this;
	}
	path_node get_node(int row,int column){
		int current_column=req.result_paths.size()-1;
		while(column<current_column){
			row=req.result_paths[current_column][row].prev;
			current_column--;
		}
		return req.result_paths[column][row];
	}
	int get_path_length(){
		return req.result_paths.size();
	}
	int get_path_num(){
		int path_len=req.result_paths.size();
		return req.result_paths[path_len-1].size();
	}

	traverser_keeppath& merge(traverser_keeppath& other, int split_length){
		sort();
		other.sort();
		int my_row=0;
		int other_start=0;
		vector<vector<path_node> >new_path;
		for(int i=split_length;i<other.get_path_length();i++){
			new_path.push_back(vector<path_node>());
		}
		while(my_row<get_path_num()){
			path_node my_path_begin=get_node(my_row,split_length-1);
			path_node my_path_end  =get_node(my_row,get_path_length()-1);
			int other_row=other_start;
			while(other_row<other.get_path_num()){
				path_node other_path_begin=other.get_node(other_row,split_length-1);
				path_node other_path_end  =other.get_node(other_row,other.get_path_length()-1);
				if(other_path_end.id<my_path_end.id){
					//skip this other.row forever 
					other_row++;
					other_start++;
					continue;
				} else if(other_path_end.id == my_path_end.id){
					if(other_path_begin.id  == my_path_begin.id){
						// this row match
						path_node node=other.get_node(other_row,split_length);
						node.prev=my_row;
						new_path[0].push_back(node);
						for(int column=split_length+1;column<other.get_path_length();column++){
							path_node node=other.get_node(other_row,column);
							node.prev=new_path[0].size()-1;
							new_path[column-split_length].push_back(node);
						}
						other_row++;
					} else {
						// begin of path doesn't match
						other_row++;
					}
				} else {
					// we already check all possible other.rows for my_row
					break;
				}
			}
			my_row++;
		}
		for(int i=0;i<new_path.size();i++){
			req.result_paths.push_back(new_path[i]);
		}
		return *this;
	}
	traverser_keeppath& print_count(){
		int path_len=req.result_paths.size();
		cout<<req.result_paths[path_len-1].size()<<endl;
		return *this;
	}
	traverser_keeppath& print_property(){
		int path_len=req.result_paths.size();
		for(int i=0;i<req.result_paths[path_len-1].size();i++){
			for(int column=0;column<path_len;column++){
				if(i>=req.result_paths[column].size()){
					cout<<"\t";
				} else {
					cout<<req.result_paths[column][i].id<<","
						<<req.result_paths[column][i].prev<<"\t";
				}
			}
			cout<<endl;
		}
		return *this;
	}

	traverser_keeppath& execute(){
		// reverse cmd_chains
		// so we can easily pop the cmd and do recursive operation
		reverse(req.cmd_chains.begin(),req.cmd_chains.end()); 
		do_execute(&req);
		req.cmd_chains.clear();
		return *this;
	}


};