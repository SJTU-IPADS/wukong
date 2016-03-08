#include "client_mode.h"

boost::unordered_map<string,vector<int>* > type_to_idvec;
void translate_req_template(client* clnt,request_template& req_template){
	req_template.place_holder_vecptr.resize(req_template.place_holder_str.size());
	for(int i=0;i<req_template.place_holder_str.size();i++){
		string type=req_template.place_holder_str[i];
		if(type_to_idvec.find(type)!=type_to_idvec.end()){
			// do nothing
		} else {
			request_or_reply type_request;
			assert(clnt->parser.find_type_of(type,type_request));
			request_or_reply reply;
			clnt->Send(type_request);
			reply=clnt->Recv();
			vector<int>* ptr=new vector<int>();
			*ptr =reply.result_table;
			type_to_idvec[type]=ptr;
			cout<<type<<" has "<<ptr->size()<<" objects"<<endl;
		}
		req_template.place_holder_vecptr[i]=type_to_idvec[type];
	}
}

void instantiate_request(client* clnt,request_template& req_template,request_or_reply& r){
	for(int i=0;i<req_template.place_holder_position.size();i++){
		int pos=req_template.place_holder_position[i];
		vector<int>* vecptr=req_template.place_holder_vecptr[i];
		if(vecptr==NULL || vecptr->size()==0){
			assert(false);
		}
		r.cmd_chains[pos]=(*vecptr)[clnt->cfg->get_random()% (vecptr->size())];
	}
	return ;
}

void interactive_execute(client* clnt,string filename,int execute_count){
	int sum=0;
    int result_count;
    request_or_reply request;
	bool success=clnt->parser.parse(filename,request);
	if(!success){
		cout<<"sparql parse error"<<endl;
		return ;
	}
	request.silent=global_silent;
	request_or_reply reply;
	for(int i=0;i<execute_count;i++){
		uint64_t t1=timer::get_usec();
        clnt->Send(request);
        reply=clnt->Recv();
		uint64_t t2=timer::get_usec();
		sum+=t2-t1;
	}
	cout<<"result size:"<<reply.silent_row_num<<endl;
	int row_to_print=min(reply.row_num(),global_max_print_row);
	if(row_to_print>0){
		clnt->print_result(reply,row_to_print);
	}
	cout<<"average latency "<<sum/execute_count<<" us"<<endl;
};

void interactive_mode(client* clnt){
    while(true){
        MPI_Barrier(MPI_COMM_WORLD);
        string iterative_filename;
        if(clnt->cfg->m_id==0 && clnt->cfg->t_id==0){
            cout<<"iterative mode (iterative file + [count]):"<<endl;
            string input_str;
            std::getline(std::cin,input_str);
            istringstream iss(input_str);
            string iterative_filename;
            int iterative_count;
            iss>>iterative_filename;
    		iss>>iterative_count;
            if(iterative_count<1){
    			iterative_count=1;
    		}
            //interactive_execute(clnt,iterative_filename,iterative_count);
			batch_execute(clnt,iterative_filename);
        }
    }
};

void noblocking_execute(client* clnt,string mix_config){
	ifstream configfile(mix_config);
	if(!configfile){
		cout<<"File "<<mix_config<<" not exist"<<endl;
		return ;
	}
	int total_query_type;
	int total_request;
	int sleep_round=1;
	configfile>>total_query_type>>total_request>>sleep_round;
	vector<int > distribution;
	vector<request_template > vec_template;
	vector<request_or_reply > vec_req;
	vec_template.resize(total_query_type);
	vec_req.resize(total_query_type);
	for(int i=0;i<total_query_type;i++){
		string filename;
		configfile>>filename;
		int current_dist;
		configfile>>current_dist;
		bool success=clnt->parser.parse_template(filename,vec_template[i]);
		translate_req_template(clnt,vec_template[i]);
		vec_req[i].cmd_chains=vec_template[i].cmd_chains;
		if(!success){
			cout<<"sparql parse error"<<endl;
			return ;
		}
		vec_req[i].silent=global_silent;
	}
	uint64_t t1=timer::get_usec();
	assert(false);
	// for(int i=0;i<total_request;i++){
	// 	int idx=clnt->cfg->get_random() %total_query_type;
	// 	instantiate_request(clnt,vec_template[idx],vec_req[idx]);
	// 	clnt->Send(vec_req[idx]);
    //     request_or_reply reply=clnt->Recv();
	// }
	uint64_t t2=timer::get_usec();
	cout<<"average latency "<<(t2-t1)/total_request<<" us"<<endl;
}
void batch_execute(client* clnt,string mix_config){
	ifstream configfile(mix_config);
	if(!configfile){
		cout<<"File "<<mix_config<<" not exist"<<endl;
		return ;
	}
	int total_query_type;
	int total_request;
	int sleep_round=1;
	configfile>>total_query_type>>total_request>>sleep_round;
	vector<int > distribution;
	vector<request_template > vec_template;
	vector<request_or_reply > vec_req;
	vec_template.resize(total_query_type);
	vec_req.resize(total_query_type);
	for(int i=0;i<total_query_type;i++){
		string filename;
		configfile>>filename;
		int current_dist;
		configfile>>current_dist;
		bool success=clnt->parser.parse_template(filename,vec_template[i]);
		translate_req_template(clnt,vec_template[i]);
		vec_req[i].cmd_chains=vec_template[i].cmd_chains;
		if(!success){
			cout<<"sparql parse error"<<endl;
			return ;
		}
		vec_req[i].silent=global_silent;
	}
	uint64_t t1=timer::get_usec();
	for(int i=0;i<global_batch_factor;i++){
		int idx=clnt->cfg->get_random() %total_query_type;
		instantiate_request(clnt,vec_template[idx],vec_req[idx]);
		clnt->Send(vec_req[idx]);
	}
	for(int i=0;i<total_request;i++){
		request_or_reply reply=clnt->Recv();
		int idx=clnt->cfg->get_random() %total_query_type;
		instantiate_request(clnt,vec_template[idx],vec_req[idx]);
		clnt->Send(vec_req[idx]);
	}
	for(int i=0;i<global_batch_factor;i++){
		request_or_reply reply=clnt->Recv();
	}
	uint64_t t2=timer::get_usec();
	cout<<"Throughput "<<total_request/((t2-t1)/1000.0)<<" Kops"<<endl;

};
