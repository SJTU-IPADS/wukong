#include "client_mode.h"

void translate_req_template(client* clnt,request_template& req_template){
	req_template.place_holder_vecptr.resize(req_template.place_holder_str.size());
	for(int i=0;i<req_template.place_holder_str.size();i++){
		string type=req_template.place_holder_str[i];
		if(clnt->parser.type_to_idvec.find(type)!=clnt->parser.type_to_idvec.end()){
			// do nothing
		} else {
			request_or_reply type_request;
			assert(clnt->parser.find_type_of(type,type_request));
			request_or_reply reply;
			clnt->Send(type_request);
			reply=clnt->Recv();
			vector<int>* ptr=new vector<int>();
			*ptr =reply.result_table;
			clnt->parser.type_to_idvec[type]=ptr;
			cout<<type<<" has "<<ptr->size()<<" objects"<<endl;
		}
		req_template.place_holder_vecptr[i]=clnt->parser.type_to_idvec[type];
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

__thread int local_barrier_val=1;
int global_barrier_val=0;
void ClientBarrier(struct thread_cfg *cfg){
	if(cfg->t_id==0){
		MPI_Barrier(MPI_COMM_WORLD);
		__sync_fetch_and_add(&global_barrier_val,1);
	}  else {
		__sync_fetch_and_add(&global_barrier_val,1);
	}
	while(global_barrier_val < local_barrier_val*cfg->client_num){
		usleep(1);
	}
	local_barrier_val+=1;
}

void single_execute(client* clnt,string filename,int execute_count){
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

void display_help(client* clnt){
	if(clnt->cfg->m_id==0 && clnt->cfg->t_id==0){
		cout<<"> reconfig: reload config file"<<endl;
		cout<<"> switch_single: execute one query at a time (singlefile [ + count])"<<endl;
		cout<<"> switch_batch: execute concurrent queries (batchfile)"<<endl;
		cout<<"> switch_mix:  (batch + singlefile [ + count] )"<<endl;
		cout<<"> help: display help infomation"<<endl;
	}
}


void iterative_shell(client* clnt){
	//ClientBarrier(clnt->cfg);
	struct thread_cfg *cfg=clnt->cfg;
	string mode_str[3];
	mode_str[0]="single mode (singlefile [ + count]):";
	mode_str[1]="batch mode (batchfile):";
	mode_str[2]="mix mode (batchfile + singlefile [ + count]):";
	if(cfg->m_id==0 && cfg->t_id==0){
		cout<<"input help to get more infomation about the shell"<<endl;
		cout<<mode_str[global_client_mode]<<endl;
	}
	while(true){
		ClientBarrier(clnt->cfg);
		string input_str;
		//exchange input
		if(cfg->m_id==0 && cfg->t_id==0){
			//cout<<mode_str[global_client_mode]<<endl;
			cout<<"> ";
			std::getline(std::cin,input_str);
			for(int i=0;i<cfg->m_num;i++){
				for(int j=0;j<cfg->client_num;j++){
					if(i==0 && j==0){
						continue;
					}
					cfg->node->Send(i,j,input_str);
				}
			}
		} else {
			input_str=cfg->node->Recv();
		}
		//end of exchange input

		if(input_str=="help"){
			display_help(clnt);
		} else if(input_str=="reconfig"){
			if(cfg->t_id==0){
				load_changeable_cfg();
			}
		} else if(input_str=="quit"){
			if(cfg->t_id==0){
				exit(0);
			}
		} else if(input_str=="switch_single"){
			if(cfg->t_id==0){
				global_client_mode=0;
				if(cfg->m_id==0){
					cout<<mode_str[global_client_mode]<<endl;
				}
			}
		} else if(input_str=="switch_batch"){
			if(cfg->t_id==0){
				global_client_mode=1;
				if(cfg->m_id==0){
					cout<<mode_str[global_client_mode]<<endl;
				}
			}
		} else if(input_str=="switch_mix"){
			if(cfg->t_id==0){
				global_client_mode=2;
				if(cfg->m_id==0){
					cout<<mode_str[global_client_mode]<<endl;
				}
			}
		} else {
			//handle queries here
			if(global_client_mode==0){
				if(cfg->m_id==0 && cfg->t_id==0){
					istringstream iss(input_str);
					string filename;
					int count=1;
					iss>>filename;
					iss>>count;
					if(count<1){
						count=1;
					}
					single_execute(clnt,filename,count);
				}
			} else if(global_client_mode==1){
				istringstream iss(input_str);
	            string filename;
	            iss>>filename;
				batch_logger logger;
				logger.init();
				batch_execute(clnt,filename,logger);
				logger.finish();
				ClientBarrier(clnt->cfg);
				//MPI_Barrier(MPI_COMM_WORLD);
				if(cfg->m_id==0 && cfg->t_id==0){
					for(int i=0;i<cfg->m_num*cfg->client_num -1 ;i++){
						batch_logger r=RecvObject<batch_logger>(clnt->cfg);
						logger.merge(r);
					}
					logger.print();
				} else {
					SendObject<batch_logger>(clnt->cfg,0,0,logger);
				}
			} else if(global_client_mode==2){
				istringstream iss(input_str);
				string batchfile;
				string singlefile;
				int count=1;
				iss>>batchfile>>singlefile>>count;
				if(count<1){
					count=1;
				}
				batch_logger logger;
				if(cfg->m_id==0 && cfg->t_id==0){
					single_execute(clnt,singlefile,count);
				} else {
					logger.init();
					batch_execute(clnt,batchfile,logger);
					logger.finish();
				}
				ClientBarrier(clnt->cfg);
				//MPI_Barrier(MPI_COMM_WORLD);
				if(cfg->m_id==0 && cfg->t_id==0){
					for(int i=0;i<cfg->m_num*cfg->client_num -1 ;i++){
						batch_logger r=RecvObject<batch_logger>(clnt->cfg);
						logger.merge(r);
					}
					logger.print();
				} else {
					SendObject<batch_logger>(clnt->cfg,0,0,logger);
				}
			}
		}


	}
}

void batch_execute(client* clnt,string mix_config,batch_logger& logger){
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
		distribution.push_back(current_dist);
		bool success=clnt->parser.parse_template(filename,vec_template[i]);
		translate_req_template(clnt,vec_template[i]);
		vec_req[i].cmd_chains=vec_template[i].cmd_chains;
		if(!success){
			cout<<"sparql parse error"<<endl;
			return ;
		}
		vec_req[i].silent=global_silent;
	}
	uint64_t start_time=timer::get_usec();
	for(int i=0;i<global_batch_factor;i++){
		int idx=mymath::get_distribution(clnt->cfg->get_random(),distribution);
		instantiate_request(clnt,vec_template[idx],vec_req[idx]);
		clnt->GetId(vec_req[idx]);
		logger.start_record(vec_req[idx].parent_id,idx);
		clnt->Send(vec_req[idx]);
	}
	for(int i=0;i<total_request;i++){
		request_or_reply reply=clnt->Recv();
		logger.end_record(reply.parent_id);
		int idx=mymath::get_distribution(clnt->cfg->get_random(),distribution);
		instantiate_request(clnt,vec_template[idx],vec_req[idx]);
		clnt->GetId(vec_req[idx]);
		logger.start_record(vec_req[idx].parent_id,idx);
		clnt->Send(vec_req[idx]);
	}
	for(int i=0;i<global_batch_factor;i++){
		request_or_reply reply=clnt->Recv();
		logger.end_record(reply.parent_id);
	}
	uint64_t end_time=timer::get_usec();
	cout<< 1000.0*(total_request+global_batch_factor)/(end_time-start_time)<<" Kops"<<endl;
};
