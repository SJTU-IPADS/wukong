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
	int row_to_print=min(reply.row_num(),(uint64_t)global_max_print_row);
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

bool simulate_execute_first_step(client* clnt,string cmd,request_or_reply& reply){
	request_or_reply request;
	bool success=clnt->parser.parse_string(cmd,request);
	if(!success){
		cout<<"sparql parse_string error"<<endl;
		return false;
	}
	request.silent=false;
    clnt->Send(request);
    reply=clnt->Recv();
	cout<<"result size:"<<reply.silent_row_num<<endl;
	return true;
}
bool simulate_execute_other_step(client* clnt,string cmd,request_or_reply& reply,set<int>& s){
	vector<vector<request_or_reply> > request_vec;
	int num_thread=global_num_server;

	request_vec.resize(clnt->cfg->m_num);
	for(int i=0;i<request_vec.size();i++){
		request_vec[i].resize(num_thread);
		for(int j=0;j<num_thread;j++){
			bool success=clnt->parser.parse_string(cmd,request_vec[i][j]);
			request_vec[i][j].set_column_num(1);
			request_vec[i][j].silent=false;
			if(!success){
				cout<<"sparql parse_string error"<<endl;
				return false;
			}
		}
	}

	for(set<int>::iterator iter=s.begin();iter!=s.end();iter++){
		int m_id= mymath::hash_mod(*iter,clnt->cfg->m_num);
		int t_id= mymath::hash_mod( (*iter)/clnt->cfg->m_num , num_thread);
		request_vec[m_id][t_id].result_table.push_back(*iter);
	}
	for(int i=0;i<request_vec.size();i++){
		for(int j=0;j<num_thread;j++){
			clnt->GetId(request_vec[i][j]);
			SendR(clnt->cfg, i , j+clnt->cfg->client_num,request_vec[i][j]);
		}
	}
	reply= RecvR(clnt->cfg);
	for(int i=0;i<clnt->cfg->m_num * num_thread -1;i++){
		request_or_reply r = RecvR(clnt->cfg);
		reply.silent_row_num +=r.silent_row_num;
		int new_size=r.result_table.size()+reply.result_table.size();
		reply.result_table.reserve(new_size);
		reply.result_table.insert( reply.result_table.end(), r.result_table.begin(), r.result_table.end());
	}
	cout<<"result size:"<<reply.silent_row_num<<endl;
	return true;
}
set<int> remove_dup(request_or_reply& reply,int col){
	set<int> s;
	for(int i=0;i<reply.row_num();i++){
        int id=reply.get_row_column(i,col);
        s.insert(id);
    }
	return s;
}

void simulate_trinity_q6(client* clnt){
	if(clnt->cfg->m_id!=0 || clnt->cfg->t_id!=0){
		return ;
	}
	string header="PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "
	"PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#> SELECT * WHERE { ";

	string part1="?Y  ub:subOrganizationOf   <http://www.University0.edu>    <-  "
				;
	string part2= "?Y  rdf:type   ub:Department . "
				  "?X  ub:worksFor ?Y    <- "
				;
	string part3="?X  rdf:type ub:FullProfessor . "
				;

	request_or_reply r1;
	request_or_reply r2;
	request_or_reply r3;

	if(!simulate_execute_first_step(clnt,header+part1+" }",r1)){
		return ;
	}
	set<int> s1=remove_dup(r1,0);
	cout<<"result size after remove_dup:"<<s1.size()<<endl;

	if(!simulate_execute_other_step(clnt,header+part2+" }",r2,s1)){
		return ;
	}
	set<int> s2=remove_dup(r2,1);
	cout<<"result size after remove_dup:"<<s2.size()<<endl;

	if(!simulate_execute_other_step(clnt,header+part3+" }",r3,s2)){
		return ;
	}
	set<int> s3=remove_dup(r3,0);
	cout<<"result size after remove_dup:"<<s3.size()<<endl;
//two-step join
	boost::unordered_map<int,vector<int> > hashtable1;
	for(int i=0;i<r2.row_num();i++){
		int v1=r2.get_row_column(i,0);
		int v2=r2.get_row_column(i,1);
		hashtable1[v1].push_back(v2);
	}
	vector<int> updated_result_table;
	for(int i=0;i<r1.row_num();i++){
		int vid=r1.get_row_column(i,0);
		if(hashtable1.find(vid)!=hashtable1.end()){
			for(int k=0;k<hashtable1[vid].size();k++){
				r1.append_row_to(i,updated_result_table);
	            updated_result_table.push_back(hashtable1[vid][k]);
			}
		}
	}
	r1.set_column_num(r1.column_num()+1);
    r1.result_table.swap(updated_result_table);
	updated_result_table.clear();

	boost::unordered_set<int > hashset2;
	for(int i=0;i<r3.row_num();i++){
		int v1=r3.get_row_column(i,0);
		hashset2.insert(v1);
	}
	for(int i=0;i<r1.row_num();i++){
		int v1=r1.get_row_column(i,0);
		int v2=r1.get_row_column(i,1);
		if(hashset2.find(v2)!=hashset2.end()){
			r1.append_row_to(i,updated_result_table);
		}
	}
	r1.result_table.swap(updated_result_table);
	cout<<"final join result size:"<<r1.row_num()<<endl;

}
void simulate_trinity_q7(client* clnt){
	if(clnt->cfg->m_id!=0 || clnt->cfg->t_id!=0){
		return ;
	}
	string header="PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "
	"PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#> SELECT * WHERE { ";

	string part1="?Y  rdf:type ub:FullProfessor <-  "
				"?Y  ub:teacherOf ?Z . "
				;
	string part2=
				"?Z  rdf:type  ub:Course . "
				 "?X  ub:takesCourse ?Z <- "
				;
	string part3="?X  ub:advisor    ?Y . "
				 "?X  rdf:type ub:UndergraduateStudent . "
				;

	request_or_reply r1;
	request_or_reply r2;
	request_or_reply r3;
	uint64_t t[20];
	t[0]=timer::get_usec();
	if(!simulate_execute_first_step(clnt,header+part1+" }",r1)){
		return ;
	}
	t[1]=timer::get_usec();
	set<int> s1=remove_dup(r1,1);
	t[2]=timer::get_usec();
	cout<<"result size after remove_dup:"<<s1.size()<<endl;

	t[3]=timer::get_usec();
	if(!simulate_execute_other_step(clnt,header+part2+" }",r2,s1)){
		return ;
	}
	t[4]=timer::get_usec();
	set<int> s2=remove_dup(r2,1);
	t[5]=timer::get_usec();
	cout<<"result size after remove_dup:"<<s2.size()<<endl;

	t[6]=timer::get_usec();
	if(!simulate_execute_other_step(clnt,header+part3+" }",r3,s2)){
		return ;
	}
	t[7]=timer::get_usec();
	set<int> s3=remove_dup(r3,1);
	t[8]=timer::get_usec();
	cout<<"result size after remove_dup:"<<s3.size()<<endl;

	cout<<t[1]-t[0]<<"   "<<t[2]-t[1]<<" usec"<<endl;
	cout<<t[4]-t[3]<<"   "<<t[5]-t[4]<<" usec"<<endl;
	cout<<t[7]-t[6]<<"   "<<t[8]-t[7]<<" usec"<<endl;

	t[9]=timer::get_usec();

	ofstream f1("file_yz");
	ofstream f2("file_zx");
	ofstream f3("file_xy");
	for(int i=0;i<r1.row_num();i++){
		int v1=r1.get_row_column(i,0);
		int v2=r1.get_row_column(i,1);
		f1<<v1<<"\t"<<v2<<endl;
	}
	for(int i=0;i<r2.row_num();i++){
		int v1=r2.get_row_column(i,0);
		int v2=r2.get_row_column(i,1);
		f2<<v1<<"\t"<<v2<<endl;
	}
	for(int i=0;i<r3.row_num();i++){
		int v1=r3.get_row_column(i,0);
		int v2=r3.get_row_column(i,1);
		f3<<v1<<"\t"<<v2<<endl;
	}

//two-step join
	boost::unordered_map<int,vector<int> > hashtable1;
	for(int i=0;i<r2.row_num();i++){
		int v1=r2.get_row_column(i,0);
		int v2=r2.get_row_column(i,1);
		hashtable1[v1].push_back(v2);
	}
	vector<int> updated_result_table;
	for(int i=0;i<r1.row_num();i++){
		int vid=r1.get_row_column(i,1);
		if(hashtable1.find(vid)!=hashtable1.end()){
			for(int k=0;k<hashtable1[vid].size();k++){
				r1.append_row_to(i,updated_result_table);
	            updated_result_table.push_back(hashtable1[vid][k]);
			}
		}
	}
	r1.set_column_num(r1.column_num()+1);
    r1.result_table.swap(updated_result_table);
	updated_result_table.clear();

	boost::unordered_map<int,vector<int> > hashtable2;
	for(int i=0;i<r3.row_num();i++){
		int v1=r3.get_row_column(i,0);
		int v2=r3.get_row_column(i,1);
		hashtable2[v1].push_back(v2);
	}
	for(int i=0;i<r1.row_num();i++){
		int v1=r1.get_row_column(i,0);
		int v2=r1.get_row_column(i,2);
		if(hashtable2.find(v2)!=hashtable2.end()){
			for(int k=0;k<hashtable2[v2].size();k++){
				if(v1==hashtable2[v2][k]){
					r1.append_row_to(i,updated_result_table);
				}
			}
		}
	}
	r1.result_table.swap(updated_result_table);
	t[10]=timer::get_usec();
	cout<<"final join result size:"<<r1.row_num()<<endl;
	cout<<t[10]-t[9]<<" usec for join-time"<<endl;

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
		} else if(input_str=="trinity_q6"){
			simulate_trinity_q6(clnt);
		} else if(input_str=="trinity_q7"){
			simulate_trinity_q7(clnt);
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
	            string batchfile;
	            iss>>batchfile;
				batch_logger logger;
				logger.init();
				nonblocking_execute(clnt,batchfile,logger);
				//batch_execute(clnt,filename,logger);
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
					nonblocking_execute(clnt,batchfile,logger);
					//batch_execute(clnt,batchfile,logger);
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


void nonblocking_execute(client* clnt,string mix_config,batch_logger& logger){
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

	int send_request=0;
	int recv_request=0;
	while(recv_request!=total_request){
		for(int t=0;t<10;t++){
			if(send_request<total_request){
				send_request++;
				int idx=mymath::get_distribution(clnt->cfg->get_random(),distribution);
				instantiate_request(clnt,vec_template[idx],vec_req[idx]);
				clnt->GetId(vec_req[idx]);
				logger.start_record(vec_req[idx].parent_id,idx);
				clnt->Send(vec_req[idx]);
			}
		}
		for(int i=0;i<sleep_round;i++){
			timer::myusleep(100);
			request_or_reply reply;
			bool success=TryRecvR(clnt->cfg,reply);
			while(success){
				recv_request++;
				logger.end_record(reply.parent_id);
				success=TryRecvR(clnt->cfg,reply);
			}
		}
	}
};
