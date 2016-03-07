#include "client_mode.h"

boost::unordered_map<string,vector<int> > type_to_idvec;
void insert_type(client* clnt,string type){
	if(type_to_idvec.find(type)!=type_to_idvec.end()){
		return ;
	}
	request_or_reply type_request;
	assert(clnt->parser.find_type_of(type,type_request));
	request_or_reply reply;
	clnt->Send(type_request);
	reply=clnt->Recv();
	type_to_idvec[type]=reply.result_table;
	cout<<type<<" has "<<type_to_idvec[type].size()<<" objects"<<endl;
}
void instantiate_request(client* clnt,request_or_reply& req){
	for(int i=0;i<req.place_holder_str.size();i++){
		int pos=req.place_holder_position[i];
		vector<int>& idvec=type_to_idvec[req.place_holder_str[i]];
		req.cmd_chains[pos]=idvec[clnt->cfg->get_random()%idvec.size()];
	}
	
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
			batch_execute(clnt,iterative_filename,iterative_count);
        }
    }
};

void batch_execute(client* clnt,string filename,int execute_count){
	int sum=0;
    int result_count;
    request_or_reply request;
	bool success=clnt->parser.parse(filename,request);
	if(!success){
		cout<<"sparql parse error"<<endl;
		return ;
	}
	for(int i=0;i<request.place_holder_str.size();i++){
		insert_type(clnt,request.place_holder_str[i]);
	}

	request.silent=global_silent;
	request_or_reply reply;
	for(int i=0;i<execute_count;i++){
		instantiate_request(clnt,request);
		uint64_t t1=timer::get_usec();
        clnt->Send(request);
        reply=clnt->Recv();
		uint64_t t2=timer::get_usec();
		cout<<"result size:"<<reply.silent_row_num<<endl;
		sum+=t2-t1;
	}
	cout<<"average latency "<<sum/execute_count<<" us"<<endl;
};
