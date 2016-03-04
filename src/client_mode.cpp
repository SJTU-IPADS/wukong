#include "client_mode.h"

void interactive_execute(client* clnt,string filename,int execute_count){
	int sum=0;
    int result_count;
    request_or_reply request;
	bool success=clnt->parser.parse(filename,request);
	if(!success){
		cout<<"sparql parse error"<<endl;
		return ;
	}
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
            interactive_execute(clnt,iterative_filename,iterative_count);
        }
    }
};
