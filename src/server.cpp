#include "server.h"

server::server(distributed_graph& _g,thread_cfg* _cfg):g(_g),cfg(_cfg){
    last_time=-1;
    pthread_spin_init(&recv_lock,0);
    pthread_spin_init(&wqueue_lock,0);
}
void server::const_to_unknown(request_or_reply& req){
    int start       =req.cmd_chains[req.step*4];
    int predict     =req.cmd_chains[req.step*4+1];
    int direction   =req.cmd_chains[req.step*4+2];
    int end         =req.cmd_chains[req.step*4+3];
    vector<int> updated_result_table;

    if(!(req.column_num()==0 && req.column_num() == req.var2column(end)  )){
        //it means the query plan is wrong
        assert(false);
    }
    int edge_num=0;
    edge* edge_ptr;
    edge_ptr=g.get_edges_global(cfg->t_id,start,direction,predict,&edge_num);
    for(int k=0;k<edge_num;k++){
        updated_result_table.push_back(edge_ptr[k].val);
    }
    req.result_table.swap(updated_result_table);
    req.set_column_num(1);
    req.step++;
};
void server::const_to_known(request_or_reply& req){
    //TODO
};
void server::known_to_unknown(request_or_reply& req){
    int start       =req.cmd_chains[req.step*4];
    int predict     =req.cmd_chains[req.step*4+1];
    int direction   =req.cmd_chains[req.step*4+2];
    int end         =req.cmd_chains[req.step*4+3];
    vector<int> updated_result_table;

    if(req.column_num() != req.var2column(end) ){
        //it means the query plan is wrong
        assert(false);
    }
    for(int i=0;i<req.row_num();i++){
        int prev_id=req.get_row_column(i,req.var2column(start));
        int edge_num=0;
        edge* edge_ptr;
        edge_ptr=g.get_edges_global(cfg->t_id, prev_id,direction,predict,&edge_num);
        for(int k=0;k<edge_num;k++){
            req.append_row_to(i,updated_result_table);
            updated_result_table.push_back(edge_ptr[k].val);
        }
    }
    req.set_column_num(req.column_num()+1);
    req.result_table.swap(updated_result_table);
    req.step++;
};
void server::known_to_known(request_or_reply& req){
    int start       =req.cmd_chains[req.step*4];
    int predict     =req.cmd_chains[req.step*4+1];
    int direction   =req.cmd_chains[req.step*4+2];
    int end         =req.cmd_chains[req.step*4+3];
    vector<int> updated_result_table;

    for(int i=0;i<req.row_num();i++){
        int prev_id=req.get_row_column(i,req.var2column(start));
        int edge_num=0;
        edge* edge_ptr;
        edge_ptr=g.get_edges_global(cfg->t_id, prev_id,direction,predict,&edge_num);
        int end_id=req.get_row_column(i,req.var2column(end));
        for(int k=0;k<edge_num;k++){
            if(edge_ptr[k].val == end_id){
                req.append_row_to(i,updated_result_table);
                break;
            }
        }
    }
    req.result_table.swap(updated_result_table);
    req.step++;
};
void server::known_to_const(request_or_reply& req){
    int start       =req.cmd_chains[req.step*4];
    int predict     =req.cmd_chains[req.step*4+1];
    int direction   =req.cmd_chains[req.step*4+2];
    int end         =req.cmd_chains[req.step*4+3];
    vector<int> updated_result_table;

    for(int i=0;i<req.row_num();i++){
        int prev_id=req.get_row_column(i,req.var2column(start));
        int edge_num=0;
        edge* edge_ptr;
        edge_ptr=g.get_edges_global(cfg->t_id, prev_id,direction,predict,&edge_num);
        for(int k=0;k<edge_num;k++){
            if(edge_ptr[k].val == end){
                req.append_row_to(i,updated_result_table);
                break;
            }
        }
    }
    req.result_table.swap(updated_result_table);
    req.step++;
};

void server::index_to_unknown(request_or_reply& req){
    int index_vertex=req.cmd_chains[req.step*4];
    int nothing     =req.cmd_chains[req.step*4+1];
    int direction   =req.cmd_chains[req.step*4+2];
    int var         =req.cmd_chains[req.step*4+3];
    vector<int> updated_result_table;

    if(!(req.column_num()==0 && req.column_num() == req.var2column(var)  )){
        //it means the query plan is wrong
        assert(false);
    }

    int edge_num=0;
    edge* edge_ptr;
    edge_ptr=g.local_storage.get_index_edges_local(cfg->t_id,index_vertex,direction,&edge_num);
    int start_id=req.mt_current_thread;
    for(int k=start_id;k<edge_num;k+=req.mt_total_thread){
        updated_result_table.push_back(edge_ptr[k].val);
    }

    req.result_table.swap(updated_result_table);
    req.set_column_num(1);
    req.step++;
};

bool server::execute_one_step(request_or_reply& req){
    if(req.is_finished()){
        return false;
    }
    if(req.step==0 && req.use_index_vertex()){
        index_to_unknown(req);
        return true;
    }
    int start       =req.cmd_chains[req.step*4];
    int predict     =req.cmd_chains[req.step*4+1];
    int direction   =req.cmd_chains[req.step*4+2];
    int end         =req.cmd_chains[req.step*4+3];
    switch (var_pair(req.variable_type(start),req.variable_type(end))) {
        ///start from const_var
        case var_pair(const_var,const_var):
            cout<<"error:const_var->const_var"<<endl;
            assert(false);
            break;
        case var_pair(const_var,unknown_var):
            const_to_unknown(req);
            break;
        case var_pair(const_var,known_var):
            cout<<"error:const_var->known_var"<<endl;
            assert(false);
            break;

        ///start from known_var
        case var_pair(known_var,const_var):
            known_to_const(req);
            break;
        case var_pair(known_var,known_var):
            known_to_known(req);
            break;
        case var_pair(known_var,unknown_var):
            known_to_unknown(req);
            break;

        ///start from unknown_var
        case var_pair(unknown_var,const_var):
        case var_pair(unknown_var,known_var):
        case var_pair(unknown_var,unknown_var):
            cout<<"error:unknown_var->"<<endl;
            assert(false);
        default :
            cout<<"default"<<endl;
            break;
    }
    return true;
};
vector<request_or_reply> server::generate_sub_requests(request_or_reply& req){
    int start       =req.cmd_chains[req.step*4];
    int end         =req.cmd_chains[req.step*4+3];

	vector<request_or_reply> sub_reqs;
	int num_sub_request=cfg->m_num;
	sub_reqs.resize(num_sub_request);
	for(int i=0;i<sub_reqs.size();i++){
		sub_reqs[i].parent_id=req.id;
		sub_reqs[i].cmd_chains=req.cmd_chains;
        sub_reqs[i].step=req.step;
        sub_reqs[i].col_num=req.col_num;
        sub_reqs[i].silent=req.silent;
        sub_reqs[i].local_var=end;
	}
	for(int i=0;i<req.row_num();i++){
		int machine = mymath::hash_mod(req.get_row_column(i,req.var2column(start)), num_sub_request);
		req.append_row_to(i,sub_reqs[machine].result_table);
	}
	return sub_reqs;
}
bool server::need_sub_requests(request_or_reply& req){
    int start       =req.cmd_chains[req.step*4];
    if(req.local_var==start){
        return false;
    }
    if(req.row_num()<global_rdma_threshold){
        return false;
    }
    return true;
};
void server::execute(request_or_reply& req){
    while(true){
        execute_one_step(req);
        if(req.is_finished()){
            req.silent_row_num=req.row_num();
            if(req.silent){
                req.clear_data();
            }
            SendR(cfg,cfg->mid_of(req.parent_id),cfg->tid_of(req.parent_id),req);
            return ;
        }
        if(need_sub_requests(req)){
            vector<request_or_reply> sub_reqs=generate_sub_requests(req);
            wqueue.put_parent_request(req,sub_reqs.size());
			for(int i=0;i<sub_reqs.size();i++){
				SendR(cfg,i,cfg->t_id,sub_reqs[i]);
			}
            return ;
        }
    }
    return;
};

void server::run(){
    int own_id=cfg->t_id - cfg->client_num ;
    int possible_array[2]={own_id , cfg->server_num-1 - own_id};
    uint64_t try_count=0;
    while(true){
        last_time=timer::get_usec();
        request_or_reply r;
        int recvid;
        // step 1: pool message
        while(true){
            //recvid=own_id;
            recvid=possible_array[try_count%2];
            try_count++;
            if(recvid==own_id){
                // tryrecv
            } else {
                uint64_t last_of_other=s_array[recvid]->last_time;
                if(last_time > last_of_other && (last_time - last_of_other)> 100000 ){
                    // tryrecv
                } else {
                    continue;
                }
            }

            bool success;
            pthread_spin_lock(&s_array[recvid]->recv_lock);
            success=TryRecvR(s_array[recvid]->cfg,r);
            pthread_spin_unlock(&s_array[recvid]->recv_lock);
            if(success){
                break;
            }
        }

        // step 2: handle it
        if(r.is_request()){
            r.id=cfg->get_inc_id();
            execute(r);
        } else {
            //r is reply
            pthread_spin_lock(&s_array[recvid]->wqueue_lock);
            s_array[recvid]->wqueue.put_reply(r);
            if(s_array[recvid]->wqueue.is_ready(r.parent_id)){
                request_or_reply reply=s_array[recvid]->wqueue.get_merged_reply(r.parent_id);
                pthread_spin_unlock(&s_array[recvid]->wqueue_lock);
                SendR(cfg,cfg->mid_of(reply.parent_id),cfg->tid_of(reply.parent_id),reply);
            }
            pthread_spin_unlock(&s_array[recvid]->wqueue_lock);
        }
    }
}

//
//
// void server::run(){
//     while(true){
//         request_or_reply r=RecvR(cfg);
//         if(r.is_request()){
//             r.id=cfg->get_inc_id();
//             execute(r);
//         } else {
//             //r is reply
//             wqueue.put_reply(r);
//             if(wqueue.is_ready(r.parent_id)){
//                 request_or_reply reply=wqueue.get_merged_reply(r.parent_id);
//                 SendR(cfg,cfg->mid_of(reply.parent_id),cfg->tid_of(reply.parent_id),reply);
//             }
//         }
//     }
// }
