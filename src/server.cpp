#include "server.h"
#include <stdlib.h> //qsort



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
    updated_result_table.reserve(req.result_table.size());
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

    if(!global_enable_index_partition){
        const_to_unknown(req);
        return ;
    }
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
    req.local_var=-1;
};

void server::const_unknown_unknown(request_or_reply& req){
    int start       =req.cmd_chains[req.step*4];
    int predict     =req.cmd_chains[req.step*4+1];
    int direction   =req.cmd_chains[req.step*4+2];
    int end         =req.cmd_chains[req.step*4+3];
    vector<int> updated_result_table;

    if(req.column_num()!=0 ){
        //it means the query plan is wrong
        assert(false);
    }
    int npredict=0;
    edge* predict_ptr=g.get_edges_global(cfg->t_id,start,direction,0,&npredict);
    // foreach possible predict
    for(int p=0;p<npredict;p++){
        int edge_num=0;
        edge* edge_ptr;
        edge_ptr=g.get_edges_global(cfg->t_id, start,direction,predict_ptr[p].val,&edge_num);
        for(int k=0;k<edge_num;k++){
            updated_result_table.push_back(predict_ptr[p].val);
            updated_result_table.push_back(edge_ptr[k].val);
        }
    }
    req.result_table.swap(updated_result_table);
    req.set_column_num(2);
    req.step++;
};

void server::known_unknown_unknown(request_or_reply& req){
    int start       =req.cmd_chains[req.step*4];
    int predict     =req.cmd_chains[req.step*4+1];
    int direction   =req.cmd_chains[req.step*4+2];
    int end         =req.cmd_chains[req.step*4+3];
    vector<int> updated_result_table;

    // foreach vertex
    for(int i=0;i<req.row_num();i++){
        int prev_id=req.get_row_column(i,req.var2column(start));
        int npredict=0;
        edge* predict_ptr=g.get_edges_global(cfg->t_id,prev_id,direction,0,&npredict);
        // foreach possible predict
        for(int p=0;p<npredict;p++){
            int edge_num=0;
            edge* edge_ptr;
            edge_ptr=g.get_edges_global(cfg->t_id, prev_id,direction,predict_ptr[p].val,&edge_num);
            for(int k=0;k<edge_num;k++){
                req.append_row_to(i,updated_result_table);
                updated_result_table.push_back(predict_ptr[p].val);
                updated_result_table.push_back(edge_ptr[k].val);
            }
        }
    }

    req.set_column_num(req.column_num()+2);
    req.result_table.swap(updated_result_table);
    req.step++;
};

void server::known_unknown_const(request_or_reply& req){
    int start       =req.cmd_chains[req.step*4];
    int predict     =req.cmd_chains[req.step*4+1];
    int direction   =req.cmd_chains[req.step*4+2];
    int end         =req.cmd_chains[req.step*4+3];
    vector<int> updated_result_table;

    // foreach vertex
    for(int i=0;i<req.row_num();i++){
        int prev_id=req.get_row_column(i,req.var2column(start));
        int npredict=0;
        edge* predict_ptr=g.get_edges_global(cfg->t_id,prev_id,direction,0,&npredict);
        // foreach possible predict
        for(int p=0;p<npredict;p++){
            int edge_num=0;
            edge* edge_ptr;
            edge_ptr=g.get_edges_global(cfg->t_id, prev_id,direction,predict_ptr[p].val,&edge_num);
            for(int k=0;k<edge_num;k++){
                if(edge_ptr[k].val == end){
                    req.append_row_to(i,updated_result_table);
                    updated_result_table.push_back(predict_ptr[p].val);
                    break;
                }
            }
        }
    }

    req.set_column_num(req.column_num()+1);
    req.result_table.swap(updated_result_table);
    req.step++;
}
typedef std::pair<int,int> v_pair;
size_t hash_pair(const v_pair &x){
	size_t r=x.first;
	r=r<<32;
	r+=x.second;
	return hash<size_t>()(r);
}
void server::handle_join(request_or_reply& req){
    // step.1 remove dup;
    uint64_t t0=timer::get_usec();

    boost::unordered_set<int> remove_dup_set;
    int dup_var=req.cmd_chains[req.step*4+4];
    assert(dup_var<0);
    for(int i=0;i<req.row_num();i++){
        remove_dup_set.insert(req.get_row_column(i,req.var2column(dup_var)));
    }

    // step.2 generate cmd_chain for sub-req
    vector<int> sub_chain;
    boost::unordered_map<int,int> var_mapping;
    vector<int> reverse_mapping;
    int join_step=req.cmd_chains[req.step*4+3];
    for(int i=req.step*4+4;i<join_step*4;i++){
        if(req.cmd_chains[i]<0 && ( var_mapping.find(req.cmd_chains[i]) ==  var_mapping.end()) ){
            int new_id=-1-var_mapping.size();
            var_mapping[req.cmd_chains[i]]=new_id;
            reverse_mapping.push_back(req.var2column(req.cmd_chains[i]));
        }
        if(req.cmd_chains[i]<0){
            sub_chain.push_back(var_mapping[req.cmd_chains[i]]);
        } else {
            sub_chain.push_back(req.cmd_chains[i]);
        }
    }

    // step.3 make sub-req
    request_or_reply sub_req;
    {
        boost::unordered_set<int>::iterator iter;
        for(iter=remove_dup_set.begin();iter!=remove_dup_set.end();iter++){
            sub_req.result_table.push_back(*iter);
        }
        sub_req.cmd_chains=sub_chain;
        sub_req.silent=false;
        sub_req.col_num=1;
    }

    uint64_t t1=timer::get_usec();
    // step.4 execute sub-req
    while(true){
        execute_one_step(sub_req);
        if(sub_req.is_finished()){
            break;
        }
    }
    uint64_t t2=timer::get_usec();

    uint64_t t3,t4;
    vector<int> updated_result_table;

    if(sub_req.column_num()>2){
    //if(true){ // always use qsort
        mytuple::qsort_tuple(sub_req.column_num(),sub_req.result_table);
        vector<int> tmp_vec;
        tmp_vec.resize(sub_req.column_num());
        t3=timer::get_usec();
        for(int i=0;i<req.row_num();i++){
            for(int c=0;c<reverse_mapping.size();c++){
                tmp_vec[c]=req.get_row_column(i,reverse_mapping[c]);
            }
            if(mytuple::binary_search_tuple(sub_req.column_num(),sub_req.result_table,tmp_vec)){
                req.append_row_to(i,updated_result_table);
            }
        }
        t4=timer::get_usec();
    } else { // hash join
        boost::unordered_set<v_pair> remote_set;
        for(int i=0;i<sub_req.row_num();i++){
            remote_set.insert(v_pair(sub_req.get_row_column(i,0),sub_req.get_row_column(i,1)));
        }
        vector<int> tmp_vec;
        tmp_vec.resize(sub_req.column_num());
        t3=timer::get_usec();
        for(int i=0;i<req.row_num();i++){
            for(int c=0;c<reverse_mapping.size();c++){
                tmp_vec[c]=req.get_row_column(i,reverse_mapping[c]);
            }
            v_pair target=v_pair(tmp_vec[0],tmp_vec[1]);
            if(remote_set.find(target)!= remote_set.end()){
                req.append_row_to(i,updated_result_table);
            }
        }
        t4=timer::get_usec();
    }
    if(cfg->m_id==0 && cfg->t_id==cfg->client_num){
        cout<<"prepare "<<(t1-t0) <<" us"<<endl;
        cout<<"execute sub-request "<<(t2-t1) <<" us"<<endl;
        cout<<"sort "<<(t3-t2) <<" us"<<endl;
        cout<<"lookup "<<(t4-t3) <<" us"<<endl;
    }

    req.result_table.swap(updated_result_table);
    req.step=join_step;
}
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

    if(predict<0){
        switch (var_pair(req.variable_type(start),req.variable_type(end))) {
            case var_pair(const_var,unknown_var):
                const_unknown_unknown(req);
                break;
            case var_pair(known_var,unknown_var):
                known_unknown_unknown(req);
                break;
            default :
                assert(false);
                break;
        }
        return true;
    }

    // known_predict
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
        sub_reqs[i].local_var=start;
	}
	for(int i=0;i<req.row_num();i++){
		int machine = mymath::hash_mod(req.get_row_column(i,req.var2column(start)), num_sub_request);
		req.append_row_to(i,sub_reqs[machine].result_table);
	}
	return sub_reqs;
}
vector<request_or_reply> server::generate_mt_sub_requests(request_or_reply& req){
    int start       =req.cmd_chains[req.step*4];
    int end         =req.cmd_chains[req.step*4+3];
    int nthread=max(1,min(global_multithread_factor,global_num_server));

	vector<request_or_reply> sub_reqs;
	int num_sub_request=cfg->m_num * nthread ;
	sub_reqs.resize(num_sub_request );
	for(int i=0;i<sub_reqs.size();i++){
		sub_reqs[i].parent_id=req.id;
		sub_reqs[i].cmd_chains=req.cmd_chains;
        sub_reqs[i].step=req.step;
        sub_reqs[i].col_num=req.col_num;
        sub_reqs[i].silent=req.silent;
        sub_reqs[i].local_var=start;
	}
	for(int i=0;i<req.row_num();i++){
        // id = t_id*cfg->m_num + m_id
        //so  m_id = id % cfg->m_num
        //    t_id = id / cfg->m_num
		int id = mymath::hash_mod(req.get_row_column(i,req.var2column(start)), num_sub_request);
		req.append_row_to(i,sub_reqs[id].result_table);
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
    uint64_t t1;
    uint64_t t2;
    while(true){
        t1=timer::get_usec();
        execute_one_step(req);
        t2=timer::get_usec();
        if(cfg->m_id==0 && cfg->t_id==cfg->client_num){
            //cout<<"step "<<req.step <<" "<<t2-t1<<" us"<<endl;
        }
        if(!req.is_finished() && req.cmd_chains[req.step*4+2]==join_cmd){
            t1=timer::get_usec();
            handle_join(req);
            t2=timer::get_usec();
            if(cfg->m_id==0 && cfg->t_id==cfg->client_num){
                //cout<<"handle join "<<" "<<t2-t1<<" us"<<endl;
            }
        }
        if(req.is_finished()){
            req.silent_row_num=req.row_num();
            if(req.silent){
                req.clear_data();
            }
            SendR(cfg,cfg->mid_of(req.parent_id),cfg->tid_of(req.parent_id),req);
            return ;
        }
        if(req.step==1 && req.use_index_vertex() && !global_enable_index_partition){
            assert(!global_enable_workstealing);
            vector<request_or_reply> sub_reqs=generate_mt_sub_requests(req);
            wqueue.put_parent_request(req,sub_reqs.size());
            //so  m_id = id % cfg->m_num
            //    t_id = id / cfg->m_num + cfg->client_num
            for(int i=0;i<sub_reqs.size();i++){
                SendR(cfg,i%cfg->m_num ,i / cfg->m_num+ cfg->client_num,sub_reqs[i]);
			}
            return ;
        }
        if(need_sub_requests(req)){
            vector<request_or_reply> sub_reqs=generate_sub_requests(req);
            wqueue.put_parent_request(req,sub_reqs.size());
			for(int i=0;i<sub_reqs.size();i++){
                if(i!=cfg->m_id){
                    SendR(cfg,i,cfg->t_id,sub_reqs[i]);
                } else {
                    pthread_spin_lock(&recv_lock);
                    msg_fast_path.push_back(sub_reqs[i]);
                    pthread_spin_unlock(&recv_lock);
                }
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
            //check fast path first
            bool get_from_fast_path=false;
            pthread_spin_lock(&recv_lock);
            if(msg_fast_path.size()>0){
                r=msg_fast_path.back();
                msg_fast_path.pop_back();
                get_from_fast_path=true;
            }
            pthread_spin_unlock(&recv_lock);
            if(get_from_fast_path){
                break;
            }


            int size=global_enable_workstealing?2:1;
            recvid=possible_array[try_count % size];
            try_count++;
            if(recvid==own_id){
                // tryrecv
            } else {
                uint64_t last_of_other=s_array[recvid]->last_time;
                if(last_time > last_of_other && (last_time - last_of_other)> 10000 ){
                    // tryrecv
                } else {
                    continue;
                }
            }

            bool success;
            pthread_spin_lock(&s_array[recvid]->recv_lock);
            success=TryRecvR(s_array[recvid]->cfg,r);
            if(success && recvid!=own_id && r.use_index_vertex()){
                s_array[recvid]->msg_fast_path.push_back(r);
                success=false;
            }
            pthread_spin_unlock(&s_array[recvid]->recv_lock);
            if(success){
                break;
            }
        }

        // step 2: handle it
        if(r.is_request()){
            r.id=cfg->get_inc_id();
            int before=r.row_num();
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
