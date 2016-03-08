#include "sparql_parser.h"

inline static bool is_upper(string str1,string str2){
    return boost::to_upper_copy<std::string>(str1)==str2;
}

sparql_parser::sparql_parser(string_server* _str_server):str_server(_str_server){

};


void sparql_parser::clear(){
    prefix_map.clear();
    variable_map.clear();
    req_template =  request_template();
};

bool sparql_parser::readFile(string filename){
    ifstream file(filename);
	if(!file){
        return false;
	}
    string cmd;
    vector<string> token_vec;
	while(file>>cmd){
        token_vec.push_back(cmd);
        if(cmd=="}"){
            break;
        }
	}
    file.close();
    int iter=0;

    boost::unordered_map<string,int>* id_maps[3]={
        &str_server->subject_to_id,
        &str_server->predict_to_id,
        &str_server->subject_to_id,
    };
    //A more reasonable parser should be implemented
    while(token_vec[iter]!="{"){
        iter++;
    }
    iter++;
    //handle possible index vertex
    if(token_vec[iter]=="PI" || token_vec[iter]=="TI"){
        string str_index=token_vec[iter+1];
        string str_var=token_vec[iter+2];
        int id_index;
        int id_var;
        int dir;
        if(token_vec[iter]=="PI"){
            if(id_maps[1]->find(str_index)==id_maps[1]->end()){
                return false;
            }
            id_index=(*id_maps[1])[str_index];
            if(token_vec[iter+3]=="->" || token_vec[iter+3]=="."){
                dir=pindex_out;
            } else {
                dir=pindex_in;
            }
        } else {
            if(id_maps[0]->find(str_index)==id_maps[0]->end()){
                return false;
            }
            id_index=(*id_maps[0])[str_index];
            dir=tindex_in;
        }
        if(variable_map.find(str_var)==variable_map.end()){
            int new_id=-1-variable_map.size();
            variable_map[str_var]=new_id;
        }
        id_var=variable_map[str_var];
        req_template.cmd_chains.push_back(id_index);
        req_template.cmd_chains.push_back(0);//useless
        req_template.cmd_chains.push_back(dir);
        req_template.cmd_chains.push_back(id_var);
        iter+=4;
    }
    while(token_vec[iter]!="}"){
        string str_s=token_vec[iter];
        string str_p=token_vec[iter+1];
        string str_o=token_vec[iter+2];
        string strs[3]={str_s,str_p,str_o};
        int ids[3];
        for(int i=0;i<3;i++){
            if(strs[i][0]=='?'){
                if(i==1){
                    //doesn't support predict varibale
                    return false;
                }
                if(variable_map.find(strs[i])==variable_map.end()){
                    int new_id=-1-variable_map.size();
                    variable_map[strs[i]]=new_id;
                }
                ids[i]=variable_map[strs[i]];
            } else if(strs[i][0]=='%'){
                ids[i]=place_holder;
                req_template.place_holder_str.push_back(strs[i].substr(1));
            } else {
                if(id_maps[i]->find(strs[i])==id_maps[i]->end()){
                    return false;
                }
                ids[i]=(*id_maps[i])[strs[i]];
            }
        }
        if(token_vec[iter+3]=="." || token_vec[iter+3]=="->"){
            req_template.cmd_chains.push_back(ids[0]);
            req_template.cmd_chains.push_back(ids[1]);
            req_template.cmd_chains.push_back(direction_out);
            req_template.cmd_chains.push_back(ids[2]);
            iter+=4;
        } else if(token_vec[iter+3]=="<-"){
            req_template.cmd_chains.push_back(ids[2]);
            req_template.cmd_chains.push_back(ids[1]);
            req_template.cmd_chains.push_back(direction_in);
            req_template.cmd_chains.push_back(ids[0]);
            iter+=4;
        } else {
            return false;
        }
    }
    for(int i=0;i<req_template.cmd_chains.size();i++){
        if(req_template.cmd_chains[i]==place_holder){
            req_template.place_holder_position.push_back(i);
        }
    }
    return true;
}

bool sparql_parser::parse(string filename,request_or_reply& r){
    clear();
    if(!readFile(filename)){
        return false;
    }
    r=request_or_reply();
    r.cmd_chains=req_template.cmd_chains;
    return true;
};

bool sparql_parser::parse_template(string filename,request_template& r){
    clear();
    if(!readFile(filename)){
        return false;
    }
    r=req_template;
    return true;
};

bool sparql_parser::find_type_of(string type,request_or_reply& r){
    clear();
    r=request_or_reply();
    r.cmd_chains.push_back(str_server->subject_to_id[type]);
    r.cmd_chains.push_back(global_rdftype_id);
    r.cmd_chains.push_back(direction_in);
    r.cmd_chains.push_back(-1);
    return true;
};
