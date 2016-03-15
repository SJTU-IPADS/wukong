#include "string_server.h"


string_server::string_server(string dir_name){
    struct dirent *ptr;
    DIR *dir;
    dir=opendir(dir_name.c_str());
    vector<string> filenames;
    if(dir==NULL){
        cout<<"Error folder: "<<dir_name<<endl;
        exit(-1);
    }

    global_rdftype_id=1;
    while((ptr=readdir(dir))!=NULL){
        if(ptr->d_name[0] == '.'){
            continue;
        }
        string fname(ptr->d_name);
        string complete_fname=dir_name+"/"+fname;
        if(fname == "str_normal" ){
            if(!global_load_minimal_index){
                load_index(complete_fname);
            }
        } else if(fname == "str_normal_minimal"){
            if(global_load_minimal_index){
                load_index(complete_fname);
            }
        }else if(fname == "str_index"){
            load_index(complete_fname);
        } else{
            continue;
        }
    }
}


void string_server::load_index(string filename){
	ifstream file(filename.c_str());
	string str;
	int id;
	while(file>>str>>id){
        assert(str2id.find(str)==str2id.end());
        assert(id2str.find(id)==id2str.end());
		str2id[str]=id;
        id2str[id]=str;
	}
	file.close();
}
