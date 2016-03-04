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

    while((ptr=readdir(dir))!=NULL){
        if(ptr->d_name[0] == '.'){
            continue;
        }
        string fname(ptr->d_name);
        string complete_fname=dir_name+"/"+fname;
        string index_prefix="index";
        if(fname == "index_subject" ){
            if(!global_load_minimal_index){
                load_index(complete_fname,subject_to_id,id_to_subject);
            }
        } else if(fname == "minimal_index_subject"){
            if(global_load_minimal_index){
                load_index(complete_fname,subject_to_id,id_to_subject);
            }
        }else if(fname == "index_predict"){
            load_index_predict(complete_fname,predict_to_id,id_to_predict);
        } else{
            continue;
        }
    }
}


void string_server::load_index(string filename,boost::unordered_map<string,int>& str2id,
				boost::unordered_map<int,string>& id2str){
	ifstream file(filename.c_str());
	string str;
	int id;
	while(file>>str>>id){
		str2id[str]=id;
        id2str[id]=str;
	}
	file.close();
}
void string_server::load_index_predict(string filename,boost::unordered_map<string,int>& str2id,
				boost::unordered_map<int,string>& id2str){
    //TODO
    //predict file doesn't have id;
    //shoule be re-designed
    ifstream file(filename.c_str());
	string str;
	int id=0;
	while(file>>str){
        if(str=="<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"){
			global_rdftype_id=id;
		}
		str2id[str]=id;
        id2str[id]=str;
        id++;
	}
	file.close();
}
