// g++ -std=c++11 generate_data.cpp -o generate_data
// /home/sjx/graph-query/generate/generate_data lubm_raw_40 id_lubm_40

#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <assert.h>
using namespace std;

int main(int argc,char** argv){
    unordered_map<string,int> str_to_id;
    vector<string> normal_str;
    vector<string> index_str;


    struct dirent *ptr;
    DIR *dir;
    if(argc<=2){
    	printf("usage: ./generate_data src_dir dst_dir\n");
    	return 0;
    }
    dir=opendir(argv[1]);

    //init

    str_to_id["__PREDICT__"]=0;
    index_str.push_back("__PREDICT__");
    str_to_id["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"]=1;
    index_str.push_back("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>");

    int nbit_predict=15;
    size_t next_index_id=2;
    size_t next_normal_id=1<<nbit_predict;
    int count=1;
    while((ptr=readdir(dir))!=NULL){
        if(ptr->d_name[0] == '.')
            continue;
        ifstream file((string(argv[1])+"/"+string(ptr->d_name)).c_str());
        ofstream output((string(argv[2])+"/id_"+string(ptr->d_name)).c_str());
        printf("No.%d, loading %s ...\n",count,ptr->d_name);
        count++;
        string subject;
		string predict;
		string object;
		string useless_dot;
        while(file>>subject>>predict>>object>>useless_dot){
			int id[3];
            if(str_to_id.find(subject)==str_to_id.end()){
    			str_to_id[subject]=next_normal_id;
                next_normal_id++;
                normal_str.push_back(subject);
    		}
            if(str_to_id.find(predict)==str_to_id.end()){
    			str_to_id[predict]=next_index_id;
                next_index_id++;
                index_str.push_back(predict);
    		}
            if(predict=="<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"){
                if(str_to_id.find(object)==str_to_id.end()){
        			str_to_id[object]=next_index_id;
                    next_index_id++;
                    index_str.push_back(object);
        		}
            } else {
                if(str_to_id.find(object)==str_to_id.end()){
        			str_to_id[object]=next_normal_id;
                    next_normal_id++;
                    normal_str.push_back(object);
        		}
            }
            id[0]=str_to_id[subject];
			id[1]=str_to_id[predict];
            id[2]=str_to_id[object];
            output<<id[0]<<"\t"<<id[1]<<"\t"<<id[2]<<endl;
        }
    }
    {
        ofstream f_normal((string(argv[2])+"/str_normal").c_str());
        for(int i=0;i<normal_str.size();i++){
            f_normal<<normal_str[i]<<"\t"<<str_to_id[normal_str[i]]<<endl;
        }
    }
    {
        ofstream f_index((string(argv[2])+"/str_index").c_str());
        for(int i=0;i<index_str.size();i++){
            f_index<<index_str[i]<<"\t"<<str_to_id[index_str[i]]<<endl;
        }
    }
    cout<<"sizeof str_to_id="<<str_to_id.size()<<endl;
    cout<<"sizeof normal_str="<<normal_str.size()<<endl;
    cout<<"sizeof index_str="<<index_str.size()<<endl;
}
