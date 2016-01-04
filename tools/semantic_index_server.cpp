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
#include <string>

using namespace std;

unsigned long semantic_hash(string str){
	if(str==""){
		return 0;
	}
	int start=0;
	size_t end=str.find_last_of('.');
	if(end==string::npos){
		end=str.size();
	} else {
		start=end-1;
		while(start>=0 && str[start]!='.'){
			start--;
		}
	}
	unsigned long val = 5381;
	for(int i=start;i<end;i++){
		val=val*37 + str[i];
	}
	return hash<unsigned long>()(val);
}
int main(int argc,char** argv){

	if(argc<=1){
    	printf("usage: ./semantic_index_server origin_dir\n");
    	printf("data will generated in current dir\n");
    	return 0;
    }
    struct dirent *ptr;    
    DIR *dir;
    dir=opendir(argv[1]);
    printf("files:\n");

    vector<string> datafile_vec;

	while((ptr=readdir(dir))!=NULL){
        if(ptr->d_name[0] == '.')
            continue;
        string filename=string(ptr->d_name);
        string prefix="id";
		if(equal(prefix.begin(), prefix.end(), filename.begin())){
			datafile_vec.push_back(filename);
		}
    }
    int total_partition=12;
    unordered_map<int,int> id_mapping;
    vector<int> new_id;
    new_id.resize(total_partition);
    for(int i=0;i<total_partition;i++){
    	new_id[i]=i;
    }
    // build mapping
    {
    	ifstream finput(string(argv[1])+"/"+string("index_subject"));
    	ofstream foutput("index_subject");
    	int id=0;
    	string subject;
    	while(finput>>subject>>id){
			unsigned long hash=semantic_hash(subject);
			if(new_id[hash%total_partition]<0){
				assert(false);
			}
			//cout<<subject<<"~"<<id<<"~"<<hash<<new_id[hash%total_partition]<<endl;
			id_mapping[id]=new_id[hash%total_partition];
			new_id[hash%total_partition]+=total_partition;

			foutput<< subject<<"\t"<<id_mapping[id]<<endl;
		}
    }

    //rewrite data 
	{
		for(int i=0;i<datafile_vec.size();i++){
			ifstream finput(string(argv[1])+"/"+datafile_vec[i]);
			ofstream foutput(datafile_vec[i]);
			int s,p,o;
			while(finput>>s>>p>>o){
				foutput<< id_mapping[s]<<"\t"<<p<<"\t"<<id_mapping[o]<<endl;
			}
			finput.close();
			foutput.close();
		}
	}

	{
    	ifstream finput(string(argv[1])+"/"+string("minimal_index_subject"));
    	if(finput){
    		ofstream foutput("minimal_index_subject");
    		int id=0;
	    	string subject;
	    	while(finput>>subject>>id){
				foutput<< subject<<"\t"<<id_mapping[id]<<endl;
			}
    	}
    } 
    for(int i=0;i<total_partition;i++){
    	cout<<"partition "<<i<<" = "<<new_id[i]<<endl;
    }
    //copy files
    {
    	ifstream finput(string(argv[1])+"/"+string("index_predict"),ios::binary);
    	ofstream foutput("index_predict");
    	foutput << finput.rdbuf();
    }
}