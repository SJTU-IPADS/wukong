// this small program is used to generate string index files

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


int main(int argc,char** argv){

	if(argc<=1){
    	printf("usage: ./convert_format split_number \n");
    	return 0;
    }
	
    int n_split=atoi(argv[1]);

	struct dirent *ptr;    
    DIR *dir;
    dir=opendir("./");

    vector<ofstream> s_file(n_split);
    vector<ofstream> o_file(n_split);
    for(int i=0;i<n_split;i++){
    	string s_name="s_";
    	s_name=s_name+to_string(i);    	
    	s_file[i].open(s_name.c_str());
    	string o_name="o_";
    	o_name=o_name+to_string(i);
    	o_file[i].open(o_name.c_str());
    }
    while((ptr=readdir(dir))!=NULL){
        if(ptr->d_name[0] == '.')
            continue;
        string fname(ptr->d_name);
        
        string data_prefix="id";
		if(equal(data_prefix.begin(), data_prefix.end(), fname.begin())){
			cout<<fname<<endl;
			ifstream file(fname.c_str());
			uint64_t s,p,o;
			while(file>>s>>p>>o){
				int s_file_id=s%n_split;
				int o_file_id=o%n_split;
				s_file[s_file_id]<<s<<" "<<p<<" "<<o<<endl;
				o_file[o_file_id]<<s<<" "<<p<<" "<<o<<endl;
			}
			//filenames.push_back(string(dir_name)+"/"+fname);
		} else{
		
		}
    }



}