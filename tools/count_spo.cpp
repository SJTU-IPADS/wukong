#include <string>  
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <assert.h> 
#include <string>

using namespace std;


int main(int argc,char** argv){

	if(argc<=1){
    	printf("usage: ./count_spo origin_dir\n");
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
    unordered_set<int> s_set;
    unordered_set<int> o_set;

    int triples=0;
	{
		for(int i=0;i<datafile_vec.size();i++){
			cout<<"checking file "<<i<<endl;
			ifstream finput(string(argv[1])+"/"+datafile_vec[i]);
			int s,p,o;
			while(finput>>s>>p>>o){
				s_set.insert(s);
				o_set.insert(o);
				triples++;
			}
			finput.close();
		}
	}
	int intersect=0;
	for (auto id : s_set){
		if(o_set.find(id)!=o_set.end()){
			intersect++;
		}
	}
	cout<<"Triples# "<<triples<<endl;
	cout<<"S# "<<s_set.size()<<endl;
	cout<<"O# "<<o_set.size()<<endl;
	cout<<"S^O# "<<intersect<<endl;
}