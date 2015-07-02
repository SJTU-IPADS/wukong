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
using namespace std;

int main(int argc,char** argv){

	unordered_map<string,int> subject_to_id;
	unordered_map<string,int> predict_to_id;
	vector<string> id_to_subject;
	vector<string> id_to_predict;

	struct dirent *ptr;    
    DIR *dir;
    if(argc<=1){
    	printf("usage: ./index_server dir\n");
    	return 0;
    }
    dir=opendir(argv[1]);
    printf("files:\n");
    
    while((ptr=readdir(dir))!=NULL){
        if(ptr->d_name[0] == '.')
            continue;
        string filename=string(argv[1])+"/"+string(ptr->d_name);
        printf("loading %s ...\n",ptr->d_name);
        ifstream file(filename.c_str());
        ofstream output(("id_"+string(ptr->d_name)).c_str());
        // S P O .
		string subject;
		string predict;
		string object;
		string useless_dot;
        while(file>>subject>>predict>>object>>useless_dot){
			int id[3];
			//replace prefix
			string prefix="<http://swat.cse.lehigh.edu/onto/univ-bench.owl";
			if(equal(prefix.begin(), prefix.end(), subject.begin()))
				subject="<ub"+subject.substr(prefix.size());
			if(equal(prefix.begin(), prefix.end(), predict.begin()))
				predict="<ub"+predict.substr(prefix.size());
			if(equal(prefix.begin(), prefix.end(), object.begin()))
				object="<ub"+object.substr(prefix.size());

			if(subject_to_id.find(subject)==subject_to_id.end()){
				int size =subject_to_id.size();
				subject_to_id[subject]=size;
				id_to_subject.push_back(subject);
			}
			if(predict_to_id.find(predict)==predict_to_id.end()){
				int size =predict_to_id.size();
				predict_to_id[predict]=size;
				id_to_predict.push_back(predict);
			}
			if(subject_to_id.find(object)==subject_to_id.end() ){
				int size =subject_to_id.size();
				subject_to_id[object]=size;
				id_to_subject.push_back(object);
			}
			id[0]=subject_to_id[subject];
			id[1]=predict_to_id[predict];
			id[2]=subject_to_id[object];
			output<<id[0]<<"\t"<<id[1]<<"\t"<<id[2]<<endl;		
		}
		file.close();
		output.close();        
    }
    closedir(dir);
    cout<<id_to_subject.size()<<endl;
    ofstream f1("index_subject");
    for(int i=0;i<id_to_subject.size();i++){
    	f1<<id_to_subject[i]<<endl;
    }
    f1.close();

    ofstream f2("index_predict");
    cout<<id_to_predict.size()<<endl;
    for(int i=0;i<id_to_predict.size();i++){
    	f2<<id_to_predict[i]<<endl;
    }
    f2.close();
}

