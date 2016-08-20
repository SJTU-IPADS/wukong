#include "hdfs.h"
#include <sstream>
#include "string_server.h"

string_server::string_server(string dir_name){
    if (!global_use_hdfs){
        // use NFS
        struct dirent *ptr;
        DIR *dir;
        dir=opendir(dir_name.c_str());
        
        if(dir==NULL){
            cout<<"Error folder string server: "<<dir_name<<endl;
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
                    load_index_nfs(complete_fname);
                }
            } else if(fname == "str_normal_minimal"){
                if(global_load_minimal_index){
                    load_index_nfs(complete_fname);
                }
            }else if(fname == "str_index"){
                load_index_nfs(complete_fname);
            } else{
                continue;
            }
        }
    } else{
        // use HDFS
        int numEntries;
        hdfsFS fs = hdfsConnect("default", 0);
        hdfsFileInfo *list = hdfsListDirectory(fs, dir_name.c_str(), &numEntries);;
        
        for(int i = 0; i < numEntries; i++){
            string fname = list[i].mName;

            // str_normal
            std::size_t found = fname.find("str_normal");
            if (found!=std::string::npos &&
                !global_load_minimal_index){
                load_index_hdfs(fname);
            }
            // str_normal_minimal
            found = fname.find("str_normal_minimal");
            if (found!=std::string::npos &&
                global_load_minimal_index){
                load_index_hdfs(fname);
            }
            // str_index
            found = fname.find("str_index");
            if (found!=std::string::npos){
                load_index_hdfs(fname);
            }
        }

        hdfsDisconnect(fs);
    }

}


void string_server::load_index_nfs(string filename){
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

void string_server::load_index_hdfs(string filename){
    cout << "loading " << filename << endl;

    hdfsFS fs = hdfsConnect("default", 0);
    hdfsFile readFile = hdfsOpenFile(fs, filename.c_str(), O_RDONLY, 0, 0, 0);
    if(!readFile) {
        cout << "Failed to open index file!\n";
        exit(-1);
    }

    const int buffer_size = 256;
    char buffer[buffer_size];
    int start = 0, offset;
    while(true){
        offset = hdfsRead(fs, readFile, 
                          buffer + start, buffer_size - start);
        // buffer[0..start+offset)
        string line="";
        for(int i = 0; i < start+offset; i++){
            if (buffer[i] == '\n'){
                // current line finish
                int id;
                string str;
                stringstream ss(line);
                ss >> str >> id;

                assert(str2id.find(str)==str2id.end());
                assert(id2str.find(id)==id2str.end());
                str2id[str]=id;
                id2str[id]=str;

                // debug output
                //if (id < (1 << 17))
                //    cout << line << endl << endl;
                line = "";
            } else{
                line = line + buffer[i];
            }
        }
        if (offset <= 0)
            break;

        // copy the rest of buffer to the front for next round
        start = line.length();
        memcpy(buffer, line.c_str(), start);
    }

    hdfsDisconnect(fs);
}