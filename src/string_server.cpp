#include "string_server.h"
#include "hdfs.hpp"

/**
 * load ID mapping files from a shared filesystem (e.g., NFS)
 */
void
string_server::load_from_posixfs(string dname)
{
    DIR *dir = opendir(dname.c_str());
    if (dir == NULL) {
        cout << "ERROR: failed to open the directory of ID-mapping files ("
             << dname << ")." << endl;
        exit(-1);
    }

    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.')
            continue;

        string fname(dname + ent->d_name);
        if ((boost::ends_with(fname, "/str_index"))
                || (boost::ends_with(fname, "/str_normal") && !global_load_minimal_index)
                || (boost::ends_with(fname, "/str_normal_minimal") && global_load_minimal_index)) {
            cout << "loading ID-mapping file: " << fname << endl;

            ifstream file(fname.c_str());
            string str;
            int64_t id;
            while (file >> str >> id) {
                // both string and ID are unique
                assert(str2id.find(str) == str2id.end());
                assert(id2str.find(id) == id2str.end());

                str2id[str] = id;
                id2str[id] = str;
            }
            file.close();
        }
    }
}

/**
 * load ID mapping files from HDFS
 */
void
string_server::load_from_hdfs(string dname)
{
    wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
    vector<string> files = hdfs.list_files(dname);

    for (int i = 0; i < files.size(); i++) {
        string fname = files[i];
        // NOTE: users may use a short path (w/o ip:port)
        // e.g., hdfs:/xxx/xxx/
        if ((boost::ends_with(fname, "/str_index"))
                || (boost::ends_with(fname, "/str_normal") && !global_load_minimal_index)
                || (boost::ends_with(fname, "/str_normal_minimal") && global_load_minimal_index)) {
            cout << "loading ID-mapping file from HDFS: " << fname << endl;

            wukong::hdfs::fstream file(hdfs, fname);
            string str;
            int64_t id;
            while (file >> str >> id) {
                // both string and ID are unique
                assert(str2id.find(str) == str2id.end());
                assert(id2str.find(id) == id2str.end());

                str2id[str] = id;
                id2str[id] = str;
            }
        }
    }
}

string_server::string_server(string dname)
{
    if (boost::starts_with(dname, "hdfs:")) {
        if (!wukong::hdfs::has_hadoop()) {
            cout << "ERORR: attempting to load ID mapping files from HDFS "
                 << "but Wukong was built without HDFS."
                 << endl;
            exit(-1);
        }
        load_from_hdfs(dname);
    } else {
        load_from_posixfs(dname);
    }
}
