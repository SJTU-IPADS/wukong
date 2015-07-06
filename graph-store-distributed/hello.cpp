#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <string>
#include <vector>
#include <iostream>
#include "timer.h"
#include "request.h"
/*
using namespace boost::archive;
struct path_node{
  path_node():id(-1),prev(-1){

  }
  path_node(int _id,int _prev):id(_id),prev(_prev){
  }
  int id;
  int prev;
  template <typename Archive>
  void serialize(Archive &ar, const unsigned int version) { 
    ar & id; 
    ar & prev; 
  }
};
struct request{
  std::vector<int> cmd_chains;
  std::vector<std::vector<path_node> >result_paths;
  void clear(){
    cmd_chains.clear();
    result_paths.clear();
  }
  template <typename Archive>
  void serialize(Archive &ar, const unsigned int version) { 
    ar & cmd_chains; 
    ar & result_paths; 
  }
};
*/
int main(int argc, char *argv[])
{
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  if (world.rank() == 0)
  {
    std::string s;
    std::vector<int> v;
    request r;
    path_node p(1,2);
    world.recv(boost::mpi::any_source, 16, r);
//    std::cout << s << '\n';
    std::cout << r.result_paths.size()<<std::endl;
    //for(int i=0;i<v.size();i++){
      //std::cout << v[i]<<'\n';
    //}
    request r2;
    world.send(0, 17, r);
    world.recv(boost::mpi::any_source, 17, r2);
    std::cout << r2.result_paths.size()<<std::endl;

    world.send(1, 17, r);
  }
  else
  {
    std::string s = "Hello, world!";
    std::vector<int> v;
    for(int i=0;i<10000;i++){
      v.push_back(i);
    }
    timer t1;
    request r;
    r.cmd_chains.push_back(1);
    r.cmd_chains.push_back(3);
    for(int i=0;i<10;i++){
      r.result_paths.push_back(std::vector<path_node>());
      for(int j=0;j<100;j++){
        r.result_paths[i].push_back(path_node(i,j*i));
      }
    }
    path_node p(2,3);

    world.send(0, 16, r);
    world.recv(boost::mpi::any_source, 17, r);
    timer t2;
    std::cout << t2.diff(t1)<<std::endl;
  }
}