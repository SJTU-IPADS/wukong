#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>


#include <string>
#include <vector>
#include <iostream>
#include "../graph-store-distributed/timer.h"
#include "../graph-store-distributed/request.h"
#include <sstream> 
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
using namespace std;
class person 
{ 
public: 
  person() { } 

  person(int age) : age_(age) {  } 

  int age() const { return age_; } 

private:
  friend class boost::serialization::access;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & age_;
    ar & fff_;
  }

  int age_;
  int fff_;
}; 
string mysave(request& r)
{
    stringstream ss;
    boost::archive::text_oarchive oa(ss);
    oa << r;

    return ss.str();
}
void myload(string str)
{
    stringstream s;
    s << str;
    boost::archive::text_iarchive ia(s);
    request p;
    ia >> p;
    std::cout << p.result_paths.size() << std::endl;
}
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
    std::cout << "r.result_paths.size:"<<r.result_paths.size()<<std::endl;
    myload(mysave(r));

    //convert to string
    std::stringstream ss;
    boost::archive::text_oarchive oa(ss); 
    oa << r; 
    s=ss.str();
    std::cout<<"s.size:"<<s.size()<<std::endl;

    std::stringstream ss2;
    ss2<<s;
    boost::archive::text_iarchive ia(ss2); 
    request r2;
    ia >> r2; 
    world.send(1, 17, s);
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
    world.recv(boost::mpi::any_source, 17, s);
    timer t2;
    std::cout<<"s.size:"<<s.size()<<std::endl;
    std::stringstream ss;
    ss<<s;
    boost::archive::text_iarchive ia(ss); 
    request r2;
    ia >> r2; 

    std::cout <<"round trip:"<< t2.diff(t1)<<" ms"<<std::endl;
    std::cout << "r2.result_paths.size:"<<r2.result_paths.size()<<std::endl;
  }
}