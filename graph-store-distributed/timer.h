#pragma once 

#include <sys/time.h>
#include <stdint.h>

/*
class timer{
public:
  struct  timeval t;
  timer(){
    gettimeofday(&t,0);
  }
  void reset(){
    gettimeofday(&t,0);
  }
  int diff(timer& t2){
    return (1000000 * (t.tv_sec-t2.t.tv_sec)+ t.tv_usec-t2.t.tv_usec)/1000;
  }
};
*/

class timer{
public:
  struct timespec ts;
  timer(){
    clock_gettime(CLOCK_MONOTONIC,&ts);
  }
  void reset(){
    clock_gettime(CLOCK_MONOTONIC,&ts);
  }
  int diff(timer& t2){
    return 1000 * (ts.tv_sec-t2.ts.tv_sec)+ (ts.tv_nsec-t2.ts.tv_nsec)/1000000;
  }
  static uint64_t get_usec(){
    struct timespec tmp;
    clock_gettime(CLOCK_MONOTONIC,&tmp);
    uint64_t result=tmp.tv_sec;
    result*=1000*1000;
    result+=tmp.tv_nsec/1000;
    return result;
  }
};
