#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

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

#endif