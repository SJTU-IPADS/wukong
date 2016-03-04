#pragma once

#include <sys/time.h>
#include <stdint.h>

class timer{
public:
    struct timespec ts;
    static uint64_t get_usec(){
        struct timespec tmp;
        clock_gettime(CLOCK_MONOTONIC,&tmp);
        uint64_t result=tmp.tv_sec;
        result*=1000*1000;
        result+=tmp.tv_nsec/1000;
        return result;
    }
};
