#pragma once

#include <boost/variant.hpp>
enum {
    SID_t = 0,
    INT_t = 1,
    FLOAT_t = 2,
    DOUBLE_t = 3,
};
// int 1, flot 2, double 3
class variant_type : public boost::static_visitor<int> {

    public:
        int operator ()(int i) const
        {
            return INT_t;
        }

        int operator () (float f ) const
        {
            return FLOAT_t;
        }

        int operator ()(double d) const
        {
            return DOUBLE_t;
        }
};

variant_type get_type;

size_t get_size(int type){
    switch (type){
        case INT_t :
            return sizeof(int);
        case FLOAT_t :
            return sizeof(float); 
        case DOUBLE_t :
            return sizeof(double);
        default :
            return 0;
    }

}

