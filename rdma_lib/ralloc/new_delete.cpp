#include <new>

using namespace std;

extern "C" {
    void* malloc(size_t);
    void free(void*);
}

void* operator new(size_t size) throw (std::bad_alloc)
{
    return malloc(size);
}

void * operator new(size_t size, const std::nothrow_t&) throw()
{
    return malloc(size);
}

void operator delete(void *ptr)
{
    free(ptr);
}

void* operator new[](size_t size) throw (std::bad_alloc)
{
    return malloc(size);
}

void * operator new[](size_t size, const std::nothrow_t&) throw()
{
    return malloc(size);
}

void operator delete[](void *ptr)
{
    free(ptr);
}

