#ifndef RDMA_MALLOC
#define RDMA_MALLOC

#include <stddef.h>
#include <stdint.h>

/*   This file provides interfaces of a malloc for manging registered RDMA regions. 
   It shall be linked to the dedicated ssmalloc library which can be installed 
   by following instructions in ../ralloc/README.md. 
   
   Usage:
     To manage allocation in RDMA registered region, just pass the start pointer and the 
   size to RInit() for initlization. 
     Before Each thread can alloc memory, they shall call RThreadLocalInit() at first. 
     
       Rmalloc and Rfree works as the same as standard malloc and free. The addresses returned 
     is in the registered memory region. 
     
   Limitation:
     We assume there is exactly one RDMA region on one machine.  Which is enough most of the time. 
*/

extern "C"  {
  /* Initilize the lib with the dedicated memroy buffer. Can only be called exactly once. 
     @ret
       NULL - An error occured. This is because the memory region size is not large enough.
       A size - The actual size of memory region shall be allocaed .This typicall is less than size for algiment 
       reasons. 
   */
  uint64_t  RInit(char *buffer, uint64_t size);
  /*
    Initilize thread local data structure. 
    Shall be called exactly after RInit and before the first call of Rmalloc or Rfree at this thread. 
   */
  void  RThreadLocalInit(void);
  void *Rmalloc(size_t __size);
  void  Rfree(void *__ptr);
}

#endif
