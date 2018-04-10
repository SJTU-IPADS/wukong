#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <sched.h>
#include <time.h>

#include <assert.h>
#include <execinfo.h>
#include <signal.h>

#include "atomic.h"
#include "bitops.h"
#include "queue.h"
#include "double-list.h"
#include "cpu.h"

/* Configurations */
#define CHUNK_DATA_SIZE     (16*PAGE_SIZE)
#define ALLOC_UNIT          (4*1024*1024)
#define MAX_FREE_SIZE       (4*1024*1024)
#define RAW_POOL_START      ((void*)((0x600000000000/CHUNK_SIZE+1)*CHUNK_SIZE))

#define BLOCK_BUF_CNT       (16)

// #define RETURN_MEMORY
// #define DEBUG

/* Other */
#define CHUNK_SIZE          (CHUNK_DATA_SIZE+sizeof(dchunk_t))
#define CHUNK_MASK          (~(CHUNK_SIZE-1))
#define LARGE_CLASS         (100)
#define DUMMY_CLASS         (101)
#define DCH                 (sizeof(dchunk_t))
#define MAX_FREE_CHUNK      (MAX_FREE_SIZE/CHUNK_SIZE)
#define LARGE_OWNER         ((void*)0xDEAD)
#define ACTIVE              ((void*)1)

/* Utility Macros */
#define ROUNDUP(x,n)        ((x+n-1)&(~(n-1)))
#define ROUNDDOWN(x,n)      (((x-n)&(~(n-1)))+1)
#define PAGE_ROUNDUP(x)     (ROUNDUP((uintptr_t)x,PAGE_SIZE))
#define PAGE_ROUNDDOWN(x)   (ROUNDDOWN((uintptr_t)x,PAGE_SIZE))
#define CACHE_ALIGN __attribute__ ((aligned (CACHE_LINE_SIZE)))
#define THREAD_LOCAL __attribute__ ((tls_model ("initial-exec"))) __thread
#define likely(x)           __builtin_expect(!!(x),1)
#define unlikely(x)         __builtin_expect(!!(x),0)

/* Multi consumer queue */
#define queue_init(head)\
    mc_queue_init(head)
#define queue_put(head,elem)\
    mc_enqueue(head,elem,0)
#define queue_fetch(head)\
    mc_dequeue(head,0)
typedef queue_head_t Queue;

/* Single consumer queue */
#define fast_queue_init(head)\
    sc_queue_init(head)
#define fast_queue_put(head,elem)\
    sc_enqueue(head,elem,0)
#define fast_queue_fetch(head)\
    sc_dequeue(head,0)
#define fast_queue_chain_fetch(head)\
    sc_chain_dequeue(head)
typedef queue_head_t FastQueue;

/* Sequencial queue */
#define seq_queue_init(head)\
    seq_queue_init(head)
#define seq_queue_put(head,elem)\
    seq_enqueue(head,elem)
#define seq_queue_fetch(head)\
    seq_dequeue(head)
#define fast_queue_chain_put(head)\
    seq_chain_enqueue(head)
typedef seq_queue_head_t SeqQueue;

/* Type definations */
typedef enum {
    UNINITIALIZED,
    READY
} init_state;

typedef enum {
    FOREGROUND,
    BACKGROUND,
    FULL
} dchunk_state;

typedef struct lheap_s lheap_t;
typedef struct gpool_s gpool_t;
typedef struct dchunk_s dchunk_t;
typedef struct chunk_s chunk_t;
typedef struct obj_buf_s obj_buf_t;
typedef struct large_header_s large_header_t;

typedef double_list_t LinkedList;
typedef double_list_elem_t LinkedListElem;

struct large_header_s {
    CACHE_ALIGN size_t alloc_size;
    void* mem;
    CACHE_ALIGN lheap_t *owner;
};

struct chunk_s {
    CACHE_ALIGN LinkedListElem active_link;
    uint32_t numa_node;
};

/* Data chunk header */
struct dchunk_s {
    /* Local Area */
    CACHE_ALIGN LinkedListElem active_link;
    uint32_t numa_node;

    /* Read Area */
    CACHE_ALIGN lheap_t * owner;
    uint32_t size_cls;

    /* Local Write Area */
     CACHE_ALIGN dchunk_state state;
    uint32_t free_blk_cnt;
    uint32_t blk_cnt;
    SeqQueue free_head;
    uint32_t block_size;
    char *free_mem;

    /* Remote Write Area */
     CACHE_ALIGN FastQueue remote_free_head;
};

struct gpool_s {
    pthread_mutex_t lock;
    volatile char *pool_start;
    volatile char *pool_end;
    volatile char *free_start;
    Queue free_dc_head[MAX_CORE_ID];
    Queue free_lh_head[MAX_CORE_ID];
    Queue released_dc_head[MAX_CORE_ID];
};

struct obj_buf_s {
    void *dc;
    void *first;
    SeqQueue free_head;
    int count;
};

/* Per-thread data chunk pool */
struct lheap_s {
    CACHE_ALIGN LinkedListElem active_link;
    uint32_t numa_node;
    SeqQueue free_head;
    uint32_t free_cnt;

    dchunk_t *foreground[DEFAULT_BLOCK_CLASS];
    LinkedList background[DEFAULT_BLOCK_CLASS];
    dchunk_t dummy_chunk;
    obj_buf_t block_bufs[BLOCK_BUF_CNT];

     CACHE_ALIGN FastQueue need_gc[DEFAULT_BLOCK_CLASS];
};

static inline int max(int a, int b)
{
    return (a > b) ? a : b;
}


/* The new interfaces which is used for RDMA buffer malloc usage */
/* Shall only be called once! */

/* Return the actual size used. If the return size is 0, then the allocation is failed */
uint64_t  RInit(char *buffer, uint64_t size);
void  RThreadLocalInit(void);
void *Rmalloc(size_t __size);
void  Rfree(void *__ptr);

void *malloc(size_t __size);
void *realloc(void *__ptr, size_t __size);
void free(void *__ptr);
