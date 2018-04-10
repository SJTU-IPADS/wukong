#define _GNU_SOURCE
#include "ssmalloc.h"

/* Global metadata */
init_state global_state = UNINITIALIZED;
init_state r_global_state = UNINITIALIZED;
gpool_t global_pool;
gpool_t r_global_pool;
char *register_rdma_buffer;
uint64_t register_rdma_buffer_size = 0;

pthread_key_t destructor;
pthread_key_t r_destructor ;
pthread_once_t init_once = PTHREAD_ONCE_INIT;

/* Mappings used in ssmalloc */
CACHE_ALIGN int cls2size[128];
char sizemap[256];
char sizemap2[128];

/* Mappings used in r_malloc */
CACHE_ALIGN int r_cls2size[128];
char r_sizemap[256];
char r_sizemap2[128];


/* Private metadata in ssmalloc */
THREAD_LOCAL init_state thread_state = UNINITIALIZED;
THREAD_LOCAL lheap_t *local_heap = NULL;

/* Private metadata for r_malloc */
THREAD_LOCAL init_state r_thread_state = UNINITIALIZED;
THREAD_LOCAL lheap_t * r_local_heap = NULL;

/* System init functions */
static void maps_init(void);
static void thread_init(void);
static void thread_exit(void *dummy);
static void r_thread_exit(void *dummy);
static void global_init(void);
inline static void check_init(void);

/* Global pool management functions */
inline static void gpool_check_size(void *target);
/* Unlike gpool check size which can grow unlimited, RDMA resources is scarce */
inline static int  rpool_check_size(void *target);
static int gpool_grow(void);
static void gpool_init(void);
static void *gpool_make_raw_chunk(void);
static void *rpool_make_raw_chunk(void);

inline static chunk_t *gpool_acquire_chunk(void);
inline static chunk_t *rpool_acquire_chunk(void);
inline static void gpool_release_chunk(dchunk_t *dc);

static lheap_t *gpool_acquire_lheap(void);
static lheap_t *r_acquire_lheap(void);
static void gpool_release_lheap(lheap_t *lh);
static void rpool_release_lheap(lheap_t *lh);


/* Local heap management functions */
inline static void lheap_init(lheap_t *lh);
inline static void lheap_replace_foreground(lheap_t *lh, int size_cls);
inline static int r_lheap_replace_foreground(lheap_t *lh,int size_cls);

/* Data chunk management functions */
inline static void dchunk_change_cls(dchunk_t *dc, int size_cls);
inline static void dchunk_init(dchunk_t *dc, int size_cls);
inline static void dchunk_collect_garbage(dchunk_t *dc);
inline static void *dchunk_alloc_obj(dchunk_t *dc);
inline static dchunk_t* dchunk_extract(void *ptr);

/* Object buffer management functions */
inline static void obj_buf_flush(obj_buf_t *obuf);
inline static void obj_buf_flush_all(lheap_t *lh);
inline static void obj_buf_put(obj_buf_t *bbuf, dchunk_t * dc, void *ptr);

/* Allocator helpers */
inline static void *r_large_malloc(size_t size);
inline static void *r_small_malloc(int size_cls);
inline static void *large_malloc(size_t size);
inline static void *small_malloc(int size_cls);
inline static void large_free(void *ptr);
//inline static void *r_large_free(void *ptr);
inline static void local_free(lheap_t *lh, dchunk_t *dc, void *ptr);
inline static void remote_free(lheap_t *lh, dchunk_t *dc, void *ptr);
static void *large_memalign(size_t boundary, size_t size);

/* Misc functions */
static void* page_alloc(void *pos, size_t size);
static void page_free(void *pos, size_t size);
static void touch_memory_range(void *start, size_t len); 
inline static int size2cls(size_t size);
inline static int r_size2cls(size_t size);

#ifdef DEBUG
static void handler(int sig);
#endif

/* Interface */
void *malloc(size_t size);
void free(void* ptr);
void *realloc(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);
void *memalign(size_t boundary, size_t size);
int posix_memalign(void **memptr, size_t alignment, size_t size);
void *valloc(size_t size);
void *pvalloc(size_t size);

#ifdef RETURN_MEMORY
pthread_t gpool_gc_thread;

static void* gpool_gc(void* arg)
{
  pthread_detach(pthread_self());
  char *ptr = NULL;
  
  /* sleeptime = 100 ms */
  struct timespec sleeptime = {0, 10000000};
  
  while(1) {
    nanosleep(&sleeptime, NULL);
    ptr = (char*) queue_fetch(&global_pool.free_dc_head[get_core_id()]);
    if(ptr) {
      void *ptr_end = PAGE_ROUNDDOWN(ptr + CHUNK_SIZE);
      void *ptr_start = PAGE_ROUNDUP(ptr);
      madvise(ptr_start, (uintptr_t)ptr_end - (uintptr_t)ptr_start, MADV_DONTNEED); 
      queue_put(&global_pool.released_dc_head[get_core_id()], ptr);
    }
  }
}
#endif

static void maps_init()
{
    int size;
    int class;

    /* 8 +4 64 */
    for (size = 8, class = 0; size <= 64; size += 4, class++) {
        cls2size[class] = size;
    }

    /* 80 +16 128 */
    for (size = 64 + 16; size <= 128; size += 16, class++) {
        cls2size[class] = size;
    }

    /* 160 +32 256 */
    for (size = 128 + 32; size <= 256; size += 32, class++) {
        cls2size[class] = size;
    }

    for (size = 256; size < 65536; size <<= 1) {
        cls2size[class++] = size + (size >> 1);
        cls2size[class++] = size << 1;
    }

    int cur_class = 0;
    int cur_size = 0;

    /* init sizemap */
    for (cur_size = 4; cur_size <= 1024; cur_size += 4) {
        if (cur_size > cls2size[cur_class])
            cur_class++;
        sizemap[(cur_size - 1) >> 2] = cur_class;
    }
    
    /* init sizemap2 */
    for (cur_size = 1024; cur_size <= 65536; cur_size += 512) {
        if (cur_size > cls2size[cur_class])
            cur_class++;
        sizemap2[(cur_size - 1) >> 9] = cur_class;
    }
}

static void thread_init()
{
    /* Register the destructor */
  pthread_setspecific(destructor, ACTIVE);
  /* Initialize thread pool */
  local_heap = gpool_acquire_lheap();
  thread_state = READY;
}

static void r_thread_exit (void *dummy) {
  rpool_release_lheap(r_local_heap);
}
static void thread_exit(void *dummy)
{
  gpool_release_lheap(local_heap);
}


uint64_t RInit(char *buffer,uint64_t size) {


  pthread_key_create(&r_destructor, r_thread_exit);

  /* Rounding to chunk size */
  uint64_t add_off = CHUNK_SIZE - ((uint64_t )buffer) % CHUNK_SIZE;
  if(add_off >= size)
    return 0;
  size -= add_off;
  if(size < 16 * CHUNK_SIZE) {
    /* We shall ensure the register rdma area is large enough! */
    return 0;
  }
  
  register_rdma_buffer = buffer + add_off;
  register_rdma_buffer_size = size;
  
  r_global_pool.pool_start = register_rdma_buffer;
  r_global_pool.pool_end   = register_rdma_buffer + register_rdma_buffer_size;
  r_global_pool.free_start = register_rdma_buffer;

  pthread_mutex_init(&r_global_pool.lock, NULL);

  {
    /* maps init */

    int size;
    int class;
    
    /* 8 +4 64 */
    for (size = 8, class = 0; size <= 64; size += 4, class++) {
      r_cls2size[class] = size;
    }
    
    /* 80 +16 128 */
    for (size = 64 + 16; size <= 128; size += 16, class++) {
      r_cls2size[class] = size;
    }
    
    /* 160 +32 256 */
    for (size = 128 + 32; size <= 256; size += 32, class++) {
      r_cls2size[class] = size;
    }
    
    for (size = 256; size < 65536; size <<= 1) {
      r_cls2size[class++] = size + (size >> 1);
      r_cls2size[class++] = size << 1;
    }
    
    int cur_class = 0;
    int cur_size = 0;
    
    /* init sizemap */
    for (cur_size = 4; cur_size <= 1024; cur_size += 4) {
      if (cur_size > r_cls2size[cur_class])
	cur_class++;
      r_sizemap[(cur_size - 1) >> 2] = cur_class;
    }
    
    /* init sizemap2 */
    for (cur_size = 1024; cur_size <= 65536; cur_size += 512) {
      if (cur_size > r_cls2size[cur_class])
	cur_class++;
      r_sizemap2[(cur_size - 1) >> 9] = cur_class;
    }    
  }
  r_global_state = READY;
  return size;
}

void RThreadLocalInit () {
  
  if(unlikely(r_thread_state != READY)) {
    pthread_setspecific(r_destructor, ACTIVE);
    //    r_local_heap = gp
    r_local_heap = r_acquire_lheap();
    r_thread_state = READY;
  }
}

static void global_init()
{
#ifdef DEBUG
    /* Register the signal handler for backtrace*/
    signal(SIGSEGV, handler);
#endif
    pthread_key_create(&destructor, thread_exit);
    /* Initialize global data */
    gpool_init();
    maps_init();

    global_state = READY;
#ifdef RETURN_MEMORY
    /* Create the gc thread */
    pthread_create(&gpool_gc_thread, NULL, gpool_gc, NULL);
#endif
}

inline static void check_init()
{
  if (unlikely(thread_state != READY)) {
    if (global_state != READY) {
      pthread_once(&init_once, global_init);
    }
    thread_init();
  }
}

inline static int rpool_check_size(void *target) {
  if(r_global_pool.pool_end <= (char *)target) {
    return 0 ;//false
  }
  return 1; //true
}

inline static void gpool_check_size(void *target)
{
  if (global_pool.pool_end <= (char *)target) {
    /* Global Pool Full */
    pthread_mutex_lock(&global_pool.lock);
    while (global_pool.pool_end <= (char *)target) {
      gpool_grow();
    }
    pthread_mutex_unlock(&global_pool.lock);
  }
}

static int gpool_grow()
{
    /* Enlarge the raw memory pool */
    static int last_alloc = 8;
    int alloc_size = ALLOC_UNIT * last_alloc;
    if (last_alloc < 32) {
        last_alloc *= 2;
    }

    void *mem = page_alloc((void *)global_pool.pool_end, alloc_size);
    if (mem == MAP_FAILED) {
        exit(-1);
        return -1;
    }

    /* Increase the global pool size */
    global_pool.pool_end += alloc_size;
    return 0;
}



/* Initialize the global memory pool */
static void gpool_init()
{
    global_pool.pool_start = RAW_POOL_START;
    global_pool.pool_end = RAW_POOL_START;
    global_pool.free_start = RAW_POOL_START;
    //queue_init(&global_pool.free_dc_head);
    pthread_mutex_init(&global_pool.lock, NULL);
    gpool_grow();
}


inline static chunk_t *gpool_acquire_chunk()
{
  void *ptr = NULL;
  
  /* Try to alloc a freed chunk from the free list */
  ptr = queue_fetch(&global_pool.free_dc_head[get_core_id()]);
  if (ptr) {
    return (chunk_t *) ptr;
  }
  
#ifdef RETURN_MEMORY
  ptr = queue_fetch(&global_pool.released_dc_head[get_core_id()]);
  if (ptr) {
    // XXX: Fix me
    ((chunk_t *) ptr)->numa_node = get_core_id();
    touch_memory_range(ptr, CHUNK_SIZE);
    return (chunk_t *) ptr;
  }
#endif
  
  /* Or just alloc a new chunk */
  ptr = gpool_make_raw_chunk();
  gpool_check_size(ptr);
  ptr -= CHUNK_SIZE;
  ((chunk_t *) ptr)->numa_node = get_core_id();
  touch_memory_range(ptr, CHUNK_SIZE);
  return (chunk_t *) ptr;
}


inline static chunk_t *rpool_acquire_chunk()
{
  void *ptr = NULL;
  
  /* Try to alloc a freed chunk from the free list */
  ptr = queue_fetch(&r_global_pool.free_dc_head[get_core_id()]);
  if (ptr) {
    return (chunk_t *) ptr;
  }
  
#ifdef RETURN_MEMORY
  ptr = queue_fetch(&r_global_pool.released_dc_head[get_core_id()]);
  if (ptr) {
    // XXX: Fix me
    ((chunk_t *) ptr)->numa_node = get_core_id();
    touch_memory_range(ptr, CHUNK_SIZE);
    return (chunk_t *) ptr;
  }
#endif
  
  /* Or just alloc a new chunk */
  ptr = rpool_make_raw_chunk();
  if((char *)ptr > (char *)r_global_pool.pool_end) {
    return NULL;
  }
    
  //  rpool_check_size(ptr);
  ptr -= CHUNK_SIZE;
  ((chunk_t *) ptr)->numa_node = get_core_id();
  touch_memory_range(ptr, CHUNK_SIZE);
  return (chunk_t *) ptr;
}


static void *rpool_make_raw_chunk() {
  void *ret = (void *)(atmc_fetch_and_add64((unsigned long long *)
					     &r_global_pool.free_start,
					    CHUNK_SIZE)) ;
  return ret;
}

static void *gpool_make_raw_chunk()
{
  /* Atomic increse the global pool size */
  void *ret = (void *)(atmc_fetch_and_add64((unsigned long long *)
					    &global_pool.free_start,
					    CHUNK_SIZE));
  return ret;
}
  
  
inline static void gpool_release_chunk(dchunk_t *dc)
{
  queue_put(&global_pool.free_dc_head[dc->numa_node], dc);
}

inline static void rpool_release_chunk(dchunk_t *dc) {
  queue_put(&r_global_pool.free_dc_head[dc->numa_node], dc);
}


static lheap_t *r_acquire_lheap() {
  lheap_t *lh;
  lh = queue_fetch(&(r_global_pool.free_lh_head[get_core_id()]));
  /* Alloc a new one */
  if (!lh) {
    lh = (lheap_t *) rpool_acquire_chunk();
    if(lh == NULL) {
      fprintf(stderr,"panic, cannot acquire local heap\n");
      return NULL;
    }
    lheap_init(lh);
  }
  return lh;  
}



static lheap_t *gpool_acquire_lheap()
{
  lheap_t *lh;
  lh = queue_fetch(&(global_pool.free_lh_head[get_core_id()]));
  /* Alloc a new one */
  if (!lh) {
    lh = (lheap_t *) gpool_acquire_chunk();
    lheap_init(lh);
  }
  return lh;  
}

static void gpool_release_lheap(lheap_t *lh)
{
  queue_put(&global_pool.free_lh_head[local_heap->numa_node], lh);
}

static void rpool_release_lheap(lheap_t *lh)
{
  queue_put(&r_global_pool.free_lh_head[local_heap->numa_node], lh);
}


inline static void lheap_init(lheap_t * lh)
{
  memset(&lh->free_head, 0, sizeof(lheap_t));

  int size_cls;
  lh->dummy_chunk.size_cls = DUMMY_CLASS;
  lh->dummy_chunk.free_blk_cnt = 1;
  
  for (size_cls = 0; size_cls < DEFAULT_BLOCK_CLASS; size_cls++) {
    /* Install the dummy chunk */
    lh->foreground[size_cls] = &lh->dummy_chunk;
  }
}

inline static int r_lheap_replace_foreground
(lheap_t * lh, int size_cls) {

  dchunk_t *dc;
  
  /* Try to acquire the block from background list */
  dc = (dchunk_t *) lh->background[size_cls].head;
  if (dc != NULL) {
    double_list_remove(dc, &lh->background[size_cls]);
    goto finish;
  }
  
  /* Try to acquire a block in the remote freed list */
  dc = fast_queue_fetch(&lh->need_gc[size_cls]);
  if (dc != NULL) {
    dchunk_collect_garbage(dc);
    goto finish;
  }
  
  /* Try to acquire the chunk from local pool */
  dc = (dchunk_t *) seq_queue_fetch(&lh->free_head);
  if (dc != NULL) {
    //    fprintf(stdout,"get free head\n");
    lh->free_cnt--;
    dchunk_change_cls(dc, size_cls);
    goto finish;
  }
  
  /* Acquire the chunk from global pool */
  
  dc = (dchunk_t *) rpool_acquire_chunk();
  //  fprintf(stdout,"acquire raw pool\n");
  if(unlikely(dc == NULL)) {
    return 0; // false
  }
  //  fprintf(stdout,"owner %p\n",lh);
  dc->owner = lh;
  fast_queue_init((FastQueue *) & (dc->remote_free_head));
  dchunk_init(dc, size_cls);
  
 finish:
  /* Set the foreground chunk */
  lh->foreground[size_cls] = dc;
  dc->state = FOREGROUND;
  return 1; // true
}


inline static void lheap_replace_foreground
(lheap_t * lh, int size_cls) {
  dchunk_t *dc;

  /* Try to acquire the block from background list */
  dc = (dchunk_t *) lh->background[size_cls].head;
  if (dc != NULL) {
    double_list_remove(dc, &lh->background[size_cls]);
    goto finish;
  }
  
  /* Try to acquire a block in the remote freed list */
  dc = fast_queue_fetch(&lh->need_gc[size_cls]);
  if (dc != NULL) {
    dchunk_collect_garbage(dc);
    goto finish;
  }
  
  /* Try to acquire the chunk from local pool */
  dc = (dchunk_t *) seq_queue_fetch(&lh->free_head);
  if (dc != NULL) {
    lh->free_cnt--;
    dchunk_change_cls(dc, size_cls);
    goto finish;
  }
  
  /* Acquire the chunk from global pool */
  
  dc = (dchunk_t *) gpool_acquire_chunk();
  dc->owner = lh;
  fast_queue_init((FastQueue *) & (dc->remote_free_head));
  dchunk_init(dc, size_cls);
  
 finish:
  /* Set the foreground chunk */
  lh->foreground[size_cls] = dc;
  dc->state = FOREGROUND;
}

inline static void dchunk_change_cls(dchunk_t * dc, int size_cls)
{
    int size = cls2size[size_cls];
    int data_offset = DCH;
    dc->blk_cnt = (CHUNK_SIZE - data_offset) / size;
    dc->free_blk_cnt = dc->blk_cnt;
    dc->block_size = size;
    dc->free_mem = (char *)dc + data_offset;
    dc->size_cls = size_cls;
    seq_queue_init(&dc->free_head);
}

inline static void dchunk_init(dchunk_t * dc, int size_cls)
{
    dc->active_link.next = NULL;
    dc->active_link.prev = NULL;
    dchunk_change_cls(dc, size_cls);
}

inline static void dchunk_collect_garbage(dchunk_t * dc)
{
    seq_head(dc->free_head) =
        counted_chain_dequeue(&dc->remote_free_head, &dc->free_blk_cnt);
}

inline static void *dchunk_alloc_obj(dchunk_t * dc)
{
    void *ret;

    /* Dirty implementation of dequeue, avoid one branch */
    ret = seq_head(dc->free_head);

    if (unlikely(!ret)) {
        ret = dc->free_mem;
        dc->free_mem += dc->block_size;
    } else {
        seq_head(dc->free_head) = *(void**)ret;
    }

#if 0
    /* A clearer implementation with one more branch*/
    ret = seq_lifo_dequeue(&dc->free_head);
    if (unlikely(!ret)) {
        ret = dc->free_mem;
        dc->free_mem += dc->block_size;
    }
#endif

    return ret;
}

inline static dchunk_t *dchunk_extract(void *ptr)
{
    return (dchunk_t *) ((uintptr_t)ptr - ((uintptr_t)ptr % CHUNK_SIZE));
}

inline static void obj_buf_flush(obj_buf_t * bbuf)
{
    void *prev;

    dchunk_t *dc = bbuf->dc;
    lheap_t *lh = dc->owner;

    prev = counted_chain_enqueue(&(dc->remote_free_head),
                                 seq_head(bbuf->free_head), bbuf->first, bbuf->count);
    bbuf->count = 0;
    bbuf->dc = NULL;
    bbuf->first = NULL;
    seq_head(bbuf->free_head) = NULL;

    /* If I am the first thread done remote free in this memory chunk*/
    if ((unsigned long long)prev == 0L) {
      fast_queue_put(&(lh->need_gc[dc->size_cls]), dc);
    }
    return;
}

inline static void obj_buf_flush_all(lheap_t *lh) {
    int i;
    for (i = 0; i < BLOCK_BUF_CNT; i++) {
        obj_buf_t *buf = &lh->block_bufs[i];
        if (buf->count == 0)
            continue;
        obj_buf_flush(buf);
        buf->dc = NULL;
    }
}

inline static void obj_buf_put(obj_buf_t *bbuf, dchunk_t * dc, void *ptr) {
    if (unlikely(bbuf->dc != dc)) {
        if (bbuf->dc != NULL) {
            obj_buf_flush(bbuf);
        }
        bbuf->dc = dc;
        bbuf->first = ptr;
        bbuf->count = 0;
        seq_head(bbuf->free_head) = NULL;
    }

    seq_queue_put(&bbuf->free_head, ptr);
    bbuf->count++;
}

inline static void *r_large_malloc(size_t size) {

  /* round up the size */
  size_t real_size = size + CHUNK_SIZE - size % CHUNK_SIZE;
  void *ret = (void *)(atmc_fetch_and_add64((unsigned long long *)
					     &r_global_pool.free_start,
					    real_size)) ;
  if(ret > (void *)(r_global_pool.pool_end))
    return NULL;
  //void *mem_start = (char *)ret + CHUNK_SIZE - CACHE_LINE_SIZE;
  large_header_t *header = (large_header_t *)dchunk_extract(ret);
  
  header->alloc_size = real_size;
  header->mem = ret;
  header->owner = LARGE_OWNER;
  
  return ret;
}

inline static void *large_malloc(size_t size)
{
    size_t alloc_size = PAGE_ROUNDUP(size + CHUNK_SIZE);
    void *mem = page_alloc(NULL, alloc_size);
    void *mem_start = (char*)mem + CHUNK_SIZE - CACHE_LINE_SIZE;
    large_header_t *header = (large_header_t *)dchunk_extract(mem_start);

    /* If space is enough for the header of a large block */
    intptr_t distance = (intptr_t)mem_start - (intptr_t)header;
    if (distance >= sizeof(large_header_t)) {
        header->alloc_size = alloc_size;
        header->mem = mem;
        header->owner = LARGE_OWNER;
        return mem_start;
    }

    /* If not, Retry Allocation */
    void *ret = large_malloc(size);
    page_free(mem, alloc_size);
    return ret;
}

inline static void *r_small_malloc(int size_cls) {
  
  lheap_t *lh = r_local_heap;
  dchunk_t *dc;
  void *ret;
 retry:
  dc = lh->foreground[size_cls];
  ret = dchunk_alloc_obj(dc);
  //  fprintf(stdout,"alloc owner %p\n",dc->owner);
  /* Check if the datachunk is full */
  if (unlikely(--dc->free_blk_cnt == 0)) {
    dc->state = FULL;
    /* There is not enough memory in RDMA region */
    if(unlikely(r_lheap_replace_foreground(lh, size_cls) == 0))
      return NULL;
    if (unlikely(dc->size_cls == DUMMY_CLASS)) {
      /* A dummy chunk */
      dc->free_blk_cnt = 1;
      goto retry;
    }
  }
  
  return ret;  
}

inline static void *small_malloc(int size_cls)
{
    lheap_t *lh = local_heap;
    dchunk_t *dc;
    void *ret;
  retry:
    dc = lh->foreground[size_cls];
    ret = dchunk_alloc_obj(dc);

    /* Check if the datachunk is full */
    if (unlikely(--dc->free_blk_cnt == 0)) {
        dc->state = FULL;
        lheap_replace_foreground(lh, size_cls);
        if (unlikely(dc->size_cls == DUMMY_CLASS)) {
            /* A dummy chunk */
            dc->free_blk_cnt = 1;
            goto retry;
        }
    }

    return ret;
}

inline static void large_free(void *ptr)
{
    large_header_t *header = (large_header_t*)dchunk_extract(ptr);
    page_free(header->mem, header->alloc_size);
}

inline static void local_free(lheap_t * lh, dchunk_t * dc, void *ptr)
{
    unsigned int free_blk_cnt = ++dc->free_blk_cnt;
    seq_queue_put(&dc->free_head, ptr);

    switch (dc->state) {
    case FULL:
        double_list_insert_front(dc, &lh->background[dc->size_cls]);
        dc->state = BACKGROUND;
        break;
    case BACKGROUND:
        if (unlikely(free_blk_cnt == dc->blk_cnt)) {
            int free_cnt = lh->free_cnt;
            double_list_remove(dc, &lh->background[dc->size_cls]);

            if (free_cnt >= MAX_FREE_CHUNK) {
                gpool_release_chunk(dc);
            } else {
                seq_queue_put(&lh->free_head, dc);
                lh->free_cnt = free_cnt + 1;
            }
        }
        break;
    case FOREGROUND:
        /* Tada.. */
        break;
    }
}

THREAD_LOCAL int buf_cnt;
inline static void remote_free(lheap_t * lh, dchunk_t * dc, void *ptr)
{
    /* Put the object in a local buffer rather than return it to owner */
  int tag = ((unsigned long long)dc / CHUNK_SIZE) % BLOCK_BUF_CNT;
  obj_buf_t *bbuf = &lh->block_bufs[tag];
  obj_buf_put(bbuf, dc, ptr);
  
    /* Periodically flush buffered remote objects */
  if ((buf_cnt++ & 0xFFFF) == 0) {
    obj_buf_flush_all(lh);
  }
}

static void touch_memory_range(void *addr, size_t len)
{
    char *ptr = (char *)addr;
    char *end = ptr + len;

    for (; ptr < end; ptr += PAGE_SIZE) {
        *ptr = 0;
    }
}

static void *large_memalign(size_t boundary, size_t size) {
    /* Alloc a large enough memory block */
    size_t padding = boundary + CHUNK_SIZE;
    size_t alloc_size = PAGE_ROUNDUP(size + padding);
    void *mem = page_alloc(NULL, alloc_size);

    /* Align up the address to boundary */
    void *mem_start = 
        (void*)((uintptr_t)((char*)mem + padding) & ~(boundary - 1));

    /* Extract space for an header */
    large_header_t *header = 
        (large_header_t *)dchunk_extract(mem_start);

    /* If space is enough for the header of a large block */
    intptr_t distance = (intptr_t)mem_start - (intptr_t)header;
    if (distance >= sizeof(large_header_t)) {
        header->alloc_size = alloc_size;
        header->mem = mem;
        header->owner = LARGE_OWNER;
        return mem_start;
    }

    /* If not, retry allocation */
    void *ret = NULL;

    /* Avoid infinite loop if application call memalign(CHUNK_SIZE,size),
     * althrough it is actually illegal
     */
    if (boundary % CHUNK_SIZE != 0) {
        ret = large_memalign(boundary, size);
    }
    page_free(mem, alloc_size);
    return ret;
}

#ifdef DEBUG
/* Signal handler for debugging use */
static void handler(int sig)
{
    void *array[10];
    size_t size;

    /* get void*'s for all entries on the stack */
    size = backtrace(array, 10);

    /* print out all the frames to stderr */
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, 2);
    exit(1);
}
#endif

static void *page_alloc(void *pos, size_t size)
{
    return mmap(pos,
                size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
}

static void page_free(void *pos, size_t size)
{   
    munmap(pos, size);
}


inline static int size2cls(size_t size)
{
    int ret;
    if (likely(size <= 1024)) {
        ret = sizemap[(size - 1) >> 2];
    } else if (size <= 65536) {
        ret = sizemap2[(size - 1) >> 9];
    } else {
        ret = LARGE_CLASS;
    }
    return ret;
}

inline static int r_size2cls(size_t size) {
  int ret;
  if (likely(size <= 1024)) {
    ret = r_sizemap[(size - 1) >> 2];
  } else if (size <= 65536) {
    ret = r_sizemap2[(size - 1) >> 9];
  } else {
    ret = LARGE_CLASS;
  }
  return ret;  
}

void *malloc(size_t size)
{
    void *ret = NULL;

    /* Initialize the allocator */
    check_init();

    /* Deal with zero-size allocation */
    size += (size == 0);

#if 0
    /* The expression above is equivalent to the code below */
    if (unlikely(size == 0)) {
        size = 1;
    }
#endif

    int size_cls = size2cls(size);
    if (likely(size_cls < DEFAULT_BLOCK_CLASS)) {
      ret = small_malloc(size_cls);
      //      return NULL;
    } else {
      ret = large_malloc(size);
    }
    if(unlikely(ret == NULL))
      assert(0);
    return ret;
}

void *Rmalloc(size_t size) {
  void *ret = NULL;
  
  /* Deal with zero-size allocation */
  size += (size == 0);
  
  int size_cls = r_size2cls(size);
  if (likely(size_cls < DEFAULT_BLOCK_CLASS)) {
    ret = r_small_malloc(size_cls);
  } else {
    ret = r_large_malloc(size);
  }
  return ret;  
}


void free(void *ptr)
{
    if(ptr == NULL) {
        return;
    }

    dchunk_t *dc = dchunk_extract(ptr);
    lheap_t *lh = local_heap;
    lheap_t *target_lh = dc->owner;

    if (likely(target_lh == lh)) {
        local_free(lh, dc, ptr);
    } else if(likely(target_lh != LARGE_OWNER)){
        check_init();
        lh = local_heap;
        remote_free(lh, dc, ptr);
    } else {
        large_free(ptr);
    }
}

void Rfree(void *ptr) {
  
  if(ptr == NULL) {
    return;
  }
  
  dchunk_t *dc = dchunk_extract(ptr);
  lheap_t *lh = r_local_heap;
  lheap_t *target_lh = dc->owner;
  //  fprintf(stdout,"check owner %p\n",target_lh);
  
  if (likely(target_lh == lh)) {
    local_free(lh, dc, ptr);
  } else if(likely(target_lh != LARGE_OWNER)) {
    //    check_init();
    lh = r_local_heap;
    remote_free(lh, dc, ptr);
  } else {    
    //    large_free(ptr);
  }
}


void *realloc(void* ptr, size_t size)
{
    /* Handle special cases */
    if (ptr == NULL) {
        void *ret = malloc(size);
        return ret;
    }

    if (size == 0) {
        free(ptr);
    }

    dchunk_t *dc = dchunk_extract(ptr);
    if (dc->owner != LARGE_OWNER) {
        int old_size = cls2size[dc->size_cls];

        /* Not exceed the current size, return */
        if (size <= old_size) {
            return ptr;
        }

        /* Alloc a new block */
        void *new_ptr = malloc(size);
        memcpy(new_ptr, ptr, old_size);
        free(ptr);
        return new_ptr;
    } else {
        large_header_t *header = (large_header_t *)dc;
        size_t alloc_size = header->alloc_size;
        void* mem = header->mem;
        size_t offset = (uintptr_t)ptr - (uintptr_t)mem;
        size_t old_size = alloc_size - offset;

        /* Not exceed the current size, return */
        if(size <= old_size) {
            return ptr;
        }
        
        /* Try to do mremap */
        int new_size = PAGE_ROUNDUP(size + CHUNK_SIZE);
        mem = mremap(mem, alloc_size, new_size, MREMAP_MAYMOVE);
        void* mem_start = (void*)((uintptr_t)mem + offset);
        header = (large_header_t*)dchunk_extract(mem_start);

        intptr_t distance = (intptr_t)mem_start - (intptr_t)header;
        if (distance >= sizeof(large_header_t)) {
            header->alloc_size = new_size;
            header->mem = mem;
            header->owner = LARGE_OWNER;
            return mem_start;
        }

        void* new_ptr = large_malloc(size);
        memcpy(new_ptr, mem_start, old_size);
        free(mem);
        return new_ptr; 
    }
}

void * __attribute__((optimize("O0"))) calloc(size_t nmemb, size_t size)
{
  void *ptr;
    size_t m_size = nmemb * size;
    //    ptr = malloc(nmemb * size);
    ptr = malloc(m_size);
    if (!ptr) {
      //      assert(0);
        return NULL;
    }
    return memset(ptr, 0, nmemb * size);
}

void *memalign(size_t boundary, size_t size) {
    /* Deal with zero-size allocation */
    size += (size == 0);
    if(boundary <= 256 && size <= 65536) {
        /* In this case, we handle it as small allocations */
        int boundary_cls = size2cls(boundary);
        int size_cls = size2cls(size);
        int alloc_cls = max(boundary_cls, size_cls);
        return small_malloc(alloc_cls);
    } else {
        /* Handle it as a special large allocation */
        return large_memalign(boundary, size);
    }
}

int posix_memalign(void **memptr, size_t alignment, size_t size)
{
    *memptr = memalign(alignment, size);
    if (*memptr) {
        return 0;
    } else {
        /* We have to "personalize" the return value according to the error */
        return -1;
    }
}

void *valloc(size_t size)
{
    return memalign(PAGE_SIZE, size);
}

void *pvalloc(size_t size)
{
    fprintf(stderr, "pvalloc() called. Not implemented! Exiting.\n");
    exit(1);
}
