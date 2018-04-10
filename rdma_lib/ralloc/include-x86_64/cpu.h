#ifndef __CPU_H_
#define __CPU_H_

/* Machine related macros*/
#define PAGE_SIZE           (4096)
#define SUPER_PAGE_SIZE     (4*1024*1024)
#define CACHE_LINE_SIZE     (64)
#define DEFAULT_BLOCK_CLASS (100)
#define MAX_CORE_ID         (8)

static inline int get_core_id(void) {
    return 0;
    int result;
    __asm__ __volatile__ (
        "mov $1, %%eax\n"
        "cpuid\n"
        :"=b"(result)
        :
        :"eax","ecx","edx");
    return (result>>24)%8;
}

static inline unsigned long read_tsc(void)
{
    unsigned a, d;
    __asm __volatile("rdtsc":"=a"(a), "=d"(d));
    return ((unsigned long)a) | (((unsigned long) d) << 32);
}

#endif
