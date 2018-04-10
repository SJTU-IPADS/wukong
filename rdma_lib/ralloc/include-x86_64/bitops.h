#ifndef __X86_64_BITOPS_H_
#define __X86_64_BITOPS_H_

/*
 * Copyright 1992, Linus Torvalds.
 */

#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 1)
/* Technically wrong, but this avoids compilation errors on some gcc
   versions. */
#define ADDR "=m" (*(volatile long *) addr)
#else
#define ADDR "+m" (*(volatile long *) addr)
#endif

/**
 * __change_bit - Toggle a bit in memory
 * @nr: the bit to change
 * @addr: the address to start counting from
 *
 * Unlike change_bit(), this function is non-atomic and may be reordered.
 * If it's called on the same region of memory simultaneously, the effect
 * may be that only one operation succeeds.
 */
static __inline__ void __change_bit(int nr, volatile void * addr)
{
	__asm__ __volatile__(
		"btcl %1,%0"
		:ADDR
		:"dIr" (nr));
}

/* WARNING: non atomic and it can be reordered! */
static __inline__ int __test_and_change_bit(int nr, volatile void * addr)
{
	int oldbit;

	__asm__ __volatile__(
		"btcl %2,%1\n\tsbbl %0,%0"
		:"=r" (oldbit),ADDR
		:"dIr" (nr) : "memory");
	return oldbit;
}

static inline unsigned long __fls(unsigned long word)
{
	asm("bsr %1,%0"
	    : "=r" (word)
	    : "rm" (word));
	return word;
}

static __inline__ unsigned int __get_size_class(unsigned int word) {
	asm("dec %1\n"
        "shr $2,%1\n"
        "bsr %1,%0\n"
        "cmovz %2,%0\n"
	    : "=r" (word)
	    : "rm" (word), "r" (0));
	return word;    
}

#endif /* _X86_64_BITOPS_H */

