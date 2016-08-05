#ifndef PLATFORM_H_
#define PLATFORM_H_

#undef __noinline
#undef __forceinline
#define __noinline             __attribute__((noinline))
#define __forceinline          inline __attribute__((always_inline))

#endif
