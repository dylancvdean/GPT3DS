#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* rsqrtf may already be provided by the platform math library */
#ifndef HAVE_RSQRTF
static inline float rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}
#endif

static inline float maxf(float a, float b) {
    return a > b ? a : b;
}

static inline int maxi(int a, int b) {
    return a > b ? a : b;
}

static inline int mini(int a, int b) {
    return a < b ? a : b;
}

#endif
