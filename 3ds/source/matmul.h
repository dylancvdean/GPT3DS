#ifndef MATMUL_H
#define MATMUL_H

#include <stdint.h>

/* fp16 -> fp32 conversion for ARM11 without fp16 hardware */
static inline float fp16_to_f32(uint16_t h) {
    uint32_t h_exp = (h >> 10) & 0x1f;
    uint32_t h_mant = h & 0x3ff;
    uint32_t f32;

    if (h_exp == 0) {
        if (h_mant == 0) {
            f32 = (h & 0x8000) << 16;
        } else {
            int e = -1;
            uint32_t m = h_mant;
            do {
                m <<= 1;
                e++;
            } while ((m & 0x400) == 0);
            m &= 0x3ff;
            f32 = ((h & 0x8000) << 16) | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (h_exp == 31) {
        f32 = ((h & 0x8000) << 16) | (0xff << 23) | (h_mant << 13);
    } else {
        f32 = ((h & 0x8000) << 16) | ((h_exp + 127 - 15) << 23) | (h_mant << 13);
    }
    union { uint32_t u; float f; } conv = { .u = f32 };
    return conv.f;
}

void matmul_q8_fp32(int M, int K, const int8_t* weight, const float* scales,
                    const float* input, float* output);

void matmul_fp16_fp32(int M, int K, const uint16_t* weight,
                      const float* input, float* output);

void matmul_fp32_fp32(int M, int K, const float* weight,
                      const float* input, float* output);

#endif
