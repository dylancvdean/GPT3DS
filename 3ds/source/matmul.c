#include "matmul.h"
#include "common.h"

/*
 * Quantized int8 matmul with per-row fp32 scales.
 * weight: [M, K] int8
 * scales: [M] fp32
 * input:  [K] fp32
 * output: [M] fp32
 * output[i] = scales[i] * sum_j(weight[i,j] * input[j])
 */
void matmul_q8_fp32(int M, int K, const int8_t* weight, const float* scales,
                    const float* input, float* output) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        const int8_t* w_row = weight + i * K;
        for (int j = 0; j < K; j++) {
            sum += (float)w_row[j] * input[j];
        }
        output[i] = sum * scales[i];
    }
}

void matmul_fp16_fp32(int M, int K, const uint16_t* weight,
                      const float* input, float* output) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        const uint16_t* w_row = weight + i * K;
        for (int j = 0; j < K; j++) {
            sum += fp16_to_f32(w_row[j]) * input[j];
        }
        output[i] = sum;
    }
}

void matmul_fp32_fp32(int M, int K, const float* weight,
                      const float* input, float* output) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        const float* w_row = weight + i * K;
        for (int j = 0; j < K; j++) {
            sum += w_row[j] * input[j];
        }
        output[i] = sum;
    }
}
