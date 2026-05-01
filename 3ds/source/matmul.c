#include "matmul.h"
#include "common.h"
#include "model_desc.h"

static void matmul_q8_fp32_exact(int M, int K, const int8_t* weight,
                                 const float* scales, const float* input,
                                 float* output) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        const int8_t* w_row = weight + i * K;
        for (int j = 0; j < K; j++) {
            sum += (float)w_row[j] * input[j];
        }
        output[i] = sum * scales[i];
    }
}

/*
 * Quantized int8 matmul with per-row fp32 scales.
 * weight: [M, K] int8
 * scales: [M] fp32
 * input:  [K] fp32, dynamically quantized to int8 unless
 *         GPT3DS_EXACT_Q8_MATMUL is defined
 * output: [M] fp32
 * output[i] = scales[i] * sum_j(weight[i,j] * input[j])
 */
void matmul_q8_fp32(int M, int K, const int8_t* weight, const float* scales,
                    const float* input, float* output) {
#ifdef GPT3DS_EXACT_Q8_MATMUL
    matmul_q8_fp32_exact(M, K, weight, scales, input, output);
#else
    if (K > MODEL_MLP_HIDDEN) {
        matmul_q8_fp32_exact(M, K, weight, scales, input, output);
        return;
    }

    int8_t input_q[MODEL_MLP_HIDDEN];
    float max_abs = 0.0f;

    for (int j = 0; j < K; j++) {
        float a = fabsf(input[j]);
        if (a > max_abs) max_abs = a;
    }

    if (max_abs <= 1e-12f) {
        memset(output, 0, (size_t)M * sizeof(float));
        return;
    }

    float q_scale = 127.0f / max_abs;
    float dequant = max_abs / 127.0f;
    for (int j = 0; j < K; j++) {
        float x = input[j] * q_scale;
        int q = (int)(x + (x >= 0.0f ? 0.5f : -0.5f));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        input_q[j] = (int8_t)q;
    }

    for (int i = 0; i < M; i++) {
        const int8_t* w_row = weight + i * K;
        int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int j = 0;
        for (; j + 15 < K; j += 16) {
            s0 += (int32_t)w_row[j +  0] * input_q[j +  0];
            s1 += (int32_t)w_row[j +  1] * input_q[j +  1];
            s2 += (int32_t)w_row[j +  2] * input_q[j +  2];
            s3 += (int32_t)w_row[j +  3] * input_q[j +  3];
            s0 += (int32_t)w_row[j +  4] * input_q[j +  4];
            s1 += (int32_t)w_row[j +  5] * input_q[j +  5];
            s2 += (int32_t)w_row[j +  6] * input_q[j +  6];
            s3 += (int32_t)w_row[j +  7] * input_q[j +  7];
            s0 += (int32_t)w_row[j +  8] * input_q[j +  8];
            s1 += (int32_t)w_row[j +  9] * input_q[j +  9];
            s2 += (int32_t)w_row[j + 10] * input_q[j + 10];
            s3 += (int32_t)w_row[j + 11] * input_q[j + 11];
            s0 += (int32_t)w_row[j + 12] * input_q[j + 12];
            s1 += (int32_t)w_row[j + 13] * input_q[j + 13];
            s2 += (int32_t)w_row[j + 14] * input_q[j + 14];
            s3 += (int32_t)w_row[j + 15] * input_q[j + 15];
        }
        int32_t sum = s0 + s1 + s2 + s3;
        for (; j < K; j++) {
            sum += (int32_t)w_row[j] * input_q[j];
        }
        output[i] = (float)sum * (scales[i] * dequant);
    }
#endif
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
