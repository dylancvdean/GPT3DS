#include "matmul.h"
#include "common.h"
#include "model_desc.h"

#ifdef __3DS__
#include <3ds.h>
#endif

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

#ifndef GPT3DS_EXACT_Q8_MATMUL
static int quantize_input_q8(int K, const float* input,
                             int8_t* input_q, float* dequant) {
    if (K > MODEL_MLP_HIDDEN) return -1;

    float max_abs = 0.0f;
    for (int j = 0; j < K; j++) {
        float a = fabsf(input[j]);
        if (a > max_abs) max_abs = a;
    }

    if (max_abs <= 1e-12f) {
        *dequant = 0.0f;
        memset(input_q, 0, (size_t)K);
        return 0;
    }

    float q_scale = 127.0f / max_abs;
    *dequant = max_abs / 127.0f;
    for (int j = 0; j < K; j++) {
        float x = input[j] * q_scale;
        int q = (int)(x + (x >= 0.0f ? 0.5f : -0.5f));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        input_q[j] = (int8_t)q;
    }
    return 0;
}

static inline int32_t dot_q8_row(const int8_t* w_row,
                                 const int8_t* input_q, int K) {
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
    return sum;
}

typedef struct {
    int outputs;
    int M;
    int K;
    int row_start;
    int row_end;
    const int8_t* weight[3];
    const float* scales[3];
    float* output[3];
    const int8_t* input_q;
    float dequant;
} MatmulTask;

static void matmul_task_run(const MatmulTask* task) {
    int M = task->M;
    int K = task->K;
    int row_end = task->row_end <= M ? task->row_end : M;

    if (task->dequant == 0.0f) {
        for (int out = 0; out < task->outputs; out++) {
            memset(task->output[out] + task->row_start, 0,
                   (size_t)(row_end - task->row_start) * sizeof(float));
        }
        return;
    }

    for (int i = task->row_start; i < row_end; i++) {
        for (int out = 0; out < task->outputs; out++) {
            const int8_t* w_row = task->weight[out] + i * K;
            int32_t sum = dot_q8_row(w_row, task->input_q, K);
            task->output[out][i] =
                (float)sum * (task->scales[out][i] * task->dequant);
        }
    }
}

#ifdef __3DS__
#define MATMUL_WORKER_STACK_SIZE (16 * 1024)
#define MATMUL_THREAD_MIN_OPS    (256 * 512)

static struct {
    Thread thread;
    LightEvent start_event;
    LightEvent done_event;
    volatile int stop;
    int init_attempted;
    MatmulTask task;
} g_worker;

static void matmul_worker_entry(void* arg) {
    (void)arg;
    while (1) {
        LightEvent_Wait(&g_worker.start_event);
        if (g_worker.stop) break;
        matmul_task_run(&g_worker.task);
        LightEvent_Signal(&g_worker.done_event);
    }
}

static int matmul_worker_init(void) {
    if (g_worker.thread) return 1;
    if (g_worker.init_attempted) return 0;
    g_worker.init_attempted = 1;
    g_worker.stop = 0;

    LightEvent_Init(&g_worker.start_event, RESET_ONESHOT);
    LightEvent_Init(&g_worker.done_event, RESET_ONESHOT);

    s32 prio = 0x30;
    svcGetThreadPriority(&prio, CUR_THREAD_HANDLE);
    if (prio < 0x18) prio = 0x18;
    if (prio > 0x3f) prio = 0x3f;

    g_worker.thread = threadCreate(matmul_worker_entry, NULL,
                                   MATMUL_WORKER_STACK_SIZE,
                                   prio, 2, false);
    return g_worker.thread != NULL;
}

static void matmul_run(const MatmulTask* task) {
    int ops = task->M * task->K * task->outputs;
    if (ops >= MATMUL_THREAD_MIN_OPS && task->M >= 64 && matmul_worker_init()) {
        int split = task->M / 2;
        MatmulTask main_task = *task;
        main_task.row_start = 0;
        main_task.row_end = split;

        g_worker.task = *task;
        g_worker.task.row_start = split;
        g_worker.task.row_end = task->M;
        __dmb();
        LightEvent_Signal(&g_worker.start_event);

        matmul_task_run(&main_task);
        LightEvent_Wait(&g_worker.done_event);
    } else {
        matmul_task_run(task);
    }
}

void matmul_shutdown_workers(void) {
    if (!g_worker.thread) return;
    g_worker.stop = 1;
    __dmb();
    LightEvent_Signal(&g_worker.start_event);
    threadJoin(g_worker.thread, U64_MAX);
    threadFree(g_worker.thread);
    g_worker.thread = NULL;
    g_worker.init_attempted = 0;
    g_worker.stop = 0;
}
#else
static void matmul_run(const MatmulTask* task) {
    matmul_task_run(task);
}

void matmul_shutdown_workers(void) {}
#endif
#else
void matmul_shutdown_workers(void) {}
#endif

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
    float dequant;
    quantize_input_q8(K, input, input_q, &dequant);

    MatmulTask task = {
        .outputs = 1,
        .M = M,
        .K = K,
        .row_start = 0,
        .row_end = M,
        .weight = { weight, NULL, NULL },
        .scales = { scales, NULL, NULL },
        .output = { output, NULL, NULL },
        .input_q = input_q,
        .dequant = dequant,
    };
    matmul_run(&task);
#endif
}

void matmul_q8_fp32_fused3(int M, int K,
                           const int8_t* weight0, const float* scales0, float* output0,
                           const int8_t* weight1, const float* scales1, float* output1,
                           const int8_t* weight2, const float* scales2, float* output2,
                           const float* input) {
#ifdef GPT3DS_EXACT_Q8_MATMUL
    matmul_q8_fp32_exact(M, K, weight0, scales0, input, output0);
    matmul_q8_fp32_exact(M, K, weight1, scales1, input, output1);
    matmul_q8_fp32_exact(M, K, weight2, scales2, input, output2);
#else
    if (K > MODEL_MLP_HIDDEN) {
        matmul_q8_fp32_exact(M, K, weight0, scales0, input, output0);
        matmul_q8_fp32_exact(M, K, weight1, scales1, input, output1);
        matmul_q8_fp32_exact(M, K, weight2, scales2, input, output2);
        return;
    }

    int8_t input_q[MODEL_MLP_HIDDEN];
    float dequant;
    quantize_input_q8(K, input, input_q, &dequant);

    MatmulTask task = {
        .outputs = 3,
        .M = M,
        .K = K,
        .row_start = 0,
        .row_end = M,
        .weight = { weight0, weight1, weight2 },
        .scales = { scales0, scales1, scales2 },
        .output = { output0, output1, output2 },
        .input_q = input_q,
        .dequant = dequant,
    };
    matmul_run(&task);
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
