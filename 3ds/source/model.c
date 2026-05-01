#include "model.h"
#include "matmul.h"
#include "common.h"

#include <stdio.h>
#include <stdlib.h>

/*
 * Static scratch buffers sized for max context.
 * We use a few large buffers and reuse them across layers.
 */
#define MAX_SEQ MODEL_CTX_LEN

static float buf_x     [MAX_SEQ * MODEL_D_MODEL];
static float buf_norm  [MAX_SEQ * MODEL_D_MODEL];
static float buf_q     [MAX_SEQ * MODEL_D_MODEL];
static float buf_k     [MAX_SEQ * MODEL_D_MODEL];
static float buf_v     [MAX_SEQ * MODEL_D_MODEL];
static float buf_attn  [MAX_SEQ * MODEL_D_MODEL];
static float buf_mlp   [MAX_SEQ * MODEL_MLP_HIDDEN];

static float buf_logits[MODEL_VOCAB_SIZE];

static void softmax_inplace(float* values, int n) {
    float max_val = -1e30f;
    for (int i = 0; i < n; i++) {
        if (isfinite(values[i]) && values[i] > max_val) max_val = values[i];
    }
    if (!isfinite(max_val)) {
        float uniform = 1.0f / (float)n;
        for (int i = 0; i < n; i++) values[i] = uniform;
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = values[i] - max_val;
        if (!isfinite(x) || x < -80.0f) {
            values[i] = 0.0f;
        } else {
            values[i] = expf(x);
            if (!isfinite(values[i])) values[i] = 0.0f;
        }
        sum += values[i];
    }
    if (!(sum > 0.0f) || !isfinite(sum)) {
        float uniform = 1.0f / (float)n;
        for (int i = 0; i < n; i++) values[i] = uniform;
        return;
    }

    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) values[i] *= inv;
}

/* ------------------------------------------------------------------ */
/* RMSNorm                                                            */
static void rmsnorm(float* out, const float* in, const float* w, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += in[i] * in[i];
    float rms = rsqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = in[i] * rms * w[i];
}

/* ------------------------------------------------------------------ */
/* Causal depthwise conv k=3                                          */
static void embed_conv(float* out, const float* in, int T, int C,
                       const float* w, const float* b) {
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            float w0 = w[c * 3 + 0];
            float w1 = w[c * 3 + 1];
            float w2 = w[c * 3 + 2];
            sum += in[t * C + c] * w2;              /* current  */
            if (t >= 1) sum += in[(t-1) * C + c] * w1;
            if (t >= 2) sum += in[(t-2) * C + c] * w0;
            out[t * C + c] = sum;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Causal self-attention (prefill, multiple tokens)                   */
static void attn_prefill(float* out, const float* q, const float* k,
                         const float* v, int T) {
    float scale = 1.0f / sqrtf((float)MODEL_HEAD_DIM);
    int D = MODEL_D_MODEL;
    int H = MODEL_N_HEADS;
    int HD = MODEL_HEAD_DIM;

    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++) {
            float scores[MAX_SEQ];
            /* compute scores against all causal positions */
            const float* q_vec = q + t * D + h * HD;
            for (int s = 0; s <= t; s++) {
                float dot = 0.0f;
                const float* k_vec = k + s * D + h * HD;
                for (int d = 0; d < HD; d++) dot += q_vec[d] * k_vec[d];
                scores[s] = dot * scale;
            }
            softmax_inplace(scores, t + 1);
            /* weighted sum of values */
            float* o_vec = out + t * D + h * HD;
            for (int d = 0; d < HD; d++) o_vec[d] = 0.0f;
            for (int s = 0; s <= t; s++) {
                const float* v_vec = v + s * D + h * HD;
                float sc = scores[s];
                for (int d = 0; d < HD; d++) o_vec[d] += sc * v_vec[d];
            }
        }
    }
}

static void attn_prefill_cached(float* out, const float* q,
                                const float* cache_k, const float* cache_v,
                                int cache_pos, int T) {
    float scale = 1.0f / sqrtf((float)MODEL_HEAD_DIM);
    int D = MODEL_D_MODEL;
    int H = MODEL_N_HEADS;
    int HD = MODEL_HEAD_DIM;

    for (int t = 0; t < T; t++) {
        int total = cache_pos + t + 1;
        for (int h = 0; h < H; h++) {
            float scores[MAX_SEQ];
            const float* q_vec = q + t * D + h * HD;
            for (int s = 0; s < total; s++) {
                float dot = 0.0f;
                const float* k_vec = cache_k + s * D + h * HD;
                for (int d = 0; d < HD; d++) dot += q_vec[d] * k_vec[d];
                scores[s] = dot * scale;
            }
            softmax_inplace(scores, total);

            float* o_vec = out + t * D + h * HD;
            for (int d = 0; d < HD; d++) o_vec[d] = 0.0f;
            for (int s = 0; s < total; s++) {
                const float* v_vec = cache_v + s * D + h * HD;
                float sc = scores[s];
                for (int d = 0; d < HD; d++) o_vec[d] += sc * v_vec[d];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Causal self-attention (single token with KV cache)                 */
static void attn_generate(float* out, const float* q,
                          const float* cache_k, const float* cache_v,
                          int cache_len) {
    float scale = 1.0f / sqrtf((float)MODEL_HEAD_DIM);
    int total = cache_len + 1;  /* includes current token already stored */
    int D = MODEL_D_MODEL;
    int H = MODEL_N_HEADS;
    int HD = MODEL_HEAD_DIM;

    for (int h = 0; h < H; h++) {
        float scores[MAX_SEQ];
        const float* q_vec = q + h * HD;
        for (int s = 0; s < total; s++) {
            float dot = 0.0f;
            const float* k_vec = cache_k + s * D + h * HD;
            for (int d = 0; d < HD; d++) dot += q_vec[d] * k_vec[d];
            scores[s] = dot * scale;
        }
        softmax_inplace(scores, total);

        float* o_vec = out + h * HD;
        for (int d = 0; d < HD; d++) o_vec[d] = 0.0f;
        for (int s = 0; s < total; s++) {
            const float* v_vec = cache_v + s * D + h * HD;
            float sc = scores[s];
            for (int d = 0; d < HD; d++) o_vec[d] += sc * v_vec[d];
        }
    }
}

static void model_maybe_yield(ModelCtx* ctx) {
    if (ctx && ctx->yield_cb) ctx->yield_cb(ctx->yield_user);
}

static int build_lm_head_q8(ModelCtx* ctx) {
    size_t n_weights = (size_t)MODEL_VOCAB_SIZE * MODEL_D_MODEL;
    ctx->lm_head_w_q8 = (int8_t*)malloc(n_weights);
    ctx->lm_head_s = (float*)malloc((size_t)MODEL_VOCAB_SIZE * sizeof(float));
    if (!ctx->lm_head_w_q8 || !ctx->lm_head_s) {
        free(ctx->lm_head_w_q8);
        free(ctx->lm_head_s);
        ctx->lm_head_w_q8 = NULL;
        ctx->lm_head_s = NULL;
        return -1;
    }

    for (int i = 0; i < MODEL_VOCAB_SIZE; i++) {
        const uint16_t* src = ctx->weights.token_emb + i * MODEL_D_MODEL;
        int8_t* dst = ctx->lm_head_w_q8 + i * MODEL_D_MODEL;
        float max_abs = 0.0f;

        for (int d = 0; d < MODEL_D_MODEL; d++) {
            float a = fabsf(fp16_to_f32(src[d]));
            if (a > max_abs) max_abs = a;
        }

        if (max_abs <= 1e-12f) {
            ctx->lm_head_s[i] = 1.0f;
            memset(dst, 0, MODEL_D_MODEL);
            continue;
        }

        float inv_scale = 127.0f / max_abs;
        ctx->lm_head_s[i] = max_abs / 127.0f;
        for (int d = 0; d < MODEL_D_MODEL; d++) {
            float x = fp16_to_f32(src[d]) * inv_scale;
            int q = (int)(x + (x >= 0.0f ? 0.5f : -0.5f));
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            dst[d] = (int8_t)q;
        }
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Model loading                                                      */
static uint8_t* load_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("failed to open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc(sz);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if ((long)fread(buf, 1, sz, f) != sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_size = (size_t)sz;
    return buf;
}

int model_load(ModelCtx* ctx, const char* weights_path) {
    size_t sz;
    uint8_t* base = load_file(weights_path, &sz);
    if (!base) return -1;
    if (sz != MODEL_WEIGHTS_SIZE) {
        printf("weights size mismatch: expected %d, got %zu\n", MODEL_WEIGHTS_SIZE, sz);
        free(base);
        return -1;
    }
    ctx->weights_buf = base;

#define PTR(off, type) ((const type*)(base + (off)))

    ctx->weights.token_emb    = PTR(OFF_TOKEN_EMB, uint16_t);
    ctx->weights.pos_emb      = PTR(OFF_POS_EMB, uint16_t);
    ctx->weights.embed_conv_w = PTR(OFF_EMBED_CONV_WEIGHT, float);
    ctx->weights.embed_conv_b = PTR(OFF_EMBED_CONV_BIAS, float);

    /* Manual block pointer setup */
    ctx->weights.blocks[0].attn_norm_w = PTR(OFF_BLOCKS_0_ATTN_NORM_WEIGHT, float);
    ctx->weights.blocks[0].mlp_norm_w  = PTR(OFF_BLOCKS_0_MLP_NORM_WEIGHT, float);
    ctx->weights.blocks[0].q_w = PTR(OFF_BLOCKS_0_ATTN_Q_PROJ_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[0].q_s = PTR(OFF_BLOCKS_0_ATTN_Q_PROJ_SCALES, float);
    ctx->weights.blocks[0].k_w = PTR(OFF_BLOCKS_0_ATTN_K_PROJ_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[0].k_s = PTR(OFF_BLOCKS_0_ATTN_K_PROJ_SCALES, float);
    ctx->weights.blocks[0].v_w = PTR(OFF_BLOCKS_0_ATTN_V_PROJ_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[0].v_s = PTR(OFF_BLOCKS_0_ATTN_V_PROJ_SCALES, float);
    ctx->weights.blocks[0].o_w = PTR(OFF_BLOCKS_0_ATTN_O_PROJ_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[0].o_s = PTR(OFF_BLOCKS_0_ATTN_O_PROJ_SCALES, float);
    ctx->weights.blocks[0].up_w  = PTR(OFF_BLOCKS_0_MLP_UP_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[0].up_s  = PTR(OFF_BLOCKS_0_MLP_UP_SCALES, float);
    ctx->weights.blocks[0].down_w = PTR(OFF_BLOCKS_0_MLP_DOWN_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[0].down_s = PTR(OFF_BLOCKS_0_MLP_DOWN_SCALES, float);

    ctx->weights.blocks[1].attn_norm_w = PTR(OFF_BLOCKS_1_ATTN_NORM_WEIGHT, float);
    ctx->weights.blocks[1].mlp_norm_w  = PTR(OFF_BLOCKS_1_MLP_NORM_WEIGHT, float);
    ctx->weights.blocks[1].q_w = PTR(OFF_BLOCKS_1_ATTN_Q_PROJ_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[1].q_s = PTR(OFF_BLOCKS_1_ATTN_Q_PROJ_SCALES, float);
    ctx->weights.blocks[1].k_w = PTR(OFF_BLOCKS_1_ATTN_K_PROJ_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[1].k_s = PTR(OFF_BLOCKS_1_ATTN_K_PROJ_SCALES, float);
    ctx->weights.blocks[1].v_w = PTR(OFF_BLOCKS_1_ATTN_V_PROJ_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[1].v_s = PTR(OFF_BLOCKS_1_ATTN_V_PROJ_SCALES, float);
    ctx->weights.blocks[1].o_w = PTR(OFF_BLOCKS_1_ATTN_O_PROJ_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[1].o_s = PTR(OFF_BLOCKS_1_ATTN_O_PROJ_SCALES, float);
    ctx->weights.blocks[1].up_w  = PTR(OFF_BLOCKS_1_MLP_UP_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[1].up_s  = PTR(OFF_BLOCKS_1_MLP_UP_SCALES, float);
    ctx->weights.blocks[1].down_w = PTR(OFF_BLOCKS_1_MLP_DOWN_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[1].down_s = PTR(OFF_BLOCKS_1_MLP_DOWN_SCALES, float);

    ctx->weights.blocks[2].attn_norm_w = PTR(OFF_BLOCKS_2_ATTN_NORM_WEIGHT, float);
    ctx->weights.blocks[2].mlp_norm_w  = PTR(OFF_BLOCKS_2_MLP_NORM_WEIGHT, float);
    ctx->weights.blocks[2].q_w = PTR(OFF_BLOCKS_2_ATTN_Q_PROJ_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[2].q_s = PTR(OFF_BLOCKS_2_ATTN_Q_PROJ_SCALES, float);
    ctx->weights.blocks[2].k_w = PTR(OFF_BLOCKS_2_ATTN_K_PROJ_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[2].k_s = PTR(OFF_BLOCKS_2_ATTN_K_PROJ_SCALES, float);
    ctx->weights.blocks[2].v_w = PTR(OFF_BLOCKS_2_ATTN_V_PROJ_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[2].v_s = PTR(OFF_BLOCKS_2_ATTN_V_PROJ_SCALES, float);
    ctx->weights.blocks[2].o_w = PTR(OFF_BLOCKS_2_ATTN_O_PROJ_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[2].o_s = PTR(OFF_BLOCKS_2_ATTN_O_PROJ_SCALES, float);
    ctx->weights.blocks[2].up_w  = PTR(OFF_BLOCKS_2_MLP_UP_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[2].up_s  = PTR(OFF_BLOCKS_2_MLP_UP_SCALES, float);
    ctx->weights.blocks[2].down_w = PTR(OFF_BLOCKS_2_MLP_DOWN_WEIGHT_Q8, int8_t);
    ctx->weights.blocks[2].down_s = PTR(OFF_BLOCKS_2_MLP_DOWN_SCALES, float);

    ctx->weights.final_norm_w = PTR(OFF_FINAL_NORM_WEIGHT, float);
    ctx->weights.lm_head_bias = PTR(OFF_LM_HEAD_BIAS, float);
#undef PTR

    ctx->lm_head_w_q8 = NULL;
    ctx->lm_head_s = NULL;
    if (build_lm_head_q8(ctx) != 0) {
        free(base);
        ctx->weights_buf = NULL;
        return -1;
    }

    /* Allocate KV cache */
    size_t kv_size = (size_t)MODEL_N_LAYERS * MODEL_CTX_LEN * MODEL_D_MODEL * sizeof(float);
    size_t embed_size = (size_t)MODEL_CTX_LEN * MODEL_D_MODEL * sizeof(float);
    ctx->kv_k = (float*)malloc(kv_size);
    ctx->kv_v = (float*)malloc(kv_size);
    ctx->embed_cache = (float*)malloc(embed_size);
    if (!ctx->kv_k || !ctx->kv_v || !ctx->embed_cache) {
        free(ctx->kv_k);
        free(ctx->kv_v);
        free(ctx->embed_cache);
        free(ctx->lm_head_w_q8);
        free(ctx->lm_head_s);
        free(base);
        ctx->kv_k = NULL;
        ctx->kv_v = NULL;
        ctx->embed_cache = NULL;
        ctx->lm_head_w_q8 = NULL;
        ctx->lm_head_s = NULL;
        ctx->weights_buf = NULL;
        return -1;
    }
    ctx->cache_len = 0;
    ctx->yield_cb = NULL;
    ctx->yield_user = NULL;
    return 0;
}

void model_free(ModelCtx* ctx) {
    free(ctx->kv_k);
    free(ctx->kv_v);
    free(ctx->embed_cache);
    free(ctx->lm_head_w_q8);
    free(ctx->lm_head_s);
    free(ctx->weights_buf);
    ctx->kv_k = NULL;
    ctx->kv_v = NULL;
    ctx->embed_cache = NULL;
    ctx->lm_head_w_q8 = NULL;
    ctx->lm_head_s = NULL;
    ctx->weights_buf = NULL;
    ctx->yield_cb = NULL;
    ctx->yield_user = NULL;
}

void model_set_yield_callback(ModelCtx* ctx, ModelYieldCallback cb, void* user) {
    if (!ctx) return;
    ctx->yield_cb = cb;
    ctx->yield_user = user;
}

/* ------------------------------------------------------------------ */
/* Forward pass for a single sequence                                 */
void model_forward(ModelCtx* ctx, const int* tokens, int n_tokens,
                   float* logits, int use_cache) {
    int T = n_tokens;
    int D = MODEL_D_MODEL;
    if (!ctx || !tokens || !logits || T < 1 || T > MODEL_CTX_LEN) return;

    int cache_pos = use_cache ? ctx->cache_len : 0;
    if (use_cache && cache_pos + T > MODEL_CTX_LEN) return;

    /* ---- embeddings ---- */
    for (int t = 0; t < T; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= MODEL_VOCAB_SIZE) tok = MODEL_UNK_ID;
        int pos = use_cache ? cache_pos + t : t;
        const uint16_t* te = ctx->weights.token_emb + tok * D;
        const uint16_t* pe = ctx->weights.pos_emb + pos * D;
        float* x = buf_x + t * D;
        for (int d = 0; d < D; d++) {
            x[d] = fp16_to_f32(te[d]) + fp16_to_f32(pe[d]);
        }
        if (use_cache && ctx->embed_cache) {
            memcpy(ctx->embed_cache + pos * D, x, D * sizeof(float));
        }
    }

    /* ---- causal conv ---- */
    if (use_cache && ctx->embed_cache) {
        for (int t = 0; t < T; t++) {
            int pos = cache_pos + t;
            for (int c = 0; c < D; c++) {
                float sum = ctx->weights.embed_conv_b[c];
                const float* w = ctx->weights.embed_conv_w + c * 3;
                sum += ctx->embed_cache[pos * D + c] * w[2];
                if (pos >= 1) sum += ctx->embed_cache[(pos - 1) * D + c] * w[1];
                if (pos >= 2) sum += ctx->embed_cache[(pos - 2) * D + c] * w[0];
                buf_norm[t * D + c] = sum;
            }
        }
    } else {
        embed_conv(buf_norm, buf_x, T, D,
                   ctx->weights.embed_conv_w, ctx->weights.embed_conv_b);
    }
    memcpy(buf_x, buf_norm, T * D * sizeof(float));

    /* ---- transformer layers ---- */
    for (int l = 0; l < MODEL_N_LAYERS; l++) {
        int b = l / MODEL_LOOPS_PER_PASS;
        const BlockWeights* blk = &ctx->weights.blocks[b];

        /* pre-norm attention */
        for (int t = 0; t < T; t++) {
            rmsnorm(buf_norm + t * D, buf_x + t * D, blk->attn_norm_w, D);
        }

        /* QKV projections */
        for (int t = 0; t < T; t++) {
            matmul_q8_fp32(D, D, blk->q_w, blk->q_s, buf_norm + t * D, buf_q + t * D);
            matmul_q8_fp32(D, D, blk->k_w, blk->k_s, buf_norm + t * D, buf_k + t * D);
            matmul_q8_fp32(D, D, blk->v_w, blk->v_s, buf_norm + t * D, buf_v + t * D);
        }

        /* KV cache handling */
        if (use_cache) {
            float* cache_k = ctx->kv_k + l * MODEL_CTX_LEN * D;
            float* cache_v = ctx->kv_v + l * MODEL_CTX_LEN * D;
            if (T == 1) {
                /* generation step: store at cache_pos */
                if (cache_pos < MODEL_CTX_LEN) {
                    memcpy(cache_k + cache_pos * D, buf_k, D * sizeof(float));
                    memcpy(cache_v + cache_pos * D, buf_v, D * sizeof(float));
                }
            } else {
                /* prefill: store all T positions */
                if (cache_pos + T <= MODEL_CTX_LEN) {
                    memcpy(cache_k + cache_pos * D, buf_k,
                           T * D * sizeof(float));
                    memcpy(cache_v + cache_pos * D, buf_v,
                           T * D * sizeof(float));
                }
            }
        }

        /* attention */
        if (use_cache && T == 1) {
            float* cache_k = ctx->kv_k + l * MODEL_CTX_LEN * D;
            float* cache_v = ctx->kv_v + l * MODEL_CTX_LEN * D;
            attn_generate(buf_attn, buf_q, cache_k, cache_v, cache_pos);
        } else if (use_cache) {
            float* cache_k = ctx->kv_k + l * MODEL_CTX_LEN * D;
            float* cache_v = ctx->kv_v + l * MODEL_CTX_LEN * D;
            attn_prefill_cached(buf_attn, buf_q, cache_k, cache_v,
                                cache_pos, T);
        } else {
            attn_prefill(buf_attn, buf_q, buf_k, buf_v, T);
        }

        /* output projection */
        for (int t = 0; t < T; t++) {
            matmul_q8_fp32(D, D, blk->o_w, blk->o_s, buf_attn + t * D, buf_norm + t * D);
        }

        /* residual */
        for (int i = 0; i < T * D; i++) buf_x[i] += buf_norm[i];

        /* pre-norm MLP */
        for (int t = 0; t < T; t++) {
            rmsnorm(buf_norm + t * D, buf_x + t * D, blk->mlp_norm_w, D);
        }

        /* MLP up + ReLU */
        for (int t = 0; t < T; t++) {
            matmul_q8_fp32(MODEL_MLP_HIDDEN, D, blk->up_w, blk->up_s,
                           buf_norm + t * D, buf_mlp + t * MODEL_MLP_HIDDEN);
        }
        for (int i = 0; i < T * MODEL_MLP_HIDDEN; i++) {
            if (buf_mlp[i] < 0.0f) buf_mlp[i] = 0.0f;
        }

        /* MLP down */
        for (int t = 0; t < T; t++) {
            matmul_q8_fp32(D, MODEL_MLP_HIDDEN, blk->down_w, blk->down_s,
                           buf_mlp + t * MODEL_MLP_HIDDEN, buf_norm + t * D);
        }

        /* residual */
        for (int i = 0; i < T * D; i++) buf_x[i] += buf_norm[i];

        model_maybe_yield(ctx);
    }

    if (use_cache) {
        ctx->cache_len += T;
    }

    /* ---- final norm & logits ---- */
    float* last_x = buf_x + (T - 1) * D;
    rmsnorm(buf_norm, last_x, ctx->weights.final_norm_w, D);

    /* tied head: cached int8 copy of token_emb + bias */
    matmul_q8_fp32(MODEL_VOCAB_SIZE, D, ctx->lm_head_w_q8, ctx->lm_head_s,
                   buf_norm, logits);
    for (int i = 0; i < MODEL_VOCAB_SIZE; i++) {
        logits[i] += ctx->weights.lm_head_bias[i];
    }
}

/* ------------------------------------------------------------------ */
/* Simple LCG RNG                                                     */
static uint32_t rng_state = 1337;
static uint32_t rand_u32(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return rng_state;
}
static float rand_f32(void) {
    return (float)rand_u32() / 4294967296.0f;
}

void model_seed_rng(uint32_t seed) {
    rng_state = seed;
}

/* ------------------------------------------------------------------ */
/* Greedy / temperature / top-k sampling                              */
int model_sample_logits(float* logits, float temperature, int top_k) {
    int V = MODEL_VOCAB_SIZE;

    int greedy = 0;
    float greedy_val = logits[0];
    for (int i = 1; i < V; i++) {
        if (logits[i] > greedy_val) {
            greedy_val = logits[i];
            greedy = i;
        }
    }

    if (temperature <= 0.0f || !isfinite(greedy_val)) {
        return greedy;
    }

    float temp = temperature > 1e-6f ? temperature : 1e-6f;
    int k_limit = top_k > 0 && top_k < V ? top_k : V;
    if (k_limit > 64) k_limit = 64;

    float top_vals[64];
    int top_ids[64];
    int n_top = 0;

    for (int i = 0; i < V; i++) {
        float v = logits[i];
        if (!isfinite(v)) continue;

        int insert_at = n_top;
        while (insert_at > 0 && v > top_vals[insert_at - 1]) {
            insert_at--;
        }
        if (insert_at >= k_limit) continue;

        if (n_top < k_limit) n_top++;
        for (int j = n_top - 1; j > insert_at; j--) {
            top_vals[j] = top_vals[j - 1];
            top_ids[j] = top_ids[j - 1];
        }
        top_vals[insert_at] = v;
        top_ids[insert_at] = i;
    }

    if (n_top == 0) return greedy;

    float max_logit = top_vals[0];
    float sum = 0.0f;
    for (int i = 0; i < n_top; i++) {
        float x = (top_vals[i] - max_logit) / temp;
        if (x < -80.0f) {
            top_vals[i] = 0.0f;
        } else {
            top_vals[i] = expf(x);
            if (!isfinite(top_vals[i])) {
                top_vals[i] = 0.0f;
            }
        }
        sum += top_vals[i];
    }
    if (!(sum > 0.0f) || !isfinite(sum)) return greedy;

    /* sample */
    float r = rand_f32() * sum;
    float cdf = 0.0f;
    for (int i = 0; i < n_top; i++) {
        cdf += top_vals[i];
        if (r <= cdf) return top_ids[i];
    }
    return top_ids[n_top - 1];
}

int model_generate(ModelCtx* ctx, const int* prompt, int prompt_len,
                   int* output, int max_new_tokens,
                   float temperature, int top_k, int eos_id) {
    if (!ctx || !prompt || !output) return 0;
    if (prompt_len < 1 || prompt_len >= MODEL_CTX_LEN) return 0;
    if (max_new_tokens < 0) return 0;
    if (max_new_tokens > MODEL_CTX_LEN - prompt_len) {
        max_new_tokens = MODEL_CTX_LEN - prompt_len;
    }

    /* reset cache */
    ctx->cache_len = 0;

    /* prefill */
    model_forward(ctx, prompt, prompt_len, buf_logits, 1);

    int generated = 0;
    int prev_token = model_sample_logits(buf_logits, temperature, top_k);

    for (int i = 0; i < max_new_tokens; i++) {
        if (eos_id >= 0 && prev_token == eos_id) break;
        if (ctx->cache_len >= MODEL_CTX_LEN) break;
        output[generated++] = prev_token;

        int next_input = prev_token;
        model_forward(ctx, &next_input, 1, buf_logits, 1);
        prev_token = model_sample_logits(buf_logits, temperature, top_k);
    }
    return generated;
}
