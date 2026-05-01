#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include "model_desc.h"

typedef struct {
    const float*    attn_norm_w; // [d_model]
    const float*    mlp_norm_w;  // [d_model]
    const int8_t*   q_w;         // [d_model, d_model] int8
    const float*    q_s;         // [d_model]
    const int8_t*   k_w;
    const float*    k_s;
    const int8_t*   v_w;
    const float*    v_s;
    const int8_t*   o_w;
    const float*    o_s;
    const int8_t*   up_w;        // [mlp_hidden, d_model] int8
    const float*    up_s;        // [mlp_hidden]
    const int8_t*   down_w;      // [d_model, mlp_hidden] int8
    const float*    down_s;      // [d_model]
} BlockWeights;

typedef struct {
    const uint16_t* token_emb;       // [vocab_size, d_model] fp16
    const uint16_t* pos_emb;         // [ctx_len, d_model] fp16
    const float*    embed_conv_w;    // [d_model, 3]
    const float*    embed_conv_b;    // [d_model]
    BlockWeights    blocks[MODEL_UNIQUE_BLOCKS];

    const float* final_norm_w;       // [d_model]
    const float* lm_head_bias;       // [vocab_size]
} ModelWeights;

typedef void (*ModelYieldCallback)(void* user);

typedef struct {
    ModelWeights weights;
    uint8_t*     weights_buf;        // owned, loaded from ROMFS
    int          cache_len;
    float*       kv_k;               // [n_layers, ctx_len, d_model]
    float*       kv_v;               // [n_layers, ctx_len, d_model]
    float*       embed_cache;        // [ctx_len, d_model] token + position embeddings before conv
    ModelYieldCallback yield_cb;
    void*        yield_user;
} ModelCtx;

int  model_load(ModelCtx* ctx, const char* weights_path);
void model_free(ModelCtx* ctx);
void model_set_yield_callback(ModelCtx* ctx, ModelYieldCallback cb, void* user);

/*
 * Forward pass.
 * tokens:   input token IDs
 * n_tokens: number of input tokens (1 <= n <= MODEL_CTX_LEN)
 * logits:   output buffer of size MODEL_VOCAB_SIZE, receives logits for last token
 * use_cache: if true, updates KV cache (set to false for pure eval without side effects)
 */
void model_forward(ModelCtx* ctx, const int* tokens, int n_tokens,
                   float* logits, int use_cache);

int model_sample_logits(float* logits, float temperature, int top_k);

/*
 * Prefill a prompt and then generate up to max_new_tokens.
 * prompt: input token IDs
 * prompt_len: length of prompt
 * output: buffer for generated tokens
 * max_new_tokens: max tokens to generate
 * temperature, top_k: sampling params
 * eos_id: stop token (set to -1 to ignore)
 * Returns number of generated tokens.
 */
int model_generate(ModelCtx* ctx, const int* prompt, int prompt_len,
                   int* output, int max_new_tokens,
                   float temperature, int top_k, int eos_id);

void model_seed_rng(uint32_t seed);

#endif
