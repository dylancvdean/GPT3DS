/*
 * test_host.c
 *
 * Compile and run on the host to sanity-check the C inference engine.
 *
 *   gcc -O2 -Isource -o test_host \
 *       test_host.c source/model.c source/tokenizer.c source/matmul.c -lm
 *   ./test_host
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model.h"
#include "tokenizer.h"

/* Mock romfs paths as local file paths */
#define WEIGHTS_PATH "romfs/model_weights.bin"
#define TOKENIZER_PATH "romfs/tokenizer_qwends.cbin"

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;

    printf("GPT3DS Host Test\n");
    printf("----------------\n\n");

    ModelCtx model;
    memset(&model, 0, sizeof(model));

    printf("Loading model weights...\n");
    if (model_load(&model, WEIGHTS_PATH) != 0) {
        printf("FAILED to load model\n");
        return 1;
    }
    printf("OK.  cache_len=%d\n", model.cache_len);

    printf("Loading tokenizer...\n");
    Tokenizer* tokenizer = tokenizer_load(TOKENIZER_PATH);
    if (!tokenizer) {
        printf("FAILED to load tokenizer\n");
        model_free(&model);
        return 1;
    }
    printf("OK.  bos=%d eos=%d\n", tokenizer_bos_id(tokenizer), tokenizer_eos_id(tokenizer));

    /* Test encode/decode round-trip */
    const char* test_text = "Hello world";
    int tokens[128];
    int n = tokenizer_encode(tokenizer, test_text, tokens, 128, 1);
    printf("\nEncode '%s' -> %d tokens: ", test_text, n);
    for (int i = 0; i < n && i < 10; i++) printf("%d ", tokens[i]);
    if (n > 10) printf("...");
    printf("\n");

    char decoded[256];
    tokenizer_decode(tokenizer, tokens, n, decoded, sizeof(decoded));
    printf("Decode -> '%s'\n", decoded);

    /* Test a tiny forward pass (single token) */
    printf("\nRunning forward pass on single token...\n");
    float logits[MODEL_VOCAB_SIZE];
    model.cache_len = 0;
    model_forward(&model, tokens, 1, logits, 1);
    if (model.cache_len != 1) {
        printf("FAILED: cache_len after single token = %d\n", model.cache_len);
        return 1;
    }

    /* Find top 5 logits */
    struct { float val; int idx; } top[5];
    for (int k = 0; k < 5; k++) { top[k].val = -1e30f; top[k].idx = -1; }
    for (int i = 0; i < MODEL_VOCAB_SIZE; i++) {
        float v = logits[i];
        for (int k = 0; k < 5; k++) {
            if (v > top[k].val) {
                for (int j = 4; j > k; j--) top[j] = top[j-1];
                top[k].val = v;
                top[k].idx = i;
                break;
            }
        }
    }
    printf("Top 5 next tokens:\n");
    for (int k = 0; k < 5; k++) {
        char tok_str[64];
        int single_tok[1] = { top[k].idx };
        tokenizer_decode(tokenizer, single_tok, 1, tok_str, sizeof(tok_str));
        printf("  %d (%.3f): '%s'\n", top[k].idx, top[k].val, tok_str);
    }

    printf("\nChecking cached multi-turn prefill append...\n");
    int cache_before = model.cache_len;
    model_forward(&model, tokens + 1, n - 1, logits, 1);
    if (model.cache_len != cache_before + n - 1) {
        printf("FAILED: cache_len after append = %d, expected %d\n",
               model.cache_len, cache_before + n - 1);
        return 1;
    }
    printf("OK.  cache_len=%d\n", model.cache_len);

    /* Test a short generation (stochastic) */
    printf("\nGenerating 10 tokens from prompt '%s'...\n", test_text);
    model_seed_rng((uint32_t)time(NULL));
    int out_tok[32];
    int n_gen = model_generate(&model, tokens, n, out_tok, 10, 0.9f, 40, tokenizer_eos_id(tokenizer));
    char gen_text[512];
    tokenizer_decode(tokenizer, out_tok, n_gen, gen_text, sizeof(gen_text));
    printf("Generated (%d tokens): '%s'\n", n_gen, gen_text);

    /* Test deterministic greedy generation */
    printf("\nGreedy generation from prompt '%s'...\n", test_text);
    int out_tok2[32];
    int n_gen2 = model_generate(&model, tokens, n, out_tok2, 10, 0.0f, 0, tokenizer_eos_id(tokenizer));
    char gen_text2[512];
    tokenizer_decode(tokenizer, out_tok2, n_gen2, gen_text2, sizeof(gen_text2));
    printf("Greedy (%d tokens): '%s'\n", n_gen2, gen_text2);

    printf("\nAll host tests passed.\n");

    tokenizer_free(tokenizer);
    model_free(&model);
    return 0;
}
