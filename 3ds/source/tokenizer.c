#include "tokenizer.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* ByteLevel mapping (HuggingFace tokenizers)                         */

static void build_bytelevel_maps(uint32_t* b2u, uint8_t* u2b, int u2b_size) {
    memset(u2b, 0, u2b_size);
    int bs[256];
    int cs[256];
    int n = 0;
    for (int i = 33;  i <= 126; i++) bs[n] = i, cs[n] = i, n++;
    for (int i = 161; i <= 172; i++) bs[n] = i, cs[n] = i, n++;
    for (int i = 174; i <= 255; i++) bs[n] = i, cs[n] = i, n++;
    int extra = 0;
    for (int b = 0; b < 256; b++) {
        int found = 0;
        for (int j = 0; j < n; j++) { if (bs[j] == b) { found = 1; break; } }
        if (!found) {
            bs[n] = b;
            cs[n] = 256 + extra;
            extra++;
            n++;
        }
    }
    for (int i = 0; i < 256; i++) {
        b2u[bs[i]] = (uint32_t)cs[i];
        if (cs[i] < u2b_size) u2b[cs[i]] = (uint8_t)bs[i];
    }
}

/* ------------------------------------------------------------------ */
/* UTF-8 helpers                                                      */

static int utf8_encode(uint32_t cp, uint8_t* out) {
    if (cp <= 0x7f) {
        out[0] = (uint8_t)cp;
        return 1;
    } else if (cp <= 0x7ff) {
        out[0] = (uint8_t)(0xc0 | (cp >> 6));
        out[1] = (uint8_t)(0x80 | (cp & 0x3f));
        return 2;
    } else {
        out[0] = (uint8_t)(0xe0 | (cp >> 12));
        out[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3f));
        out[2] = (uint8_t)(0x80 | (cp & 0x3f));
        return 3;
    }
}

static uint32_t utf8_decode(const uint8_t* s, int* consumed) {
    if (s[0] <= 0x7f) { *consumed = 1; return s[0]; }
    if ((s[0] & 0xe0) == 0xc0) { *consumed = 2; return ((s[0] & 0x1f) << 6) | (s[1] & 0x3f); }
    *consumed = 3;
    return ((s[0] & 0x0f) << 12) | ((s[1] & 0x3f) << 6) | (s[2] & 0x3f);
}

/* ------------------------------------------------------------------ */
/* FNV-1a string hash                                                 */

static uint32_t strhash(const uint8_t* s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= s[i];
        h *= 16777619u;
    }
    return h;
}

/* ------------------------------------------------------------------ */
/* Hash table sizes                                                   */

#define VOCAB_HASH_BITS 14
#define VOCAB_HASH_SIZE (1 << VOCAB_HASH_BITS)
#define VOCAB_HASH_MASK (VOCAB_HASH_SIZE - 1)

#define MERGE_HASH_BITS 14
#define MERGE_HASH_SIZE (1 << MERGE_HASH_BITS)
#define MERGE_HASH_MASK (MERGE_HASH_SIZE - 1)

/* ------------------------------------------------------------------ */
/* Tokenizer struct                                                   */

typedef struct {
    uint32_t hash;
    int      token_id;
    int      len;
    uint8_t* bytes;  /* points into tokenizer heap */
} VocabEntry;

typedef struct {
    int left_id;
    int right_id;
    int rank;
    int merged_id;
    int occupied;
} MergeEntry;

struct Tokenizer {
    int vocab_size;
    int merge_count;
    int bos_id;
    int eos_id;
    int pad_id;
    int unk_id;

    uint8_t* heap;
    size_t   heap_size;

    uint8_t** token_bytes;
    int*      token_len;
    VocabEntry vocab_hash[VOCAB_HASH_SIZE];

    MergeEntry merge_hash[MERGE_HASH_SIZE];
};

/* ------------------------------------------------------------------ */
/* Vocab hash ops                                                     */

static void vocab_insert(Tokenizer* t, const uint8_t* bytes, int len, int token_id) {
    uint32_t h = strhash(bytes, len);
    uint32_t idx = h & VOCAB_HASH_MASK;
    while (t->vocab_hash[idx].token_id >= 0) {
        idx = (idx + 1) & VOCAB_HASH_MASK;
    }
    t->vocab_hash[idx].hash = h;
    t->vocab_hash[idx].token_id = token_id;
    t->vocab_hash[idx].len = len;
    t->vocab_hash[idx].bytes = (uint8_t*)bytes;
}

static int vocab_lookup(const Tokenizer* t, const uint8_t* bytes, int len) {
    uint32_t h = strhash(bytes, len);
    uint32_t idx = h & VOCAB_HASH_MASK;
    while (t->vocab_hash[idx].token_id >= 0) {
        if (t->vocab_hash[idx].hash == h &&
            t->vocab_hash[idx].len == len &&
            memcmp(t->vocab_hash[idx].bytes, bytes, len) == 0) {
            return t->vocab_hash[idx].token_id;
        }
        idx = (idx + 1) & VOCAB_HASH_MASK;
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* Merge hash ops                                                     */

static void merge_insert(Tokenizer* t, int left, int right, int rank, int merged) {
    uint32_t key = ((uint32_t)left << 16) | (uint32_t)(unsigned short)right;
    uint32_t idx = key & MERGE_HASH_MASK;
    while (t->merge_hash[idx].occupied) {
        idx = (idx + 1) & MERGE_HASH_MASK;
    }
    t->merge_hash[idx].occupied = 1;
    t->merge_hash[idx].left_id = left;
    t->merge_hash[idx].right_id = right;
    t->merge_hash[idx].rank = rank;
    t->merge_hash[idx].merged_id = merged;
}

static int merge_lookup(const Tokenizer* t, int left, int right, int* out_rank, int* out_merged) {
    uint32_t key = ((uint32_t)left << 16) | (uint32_t)(unsigned short)right;
    uint32_t idx = key & MERGE_HASH_MASK;
    while (t->merge_hash[idx].occupied) {
        if (t->merge_hash[idx].left_id == left && t->merge_hash[idx].right_id == right) {
            *out_rank = t->merge_hash[idx].rank;
            *out_merged = t->merge_hash[idx].merged_id;
            return 1;
        }
        idx = (idx + 1) & MERGE_HASH_MASK;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Loading                                                            */

Tokenizer* tokenizer_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("tokenizer_load: failed to open %s\n", path); return NULL; }

    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "QDSBPE1\0", 8) != 0) {
        printf("tokenizer_load: bad magic\n"); fclose(f); return NULL;
    }

    Tokenizer* t = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    if (!t) { fclose(f); return NULL; }

    uint32_t header[6];
    if (fread(header, 4, 6, f) != 6) { free(t); fclose(f); return NULL; }
    t->vocab_size  = (int)header[0];
    t->merge_count = (int)header[1];
    t->bos_id = (int)header[2];
    t->eos_id = (int)header[3];
    t->pad_id = (int)header[4];
    t->unk_id = (int)header[5];

    for (int i = 0; i < VOCAB_HASH_SIZE; i++) t->vocab_hash[i].token_id = -1;

    /* compute total heap size for vocab strings */
    long vocab_start = ftell(f);
    size_t total_str_bytes = 0;
    for (int i = 0; i < t->vocab_size; i++) {
        uint32_t id, blen;
        if (fread(&id, 4, 1, f) != 1) break;
        if (fread(&blen, 4, 1, f) != 1) break;
        fseek(f, blen, SEEK_CUR);
        total_str_bytes += blen;
    }

    t->heap = (uint8_t*)malloc(total_str_bytes);
    t->token_bytes = (uint8_t**)calloc(t->vocab_size, sizeof(uint8_t*));
    t->token_len   = (int*)calloc(t->vocab_size, sizeof(int));
    if (!t->heap || !t->token_bytes || !t->token_len) {
        free(t->heap); free(t->token_bytes); free(t->token_len); free(t); fclose(f);
        return NULL;
    }

    /* read vocab strings into heap */
    fseek(f, vocab_start, SEEK_SET);
    size_t heap_off = 0;
    for (int i = 0; i < t->vocab_size; i++) {
        uint32_t id, blen;
        fread(&id, 4, 1, f);
        fread(&blen, 4, 1, f);
        uint8_t* dest = t->heap + heap_off;
        fread(dest, 1, blen, f);
        if (id < (uint32_t)t->vocab_size) {
            t->token_bytes[id] = dest;
            t->token_len[id] = (int)blen;
            vocab_insert(t, dest, (int)blen, (int)id);
        }
        heap_off += blen;
    }
    t->heap_size = heap_off;

    /* --- read merges --- */
    for (int r = 0; r < t->merge_count; r++) {
        uint32_t rank, llen, rlen;
        if (fread(&rank, 4, 1, f) != 1) break;
        if (fread(&llen, 4, 1, f) != 1) break;
        uint8_t* left_str = (uint8_t*)malloc(llen);
        fread(left_str, 1, llen, f);
        if (fread(&rlen, 4, 1, f) != 1) { free(left_str); break; }
        uint8_t* right_str = (uint8_t*)malloc(rlen);
        fread(right_str, 1, rlen, f);

        int left_id = vocab_lookup(t, left_str, (int)llen);
        int right_id = vocab_lookup(t, right_str, (int)rlen);
        if (left_id >= 0 && right_id >= 0) {
            int merged_len = (int)(llen + rlen);
            uint8_t* merged_str = (uint8_t*)malloc(merged_len);
            memcpy(merged_str, left_str, llen);
            memcpy(merged_str + llen, right_str, rlen);
            int merged_id = vocab_lookup(t, merged_str, merged_len);
            if (merged_id >= 0) {
                merge_insert(t, left_id, right_id, (int)rank, merged_id);
            }
            free(merged_str);
        }
        free(left_str);
        free(right_str);
    }

    fclose(f);
    return t;
}

void tokenizer_free(Tokenizer* t) {
    if (!t) return;
    free(t->heap);
    free(t->token_bytes);
    free(t->token_len);
    free(t);
}

/* ------------------------------------------------------------------ */
/* Encoding                                                           */

int tokenizer_encode(Tokenizer* t, const char* text,
                     int* tokens, int max_tokens, int add_bos) {
    if (!t || !text || max_tokens <= 0) return 0;

    uint32_t b2u[256];
    uint8_t  u2b[512];
    build_bytelevel_maps(b2u, u2b, sizeof(u2b));

    int init[512];
    int n = 0;
    if (add_bos) init[n++] = t->bos_id;

    const uint8_t* bytes = (const uint8_t*)text;
    while (*bytes && n < 500) {
        uint8_t utf8[4];
        int ulen = utf8_encode(b2u[*bytes], utf8);
        int tid = vocab_lookup(t, utf8, ulen);
        if (tid < 0) tid = t->unk_id;
        init[n++] = tid;
        bytes++;
    }

    /* BPE merges */
    while (1) {
        int best_rank = 2147483647;
        int best_idx = -1;
        int best_merged = -1;
        for (int i = 0; i < n - 1; i++) {
            int rank, merged;
            if (merge_lookup(t, init[i], init[i+1], &rank, &merged)) {
                if (rank < best_rank) {
                    best_rank = rank;
                    best_idx = i;
                    best_merged = merged;
                }
            }
        }
        if (best_idx < 0) break;
        init[best_idx] = best_merged;
        for (int i = best_idx + 1; i < n - 1; i++) init[i] = init[i + 1];
        n--;
    }

    int out_n = n < max_tokens ? n : max_tokens;
    for (int i = 0; i < out_n; i++) tokens[i] = init[i];
    return out_n;
}

/* ------------------------------------------------------------------ */
/* Decoding                                                           */

int tokenizer_decode(Tokenizer* t, const int* tokens, int n_tokens,
                     char* out, int out_size) {
    if (!t || !out || out_size <= 0) return 0;

    uint32_t b2u[256];
    uint8_t  u2b[512];
    build_bytelevel_maps(b2u, u2b, sizeof(u2b));

    int pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        int tid = tokens[i];
        if (tid < 0 || tid >= t->vocab_size) continue;
        const uint8_t* str = t->token_bytes[tid];
        int len = t->token_len[tid];
        int off = 0;
        while (off < len) {
            int consumed;
            uint32_t cp = utf8_decode(str + off, &consumed);
            off += consumed;
            if (cp < 512 && pos < out_size - 1) {
                out[pos++] = (char)u2b[cp];
            }
        }
    }
    out[pos] = '\0';
    return pos;
}

int tokenizer_bos_id(const Tokenizer* t) { return t ? t->bos_id : 0; }
int tokenizer_eos_id(const Tokenizer* t) { return t ? t->eos_id : 1; }
