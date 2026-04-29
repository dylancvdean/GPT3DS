#ifndef TOKENIZER_H
#define TOKENIZER_H

/*
 * Minimal byte-level BPE tokenizer runtime.
 * Loads the .cbin format exported by train.py.
 */

typedef struct Tokenizer Tokenizer;

Tokenizer* tokenizer_load(const char* path);
void       tokenizer_free(Tokenizer* t);

/*
 * Encode text to token IDs.
 * Returns the number of tokens written (capped at max_tokens).
 * Adds BOS at the beginning if add_bos is true.
 */
int tokenizer_encode(Tokenizer* t, const char* text,
                     int* tokens, int max_tokens, int add_bos);

/*
 * Decode token IDs to text.
 * The output buffer must be large enough; out_size is the buffer size.
 * Returns the number of bytes written (excluding null terminator).
 */
int tokenizer_decode(Tokenizer* t, const int* tokens, int n_tokens,
                     char* out, int out_size);

int tokenizer_bos_id(const Tokenizer* t);
int tokenizer_eos_id(const Tokenizer* t);

#endif
