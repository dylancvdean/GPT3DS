#include <3ds.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "tokenizer.h"

#define CHAT_HISTORY_LINES 24
#define MAX_PROMPT_LEN 256
#define MAX_RESPONSE_LEN 1024

typedef struct {
    char user[MAX_PROMPT_LEN];
    char bot[MAX_RESPONSE_LEN];
} ChatTurn;

static ChatTurn history[16];
static int history_count = 0;
static float generation_logits[MODEL_VOCAB_SIZE];

static void copy_text(char* dst, size_t dst_size, const char* src) {
    if (dst_size == 0) return;
    snprintf(dst, dst_size, "%s", src);
}

static void draw_history(void) {
    consoleClear();
    printf("\x1b[0;0H");  /* move cursor to top-left */
    printf("=== GPT3DS ===\n");
    printf("Params: %d | ctx: %d\n\n", MODEL_TOTAL_PARAMS, MODEL_CTX_LEN);

    int start = history_count > CHAT_HISTORY_LINES ? history_count - CHAT_HISTORY_LINES : 0;
    for (int i = start; i < history_count; i++) {
        printf("You: %s\n", history[i].user);
        printf("Bot: %s\n\n", history[i].bot);
    }
    printf("A: Chat | START: Exit\n");
}

int main(int argc, char* argv[]) {
    gfxInitDefault();
    PrintConsole topScreen, bottomScreen;
    consoleInit(GFX_TOP, &topScreen);
    consoleInit(GFX_BOTTOM, &bottomScreen);
    consoleSelect(&topScreen);

    printf("\x1b[2;0H");
    printf("GPT3DS - Loading model...\n");

    Result rc = romfsInit();
    if (R_FAILED(rc)) {
        printf("romfsInit failed: %08lX\n", rc);
        printf("Press START to exit.\n");
        while (aptMainLoop()) {
            hidScanInput();
            if (hidKeysDown() & KEY_START) break;
            gspWaitForVBlank();
        }
        gfxExit();
        return 1;
    }

    ModelCtx model;
    memset(&model, 0, sizeof(model));
    if (model_load(&model, "romfs:/model_weights.bin") != 0) {
        printf("Failed to load model weights.\n");
        printf("Press START to exit.\n");
        while (aptMainLoop()) {
            hidScanInput();
            if (hidKeysDown() & KEY_START) break;
            gspWaitForVBlank();
        }
        romfsExit();
        gfxExit();
        return 1;
    }

    Tokenizer* tokenizer = tokenizer_load("romfs:/tokenizer_qwends.cbin");
    if (!tokenizer) {
        printf("Failed to load tokenizer.\n");
        printf("Press START to exit.\n");
        model_free(&model);
        while (aptMainLoop()) {
            hidScanInput();
            if (hidKeysDown() & KEY_START) break;
            gspWaitForVBlank();
        }
        romfsExit();
        gfxExit();
        return 1;
    }

    printf("Model loaded.\n");
    printf("Tokenizer loaded.\n");
    svcSleepThread(1000000000LL);  /* 1 second delay to show status */

    while (aptMainLoop()) {
        hidScanInput();
        u32 kDown = hidKeysDown();

        if (kDown & KEY_START) break;

        if (kDown & KEY_A) {
            SwkbdState swkbd;
            char input_text[MAX_PROMPT_LEN];
            memset(input_text, 0, sizeof(input_text));

            swkbdInit(&swkbd, SWKBD_TYPE_NORMAL, 2, -1);
            swkbdSetHintText(&swkbd, "Enter your prompt...");
            swkbdSetButton(&swkbd, SWKBD_BUTTON_LEFT, "Cancel", false);
            swkbdSetButton(&swkbd, SWKBD_BUTTON_RIGHT, "OK", true);
            swkbdSetFeatures(&swkbd, SWKBD_DEFAULT_QWERTY);

            SwkbdButton button = swkbdInputText(&swkbd, input_text, sizeof(input_text));

            if (button == SWKBD_BUTTON_RIGHT && input_text[0] != '\0') {
                /* Seed RNG with system tick for variety */
                model_seed_rng((uint32_t)svcGetSystemTick());

                /* Encode prompt */
                int tokens[MODEL_CTX_LEN];
                int n_tokens = tokenizer_encode(tokenizer, input_text,
                                                tokens, MODEL_CTX_LEN, 1);
                if (n_tokens <= 0) {
                    consoleSelect(&bottomScreen);
                    consoleClear();
                    printf("Tokenizer returned no tokens.\n");
                    printf("A: Chat | START: Exit\n");
                    consoleSelect(&topScreen);
                    continue;
                }
                if (n_tokens >= MODEL_CTX_LEN) {
                    consoleSelect(&bottomScreen);
                    consoleClear();
                    printf("Prompt uses full context (%d tokens).\n", n_tokens);
                    printf("Shorten it and try again.\n");
                    consoleSelect(&topScreen);
                    continue;
                }

                /* Show "thinking" on bottom screen */
                consoleSelect(&bottomScreen);
                consoleClear();
                printf("Prefill: %d prompt tokens...\n", n_tokens);
                consoleSelect(&topScreen);
                gfxFlushBuffers();
                gfxSwapBuffers();
                gspWaitForVBlank();

                /* Generate one token at a time so the UI can show progress. */
                int out_tokens[MODEL_CTX_LEN];
                int max_new_tokens = MODEL_CTX_LEN - n_tokens;
                int n_out = 0;
                int eos_id = tokenizer_eos_id(tokenizer);
                char response[MAX_RESPONSE_LEN];
                response[0] = '\0';

                model.cache_len = 0;
                model_forward(&model, tokens, n_tokens, generation_logits, 1);
                int next_token = model_sample_logits(generation_logits, 0.9f, 40);

                for (int i = 0; i < max_new_tokens; i++) {
                    if (eos_id >= 0 && next_token == eos_id) break;

                    out_tokens[n_out++] = next_token;
                    tokenizer_decode(tokenizer, out_tokens, n_out,
                                     response, sizeof(response));

                    consoleSelect(&topScreen);
                    consoleClear();
                    printf("=== GPT3DS ===\n");
                    printf("You: %s\n\n", input_text);
                    printf("Bot: %s\n", response);

                    consoleSelect(&bottomScreen);
                    consoleClear();
                    printf("Generating %d/%d tokens\n", n_out, max_new_tokens);
                    printf("cache=%d last=%d eos=%d\n", model.cache_len, next_token, eos_id);
                    printf("START: wait for current token, then stop\n");
                    gfxFlushBuffers();
                    gfxSwapBuffers();
                    gspWaitForVBlank();

                    hidScanInput();
                    if (hidKeysDown() & KEY_START) break;
                    if (model.cache_len >= MODEL_CTX_LEN) break;

                    model_forward(&model, &next_token, 1, generation_logits, 1);
                    next_token = model_sample_logits(generation_logits, 0.9f, 40);
                }

                /* Store in history */
                if (history_count < 16) {
                    copy_text(history[history_count].user, sizeof(history[history_count].user), input_text);
                    copy_text(history[history_count].bot, sizeof(history[history_count].bot), response);
                    history_count++;
                } else {
                    /* shift */
                    for (int i = 1; i < 16; i++) {
                        memcpy(&history[i-1], &history[i], sizeof(ChatTurn));
                    }
                    copy_text(history[15].user, sizeof(history[15].user), input_text);
                    copy_text(history[15].bot, sizeof(history[15].bot), response);
                }

                consoleSelect(&bottomScreen);
                consoleClear();
                printf("Done: %d tokens generated.\n", n_out);
                printf("A: Chat | START: Exit\n");
                consoleSelect(&topScreen);
                draw_history();
            }
        }

        gfxFlushBuffers();
        gfxSwapBuffers();
        gspWaitForVBlank();
    }

    tokenizer_free(tokenizer);
    model_free(&model);
    romfsExit();
    gfxExit();
    return 0;
}
