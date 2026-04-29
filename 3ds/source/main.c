#include <3ds.h>
#include <citro2d.h>
#include <citro3d.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "model.h"
#include "tokenizer.h"

/* ── Screen dimensions ─────────────────────────────────── */
#define TOP_W       400
#define TOP_H       240
#define BOT_W       320
#define BOT_H       240

/* ── Layout constants ──────────────────────────────────── */
#define HEADER_H    28
#define FOOTER_H    18
#define CHAT_X      6.0f
#define CHAT_W      (TOP_W - 2.0f * CHAT_X)
#define CHAT_Y      ((float)HEADER_H + 4.0f)
#define CHAT_BOTTOM (TOP_H - FOOTER_H - 2.0f)
#define BUBBLE_PAD  5.0f
#define BUBBLE_GAP  4.0f
#define LABEL_H     12.0f
#define TEXT_SCALE   0.45f
#define SMALL_SCALE  0.38f
#define TITLE_SCALE  0.55f
#define LINE_H       13.0f
#define MAX_LINE_CHARS 52

/* ── Color palette ─────────────────────────────────────── */
#define COL_BG          C2D_Color32( 18,  18,  32, 255)
#define COL_HEADER      C2D_Color32( 28,  28,  56, 255)
#define COL_ACCENT      C2D_Color32(  0, 200, 180, 255)
#define COL_FOOTER      C2D_Color32( 28,  28,  56, 220)
#define COL_USER_BG     C2D_Color32( 42,  32,  74, 255)
#define COL_USER_BAR    C2D_Color32(130,  80, 220, 255)
#define COL_BOT_BG      C2D_Color32( 28,  34,  50, 255)
#define COL_BOT_BAR     C2D_Color32(  0, 180, 160, 255)
#define COL_TEXT         C2D_Color32(225, 225, 235, 255)
#define COL_TEXT_DIM     C2D_Color32(120, 125, 145, 255)
#define COL_LABEL_USER  C2D_Color32(180, 140, 255, 255)
#define COL_LABEL_BOT   C2D_Color32( 80, 220, 200, 255)
#define COL_GEN_PULSE   C2D_Color32(255, 200,  80, 255)

#define COL_DBG_BG      C2D_Color32( 10,  10,  22, 255)
#define COL_DBG_HEADER  C2D_Color32( 22,  22,  46, 255)
#define COL_DBG_SECT    C2D_Color32( 80, 160, 255, 255)
#define COL_DBG_KEY     C2D_Color32(140, 140, 160, 255)
#define COL_DBG_VAL     C2D_Color32(140, 255, 160, 255)
#define COL_DBG_WARN    C2D_Color32(255, 200,  80, 255)
#define COL_DBG_ERR     C2D_Color32(255, 100, 100, 255)

/* ── Chat data ─────────────────────────────────────────── */
#define MAX_PROMPT_LEN   256
#define MAX_RESPONSE_LEN 1024
#define MAX_HISTORY      16

typedef struct {
    char user[MAX_PROMPT_LEN];
    char bot[MAX_RESPONSE_LEN];
} ChatTurn;

static ChatTurn history[MAX_HISTORY];
static int history_count = 0;
static float generation_logits[MODEL_VOCAB_SIZE];

/* Live generation state (shown as in-progress turn) */
static char live_user[MAX_PROMPT_LEN];
static char live_bot[MAX_RESPONSE_LEN];
static int  live_active = 0;

/* Manual scroll offset (0 = pinned to bottom, positive = scrolled up) */
static float chat_scroll_off = 0.0f;

/* ── Hyperparameters (D-pad adjustable) ─────────────────── */
#define HP_COUNT 3
#define HP_TEMP  0
#define HP_TOPK  1
#define HP_MAXTOK 2

static int   hp_sel = 0;           /* currently selected param */
static float hp_temperature = 0.9f;
static int   hp_top_k = 40;
static int   hp_max_tokens = 0;    /* 0 = use all remaining context */

static void hp_adjust(int dir) {
    switch (hp_sel) {
    case HP_TEMP:
        hp_temperature += dir * 0.1f;
        if (hp_temperature < 0.1f) hp_temperature = 0.1f;
        if (hp_temperature > 2.0f) hp_temperature = 2.0f;
        break;
    case HP_TOPK:
        hp_top_k += dir * 5;
        if (hp_top_k < 1) hp_top_k = 1;
        if (hp_top_k > MODEL_VOCAB_SIZE) hp_top_k = MODEL_VOCAB_SIZE;
        break;
    case HP_MAXTOK:
        hp_max_tokens += dir * 8;
        if (hp_max_tokens < 0) hp_max_tokens = 0;
        if (hp_max_tokens > MODEL_CTX_LEN) hp_max_tokens = MODEL_CTX_LEN;
        break;
    }
}

/* ── Debug info ────────────────────────────────────────── */
static struct {
    const char* status;
    int   tokens_generated;
    int   max_tokens;
    int   cache_len;
    int   last_token_id;
    int   prompt_tokens;
    float tokens_per_sec;
} dbg = { "Ready", 0, 0, 0, -1, 0, 0.0f };

/* ── System info ───────────────────────────────────────── */
static u8  sys_battery_level = 0;   /* 0-5 */
static u8  sys_charging = 0;
static u8  sys_wifi = 0;            /* 0-3 */
static u32 sys_app_mem_used = 0;
static u32 sys_app_mem_total = 0;
static u32 sys_all_mem_used = 0;
static u32 sys_all_mem_total = 0;
static u64 sys_last_poll = 0;

static void poll_system_info(void) {
    /* Throttle to once per second */
    u64 now = osGetTime();
    if (now - sys_last_poll < 1000) return;
    sys_last_poll = now;

    PTMU_GetBatteryLevel(&sys_battery_level);
    PTMU_GetBatteryChargeState(&sys_charging);
    sys_wifi = osGetWifiStrength();
    sys_app_mem_used  = osGetMemRegionUsed(MEMREGION_APPLICATION);
    sys_app_mem_total = osGetMemRegionSize(MEMREGION_APPLICATION);
    sys_all_mem_used  = osGetMemRegionUsed(MEMREGION_ALL);
    sys_all_mem_total = osGetMemRegionSize(MEMREGION_ALL);
}

/* ── Render targets ────────────────────────────────────── */
static C3D_RenderTarget* tgt_top_left;
static C3D_RenderTarget* tgt_top_right;
static C3D_RenderTarget* tgt_bot;
static C2D_TextBuf       g_tbuf;

/* ── Text drawing helpers ──────────────────────────────── */

static void dt(float x, float y, float z, float sc, u32 col, const char* str) {
    C2D_Text t;
    C2D_TextParse(&t, g_tbuf, str);
    C2D_TextOptimize(&t);
    C2D_DrawText(&t, C2D_WithColor, x, y, z, sc, sc, col);
}

static void dtf(float x, float y, float z, float sc, u32 col, const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    dt(x, y, z, sc, col, buf);
}

static void dt_wrap(float x, float y, float z, float sc, u32 col,
                    float wrap_w, const char* str) {
    C2D_Text t;
    C2D_TextParse(&t, g_tbuf, str);
    C2D_TextOptimize(&t);
    C2D_DrawText(&t, C2D_WithColor | C2D_WordWrap, x, y, z, sc, sc, col, wrap_w);
}

/* Estimate number of wrapped lines for layout calculation */
static int est_lines(const char* s, int max_ch) {
    int lines = 1, col = 0;
    for (int i = 0; s[i]; i++) {
        if (s[i] == '\n') { lines++; col = 0; }
        else if (++col >= max_ch) { lines++; col = 0; }
    }
    return lines;
}

/* Height of one message bubble (label + wrapped text + padding) */
static float msg_height(const char* text) {
    int nl = est_lines(text, MAX_LINE_CHARS);
    return LABEL_H + nl * LINE_H + 2.0f * BUBBLE_PAD;
}

/* ── Top screen ────────────────────────────────────────── */

static void draw_top(float eye) {
    /* Depth-based parallax offsets */
    float d_msg = eye * 1.5f;
    float d_ui  = eye * 3.0f;

    /* Background */
    C2D_DrawRectSolid(0, 0, 0.0f, TOP_W, TOP_H, COL_BG);

    /* ── Header ─────────────────────────────────── */
    C2D_DrawRectSolid(d_ui, 0, 0.1f, TOP_W, HEADER_H, COL_HEADER);
    C2D_DrawRectSolid(d_ui, HEADER_H - 2, 0.12f, TOP_W, 2, COL_ACCENT);
    dt(8.0f + d_ui, 4.0f, 0.15f, TITLE_SCALE, COL_ACCENT, "GPT3DS");
    dtf(TOP_W - 170.0f + d_ui, 8.0f, 0.15f, SMALL_SCALE, COL_TEXT_DIM,
        "%d.%dM params | ctx %d",
        MODEL_TOTAL_PARAMS / 1000000,
        (MODEL_TOTAL_PARAMS % 1000000) / 100000,
        MODEL_CTX_LEN);

    /* ── Chat messages ──────────────────────────── */
    /* Compute total content height to determine scroll */
    float total_h = 0.0f;
    for (int i = 0; i < history_count; i++)
        total_h += msg_height(history[i].user) + BUBBLE_GAP
                 + msg_height(history[i].bot)  + BUBBLE_GAP;
    if (live_active)
        total_h += msg_height(live_user) + BUBBLE_GAP
                 + msg_height(live_bot)  + BUBBLE_GAP;

    float avail = CHAT_BOTTOM - CHAT_Y;
    float auto_scroll = (total_h > avail) ? total_h - avail : 0.0f;
    /* Clamp manual offset: 0 = bottom, auto_scroll = top of history */
    if (chat_scroll_off > auto_scroll) chat_scroll_off = auto_scroll;
    if (chat_scroll_off < 0.0f) chat_scroll_off = 0.0f;
    float scroll = auto_scroll - chat_scroll_off;
    float y = CHAT_Y - scroll;
    float text_w = CHAT_W - 2.0f * BUBBLE_PAD - 4.0f;

    /* Helper macro: draw one message bubble */
    #define DRAW_BUBBLE(label, label_col, bg_col, bar_col, text_str) do { \
        float bh = msg_height(text_str);                                   \
        if (y + bh > CHAT_Y && y < CHAT_BOTTOM) {                         \
            float bx = CHAT_X + d_msg;                                    \
            C2D_DrawRectSolid(bx, y, 0.05f, CHAT_W, bh, bg_col);         \
            C2D_DrawRectSolid(bx, y, 0.06f, 3, bh, bar_col);             \
            dt(bx + 8, y + BUBBLE_PAD, 0.07f, SMALL_SCALE,               \
               label_col, label);                                          \
            dt_wrap(bx + 8, y + BUBBLE_PAD + LABEL_H, 0.07f,             \
                    TEXT_SCALE, COL_TEXT, text_w, text_str);               \
        }                                                                  \
        y += bh + BUBBLE_GAP;                                              \
    } while (0)

    for (int i = 0; i < history_count; i++) {
        DRAW_BUBBLE("You", COL_LABEL_USER, COL_USER_BG, COL_USER_BAR,
                    history[i].user);
        DRAW_BUBBLE("Bot", COL_LABEL_BOT, COL_BOT_BG, COL_BOT_BAR,
                    history[i].bot);
    }

    if (live_active) {
        DRAW_BUBBLE("You", COL_LABEL_USER, COL_USER_BG, COL_USER_BAR,
                    live_user);
        DRAW_BUBBLE("Bot", COL_GEN_PULSE, COL_BOT_BG, COL_BOT_BAR,
                    live_bot);
    }

    #undef DRAW_BUBBLE

    /* Empty state hint */
    if (history_count == 0 && !live_active) {
        dt(TOP_W / 2.0f - 75.0f + d_msg, TOP_H / 2.0f - 8.0f, 0.05f,
           TEXT_SCALE, COL_TEXT_DIM, "Press [A] to start chatting");
    }

    /* ── Footer ─────────────────────────────────── */
    C2D_DrawRectSolid(d_ui, TOP_H - FOOTER_H, 0.1f, TOP_W, FOOTER_H, COL_FOOTER);
    dt(10.0f + d_ui, TOP_H - FOOTER_H + 3.0f, 0.15f,
       SMALL_SCALE, COL_TEXT_DIM, "[A] Chat");
    dt(TOP_W - 95.0f + d_ui, TOP_H - FOOTER_H + 3.0f, 0.15f,
       SMALL_SCALE, COL_TEXT_DIM, "[START] Exit");
}

/* ── Bottom screen (debug panel) ───────────────────────── */

static void draw_bot_screen(void) {
    C2D_DrawRectSolid(0, 0, 0.0f, BOT_W, BOT_H, COL_DBG_BG);

    /* Header bar */
    C2D_DrawRectSolid(0, 0, 0.1f, BOT_W, 22, COL_DBG_HEADER);
    C2D_DrawRectSolid(0, 20, 0.12f, BOT_W, 2, COL_ACCENT);
    dt(8, 3, 0.15f, 0.5f, COL_ACCENT, "DEBUG");

    float y = 26.0f;
    float c2 = 120.0f;  /* column 2 for values */

    /* ── System ────────────────────────────────── */
    dt(8, y, 0.1f, SMALL_SCALE, COL_DBG_SECT, "System");
    y += 13;

    /* Battery: bar + level */
    dt(16, y, 0.1f, SMALL_SCALE, COL_DBG_KEY, "Battery:");
    {
        float bx = c2, bw = 50.0f, bh = 8.0f;
        float by = y + 1.0f;
        C2D_DrawRectSolid(bx, by, 0.1f, bw, bh, C2D_Color32(40, 40, 60, 255));
        float fill = sys_battery_level / 5.0f;
        u32 bar_col = (sys_battery_level <= 1)
            ? COL_DBG_ERR
            : (sys_battery_level <= 2 ? COL_DBG_WARN : COL_DBG_VAL);
        C2D_DrawRectSolid(bx, by, 0.12f, bw * fill, bh, bar_col);
        /* Nub on right end of battery outline */
        C2D_DrawRectSolid(bx + bw, by + 2, 0.1f, 3, 4,
                          C2D_Color32(40, 40, 60, 255));
        if (sys_charging)
            dtf(bx + bw + 8, y, 0.1f, SMALL_SCALE, COL_DBG_WARN, "CHG");
        else
            dtf(bx + bw + 8, y, 0.1f, SMALL_SCALE, COL_DBG_VAL, "%d/5",
                sys_battery_level);
    }
    y += 11;

    /* App memory: bar + values */
    dt(16, y, 0.1f, SMALL_SCALE, COL_DBG_KEY, "App mem:");
    {
        float bx = c2, bw = 50.0f, bh = 8.0f;
        float by = y + 1.0f;
        float fill = (sys_app_mem_total > 0)
            ? (float)sys_app_mem_used / (float)sys_app_mem_total : 0.0f;
        if (fill > 1.0f) fill = 1.0f;
        C2D_DrawRectSolid(bx, by, 0.1f, bw, bh, C2D_Color32(40, 40, 60, 255));
        u32 mem_col = (fill > 0.85f) ? COL_DBG_ERR
                    : (fill > 0.7f)  ? COL_DBG_WARN : COL_DBG_VAL;
        C2D_DrawRectSolid(bx, by, 0.12f, bw * fill, bh, mem_col);
        dtf(bx + bw + 8, y, 0.1f, SMALL_SCALE, COL_DBG_VAL,
            "%luM/%luM", (unsigned long)(sys_app_mem_used / (1024*1024)),
            (unsigned long)(sys_app_mem_total / (1024*1024)));
    }
    y += 11;

    /* System memory */
    dt(16, y, 0.1f, SMALL_SCALE, COL_DBG_KEY, "Sys mem:");
    {
        float bx = c2, bw = 50.0f, bh = 8.0f;
        float by = y + 1.0f;
        float fill = (sys_all_mem_total > 0)
            ? (float)sys_all_mem_used / (float)sys_all_mem_total : 0.0f;
        if (fill > 1.0f) fill = 1.0f;
        C2D_DrawRectSolid(bx, by, 0.1f, bw, bh, C2D_Color32(40, 40, 60, 255));
        u32 mem_col = (fill > 0.85f) ? COL_DBG_ERR
                    : (fill > 0.7f)  ? COL_DBG_WARN : COL_DBG_VAL;
        C2D_DrawRectSolid(bx, by, 0.12f, bw * fill, bh, mem_col);
        dtf(bx + bw + 8, y, 0.1f, SMALL_SCALE, COL_DBG_VAL,
            "%luM/%luM", (unsigned long)(sys_all_mem_used / (1024*1024)),
            (unsigned long)(sys_all_mem_total / (1024*1024)));
    }
    y += 11;

    /* WiFi */
    dt(16, y, 0.1f, SMALL_SCALE, COL_DBG_KEY, "WiFi:");
    {
        /* Draw signal bars */
        float bx = c2;
        for (int i = 0; i < 4; i++) {
            float bar_h = 3.0f + i * 2.0f;
            float by = y + 9.0f - bar_h;
            u32 col = (i < (int)sys_wifi)
                ? COL_DBG_VAL
                : C2D_Color32(40, 40, 60, 255);
            C2D_DrawRectSolid(bx + i * 6.0f, by, 0.1f, 4, bar_h, col);
        }
        dtf(c2 + 30, y, 0.1f, SMALL_SCALE, COL_DBG_VAL, "%d/3", sys_wifi);
    }
    y += 14;

    /* ── Model ─────────────────────────────────── */
    dt(8, y, 0.1f, SMALL_SCALE, COL_DBG_SECT, "Model");
    y += 13;
    dt(16, y, 0.1f, SMALL_SCALE, COL_DBG_KEY, "Params:");
    dtf(c2, y, 0.1f, SMALL_SCALE, COL_DBG_VAL, "%d", MODEL_TOTAL_PARAMS);
    y += 11;
    dt(16, y, 0.1f, SMALL_SCALE, COL_DBG_KEY, "Arch:");
    dtf(c2, y, 0.1f, SMALL_SCALE, COL_DBG_VAL, "%dx%d loops, int8",
        MODEL_UNIQUE_BLOCKS, MODEL_LOOPS_PER_PASS);
    y += 11;
    dt(16, y, 0.1f, SMALL_SCALE, COL_DBG_KEY, "Vocab/Ctx:");
    dtf(c2, y, 0.1f, SMALL_SCALE, COL_DBG_VAL, "%d / %d",
        MODEL_VOCAB_SIZE, MODEL_CTX_LEN);
    y += 14;

    /* ── Generation ────────────────────────────── */
    dt(8, y, 0.1f, SMALL_SCALE, COL_DBG_SECT, "Generation");
    y += 13;
    dt(16, y, 0.1f, SMALL_SCALE, COL_DBG_KEY, "Status:");
    u32 sc = COL_DBG_VAL;
    if (strcmp(dbg.status, "Generating") == 0) sc = COL_DBG_WARN;
    else if (strcmp(dbg.status, "Prefilling") == 0) sc = COL_DBG_WARN;
    else if (strstr(dbg.status, "Error") != NULL) sc = COL_DBG_ERR;
    dt(c2, y, 0.1f, SMALL_SCALE, sc, dbg.status);
    y += 11;
    dt(16, y, 0.1f, SMALL_SCALE, COL_DBG_KEY, "Tokens:");
    dtf(c2, y, 0.1f, SMALL_SCALE, COL_DBG_VAL, "%d/%d gen  %d prompt",
        dbg.tokens_generated, dbg.max_tokens, dbg.prompt_tokens);
    y += 11;
    dt(16, y, 0.1f, SMALL_SCALE, COL_DBG_KEY, "Cache:");
    dtf(c2, y, 0.1f, SMALL_SCALE, COL_DBG_VAL, "%d / %d",
        dbg.cache_len, MODEL_CTX_LEN);
    y += 11;
    if (dbg.tokens_per_sec > 0.0f) {
        dt(16, y, 0.1f, SMALL_SCALE, COL_DBG_KEY, "Speed:");
        dtf(c2, y, 0.1f, SMALL_SCALE, COL_DBG_VAL, "%.2f tok/s",
            dbg.tokens_per_sec);
    }
    y += 14;

    /* ── Progress bar (during generation) ──────── */
    if (dbg.max_tokens > 0 && strcmp(dbg.status, "Generating") == 0) {
        float bar_w = BOT_W - 60.0f;
        float fill = (float)dbg.tokens_generated / (float)dbg.max_tokens;
        if (fill > 1.0f) fill = 1.0f;
        C2D_DrawRectSolid(16, y, 0.1f, bar_w, 8, C2D_Color32(40, 40, 60, 255));
        C2D_DrawRectSolid(16, y, 0.12f, bar_w * fill, 8, COL_ACCENT);
        dtf(16 + bar_w + 4, y - 1, 0.1f, SMALL_SCALE, COL_TEXT_DIM,
            "%d%%", (int)(fill * 100));
        y += 14;
    }

    /* ── Hyperparameters ───────────────────────── */
    dt(8, y, 0.1f, SMALL_SCALE, COL_DBG_SECT, "Config");
    y += 13;
    for (int p = 0; p < HP_COUNT; p++) {
        u32 kc = (p == hp_sel) ? COL_ACCENT : COL_DBG_KEY;
        u32 vc = (p == hp_sel) ? COL_TEXT   : COL_DBG_VAL;
        const char* arrow = (p == hp_sel) ? "> " : "  ";
        switch (p) {
        case HP_TEMP:
            dtf(12, y, 0.1f, SMALL_SCALE, kc, "%sTemperature:", arrow);
            dtf(c2, y, 0.1f, SMALL_SCALE, vc, "%.1f", hp_temperature);
            break;
        case HP_TOPK:
            dtf(12, y, 0.1f, SMALL_SCALE, kc, "%sTop-K:", arrow);
            dtf(c2, y, 0.1f, SMALL_SCALE, vc, "%d", hp_top_k);
            break;
        case HP_MAXTOK:
            dtf(12, y, 0.1f, SMALL_SCALE, kc, "%sMax tokens:", arrow);
            dtf(c2, y, 0.1f, SMALL_SCALE, vc, "%s",
                hp_max_tokens > 0 ? "" : "auto");
            if (hp_max_tokens > 0)
                dtf(c2, y, 0.1f, SMALL_SCALE, vc, "%d", hp_max_tokens);
            break;
        }
        y += 11;
    }

    /* ── Controls ──────────────────────────────── */
    y = BOT_H - 26.0f;
    C2D_DrawRectSolid(0, y - 4, 0.1f, BOT_W, BOT_H - y + 4, COL_DBG_HEADER);
    dt(8, y, 0.1f, SMALL_SCALE, COL_TEXT_DIM, "[A] Chat  [START] Exit");
    dt(8, y + 11, 0.1f, SMALL_SCALE, COL_TEXT_DIM,
       "[D-pad] Select/adjust config");
}

/* ── Render one complete frame ─────────────────────────── */

static void render_frame(void) {
    poll_system_info();
    C2D_TextBufClear(g_tbuf);
    C3D_FrameBegin(C3D_FRAME_SYNCDRAW);

    float slider = osGet3DSliderState();
    float iod = slider * 2.0f;

    /* Left eye */
    C2D_TargetClear(tgt_top_left, COL_BG);
    C2D_SceneBegin(tgt_top_left);
    draw_top(-iod);

    /* Right eye (stereoscopic 3D) */
    if (iod > 0.0f) {
        C2D_TargetClear(tgt_top_right, COL_BG);
        C2D_SceneBegin(tgt_top_right);
        draw_top(iod);
    }

    /* Bottom screen */
    C2D_TargetClear(tgt_bot, COL_DBG_BG);
    C2D_SceneBegin(tgt_bot);
    draw_bot_screen();

    C3D_FrameEnd(0);
}

/* ── Loading / error screens ───────────────────────────── */

static void show_loading(const char* msg) {
    C2D_TextBufClear(g_tbuf);
    C3D_FrameBegin(C3D_FRAME_SYNCDRAW);

    C2D_TargetClear(tgt_top_left, COL_BG);
    C2D_SceneBegin(tgt_top_left);
    C2D_DrawRectSolid(0, 0, 0.0f, TOP_W, TOP_H, COL_BG);
    dt(TOP_W / 2.0f - 30.0f, 80.0f, 0.5f, TITLE_SCALE, COL_ACCENT, "GPT3DS");
    dt(TOP_W / 2.0f - 60.0f, 120.0f, 0.5f, TEXT_SCALE, COL_TEXT_DIM, msg);

    C2D_TargetClear(tgt_bot, COL_DBG_BG);
    C2D_SceneBegin(tgt_bot);
    C2D_DrawRectSolid(0, 0, 0.0f, BOT_W, BOT_H, COL_DBG_BG);
    dt(8, 8, 0.5f, TEXT_SCALE, COL_TEXT_DIM, msg);

    C3D_FrameEnd(0);
}

static void show_error_and_wait(const char* msg) {
    while (aptMainLoop()) {
        hidScanInput();
        if (hidKeysDown() & KEY_START) break;

        C2D_TextBufClear(g_tbuf);
        C3D_FrameBegin(C3D_FRAME_SYNCDRAW);

        C2D_TargetClear(tgt_top_left, COL_BG);
        C2D_SceneBegin(tgt_top_left);
        C2D_DrawRectSolid(0, 0, 0, TOP_W, TOP_H, COL_BG);
        dt(TOP_W / 2.0f - 30.0f, 80.0f, 0.5f, TITLE_SCALE, COL_ACCENT, "GPT3DS");
        dt(20, 120, 0.5f, TEXT_SCALE, COL_DBG_ERR, msg);
        dt(20, 150, 0.5f, SMALL_SCALE, COL_TEXT_DIM, "Press START to exit.");

        C2D_TargetClear(tgt_bot, COL_DBG_BG);
        C2D_SceneBegin(tgt_bot);
        C2D_DrawRectSolid(0, 0, 0, BOT_W, BOT_H, COL_DBG_BG);
        dt(8, 8, 0.5f, TEXT_SCALE, COL_DBG_ERR, msg);

        C3D_FrameEnd(0);
    }
}

/* ── Utility ───────────────────────────────────────────── */

static void copy_text(char* dst, size_t dst_size, const char* src) {
    if (dst_size == 0) return;
    snprintf(dst, dst_size, "%s", src);
}

/* ── Main ──────────────────────────────────────────────── */

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;

    /* Init system services */
    ptmuInit();

    /* Init graphics subsystems */
    gfxInitDefault();
    gfxSet3D(true);
    C3D_Init(C3D_DEFAULT_CMDBUF_SIZE);
    C2D_Init(C2D_DEFAULT_MAX_OBJECTS);
    C2D_Prepare();

    tgt_top_left  = C2D_CreateScreenTarget(GFX_TOP, GFX_LEFT);
    tgt_top_right = C2D_CreateScreenTarget(GFX_TOP, GFX_RIGHT);
    tgt_bot       = C2D_CreateScreenTarget(GFX_BOTTOM, GFX_LEFT);
    g_tbuf        = C2D_TextBufNew(8192);

    /* Show loading screen */
    show_loading("Loading model...");

    /* Init ROMFS */
    Result rc = romfsInit();
    if (R_FAILED(rc)) {
        show_error_and_wait("romfsInit failed!");
        C2D_TextBufDelete(g_tbuf);
        C2D_Fini();
        C3D_Fini();
        gfxExit();
        return 1;
    }

    /* Load model */
    show_loading("Loading model weights...");
    ModelCtx model;
    memset(&model, 0, sizeof(model));
    if (model_load(&model, "romfs:/model_weights.bin") != 0) {
        show_error_and_wait("Failed to load model weights.");
        romfsExit();
        C2D_TextBufDelete(g_tbuf);
        C2D_Fini();
        C3D_Fini();
        gfxExit();
        return 1;
    }

    /* Load tokenizer */
    show_loading("Loading tokenizer...");
    Tokenizer* tokenizer = tokenizer_load("romfs:/tokenizer_qwends.cbin");
    if (!tokenizer) {
        show_error_and_wait("Failed to load tokenizer.");
        model_free(&model);
        romfsExit();
        C2D_TextBufDelete(g_tbuf);
        C2D_Fini();
        C3D_Fini();
        gfxExit();
        return 1;
    }

    dbg.status = "Ready";
    render_frame();

    /* ── Main loop ─────────────────────────────── */
    while (aptMainLoop()) {
        hidScanInput();
        u32 kDown = hidKeysDown();

        /* Circle pad: scroll chat history */
        circlePosition cpad;
        hidCircleRead(&cpad);
        if (cpad.dy > 30 || cpad.dy < -30) {
            chat_scroll_off += (float)cpad.dy * 0.06f;
            /* Clamped in draw_top */
        }

        if (kDown & KEY_START) break;

        /* D-pad: adjust hyperparameters */
        if (kDown & KEY_DDOWN) hp_sel = (hp_sel + 1) % HP_COUNT;
        if (kDown & KEY_DUP)   hp_sel = (hp_sel + HP_COUNT - 1) % HP_COUNT;
        if (kDown & KEY_DRIGHT) hp_adjust(1);
        if (kDown & KEY_DLEFT)  hp_adjust(-1);

        if (kDown & KEY_A) {
            /* Software keyboard */
            SwkbdState swkbd;
            char input_text[MAX_PROMPT_LEN];
            memset(input_text, 0, sizeof(input_text));

            swkbdInit(&swkbd, SWKBD_TYPE_NORMAL, 2, -1);
            swkbdSetHintText(&swkbd, "Enter your prompt...");
            swkbdSetButton(&swkbd, SWKBD_BUTTON_LEFT, "Cancel", false);
            swkbdSetButton(&swkbd, SWKBD_BUTTON_RIGHT, "OK", true);
            swkbdSetFeatures(&swkbd, SWKBD_DEFAULT_QWERTY);

            SwkbdButton button = swkbdInputText(&swkbd, input_text,
                                                sizeof(input_text));

            if (button == SWKBD_BUTTON_RIGHT && input_text[0] != '\0') {
                model_seed_rng((uint32_t)svcGetSystemTick());

                /* Encode prompt */
                int tokens[MODEL_CTX_LEN];
                int n_tokens = tokenizer_encode(tokenizer, input_text,
                                                tokens, MODEL_CTX_LEN, 1);
                if (n_tokens <= 0) {
                    dbg.status = "Error: no tokens";
                    render_frame();
                    continue;
                }
                if (n_tokens >= MODEL_CTX_LEN) {
                    dbg.status = "Error: prompt too long";
                    render_frame();
                    continue;
                }

                /* Set up live generation display */
                copy_text(live_user, sizeof(live_user), input_text);
                live_bot[0] = '\0';
                live_active = 1;
                chat_scroll_off = 0.0f;  /* snap to bottom */

                /* Update debug info */
                dbg.status = "Prefilling";
                dbg.prompt_tokens = n_tokens;
                dbg.tokens_generated = 0;
                int ctx_remaining = MODEL_CTX_LEN - n_tokens;
                int max_new = (hp_max_tokens > 0 && hp_max_tokens < ctx_remaining)
                            ? hp_max_tokens : ctx_remaining;
                dbg.max_tokens = max_new;
                dbg.cache_len = 0;
                dbg.tokens_per_sec = 0.0f;
                render_frame();

                /* Prefill */
                model.cache_len = 0;
                model_forward(&model, tokens, n_tokens, generation_logits, 1);
                int next_token = model_sample_logits(generation_logits,
                                                     hp_temperature,
                                                     hp_top_k);

                /* Token-by-token generation */
                int out_tokens[MODEL_CTX_LEN];
                int n_out = 0;
                int eos_id = tokenizer_eos_id(tokenizer);
                char response[MAX_RESPONSE_LEN];
                response[0] = '\0';

                dbg.status = "Generating";
                u64 gen_start = osGetTime();

                for (int i = 0; i < max_new; i++) {
                    if (eos_id >= 0 && next_token == eos_id) break;

                    out_tokens[n_out++] = next_token;
                    tokenizer_decode(tokenizer, out_tokens, n_out,
                                     response, sizeof(response));

                    /* Update live display */
                    copy_text(live_bot, sizeof(live_bot), response);

                    /* Update debug info */
                    dbg.tokens_generated = n_out;
                    dbg.cache_len = model.cache_len;
                    dbg.last_token_id = next_token;
                    u64 elapsed = osGetTime() - gen_start;
                    if (elapsed > 0)
                        dbg.tokens_per_sec = (float)n_out
                                           / ((float)elapsed / 1000.0f);

                    render_frame();

                    /* Check for early stop */
                    hidScanInput();
                    if (hidKeysDown() & KEY_START) break;
                    if (model.cache_len >= MODEL_CTX_LEN) break;

                    /* Next token */
                    model_forward(&model, &next_token, 1,
                                  generation_logits, 1);
                    next_token = model_sample_logits(generation_logits,
                                                     hp_temperature,
                                                     hp_top_k);
                }

                /* Store in history */
                live_active = 0;
                if (history_count < MAX_HISTORY) {
                    copy_text(history[history_count].user,
                              sizeof(history[history_count].user),
                              input_text);
                    copy_text(history[history_count].bot,
                              sizeof(history[history_count].bot),
                              response);
                    history_count++;
                } else {
                    for (int i = 1; i < MAX_HISTORY; i++)
                        memcpy(&history[i-1], &history[i], sizeof(ChatTurn));
                    copy_text(history[MAX_HISTORY-1].user,
                              sizeof(history[MAX_HISTORY-1].user),
                              input_text);
                    copy_text(history[MAX_HISTORY-1].bot,
                              sizeof(history[MAX_HISTORY-1].bot),
                              response);
                }

                dbg.status = "Ready";
            }
        }

        render_frame();
    }

    /* Cleanup */
    tokenizer_free(tokenizer);
    model_free(&model);
    romfsExit();
    C2D_TextBufDelete(g_tbuf);
    C2D_Fini();
    C3D_Fini();
    gfxExit();
    ptmuExit();
    return 0;
}
