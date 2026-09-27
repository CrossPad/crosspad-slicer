/**
 * @file SlicerApp.cpp
 * @brief Slicer -- a loop chopped into up to sixteen slices, one per pad,
 *        and saved as a kit.
 *
 *   top    back, the file (tap: pick a take), the MODE, its value, and PLAY
 *          (the whole loop, round and round)
 *   wave   the loop, the slices in turn light and dark, the selected one
 *          brightest, each numbered at its start -- the MPC's view
 *   tools  SEL picks a slice (turned; each one is heard as it is reached),
 *          MOVE moves its start (a column per detent, then a millisecond with
 *          the picture zoomed in), + splits it, DEL joins it to the one
 *          before, SAVE KIT writes the slices as a kit
 *
 * The modes:
 *   HITS   a slice at every hit; the value is the sensitivity
 *   EQUAL  the loop in equal parts; the value is how many
 *   FREE   cut by ear: a dark pad cuts the playing loop where it is -- the
 *          moment the pad is hit, as heard; a lit pad plays from its cut
 *
 * Outside FREE the pads play their slices. The card and the audio are all
 * crosspad::ISampleTools; the recorder hands its loop over through it.
 */

#include "crosspad-slicer/SlicerApp.hpp"

#if __has_include("pc_stubs/PcApp.hpp")
  #include "pc_stubs/PcApp.hpp"
#else
  #include "crosspad-gui/app/App.hpp"
#endif

#include "crosspad/app/AppRegistrar.hpp"
#include "crosspad/pad/IPadLogicHandler.hpp"
#include "crosspad/pad/PadLedController.hpp"
#include "crosspad/pad/PadManager.hpp"
#include "crosspad/platform/PlatformServices.hpp"
#include "crosspad/sampletools/ISampleTools.hpp"
#include "crosspad/sampletools/SliceMath.hpp"
#include "crosspad-gui/widgets/gui_helpers.h"

#include "lvgl.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef ESP_PLATFORM
extern "C" {
  extern lv_indev_t* indev_encoderTop;
}
#define SLI_USE_ENCODER 1
#endif

namespace {

constexpr uint16_t kCols       = 300;
constexpr uint32_t kTickMs     = 40;
constexpr uint32_t kZoomFrames = 12000;
constexpr int      kPads       = 16;

const lv_color_t kBg     = lv_color_hex(0x16181B);
const lv_color_t kPanel  = lv_color_hex(0x20242A);
const lv_color_t kText   = lv_color_hex(0xE8EAED);
const lv_color_t kDim    = lv_color_hex(0x7B828B);
const lv_color_t kInk    = lv_color_hex(0x16181B);
const lv_color_t kAccent = lv_color_hex(0xFFB347);
const lv_color_t kSliceA = lv_color_hex(0x1D2126);
const lv_color_t kSliceB = lv_color_hex(0x252A31);
const lv_color_t kSel    = lv_color_hex(0x3A2F1A);
const lv_color_t kWave   = lv_color_hex(0xFFC857);
const lv_color_t kWaveSel= lv_color_hex(0xFFE3A3);
const lv_color_t kMark   = lv_color_hex(0xE8EAED);
const lv_color_t kHead   = lv_color_hex(0x4FE0B0);

enum class Mode : uint8_t { Hits, Equal, Free };
enum class Edit : uint8_t { None, Coarse, Fine };

// ── pads: events from the pad thread, read on the LVGL timer ───────────────

/// A pad hit and, for FREE, the frame being heard at that moment -- taken
/// on the pad's thread, so the cut is where the ear was, not where the
/// screen's next tick finds the playhead.
struct PadHit { uint8_t pad; bool playing; uint32_t frame; };
constexpr uint32_t kHitQueue = 32;
PadHit                s_hits[kHitQueue];
std::atomic<uint32_t> s_hitHead{0}, s_hitTail{0};

crosspad::ISampleTools* tools() { return crosspad::getPlatformServices().sampleTools; }

class SlicerPadLogic : public crosspad::IPadLogicHandler {
public:
    void onActivate(crosspad::PadManager&) override;
    void onDeactivate(crosspad::PadManager&) override {}
    void onPadPress(crosspad::PadManager&, uint8_t pad, uint8_t) override {
        auto* st = tools();
        const uint32_t h = s_hitHead.load(std::memory_order_relaxed);
        if (h - s_hitTail.load(std::memory_order_acquire) >= kHitQueue) return;
        s_hits[h % kHitQueue] = PadHit{pad, st && st->previewing(), st ? st->previewFrame() : 0};
        s_hitHead.store(h + 1, std::memory_order_release);
    }
    void onPadRelease(crosspad::PadManager&, uint8_t) override {}
    void onPadPressure(crosspad::PadManager&, uint8_t, uint8_t) override {}
};
std::shared_ptr<SlicerPadLogic> s_padLogic;

// ── state ──────────────────────────────────────────────────────────────────

lv_obj_t* s_root = nullptr;
lv_obj_t* s_wave = nullptr;
lv_obj_t* s_fileBtn = nullptr;
lv_obj_t* s_modeBtn = nullptr;
lv_obj_t* s_paramBtn = nullptr;
lv_obj_t* s_playBtn = nullptr;
lv_obj_t* s_selBtn = nullptr;
lv_obj_t* s_moveBtn = nullptr;
lv_obj_t* s_info = nullptr;
lv_obj_t* s_overlay = nullptr;
lv_obj_t* s_nameArea = nullptr;
lv_timer_t* s_timer = nullptr;

std::string s_file;
uint32_t s_rate = 48000;
uint32_t s_regA = 0, s_regB = 0;         // the loop being sliced
std::vector<uint32_t> s_starts;          // sorted; slice i is [s_starts[i], next or s_regB)
int      s_sel = 0;
Mode     s_mode = Mode::Hits;
int      s_sens = 50;
int      s_count = 8;
crosspad::WaveOverview s_ov;             // the picture drawn: the loop, or a zoom
std::vector<uint8_t> s_env;              // the loop's envelope, for HITS
uint32_t s_envHop = 0, s_envStart = 0;
bool     s_loading = false;
bool     s_ovPending = false, s_viewWanted = false;
uint32_t s_wantA = 0, s_wantB = 0;
bool     s_listPending = false, s_exportPending = false;
Edit     s_editParam = Edit::None, s_editSel = Edit::None, s_editMove = Edit::None;
bool     s_chopping = false;             // FREE: the loop is playing and pads cut it
bool     s_playing = false;              // PLAY: the whole loop, looped
int32_t  s_headX = -1;                   // playhead column as drawn, -1 = none

#ifdef SLI_USE_ENCODER
lv_group_t* s_encGroup = nullptr;
lv_group_t* s_prevEncGroup = nullptr;
lv_group_t* s_overlayGroup = nullptr;
#endif

uint32_t slice_end(int i)
{
    return i + 1 < (int)s_starts.size() ? s_starts[i + 1] : s_regB;
}

uint32_t min_gap() { return s_rate / 50; }   // 20 ms: the shortest slice worth a pad

std::string base_of(const std::string& p)
{
    const size_t s = p.find_last_of('/');
    return s == std::string::npos ? p : p.substr(s + 1);
}

void set_text(lv_obj_t* l, const char* t)
{
    if (l && std::strcmp(lv_label_get_text(l), t) != 0) lv_label_set_text(l, t);
}

/// The same hues the exported kit gives its pads (SampleToolsPort).
void slice_rgb(int i, uint8_t& r, uint8_t& g, uint8_t& b)
{
    const float h = (float)(i % 16) / 16.0f * 6.0f;
    const int k = (int)h;
    const float f = h - (float)k;
    float fr = 0, fg = 0, fb = 0;
    switch (k) {
        case 0: fr = 1; fg = f; break;
        case 1: fr = 1 - f; fg = 1; break;
        case 2: fg = 1; fb = f; break;
        case 3: fg = 1 - f; fb = 1; break;
        case 4: fr = f; fb = 1; break;
        default: fr = 1; fb = 1 - f; break;
    }
    r = (uint8_t)(fr * 255); g = (uint8_t)(fg * 255); b = (uint8_t)(fb * 255);
}

/// Pads with a slice lit in its colour, the rest dark.
void paint_pads()
{
    auto& leds = crosspad::getPadLedController();
    for (int p = 0; p < kPads; ++p) {
        uint8_t r = 0, g = 0, b = 0;
        if (p < (int)s_starts.size()) slice_rgb(p, r, g, b);
        const bool sel = p == s_sel && s_mode != Mode::Free;
        const int dim = sel ? 1 : 4;
        leds.setPadColor((uint8_t)p, r / dim, g / dim, b / dim);
    }
}

void SlicerPadLogic::onActivate(crosspad::PadManager&) { paint_pads(); }

// ── the picture ────────────────────────────────────────────────────────────

void request_view(uint32_t a, uint32_t b)
{
    auto* st = tools();
    if (!st || s_file.empty()) return;
    if (st->requestOverview(s_file, a, b, kCols)) {
        s_ovPending = true;
        s_viewWanted = false;
    } else {
        s_viewWanted = true;
        s_wantA = a;
        s_wantB = b;
    }
}

void fill(lv_layer_t* layer, int32_t x1, int32_t y1, int32_t x2, int32_t y2, lv_color_t c)
{
    lv_draw_rect_dsc_t d;
    lv_draw_rect_dsc_init(&d);
    d.bg_color = c;
    d.bg_opa = LV_OPA_COVER;
    lv_area_t a = {x1, y1, x2, y2};
    lv_draw_rect(layer, &d, &a);
}

void wave_draw_cb(lv_event_t* e)
{
    lv_layer_t* layer = lv_event_get_layer(e);
    lv_obj_t* obj = static_cast<lv_obj_t*>(lv_event_get_target(e));
    if (!layer || s_ov.lo.empty() || s_ov.end <= s_ov.start) return;
    lv_area_t box;
    lv_obj_get_content_coords(obj, &box);
    const int32_t w = lv_area_get_width(&box);
    const int32_t h = lv_area_get_height(&box);
    const int32_t top = box.y1 + 11;                      // a strip for the numbers
    const int32_t mid = top + (box.y2 - top) / 2;
    const int32_t half = (box.y2 - top) / 2;
    const uint32_t a = s_ov.start, len = s_ov.end - s_ov.start;
    auto xOf = [&](uint32_t f) -> int32_t {
        if (f <= a) return box.x1;
        if (f >= s_ov.end) return box.x2;
        return box.x1 + (int32_t)((uint64_t)(f - a) * w / len);
    };
    for (int i = 0; i < (int)s_starts.size(); ++i) {
        const int32_t x1 = xOf(s_starts[i]), x2 = xOf(slice_end(i));
        if (x2 > x1) fill(layer, x1, top, x2 - 1, box.y2, i == s_sel ? kSel : (i % 2 ? kSliceB : kSliceA));
    }
    const size_t cols = s_ov.lo.size();
    const uint32_t selA = s_starts.empty() ? 0 : s_starts[s_sel], selB = s_starts.empty() ? 0 : slice_end(s_sel);
    for (size_t c = 0; c < cols; ++c) {
        const int32_t x = box.x1 + (int32_t)(c * w / cols);
        const uint32_t f = a + (uint32_t)((uint64_t)c * len / cols);
        const int32_t y1 = mid - (int32_t)s_ov.hi[c] * half / 128;
        const int32_t y2 = mid - (int32_t)s_ov.lo[c] * half / 128;
        fill(layer, x, std::min(y1, y2), x, std::max(y1, y2), f >= selA && f < selB ? kWaveSel : kWave);
    }
    lv_draw_label_dsc_t ld;
    lv_draw_label_dsc_init(&ld);
    ld.font = &lv_font_montserrat_10;
    for (int i = 0; i < (int)s_starts.size(); ++i) {
        if (s_starts[i] < a || s_starts[i] > s_ov.end) continue;
        const int32_t x = xOf(s_starts[i]);
        uint8_t r, g, b;
        slice_rgb(i, r, g, b);
        fill(layer, x, box.y1, x, box.y2, i == s_sel ? kMark : lv_color_make(r, g, b));
        char num[12];
        std::snprintf(num, sizeof(num), "%d", i + 1);
        ld.color = i == s_sel ? kMark : lv_color_make(r, g, b);
        ld.text = num;
        ld.text_local = 1;
        lv_area_t la = {x + 2, box.y1, x + 20, box.y1 + 10};
        lv_draw_label(layer, &ld, &la);
    }
    if (s_headX >= 0) fill(layer, s_headX, top, s_headX, box.y2, kHead);
    (void)h;
}

/// The playhead's screen x in the picture, -1 when nothing plays or it is
/// off the picture. Only its old and new columns are redrawn while it
/// moves: the whole picture every tick kept the LVGL task busy enough to
/// starve everything below it on its core.
int32_t head_x()
{
    auto* st = tools();
    if (!st || !s_wave || !st->previewing() || s_ov.end <= s_ov.start) return -1;
    const uint32_t p = st->previewFrame();
    if (p < s_ov.start || p > s_ov.end) return -1;
    lv_area_t box;
    lv_obj_get_content_coords(s_wave, &box);
    return box.x1 + (int32_t)((uint64_t)(p - s_ov.start) * lv_area_get_width(&box) / (s_ov.end - s_ov.start));
}


void move_head()
{
    const int32_t x = head_x();
    if (x == s_headX || !s_wave) return;
    lv_area_t box;
    lv_obj_get_content_coords(s_wave, &box);
    for (int32_t at : {s_headX, x}) {
        if (at < 0) continue;
        lv_area_t a = {at - 1, box.y1, at + 1, box.y2};
        lv_obj_invalidate_area(s_wave, &a);
    }
    s_headX = x;
}

// ── slices ─────────────────────────────────────────────────────────────────

void show_info();

void show_buttons()
{
    static const char* kModes[] = {"HITS", "EQUAL", "FREE"};
    set_text(lv_obj_get_child(s_modeBtn, 0), kModes[(int)s_mode]);
    char buf[20];
    if (s_mode == Mode::Hits)       std::snprintf(buf, sizeof(buf), "SENS %d", s_sens);
    else if (s_mode == Mode::Equal) std::snprintf(buf, sizeof(buf), "N %d", s_count);
    else                            std::snprintf(buf, sizeof(buf), "-");
    set_text(lv_obj_get_child(s_paramBtn, 0), buf);
    std::snprintf(buf, sizeof(buf), s_editSel != Edit::None ? "S <%d>" : "SEL %d", s_sel + 1);
    set_text(lv_obj_get_child(s_selBtn, 0), buf);
    set_text(lv_obj_get_child(s_moveBtn, 0),
             s_editMove == Edit::Coarse ? "M <>" : (s_editMove == Edit::Fine ? "M fine" : "MOVE"));
    set_text(lv_obj_get_child(s_fileBtn, 0), s_file.empty() ? LV_SYMBOL_DIRECTORY " pick a take" : base_of(s_file).c_str());
    const bool lit = s_playing || s_chopping;
    lv_obj_set_style_bg_color(s_playBtn, lit ? kAccent : kPanel, 0);
    lv_obj_set_style_text_color(lv_obj_get_child(s_playBtn, 0), lit ? kInk : kText, 0);
}

void slices_changed()
{
    if (s_starts.empty()) s_starts.push_back(s_regA);
    s_sel = std::max(0, std::min(s_sel, (int)s_starts.size() - 1));
    paint_pads();
    show_buttons();
    show_info();
    if (s_wave) lv_obj_invalidate(s_wave);
}

/// The mode's own slices, from what is known of the loop.
void recompute()
{
    if (s_regB <= s_regA) return;
    if (s_mode == Mode::Hits && !s_env.empty()) {
        s_starts = crosspad::SliceMath::onsets(s_env, s_envHop, s_envStart, s_regA, s_regB, s_sens, s_rate / 20);
    } else if (s_mode == Mode::Equal) {
        s_starts = crosspad::SliceMath::equal(s_regA, s_regB, (uint8_t)s_count);
    } else {
        s_starts = {s_regA};
    }
    slices_changed();
}

void load(const std::string& file, uint32_t a, uint32_t b)
{
    if (auto* st = tools()) st->stopPreview();
    s_playing = s_chopping = false;
    s_file = file;
    s_regA = a;
    s_regB = b;
    s_starts.clear();
    s_env.clear();
    s_sel = 0;
    s_ov = crosspad::WaveOverview{};
    s_loading = true;
    request_view(a, b);
    show_buttons();
    show_info();
}

/// A slice heard once.
void audition(int i)
{
    auto* st = tools();
    if (!st || i < 0 || i >= (int)s_starts.size()) return;
    s_playing = false;
    st->preview(s_file, s_starts[i], slice_end(i), false);
}

void show_info()
{
    if (!s_info) return;
    char buf[112];
    if (s_file.empty()) {
        std::snprintf(buf, sizeof(buf), "Pick a take -- or record one in the Recorder and press SLICE there.");
    } else if (s_loading) {
        std::snprintf(buf, sizeof(buf), "Reading %s ...", base_of(s_file).c_str());
    } else if (s_mode == Mode::Free && s_starts.size() <= 1 && !s_chopping) {
        std::snprintf(buf, sizeof(buf), "FREE: hit a pad -- the loop plays -- then hit pads to cut it.");
    } else if (s_chopping) {
        std::snprintf(buf, sizeof(buf), "Cutting: %u slices -- dark pads cut, lit pads play from their cut.", (unsigned)s_starts.size());
    } else {
        const uint32_t a = s_starts[s_sel], b = slice_end(s_sel);
        std::snprintf(buf, sizeof(buf), "Slice %d/%u   %.3f - %.3f s   (%.3f s)", s_sel + 1, (unsigned)s_starts.size(),
                      (double)(a - s_regA) / s_rate, (double)(b - s_regA) / s_rate, (double)(b - a) / s_rate);
    }
    set_text(s_info, buf);
}

// ── editable buttons: the value, SEL, MOVE ─────────────────────────────────

int turn(lv_event_t* e)
{
    const uint32_t key = lv_event_get_key(e);
    return key == LV_KEY_RIGHT || key == LV_KEY_UP ? 1 : (key == LV_KEY_LEFT || key == LV_KEY_DOWN ? -1 : 0);
}

/// Shared by the three: FOCUSED in edit mode (the encoder's first click) or
/// a tap starts it; CLICKED in edit mode ends it, or with @p twoStage goes
/// from coarse to fine first. Returns true when the event was one of those.
bool edit_event(lv_event_t* e, Edit& edit, bool twoStage)
{
    lv_obj_t* b = static_cast<lv_obj_t*>(lv_event_get_current_target(e));
    lv_group_t* g = lv_obj_get_group(b);
    const lv_event_code_t code = lv_event_get_code(e);
    if (code == LV_EVENT_FOCUSED) {
        const bool editing = g && lv_group_get_editing(g);
        if (editing && edit == Edit::None) edit = Edit::Coarse;
        else if (!editing) edit = Edit::None;
        return true;
    }
    if (code != LV_EVENT_CLICKED) return false;
    if (edit == Edit::Coarse && twoStage) {
        edit = Edit::Fine;
    } else if (edit != Edit::None) {
        edit = Edit::None;
        if (g) lv_group_set_editing(g, false);
    } else {
        edit = Edit::Coarse;
        if (g) { lv_group_focus_obj(b); lv_group_set_editing(g, true); }
    }
    return true;
}

void param_cb(lv_event_t* e)
{
    if (edit_event(e, s_editParam, false)) { show_buttons(); return; }
    const int d = turn(e);
    if (!d || s_editParam == Edit::None) return;
    if (s_mode == Mode::Hits)       s_sens = std::max(0, std::min(100, s_sens + d * 5));
    else if (s_mode == Mode::Equal) s_count = std::max(2, std::min((int)crosspad::SliceMath::kMaxSlices, s_count + d));
    else return;
    recompute();
}

void sel_cb(lv_event_t* e)
{
    if (edit_event(e, s_editSel, false)) { show_buttons(); return; }
    const int d = turn(e);
    if (!d || s_editSel == Edit::None || s_starts.empty()) return;
    s_sel = std::max(0, std::min((int)s_starts.size() - 1, s_sel + d));
    audition(s_sel);
    slices_changed();
}

void move_cb(lv_event_t* e)
{
    const Edit was = s_editMove;
    if (edit_event(e, s_editMove, true)) {
        if (s_editMove == Edit::Fine && was != Edit::Fine && !s_starts.empty()) {
            const uint32_t m = s_starts[s_sel];
            request_view(m > kZoomFrames ? m - kZoomFrames : 0, m + kZoomFrames);
        } else if (was == Edit::Fine && s_editMove != Edit::Fine) {
            request_view(s_regA, s_regB);
        }
        show_buttons();
        return;
    }
    const int d = turn(e);
    if (!d || s_editMove == Edit::None || s_starts.empty()) return;
    const uint32_t viewLen = s_ov.end > s_ov.start ? s_ov.end - s_ov.start : 1;
    const uint32_t step = s_editMove == Edit::Coarse ? std::max<uint32_t>(1, viewLen / kCols)
                                                     : std::max<uint32_t>(1, s_rate / 1000);
    /* Between its neighbours, a slice's worth from each; the first slice's
     * start trims the loop's front. */
    const int64_t lo = s_sel > 0 ? (int64_t)s_starts[s_sel - 1] + min_gap() : 0;
    const int64_t hi = (int64_t)slice_end(s_sel) - min_gap();
    int64_t v = (int64_t)s_starts[s_sel] + (int64_t)d * step;
    v = std::max(lo, std::min(hi, v));
    s_starts[s_sel] = (uint32_t)v;
    if (s_sel == 0) s_regA = s_starts[0];
    const uint32_t m = s_starts[s_sel];
    if (s_editMove == Edit::Fine && (m < s_ov.start + kZoomFrames / 4 || m > s_ov.end - kZoomFrames / 4)) {
        request_view(m > kZoomFrames ? m - kZoomFrames : 0, m + kZoomFrames);
    }
    slices_changed();
}

// ── actions ────────────────────────────────────────────────────────────────

void back_cb(lv_event_t* e)
{
    auto* app = static_cast<App*>(lv_event_get_user_data(e));
    if (app) app->destroyAppAsync();
}

void mode_cb(lv_event_t*)
{
    s_mode = (Mode)(((int)s_mode + 1) % 3);
    s_chopping = false;
    if (auto* st = tools()) { st->stopPreview(); s_playing = false; }
    recompute();
}

void play_cb(lv_event_t*)
{
    auto* st = tools();
    if (!st || s_file.empty()) return;
    if (s_playing || s_chopping) { st->stopPreview(); s_playing = s_chopping = false; }
    else s_playing = st->preview(s_file, s_regA, s_regB, true);
    show_buttons();
}

/// + splits the selected slice: where the loop is playing, if it is in it,
/// else in its middle.
void add_cb(lv_event_t*)
{
    if (s_starts.empty() || s_starts.size() >= crosspad::SliceMath::kMaxSlices) return;
    const uint32_t a = s_starts[s_sel], b = slice_end(s_sel);
    if (b - a < 2 * min_gap()) return;
    uint32_t at = a + (b - a) / 2;
    auto* st = tools();
    if (st && st->previewing()) {
        const uint32_t p = st->previewFrame();
        if (p > a + min_gap() && p + min_gap() < b) at = p;
    }
    s_starts.insert(s_starts.begin() + s_sel + 1, at);
    s_sel += 1;
    slices_changed();
}

/// DEL joins the selected slice to the one before it.
void del_cb(lv_event_t*)
{
    if (s_sel <= 0 || s_sel >= (int)s_starts.size()) return;
    s_starts.erase(s_starts.begin() + s_sel);
    s_sel -= 1;
    slices_changed();
}

// ── overlays: the takes, the kit's name ────────────────────────────────────

void close_overlay()
{
    if (!s_overlay) return;
#ifdef SLI_USE_ENCODER
    if (indev_encoderTop && s_encGroup) lv_indev_set_group(indev_encoderTop, s_encGroup);
    lv_group_delete_deferred(s_overlayGroup);     // the click that closed it is still being read
    s_overlayGroup = nullptr;
#endif
    lv_obj_delete_async(s_overlay);
    s_overlay = nullptr;
    s_nameArea = nullptr;
}

void open_overlay(const char* title)
{
    s_overlay = lv_obj_create(s_root);
    lv_obj_remove_style_all(s_overlay);
    lv_obj_set_size(s_overlay, LV_PCT(100), LV_PCT(100));
    lv_obj_add_flag(s_overlay, LV_OBJ_FLAG_FLOATING);
    lv_obj_set_style_bg_color(s_overlay, kBg, 0);
    lv_obj_set_style_bg_opa(s_overlay, LV_OPA_COVER, 0);
    lv_obj_set_style_pad_all(s_overlay, 6, 0);
    lv_obj_set_style_pad_row(s_overlay, 3, 0);
    lv_obj_set_flex_flow(s_overlay, LV_FLEX_FLOW_COLUMN);
#ifdef SLI_USE_ENCODER
    s_overlayGroup = lv_group_create();
    if (indev_encoderTop) lv_indev_set_group(indev_encoderTop, s_overlayGroup);
#endif
    lv_obj_t* t = lv_label_create(s_overlay);
    lv_obj_set_style_text_color(t, kText, 0);
    lv_label_set_text(t, title);
}

void overlay_item(const char* text, lv_event_cb_t cb, void* ud)
{
    lv_obj_t* b = lv_button_create(s_overlay);
    lv_obj_set_size(b, LV_PCT(100), 24);
    lv_obj_set_style_radius(b, 6, 0);
    lv_obj_set_style_shadow_width(b, 0, 0);
    lv_obj_set_style_bg_color(b, kPanel, 0);
    lv_obj_set_style_outline_color(b, kMark, LV_STATE_FOCUS_KEY);
    lv_obj_set_style_outline_width(b, 2, LV_STATE_FOCUS_KEY);
    lv_obj_add_event_cb(b, cb, LV_EVENT_CLICKED, ud);
    lv_obj_t* l = lv_label_create(b);
    lv_obj_set_style_text_color(l, kText, 0);
    lv_label_set_long_mode(l, LV_LABEL_LONG_DOT);
    lv_obj_set_width(l, LV_PCT(100));
    lv_label_set_text(l, text);
    lv_obj_center(l);
#ifdef SLI_USE_ENCODER
    if (s_overlayGroup) lv_group_add_obj(s_overlayGroup, b);
#endif
}

/* The card, a folder at a time: FILES opens the recordings, or the folder
 * last looked in; the rows are paged, forty objects rather than a folder's
 * worth. */
constexpr uint32_t kPageRows = 40;
std::string          s_browseDir;      // "" = the recordings, until a folder is opened
crosspad::CardFolder s_folder;
uint32_t             s_page = 0;

void browse(const std::string& dir)
{
    auto* st = tools();
    if (!st) return;
    close_overlay();
    if (st->requestFolder(dir)) {
        s_listPending = true;
        set_text(s_info, "Reading the card ...");
    } else {
        set_text(s_info, "The card is busy -- try again.");
    }
}

void fill_list();

void entry_cb(lv_event_t* e)
{
    const size_t i = (size_t)(intptr_t)lv_event_get_user_data(e);
    if (i >= s_folder.size()) return;
    const std::string path = s_folder.path + "/" + s_folder.name(i);
    if (s_folder.isFolder(i)) { browse(path); return; }
    close_overlay();
    load(path, 0, 0);
}

void fill_list()
{
    auto* st = tools();
    if (!st) return;
    const std::string root = st->cardRoot();
    const bool atRoot = s_folder.path.size() <= root.size();
    std::string title = atRoot ? std::string("/") : s_folder.path.substr(root.size());
    if (title.size() > 36) title = "..." + title.substr(title.size() - 33);
    open_overlay(title.c_str());
    if (!atRoot) {
        overlay_item(LV_SYMBOL_UP " ..", [](lv_event_t*) {
            browse(s_folder.path.substr(0, s_folder.path.find_last_of('/')));
        }, nullptr);
    }
    if (s_page) {
        overlay_item(LV_SYMBOL_UP " earlier", [](lv_event_t*) {
            s_page -= std::min(s_page, kPageRows);
            close_overlay();
            fill_list();
        }, nullptr);
    }
    const size_t end = std::min<size_t>(s_folder.size(), (size_t)s_page + kPageRows);
    for (size_t i = s_page; i < end; ++i) {
        const std::string label = std::string(s_folder.isFolder(i) ? LV_SYMBOL_DIRECTORY " " : LV_SYMBOL_AUDIO " ") +
                                  s_folder.name(i);
        overlay_item(label.c_str(), entry_cb, (void*)(intptr_t)i);
    }
    if (end < s_folder.size()) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), LV_SYMBOL_DOWN " %u more", (unsigned)(s_folder.size() - end));
        overlay_item(buf, [](lv_event_t*) {
            s_page += kPageRows;
            close_overlay();
            fill_list();
        }, nullptr);
    }
    auto note = [](const char* text) {
        lv_obj_t* l = lv_label_create(s_overlay);
        lv_obj_set_style_text_color(l, kDim, 0);
        lv_label_set_text(l, text);
    };
    if (!s_folder.size()) note("No folders or WAVs here.");
    if (s_folder.truncated) note("Only the first 2000 are listed.");
    overlay_item(LV_SYMBOL_CLOSE " Cancel", [](lv_event_t*) { close_overlay(); }, nullptr);
}

void file_cb(lv_event_t*)
{
    if (s_overlay) return;
    browse(s_browseDir);
}

void name_done_cb(lv_event_t* e)
{
    const bool ok = lv_event_get_code(e) == LV_EVENT_READY;
    const std::string name = s_nameArea ? lv_textarea_get_text(s_nameArea) : "";
    close_overlay();
    auto* st = tools();
    if (!ok || !st || name.empty()) return;
    crosspad::SliceKit kit;
    kit.source = s_file;
    kit.name = name;
    kit.starts = s_starts;
    kit.end = s_regB;
    if (st->requestExportKit(kit)) {
        s_exportPending = true;
        set_text(s_info, "Writing the kit ...");
    } else {
        set_text(s_info, "Not saved: that name cannot be a folder, or the card is busy.");
    }
}

/// SAVE KIT asks for a name, the take's own by default.
void save_cb(lv_event_t*)
{
    if (s_file.empty() || s_starts.empty() || s_overlay) return;
    if (auto* st = tools()) { st->stopPreview(); s_playing = s_chopping = false; }
    std::string name = base_of(s_file);
    if (name.size() > 4) name.resize(name.size() - 4);
    for (auto& c : name) c = (char)toupper((unsigned char)c);
    open_overlay("Kit name");
    lv_obj_set_style_pad_row(s_overlay, 2, 0);
    s_nameArea = lv_textarea_create(s_overlay);
    lv_textarea_set_one_line(s_nameArea, true);
    lv_textarea_set_max_length(s_nameArea, 32);
    lv_textarea_set_text(s_nameArea, name.c_str());
    lv_obj_set_width(s_nameArea, LV_PCT(100));
    lv_obj_t* kb = lv_keyboard_create(s_overlay);
    lv_obj_set_size(kb, LV_PCT(100), LV_PCT(100));
    lv_obj_set_flex_grow(kb, 1);
    lv_keyboard_set_textarea(kb, s_nameArea);
    lv_obj_add_event_cb(kb, name_done_cb, LV_EVENT_READY, nullptr);
    lv_obj_add_event_cb(kb, name_done_cb, LV_EVENT_CANCEL, nullptr);
#ifdef SLI_USE_ENCODER
    if (s_overlayGroup) {
        lv_group_add_obj(s_overlayGroup, kb);
        lv_group_focus_obj(kb);
        lv_group_set_editing(s_overlayGroup, true);
    }
#endif
    show_buttons();
}

bool back_handler(lv_obj_t*)
{
    if (s_overlay) { close_overlay(); return true; }
    if (s_editParam != Edit::None || s_editSel != Edit::None || s_editMove != Edit::None) {
#ifdef SLI_USE_ENCODER
        if (s_encGroup) lv_group_set_editing(s_encGroup, false);
#endif
        if (s_editMove == Edit::Fine) request_view(s_regA, s_regB);
        s_editParam = s_editSel = s_editMove = Edit::None;
        show_buttons();
        return true;
    }
    return false;
}

// ── the timer ──────────────────────────────────────────────────────────────

/// Pad hits: in FREE a pad with a slice plays the loop from that slice's
/// start, a dark one cuts the loop where it is heard (or starts it, when it
/// is not playing); the other modes play the pad's slice.
void take_hits()
{
    auto* st = tools();
    uint32_t t = s_hitTail.load(std::memory_order_relaxed);
    const uint32_t h = s_hitHead.load(std::memory_order_acquire);
    bool changed = false;
    for (; t != h; ++t) {
        const PadHit hit = s_hits[t % kHitQueue];
        if (!st || s_file.empty() || s_regB <= s_regA) continue;
        if (s_mode != Mode::Free) {
            if (hit.pad < s_starts.size()) { s_sel = hit.pad; audition(hit.pad); changed = true; }
            continue;
        }
        if (hit.pad < s_starts.size() || !s_chopping || !hit.playing) {
            const int from = hit.pad < s_starts.size() ? hit.pad : 0;
            s_sel = from;
            s_chopping = st->preview(s_file, s_starts.empty() ? s_regA : s_starts[from], s_regB, false);
            s_playing = false;
        } else if (s_starts.size() < crosspad::SliceMath::kMaxSlices) {
            /* A cut after a lit pad restarted the loop lands between earlier
             * ones; the slices stay in order, a pad each. */
            const auto it = std::upper_bound(s_starts.begin(), s_starts.end(), hit.frame);
            const uint32_t next = it == s_starts.end() ? s_regB : *it;
            if (it != s_starts.begin() && hit.frame > *(it - 1) + min_gap() && hit.frame + min_gap() < next) {
                s_sel = (int)(s_starts.insert(it, hit.frame) - s_starts.begin());
            }
        }
        changed = true;
    }
    s_hitTail.store(t, std::memory_order_release);
    if (changed) slices_changed();
}

void tick_cb(lv_timer_t*)
{
    auto* st = tools();
    if (!st) return;
    take_hits();

    const crosspad::ToolJob job = st->job();
    if (job != crosspad::ToolJob::Busy) {
        if (s_ovPending) {
            s_ovPending = false;
            if (job == crosspad::ToolJob::Ready) {
                crosspad::WaveOverview ov = st->overview();
                if (ov.file == s_file) {
                    if (ov.sampleRate) s_rate = ov.sampleRate;
                    if (s_loading) {
                        if (!s_regB || s_regB > ov.frames) s_regB = ov.end;
                        s_loading = false;
                    }
                    /* The loop's own picture carries the envelope HITS cuts by;
                     * a zoom's does not replace it. Slices are made from it
                     * once, for a newly loaded loop -- after that only a mode
                     * or value turned remakes them, never a picture coming
                     * back, or it would undo the markers moved by hand. */
                    if (ov.start == s_regA && ov.end == s_regB) {
                        s_env = ov.envelope;
                        s_envHop = ov.hop;
                        s_envStart = ov.start;
                        if (s_starts.empty()) recompute();
                    }
                    s_ov = std::move(ov);
                }
            } else {
                s_loading = false;
                set_text(s_info, st->jobResult().c_str());
            }
            if (s_wave) lv_obj_invalidate(s_wave);
            show_info();
        } else if (s_listPending) {
            s_listPending = false;
            show_info();
            if (job == crosspad::ToolJob::Ready) {
                s_folder = st->folder();
                s_browseDir = s_folder.path;
                s_page = 0;
                fill_list();
            } else {
                set_text(s_info, st->jobResult().c_str());
            }
        } else if (s_exportPending) {
            s_exportPending = false;
            char buf[112];
            std::snprintf(buf, sizeof(buf), job == crosspad::ToolJob::Ready ? "Saved the kit %s -- it is in Kits now."
                                                                            : "Not saved: %s",
                          st->jobResult().c_str());
            set_text(s_info, buf);
        } else if (s_viewWanted) {
            request_view(s_wantA, s_wantB);
        }
    } else if (s_exportPending) {
        char buf[40];
        std::snprintf(buf, sizeof(buf), "Writing the kit ... %u%%", (unsigned)st->jobProgress());
        set_text(s_info, buf);
    }

    const bool heard = st->previewing();
    if ((s_playing || s_chopping) && !heard) {
        s_playing = s_chopping = false;
        show_buttons();
        show_info();
    }
    move_head();
    (void)heard;
}

// ── building ───────────────────────────────────────────────────────────────

void style_button(lv_obj_t* b, int32_t w, int32_t h)
{
    lv_obj_set_size(b, w, h);
    lv_obj_set_style_pad_all(b, 0, 0);
    lv_obj_set_style_radius(b, 6, 0);
    lv_obj_set_style_shadow_width(b, 0, 0);
    lv_obj_set_style_bg_color(b, kPanel, 0);
    lv_obj_set_style_bg_color(b, kAccent, LV_STATE_EDITED);
    lv_obj_set_style_outline_color(b, kMark, LV_STATE_FOCUS_KEY);
    lv_obj_set_style_outline_width(b, 2, LV_STATE_FOCUS_KEY);
}

lv_obj_t* add_label(lv_obj_t* b, const char* text)
{
    lv_obj_t* l = lv_label_create(b);
    lv_obj_set_style_text_font(l, &lv_font_montserrat_10, 0);
    lv_obj_set_style_text_color(l, kText, 0);
    lv_obj_set_style_text_color(l, kInk, LV_STATE_EDITED);
    lv_label_set_long_mode(l, LV_LABEL_LONG_DOT);
    lv_obj_set_style_max_width(l, LV_PCT(100), 0);
    lv_label_set_text(l, text);
    lv_obj_center(l);
    return l;
}

void add_to_group(lv_obj_t* b)
{
#ifdef SLI_USE_ENCODER
    if (s_encGroup) lv_group_add_obj(s_encGroup, b);
#else
    (void)b;
#endif
}

lv_obj_t* make_button(lv_obj_t* parent, int32_t w, const char* text, lv_event_cb_t cb, void* ud = nullptr)
{
    lv_obj_t* b = lv_button_create(parent);
    style_button(b, w, 22);
    lv_obj_add_event_cb(b, cb, LV_EVENT_CLICKED, ud);
    add_label(b, text);
    add_to_group(b);
    return b;
}

lv_obj_t* make_editable(lv_obj_t* parent, int32_t w, const char* text, lv_event_cb_t cb)
{
    lv_obj_t* b = lv_CreateEditableButton(parent);
    style_button(b, w, 22);
    lv_obj_add_event_cb(b, cb, LV_EVENT_KEY, nullptr);
    lv_obj_add_event_cb(b, cb, LV_EVENT_CLICKED, nullptr);
    lv_obj_add_event_cb(b, cb, LV_EVENT_FOCUSED, nullptr);
    add_label(b, text);
    add_to_group(b);
    return b;
}

lv_obj_t* make_row(lv_obj_t* parent, int32_t h)
{
    lv_obj_t* r = lv_obj_create(parent);
    lv_obj_remove_style_all(r);
    lv_obj_set_size(r, LV_PCT(100), h);
    lv_obj_set_flex_flow(r, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(r, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_column(r, 3, 0);
    lv_obj_remove_flag(r, LV_OBJ_FLAG_SCROLLABLE);
    return r;
}

} // namespace

lv_obj_t* Slicer_create(lv_obj_t* parent, App* app)
{
#ifdef SLI_USE_ENCODER
    if (indev_encoderTop) s_prevEncGroup = lv_indev_get_group(indev_encoderTop);
    if (!s_encGroup) s_encGroup = lv_group_create();
    if (indev_encoderTop) lv_indev_set_group(indev_encoderTop, s_encGroup);
#endif
    if (!s_padLogic) {
        s_padLogic = std::make_shared<SlicerPadLogic>();
        crosspad::getPadManager().registerPadLogic("Slicer", s_padLogic);
    }

    s_root = lv_obj_create(parent);
    lv_obj_set_size(s_root, LV_PCT(100), LV_PCT(100));
    lv_obj_set_style_bg_color(s_root, kBg, 0);
    lv_obj_set_style_border_width(s_root, 0, 0);
    lv_obj_set_style_radius(s_root, 0, 0);
    lv_obj_set_style_pad_all(s_root, 5, 0);
    lv_obj_set_style_pad_row(s_root, 4, 0);
    lv_obj_set_flex_flow(s_root, LV_FLEX_FLOW_COLUMN);
    lv_obj_remove_flag(s_root, LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t* top = make_row(s_root, 22);
    make_button(top, 30, LV_SYMBOL_LEFT, back_cb, app);
    s_fileBtn = make_button(top, 100, "", file_cb);
    s_modeBtn = make_button(top, 44, "HITS", mode_cb);
    s_paramBtn = make_editable(top, 60, "SENS 50", param_cb);
    lv_obj_t* gap = lv_obj_create(top);
    lv_obj_remove_style_all(gap);
    lv_obj_set_flex_grow(gap, 1);
    s_playBtn = make_button(top, 40, LV_SYMBOL_PLAY, play_cb);

    s_wave = lv_obj_create(s_root);
    lv_obj_remove_style_all(s_wave);
    lv_obj_set_width(s_wave, LV_PCT(100));
    lv_obj_set_flex_grow(s_wave, 1);
    lv_obj_set_style_bg_color(s_wave, kPanel, 0);
    lv_obj_set_style_bg_opa(s_wave, LV_OPA_COVER, 0);
    lv_obj_set_style_radius(s_wave, 6, 0);
    lv_obj_set_style_pad_all(s_wave, 3, 0);
    lv_obj_remove_flag(s_wave, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_add_event_cb(s_wave, wave_draw_cb, LV_EVENT_DRAW_MAIN_END, nullptr);

    s_info = lv_label_create(s_root);
    lv_obj_set_style_text_font(s_info, &lv_font_montserrat_10, 0);
    lv_obj_set_style_text_color(s_info, kDim, 0);
    lv_label_set_long_mode(s_info, LV_LABEL_LONG_DOT);
    lv_obj_set_size(s_info, LV_PCT(100), 12);     // one line: a long one ends in dots, the picture keeps its height
    lv_label_set_text(s_info, "");

    lv_obj_t* tools_row = make_row(s_root, 24);
    s_selBtn = make_editable(tools_row, 52, "SEL 1", sel_cb);
    s_moveBtn = make_editable(tools_row, 52, "MOVE", move_cb);
    make_button(tools_row, 34, LV_SYMBOL_PLUS, add_cb);
    make_button(tools_row, 40, "DEL", del_cb);
    lv_obj_t* gap2 = lv_obj_create(tools_row);
    lv_obj_remove_style_all(gap2);
    lv_obj_set_flex_grow(gap2, 1);
    make_button(tools_row, 76, LV_SYMBOL_SAVE " SAVE KIT", save_cb);

    s_hitTail.store(s_hitHead.load());
    std::string file;
    uint32_t a = 0, b = 0;
    if (auto* st = tools(); st && st->takeHandoff(file, a, b)) {
        load(file, a, b);
    } else if (!s_file.empty()) {
        s_loading = true;                   // back to the loop left last time
        request_view(s_regA, s_regB);
    }
    show_buttons();
    show_info();

#ifdef SLI_USE_ENCODER
    if (s_encGroup) lv_group_focus_obj(s_selBtn);
#endif
    s_timer = lv_timer_create(tick_cb, kTickMs, nullptr);

    if (app) {
        app->setOnBack(back_handler);
        app->setOnShow([](lv_obj_t*) {
            if (s_timer) lv_timer_resume(s_timer);
#ifdef SLI_USE_ENCODER
            if (indev_encoderTop && s_encGroup) lv_indev_set_group(indev_encoderTop, s_encGroup);
#endif
        });
        app->setOnHide([](lv_obj_t*) {
            if (s_timer) lv_timer_pause(s_timer);
            if (auto* st = tools()) st->stopPreview();
#ifdef SLI_USE_ENCODER
            if (indev_encoderTop && s_prevEncGroup) lv_indev_restore_group(indev_encoderTop, s_prevEncGroup);
#endif
        });
    }
    return s_root;
}

void Slicer_destroy(lv_obj_t* root)
{
    if (s_timer) {
        lv_timer_delete(s_timer);
        s_timer = nullptr;
    }
    if (auto* st = tools()) st->stopPreview();
#ifdef SLI_USE_ENCODER
    if (s_overlayGroup) { lv_group_delete(s_overlayGroup); s_overlayGroup = nullptr; }
#endif
    s_overlay = s_nameArea = nullptr;
    s_root = s_wave = s_fileBtn = s_modeBtn = s_paramBtn = s_playBtn = s_selBtn = s_moveBtn = s_info = nullptr;
    s_playing = s_chopping = s_ovPending = s_viewWanted = s_listPending = s_exportPending = false;
    s_editParam = s_editSel = s_editMove = Edit::None;
    s_headX = -1;

#ifdef SLI_USE_ENCODER
    if (indev_encoderTop && s_prevEncGroup) lv_indev_restore_group(indev_encoderTop, s_prevEncGroup);
    if (s_encGroup) {
        lv_group_delete(s_encGroup);
        s_encGroup = nullptr;
    }
    s_prevEncGroup = nullptr;
#endif
    lv_obj_delete_async(root);
}

// The pads are the Slicer's while it is open: each plays its slice, and in
// FREE they cut the loop.
REGISTER_APP_PL(Slicer, nullptr, "slicer.png",
                Slicer_create, Slicer_destroy,
                nullptr, nullptr, nullptr, 0, "Slicer")
