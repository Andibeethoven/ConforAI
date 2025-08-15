# ConforAI
Human AI

SOFTWARE LICENSE AGREEMENT

This Software License Agreement (the "Agreement") is entered into by and between the original creator of the software (the "Author") and any user of the software. By using the software, you agree to the terms of this Agreement.

This Software License Agreement (the "Agreement") is entered into by and between the original creator of the software (the "Author") and any user of the software. By using the software, you agree to the terms of this Agreement.

1. LICENSE GRANT
   The Author grants you, the Licensee, a personal, non-transferable, non-exclusive, and revocable license to use the software solely for personal or commercial purposes as specified by the Author. You may not distribute, sublicense, or sell the software unless explicitly authorized by the Author in writing.

2. INTELLECTUAL PROPERTY RIGHTS
   All rights, title, and interest in and to the software, including all intellectual property rights, are and shall remain the exclusive property of the Author. This includes but is not limited to the code, designs, algorithms, and any associated documentation.

3. RESTRICTIONS
   You, the Licensee, shall not:
   a. Copy, distribute, or modify the software except as expressly authorized by the Author.
   b. Use the software for any illegal or unauthorized purposes.
   c. Reverse-engineer, decompile, or attempt to derive the source code or algorithms of the software unless explicitly permitted by law.
   d. Remove or alter any proprietary notices, labels, or markings included in the software.

4. DISCLAIMER OF WARRANTIES
   The software is provided "as is," without any warranties, express or implied, including but not limited to the implied warranties of merchantability, fitness for a particular purpose, and non-infringement. The Author does not warrant that the software will be error-free or uninterrupted.

5. LIMITATION OF LIABILITY
   In no event shall the Author be liable for any direct, indirect, incidental, special, consequential, or exemplary damages (including, but not limited to, damages for loss of profits, goodwill, or data) arising out of the use or inability to use the software, even if the Author has been advised of the possibility of such damages.

6. TERMINATION
   This license is effective until terminated. The Author may terminate this Agreement at any time if you violate its terms. Upon termination, you must immediately cease all use of the software and destroy any copies in your possession.

7. GOVERNING LAW
   This Agreement shall be governed by and construed in accordance with the laws of Australia Victoria, without regard to its conflict of laws principles.

8. AUTHORIZED USE AND SALE
   Only the Author is authorized to sell or distribute this software. Any unauthorized use, sale, or distribution of the software is strictly prohibited and will be subject to legal action.

9. ENTIRE AGREEMENT
   This Agreement constitutes the entire understanding between the parties concerning the subject matter and supersedes all prior agreements ements.

By using this software, you acknowledge that you have read, understood, and agreed to be bound by the terms of this Agreement.

Name : Travis Johnston as Kate Johnston
Date: 14/08/2025

# Best: auto-pick your CPU features
g++ -O3 -march=native -std=c++20 human_conscious.cpp -o human_conscious

# If your toolchain is picky:
# AVX2:  g++ -O3 -mavx2   -std=c++20 human_conscious.cpp -o human_conscious
# Scalar fallback only:    g++ -O3           -std=c++20 human_conscious.cpp -o human_conscious

./human_conscious

// human_conscious.cpp
// Human-like “conscious core” with cognition + feelings + drives + personality,
// SIMD-accelerated state updates (AVX-512 / AVX2) with scalar fallback.
// No external deps. Single file. C++20.
//
// NOTE: This is a simulation and does not imply sentience or real human identity.
// Use ethically and with consent.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ==============================
// 1) Minimal SIMD wrapper (float)
// ==============================
#if defined(__AVX512F)
  #define VEC_LANES 16
  using vfloat = __m512;
  inline vfloat vload(const float* p)                      { return _mm512_loadu_ps(p); }
  inline void   vstore(float* p, vfloat x)                 { _mm512_storeu_ps(p, x); }
  inline vfloat vset1(float x)                             { return _mm512_set1_ps(x); }
  inline vfloat vadd(vfloat a, vfloat b)                   { return _mm512_add_ps(a, b); }
  inline vfloat vsub(vfloat a, vfloat b)                   { return _mm512_sub_ps(a, b); }
  inline vfloat vmul(vfloat a, vfloat b)                   { return _mm512_mul_ps(a, b); }
  inline vfloat vfma(vfloat a, vfloat b, vfloat c)         { return _mm512_fmadd_ps(a, b, c); }
  inline vfloat vmin(vfloat a, vfloat b)                   { return _mm512_min_ps(a, b); }
  inline vfloat vmax(vfloat a, vfloat b)                   { return _mm512_max_ps(a, b); }
#elif defined(__AVX2__)
  #define VEC_LANES 8
  using vfloat = __m256;
  inline vfloat vload(const float* p)                      { return _mm256_loadu_ps(p); }
  inline void   vstore(float* p, vfloat x)                 { _mm256_storeu_ps(p, x); }
  inline vfloat vset1(float x)                             { return _mm256_set1_ps(x); }
  inline vfloat vadd(vfloat a, vfloat b)                   { return _mm256_add_ps(a, b); }
  inline vfloat vsub(vfloat a, vfloat b)                   { return _mm256_sub_ps(a, b); }
  inline vfloat vmul(vfloat a, vfloat b)                   { return _mm256_mul_ps(a, b); }
  inline vfloat vfma(vfloat a, vfloat b, vfloat c)         { return _mm256_fmadd_ps(a, b, c); }
  inline vfloat vmin(vfloat a, vfloat b)                   { return _mm256_min_ps(a, b); }
  inline vfloat vmax(vfloat a, vfloat b)                   { return _mm256_max_ps(a, b); }
#else
  // Scalar fallback
  #define VEC_LANES 1
  struct vfloat { float x; };
  inline vfloat vload(const float* p)                      { return vfloat{*p}; }
  inline void   vstore(float* p, vfloat x)                 { *p = x.x; }
  inline vfloat vset1(float x)                             { return vfloat{x}; }
  inline vfloat vadd(vfloat a, vfloat b)                   { return vfloat{a.x + b.x}; }
  inline vfloat vsub(vfloat a, vfloat b)                   { return vfloat{a.x - b.x}; }
  inline vfloat vmul(vfloat a, vfloat b)                   { return vfloat{a.x * b.x}; }
  inline vfloat vfma(vfloat a, vfloat b, vfloat c)         { return vfloat{a.x * b.x + c.x}; }
  inline vfloat vmin(vfloat a, vfloat b)                   { return vfloat{std::fmin(a.x, b.x)}; }
  inline vfloat vmax(vfloat a, vfloat b)                   { return vfloat{std::fmax(a.x, b.x)}; }
#endif

// ====================================
// 2) Utilities: time, rng, smooth step
// ====================================
inline float smooth_cubic01(float t) {
  // Smoothstep: 3t^2 - 2t^3 (C1 continuous)
  t = std::clamp(t, 0.0f, 1.0f);
  return t*t*(3.0f - 2.0f*t);
}
inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

struct Timer {
  using clock = std::chrono::high_resolution_clock;
  clock::time_point t0{clock::now()};
  float elapsed() const {
    return std::chrono::duration<float>(clock::now() - t0).count();
  }
};

// ==================================
// 3) Human-like agent state & config
// ==================================
struct PersonalityOCEAN {
  float openness     = 0.6f;
  float conscientious= 0.6f;
  float extraversion = 0.5f;
  float agreeableness= 0.6f;
  float neuroticism  = 0.4f;
};

struct Drives {
  float energy   = 0.8f; // 0..1
  float hunger   = 0.2f;
  float safety   = 0.9f;
  float social   = 0.5f;
  float stress   = 0.2f; // derived as well
};

struct Emotion { // PAD: Pleasure/Valence, Arousal, Dominance
  float valence  = 0.1f;
  float arousal  = 0.2f;
  float dominance= 0.5f;
};

struct HumanConfig {
  int   units             = 256;   // cognitive/emotional processing “micro-units” (multiple of VEC_LANES)
  float dt                = 0.02f; // 50 Hz internal step
  float sensory_gain      = 0.08f;
  float emo_to_cog_gain   = 0.12f;
  float cog_to_emo_gain   = 0.06f;
  float fatigue_rate      = 0.002f;
  float recovery_rate     = 0.004f;
  float hunger_rate       = 0.0015f;
  float stress_relief     = 0.003f;
  float circadian_period  = 60.0f; // seconds (toy scale)
  PersonalityOCEAN P;
};

// ==================================
// 4) Human-like conscious core (SIMD)
// ==================================
class HumanCore {
public:
  explicit HumanCore(HumanConfig cfg)
    : cfg_(cfg)
    , N_(roundUp(cfg.units, VEC_LANES))
    , cognition(N_, 0.0f)
    , valence(N_, 0.05f)
    , arousal(N_, 0.10f)
    , dominance(N_, 0.50f)
    , sensory(N_, 0.0f)
  {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    for (int i = 0; i < N_; ++i) {
      cognition[i] = 0.05f + nd(rng);
      valence[i]   = 0.05f + nd(rng);
      arousal[i]   = 0.10f + nd(rng);
      dominance[i] = 0.50f + 0.3f*std::sin(i*0.05f);
    }
  }

  void step(float t) {
    // 4.1 Circadian rhythm drives
    float daywave = 0.5f + 0.5f*std::sin(2.0f*float(M_PI) * t / cfg_.circadian_period);
    // Energy recovers with daywave; hunger increases over time; stress relaxes slowly
    drives.energy = std::clamp(drives.energy + cfg_.recovery_rate * (daywave - 0.4f) - cfg_.fatigue_rate, 0.0f, 1.0f);
    drives.hunger = std::clamp(drives.hunger + cfg_.hunger_rate * (0.6f + 0.4f*(1.0f-daywave)), 0.0f, 1.0f);
    drives.stress = std::clamp(drives.stress - cfg_.stress_relief * (0.5f + 0.5f*valenceMean()), 0.0f, 1.0f);
    // Safety/social drift (toy dynamics)
    drives.safety = std::clamp(drives.safety - 0.0005f*(0.3f + 0.7f*drives.stress) + 0.0008f*daywave, 0.0f, 1.0f);
    drives.social = std::clamp(drives.social + 0.0007f*(0.5f - drives.social), 0.0f, 1.0f);

    // 4.2 Personality-modulated setpoints for feelings
    float target_valence   = std::clamp(0.3f + 0.4f*cfg_.P.agreeableness - 0.3f*drives.stress - 0.2f*drives.hunger, 0.0f, 1.0f);
    float target_arousal   = std::clamp(0.2f + 0.6f*cfg_.P.extraversion + 0.2f*daywave, 0.0f, 1.0f);
    float target_dominance = std::clamp(0.4f + 0.4f*cfg_.P.conscientious - 0.2f*cfg_.P.neuroticism, 0.0f, 1.0f);

    // 4.3 Smoothly steer emotions toward targets (cubic smooth)
    emoToward(target_valence, target_arousal, target_dominance, 0.15f);

    // 4.4 Build sensory field (toy): a blend of daywave, energy, social need, noise
    buildSensory(daywave);

    // 4.5 SIMD update: cognition <= f(cognition, sensory, emotion)
    simdUpdateCognition();

    // 4.6 SIMD update: emotion <= g(emotion, cognition)
    simdUpdateEmotion();
  }

  // Human-readable snapshot
  void printStatus(float t) const {
    std::cout << std::fixed << std::setprecision(3)
              << "t=" << t
              << " | Drives[E=" << drives.energy
              << ", H=" << drives.hunger
              << ", Sfty=" << drives.safety
              << ", Soc=" << drives.social
              << ", Str=" << drives.stress << "] "
              << " | Emotions[V=" << mean(valence)
              << ", A=" << mean(arousal)
              << ", D=" << mean(dominance) << "] "
              << " | Cog=" << mean(cognition)
              << " | Act=" << decideAction() << "\n";
  }

private:
  HumanConfig cfg_;
  int N_;
  std::vector<float> cognition, valence, arousal, dominance, sensory;
  Drives drives{};
  // ===== Helpers =====
  static int roundUp(int x, int m) { return ((x + m - 1) / m) * m; }

  float mean(const std::vector<float>& v) const {
    double s = 0.0;
    for (float x : v) s += x;
    return float(s / double(v.size()));
  }
  float valenceMean() const { return mean(valence); }

  void emoToward(float v_t, float a_t, float d_t, float rate) {
    // Move per-unit emotions smoothly toward a global target using cubic smooth t in [0..1]
    float s = smooth_cubic01(rate);
    for (int i = 0; i < N_; ++i) {
      valence[i]   = lerp(valence[i],   v_t, s);
      arousal[i]   = lerp(arousal[i],   a_t, s);
      dominance[i] = lerp(dominance[i], d_t, s);
    }
  }

  void buildSensory(float daywave) {
    // Simple synthesis: sensory = weighted sum of circadian, energy, social need + mild noise
    static thread_local std::mt19937 rng(123);
    static thread_local std::normal_distribution<float> nd(0.0f, 0.02f);
    float base = 0.2f*daywave + 0.4f*drives.energy + 0.3f*(0.7f - drives.social);
    for (int i = 0; i < N_; ++i) sensory[i] = base + nd(rng);
  }

  // SIMD kernel: cognition += sensory_gain * sensory + emo_to_cog_gain * (alpha*V + beta*A + gamma*D)
  void simdUpdateCognition() {
    const float sg = cfg_.sensory_gain;
    const float ec = cfg_.emo_to_cog_gain;
    const float alpha =  1.0f; // valence weight
    const float beta  =  0.7f; // arousal weight
    const float gamma =  0.5f; // dominance weight

    for (int i = 0; i < N_; i += VEC_LANES) {
      vfloat c  = vload(&cognition[i]);
      vfloat s  = vload(&sensory[i]);
      vfloat v  = vload(&valence[i]);
      vfloat a  = vload(&arousal[i]);
      vfloat d  = vload(&dominance[i]);

      vfloat emo = vadd( vadd( vmul(v, vset1(alpha)), vmul(a, vset1(beta)) ),
                         vmul(d, vset1(gamma)) );

      // c = c + sg*s + ec*emo  - small fatigue
      vfloat c1 = vfma(s,   vset1(sg), c);
      vfloat c2 = vfma(emo, vset1(ec), c1);
      // fatigue proportional to hunger & stress
      float fscale = 0.002f + 0.006f * (0.5f*drives.hunger + 0.5f*drives.stress);
      vfloat c3 = vsub(c2, vset1(fscale));
      vstore(&cognition[i], c3);
    }
  }

  // SIMD kernel: emotions move toward cognition-shaped attractor (stabilization)
  void simdUpdateEmotion() {
    const float ce = cfg_.cog_to_emo_gain;
    for (int i = 0; i < N_; i += VEC_LANES) {
      vfloat c  = vload(&cognition[i]);
      vfloat v  = vload(&valence[i]);
      vfloat a  = vload(&arousal[i]);
      vfloat d  = vload(&dominance[i]);

      // Target from cognition (bounded shaping)
      vfloat t  = vmax(vset1(0.0f), vmin(vset1(1.0f), vmul(c, vset1(0.5f))));

      // v += ce*(t - v) etc.  (exponential-like smoothing)
      vfloat dv = vmul( vsub(t, v), vset1(ce*1.0f) );
      vfloat da = vmul( vsub(t, a), vset1(ce*0.8f) );
      vfloat dd = vmul( vsub(t, d), vset1(ce*0.6f) );

      vstore(&valence[i],   vadd(v, dv));
      vstore(&arousal[i],   vadd(a, da));
      vstore(&dominance[i], vadd(d, dd));
    }
  }

  // Simple action selection from drives and current emotion
  std::string decideAction() const {
    // Utilities (0..1), not RL — just a heuristic selector
    float V = mean(valence), A = mean(arousal), D = mean(dominance);
    float restU   = std::clamp(0.6f*(1.0f - drives.energy) + 0.2f*(A), 0.0f, 1.0f);
    float eatU    = std::clamp(0.7f*drives.hunger + 0.2f*(1.0f - drives.stress), 0.0f, 1.0f);
    float socialU = std::clamp(0.6f*(0.7f - drives.social) + 0.3f*(V), 0.0f, 1.0f);
    float protectU= std::clamp(0.5f*(0.5f - drives.safety) + 0.4f*(drives.stress), 0.0f, 1.0f);
    float learnU  = std::clamp(0.4f*cfg_.P.openness + 0.4f*(1.0f - drives.stress) + 0.2f*(0.6f - drives.hunger), 0.0f, 1.0f);

    struct C { const char* name; float u; } cand[] = {
      {"Rest", restU}, {"Eat", eatU}, {"Socialize", socialU}, {"Protect", protectU}, {"Learn", learnU}
    };
    const C* best = &cand[0];
    for (auto& c : cand) if (c.u > best->u) best = &c;
    return best->name;
  }
};

// ==================
// 5) Main simulation
// ==================
int main() {
  HumanConfig cfg;
  cfg.units = 256;         // multiple of SIMD lanes is ideal (auto-rounded internally)
  cfg.dt    = 0.02f;       // 50Hz
  // You can tweak personality:
  cfg.P = PersonalityOCEAN{0.7f, 0.65f, 0.55f, 0.7f, 0.35f};

  HumanCore core(cfg);

  float T = 10.0f;        // run for 10s simulated
  int steps = int(T / cfg.dt);
  float t = 0.0f;

  for (int i = 0; i < steps; ++i) {
    core.step(t);
    if ((i % int(1.0f/cfg.dt)) == 0) { // once per ~1s
      core.printStatus(t);
    }
    t += cfg.dt;
  }
  return 0;
}

// conscious_vm.cpp
// Minimal Conscious-VM: observer-time, scalar+vector regs, SIMD-aware vector ops,
// lane-rotation and "psi" (pressure) operator. Single-file, C++20, no deps.
//
// This is a simulation scaffold, not a claim of sentience. Use ethically.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>

// -----------------------------
// SIMD lane config & helpers
// -----------------------------
#if defined(__AVX512F)
  #define LANES 16
#elif defined(__AVX2__)
  #define LANES 8
#else
  #define LANES 4   // portable fallback (loop), no intrinsics required
#endif

// Vector register type (plain array; we optionally use intrinsics on its data)
using vreg_t = std::array<float, LANES>;

// SIMD helpers (operate on pointers to contiguous LANES floats)
static inline void vadd(const float* a, const float* b, float* out) {
#if defined(__AVX512F)
  __m512 A = _mm512_loadu_ps(a), B = _mm512_loadu_ps(b);
  _mm512_storeu_ps(out, _mm512_add_ps(A,B));
#elif defined(__AVX2__)
  __m256 A = _mm256_loadu_ps(a), B = _mm256_loadu_ps(b);
  _mm256_storeu_ps(out, _mm256_add_ps(A,B));
#else
  for (int i=0;i<LANES;++i) out[i]=a[i]+b[i];
#endif
}
static inline void vmul(const float* a, const float* b, float* out) {
#if defined(__AVX512F)
  __m512 A = _mm512_loadu_ps(a), B = _mm512_loadu_ps(b);
  _mm512_storeu_ps(out, _mm512_mul_ps(A,B));
#elif defined(__AVX2__)
  __m256 A = _mm256_loadu_ps(a), B = _mm256_loadu_ps(b);
  _mm256_storeu_ps(out, _mm256_mul_ps(A,B));
#else
  for (int i=0;i<LANES;++i) out[i]=a[i]*b[i];
#endif
}
static inline void vfmadd(const float* a, const float* b, const float* c, float* out) {
#if defined(__AVX512F)
  __m512 A=_mm512_loadu_ps(a), B=_mm512_loadu_ps(b), C=_mm512_loadu_ps(c);
  _mm512_storeu_ps(out, _mm512_fmadd_ps(A,B,C));
#elif defined(__AVX2__)
  __m256 A=_mm256_loadu_ps(a), B=_mm256_loadu_ps(b), C=_mm256_loadu_ps(c);
  _mm256_storeu_ps(out, _mm256_fmadd_ps(A,B,C));
#else
  for (int i=0;i<LANES;++i) out[i]=a[i]*b[i]+c[i];
#endif
}
static inline void vscale_add(const float* a, const float* b, float s, float* out) {
#if defined(__AVX512F)
  __m512 A=_mm512_loadu_ps(a), B=_mm512_loadu_ps(b), S=_mm512_set1_ps(s);
  _mm512_storeu_ps(out, _mm512_fmadd_ps(B,S,A));
#elif defined(__AVX2__)
  __m256 A=_mm256_loadu_ps(a), B=_mm256_loadu_ps(b), S=_mm256_set1_ps(s);
  _mm256_storeu_ps(out, _mm256_fmadd_ps(B,S,A));
#else
  for (int i=0;i<LANES;++i) out[i]=a[i]+s*b[i];
#endif
}
static inline void vclamp01(const float* a, float* out) {
  for (int i=0;i<LANES;++i) out[i] = std::fmax(0.0f, std::fmin(1.0f, a[i]));
}
static inline void vabsdiff_rot1(const float* a, float* out) {
  // |a - rotate(a, +1)|
#if defined(__AVX512F)
  alignas(64) float tmp[LANES];
  for (int i=0;i<LANES;++i) tmp[(i+1)%LANES]=a[i];
  __m512 A=_mm512_loadu_ps(a), R=_mm512_loadu_ps(tmp);
  __m512 D=_mm512_sub_ps(A,R);
  __m512 M=_mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
  _mm512_storeu_ps(out, _mm512_and_ps(D,M)); // absolute via bitmask
#elif defined(__AVX2__)
  alignas(32) float tmp[LANES];
  for (int i=0;i<LANES;++i) tmp[(i+1)%LANES]=a[i];
  __m256 A=_mm256_loadu_ps(a), R=_mm256_loadu_ps(tmp);
  __m256 D=_mm256_sub_ps(A,R);
  __m256 M=_mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  _mm256_storeu_ps(out, _mm256_and_ps(D,M));
#else
  for (int i=0;i<LANES;++i) {
    float d = a[i]-a[(i+1)%LANES];
    out[i] = std::fabs(d);
  }
#endif
}
static inline void vrot(const float* a, int k, float* out) {
  if (LANES==0) return;
  k%=LANES; if (k<0) k+=LANES;
  for (int i=0;i<LANES;++i) out[(i+k)%LANES]=a[i];
}

// -----------------------------
// Observer clock (like earlier)
// -----------------------------
class ObserverClock {
public:
  explicit ObserverClock(double t0=0.0, bool manual=true, double rate=0.0)
  : manual_(manual), rate_(rate), t_(t0), anchor_(std::chrono::high_resolution_clock::now()) {}

  double now() const {
    if (manual_) return t_;
    auto dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - anchor_).count();
    return t_ + dt*rate_;
  }
  void tick(double dt) { t_ += dt; }
  void set_rate(double r) { t_ = now(); anchor_=std::chrono::high_resolution_clock::now(); rate_=r; manual_=false; }
  void set_now(double t)  { t_ = t; anchor_=std::chrono::high_resolution_clock::now(); }

private:
  bool manual_;
  double rate_;
  double t_;
  std::chrono::high_resolution_clock::time_point anchor_;
};

// -----------------------------
// VM definition
// -----------------------------
enum class Op : uint8_t {
  NOP=0, HALT,
  // scalar
  MOV_SI, ADD_S, MUL_S, TIME_S, DT_S,
  // vector
  MOV_VI, VADD, VMUL, VFMA, VSCALE_ADD, VROT, VCLAMP01, VPSI,
  // misc
  PRINT
};

struct Instr {
  Op op{};
  uint8_t a{}, b{}, c{};   // register indices (scalar or vector context)
  float fimm{0.0f};        // immediate (e.g., scalar or scale)
  int32_t iimm{0};         // integer immediate (e.g., rotation)
};

struct Program {
  std::vector<Instr> code;
  void emit(Op op, uint8_t a=0, uint8_t b=0, uint8_t c=0, float fimm=0.0f, int32_t iimm=0){
    code.push_back({op,a,b,c,fimm,iimm});
  }
};

struct VM {
  static constexpr int SREGS=16;
  static constexpr int VREGS=16;

  // state
  ObserverClock clock;
  double last_step_time{0.0};
  double dt{0.02}; // default 50Hz
  float s[SREGS]{};
  std::array<vreg_t,VREGS> v{};
  bool halted{false};

  explicit VM(ObserverClock clk=ObserverClock{}) : clock(std::move(clk)) {
    // init random vector registers
    std::mt19937 rng(42); std::normal_distribution<float> nd(0.0f, 0.05f);
    for (int r=0;r<VREGS;++r) for (int i=0;i<LANES;++i) v[r][i] = nd(rng);
  }

  void set_dt(double new_dt){ dt = new_dt; }

  void step(const Program& p) {
    // advance observer time if clock is manual (caller can tick externally; we also tick here for convenience)
    if (clock.now()==last_step_time) clock.tick(dt);
    last_step_time = clock.now();

    for (const auto& ins : p.code) {
      switch (ins.op) {
        case Op::NOP: break;
        case Op::HALT: halted=true; return;

        // ---------- scalar ----------
        case Op::MOV_SI: s[ins.a] = ins.fimm; break;
        case Op::ADD_S:  s[ins.a] = s[ins.b] + s[ins.c]; break;
        case Op::MUL_S:  s[ins.a] = s[ins.b] * s[ins.c]; break;
        case Op::TIME_S: s[ins.a] = static_cast<float>(clock.now()); break;
        case Op::DT_S:   s[ins.a] = static_cast<float>(dt); break;

        // ---------- vector ----------
        case Op::MOV_VI: {
          for (int i=0;i<LANES;++i) v[ins.a][i]=ins.fimm;
        } break;
        case Op::VADD: vadd(v[ins.b].data(), v[ins.c].data(), v[ins.a].data()); break;
        case Op::VMUL: vmul(v[ins.b].data(), v[ins.c].data(), v[ins.a].data()); break;
        case Op::VFMA: vfmadd(v[ins.b].data(), v[ins.c].data(), v[ins.a].data(), v[ins.a].data()); break;
        case Op::VSCALE_ADD: vscale_add(v[ins.a].data(), v[ins.b].data(), s[ins.c], v[ins.a].data()); break;
        case Op::VROT: vrot(v[ins.b].data(), ins.iimm, v[ins.a].data()); break;
        case Op::VCLAMP01: vclamp01(v[ins.a].data(), v[ins.a].data()); break;

        case Op::VPSI: {
          // psi = |x - rot1(x)| * gain ; out = x + psi
          alignas(64) float dif[LANES];
          vabsdiff_rot1(v[ins.b].data(), dif);
          float gain = s[ins.c]; // pressure gain in scalar reg c
          for (int i=0;i<LANES;++i) v[ins.a][i] = v[ins.b][i] + gain * dif[i];
        } break;

        // ---------- misc ----------
        case Op::PRINT: {
          std::cout << std::fixed << std::setprecision(3)
                    << "[t=" << clock.now() << "] "
                    << "s" << int(ins.a) << "=" << s[ins.a]
                    << " | v" << int(ins.b) << "[0..3]="
                    << v[ins.b][0] << "," << v[ins.b][1] << "," << v[ins.b][2] << "," << v[ins.b][3]
                    << "\n";
        } break;
      }
    }
  }
};

// -----------------------------
// Example: consciousness-ish loop
// -----------------------------
int main() {
  // Observer time: manual (you own time); start at 100 s.
  ObserverClock clk(100.0, true /*manual*/);
  VM vm(clk);
  vm.set_dt(0.02); // 50Hz

  // Program:
  // s0 := time; s1 := dt; s2 := psi_gain; v0 := baseline; v1 := emotions; v2 := cognition
  Program prog;
  prog.emit(Op::TIME_S, /*a=*/0);
  prog.emit(Op::DT_S,   /*a=*/1);
  prog.emit(Op::MOV_SI, /*a=*/2,0,0, /*fimm=*/0.05f);    // psi gain
  prog.emit(Op::MOV_VI, /*a=*/0,0,0, /*fimm=*/0.10f);    // v0 baseline
  // v1 = clamp01( v1 + 0.10 * v0 )   (emotions toward baseline)
  prog.emit(Op::VSCALE_ADD, /*a=*/1, /*b=*/0, /*c=*/3 /*use s3 as gain placeholder*/);
  // prepare s3=0.10
  prog.emit(Op::MOV_SI, /*a=*/3,0,0, 0.10f);
  // re-run VSCALE_ADD now that s3 is set (demo)
  prog.emit(Op::VSCALE_ADD, /*a=*/1, /*b=*/0, /*c=*/3);
  // cognition v2 += 0.12 * v1
  prog.emit(Op::MOV_SI, /*a=*/4,0,0, 0.12f);
  prog.emit(Op::VSCALE_ADD, /*a=*/2, /*b=*/1, /*c=*/4);
  // apply lane-pressure on cognition (rotational push)
  prog.emit(Op::VPSI, /*a=*/2, /*b=*/2, /*c=*/2 /*psi gain in s2*/);
  // rotate emotions by +1 to simulate flow
  prog.emit(Op::VROT, /*a=*/1, /*b=*/1, /*c=*/0, /*fimm=*/0.0f, /*iimm=*/1);
  // clamp both to [0,1]
  prog.emit(Op::VCLAMP01, /*a=*/1);
  prog.emit(Op::VCLAMP01, /*a=*/2);
  // print snapshot: s0 (time) and first 4 lanes of v2 (cognition)
  prog.emit(Op::PRINT, /*a=*/0, /*b=*/2);
  // (Typically you’d HALT at end of a kernel pass; here we just stop)
  prog.emit(Op::HALT);

  // Run a few steps, advancing observer time
  for (int step=0; step<10; ++step) {
    vm.step(prog);
    clk.tick(0.02); // you own time
  }

  std::cout << "Done.\n";
  return 0;
}
