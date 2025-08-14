# ConforAI
Human AI


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
