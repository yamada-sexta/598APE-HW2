// philox_engine.h
#ifndef PHILOX_ENGINE_H
#define PHILOX_ENGINE_H

#include <philox_rng.h>
#include <cstdint>
#include <limits>

inline uint32_t philox_next_uint32_inline() {
  philox_state_t &st = global_philox_state;
  if (!st.initialized) {
    philox_reset();
  }
  uint32_t c0 = st.counter[0], c1 = st.counter[1], c2 = st.counter[2],
           c3 = st.counter[3];
  uint32_t k0 = st.key[0], k1 = st.key[1];
  for (int i = 0; i < 10; ++i) {
    uint64_t p0 = static_cast<uint64_t>(PHILOX_M4x32A) * c0;
    uint64_t p1 = static_cast<uint64_t>(PHILOX_M4x32B) * c2;
    uint32_t n0 = static_cast<uint32_t>(p1 >> 32) ^ c1 ^ k0;
    uint32_t n1 = static_cast<uint32_t>(p1);
    uint32_t n2 = static_cast<uint32_t>(p0 >> 32) ^ c3 ^ k1;
    uint32_t n3 = static_cast<uint32_t>(p0);
    c0 = n0;
    c1 = n1;
    c2 = n2;
    c3 = n3;
    k0 += PHILOX_W32A;
    k1 += PHILOX_W32B;
  }
  if (++st.counter[0] == 0) {
    if (++st.counter[1] == 0) {
      if (++st.counter[2] == 0) {
        ++st.counter[3];
      }
    }
  }
  return c0;
}

struct PhiloxEngine {
  using result_type = uint32_t;
  explicit PhiloxEngine(uint32_t seed) {
    philox_seed(seed);
  }
  uint32_t operator()() {
    return philox_next_uint32_inline();
  }
  static constexpr uint32_t min() { return 0; }
  static constexpr uint32_t max() { return UINT32_MAX; }
};

#endif // PHILOX_ENGINE_H

