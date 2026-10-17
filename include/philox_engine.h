// philox_engine.h
#ifndef PHILOX_ENGINE_H
#define PHILOX_ENGINE_H

#include <philox_rng.h>
#include <cstdint>
#include <limits>

inline void philox_refill(philox_state_t &st) {
  if (!st.initialized) {
    philox_reset();
  }
  const uint32_t s0 = st.counter[0], s1 = st.counter[1], s2 = st.counter[2],
                 s3 = st.counter[3];
  const uint32_t key0 = st.key[0], key1 = st.key[1];
  constexpr uint32_t half = PHILOX_BUF_SIZE / 2;
  for (uint32_t j = 0; j < half; ++j) {
    uint32_t a0[2], a1[2], a2[2], a3[2];
    for (int h = 0; h < 2; ++h) {
      uint32_t lane = j + h * half;
      a0[h] = s0 + lane;
      uint32_t carry0 = a0[h] < s0;
      a1[h] = s1 + carry0;
      uint32_t carry1 = carry0 & (a1[h] == 0);
      a2[h] = s2 + carry1;
      uint32_t carry2 = carry1 & (a2[h] == 0);
      a3[h] = s3 + carry2;
    }
    uint32_t k0 = key0, k1 = key1;
#pragma GCC unroll 10
    for (int i = 0; i < 10; ++i) {
#pragma GCC unroll 2
      for (int h = 0; h < 2; ++h) {
        uint64_t p0 = static_cast<uint64_t>(PHILOX_M4x32A) * a0[h];
        uint64_t p1 = static_cast<uint64_t>(PHILOX_M4x32B) * a2[h];
        uint32_t n0 = static_cast<uint32_t>(p1 >> 32) ^ a1[h] ^ k0;
        uint32_t n1 = static_cast<uint32_t>(p1);
        uint32_t n2 = static_cast<uint32_t>(p0 >> 32) ^ a3[h] ^ k1;
        uint32_t n3 = static_cast<uint32_t>(p0);
        a0[h] = n0;
        a1[h] = n1;
        a2[h] = n2;
        a3[h] = n3;
      }
      k0 += PHILOX_W32A;
      k1 += PHILOX_W32B;
    }
    st.buf[j] = a0[0];
    st.buf[j + half] = a0[1];
  }
  st.buf_pos = 0;
  st.buf_len = PHILOX_BUF_SIZE;
}

inline uint32_t philox_next_uint32_inline() {
  philox_state_t &st = global_philox_state;
  if (__builtin_expect(st.buf_pos == st.buf_len, 0)) {
    philox_refill(st);
  }
  uint32_t v = st.buf[st.buf_pos++];
  if (++st.counter[0] == 0) {
    if (++st.counter[1] == 0) {
      if (++st.counter[2] == 0) {
        ++st.counter[3];
      }
    }
  }
  return v;
}

struct PhiloxEngine {
  using result_type = uint32_t;
  explicit PhiloxEngine(uint32_t seed) {
    philox_seed(seed);
  }
  uint32_t operator()() {
    return philox_next_uint32_inline();
  }
  static uint32_t draw() { return philox_next_uint32_inline(); }
  static constexpr uint32_t min() { return 0; }
  static constexpr uint32_t max() { return UINT32_MAX; }
};

#endif // PHILOX_ENGINE_H

