#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace genetic {
namespace detail {


#if defined(__FMA__)

namespace trig {
constexpr double hpi_inv = 0x1.45F306DC9C883p+23;
constexpr double hpi = 0x1.921FB54442D18p0;
constexpr double c0 = 0x1p0, c1 = -0x1.ffffffd0c621cp-2,
                 c2 = 0x1.55553e1068f19p-5, c3 = -0x1.6c087e89a359dp-10,
                 c4 = 0x1.99343027bf8c3p-16;
constexpr double s1 = -0x1.555545995a603p-3, s2 = 0x1.1107605230bc4p-7,
                 s3 = -0x1.994eb3774cf24p-13;

inline uint32_t abstop12(float x) {
  uint32_t i;
  std::memcpy(&i, &x, sizeof(i));
  return (i >> 20) & 0x7ff;
}
constexpr uint32_t top_tiny = 0x398;
constexpr uint32_t top_big = 0x42f;

inline uint64_t bits_of(double d) {
  uint64_t u;
  std::memcpy(&u, &d, sizeof(u));
  return u;
}
inline double double_of(uint64_t u) {
  double d;
  std::memcpy(&d, &u, sizeof(d));
  return d;
}
inline uint32_t bits_of(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}
inline float float_of(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline double sincos_core(float y, bool want_cos) {
  double x = y;
  double r = x * hpi_inv;
  int32_t n = (static_cast<int32_t>(r) + 0x800000) >> 24;
  double xr = std::fma(-static_cast<double>(n), hpi, x);
  double x2 = xr * xr;

  double x3 = xr * x2;
  double sp1 = std::fma(x2, s3, s2);
  double x7 = x3 * x2;
  double sp = std::fma(x7, sp1, std::fma(x3, s1, xr));

  double x4 = x2 * x2;
  double cp2 = std::fma(x2, c4, c3);
  double cp1 = std::fma(x2, c1, c0);
  double x6 = x4 * x2;
  double cp = std::fma(x6, cp2, std::fma(x4, c2, cp1));

  uint64_t odd = static_cast<uint64_t>(n & 1);
  uint64_t q2 = static_cast<uint64_t>((n >> 1) & 1);
  uint64_t use_cp = want_cos ? odd ^ 1 : odd;
  uint64_t neg = want_cos ? q2 ^ odd : q2;
  uint64_t m = 0 - use_cp;
  uint64_t res = (bits_of(cp) & m) | (bits_of(sp) & ~m);
  return double_of(res ^ (neg << 63));
}
}

namespace logf_impl {
struct entry {
  double invc, logc;
};
constexpr entry tab[16] = {
    {0x1.661ec79f8f3bep+0, -0x1.57bf7808caadep-2},
    {0x1.571ed4aaf883dp+0, -0x1.2bef0a7c06ddbp-2},
    {0x1.49539f0f010bp+0, -0x1.01eae7f513a67p-2},
    {0x1.3c995b0b80385p+0, -0x1.b31d8a68224e9p-3},
    {0x1.30d190c8864a5p+0, -0x1.6574f0ac07758p-3},
    {0x1.25e227b0b8eap+0, -0x1.1aa2bc79c81p-3},
    {0x1.1bb4a4a1a343fp+0, -0x1.a4e76ce8c0e5ep-4},
    {0x1.12358f08ae5bap+0, -0x1.1973c5a611cccp-4},
    {0x1.0953f419900a7p+0, -0x1.252f438e10c1ep-5},
    {0x1p+0, 0x0p+0},
    {0x1.e608cfd9a47acp-1, 0x1.aa5aa5df25984p-5},
    {0x1.ca4b31f026aap-1, 0x1.c5e53aa362eb4p-4},
    {0x1.b2036576afce6p-1, 0x1.526e57720db08p-3},
    {0x1.9c2d163a1aa2dp-1, 0x1.bc2860d22477p-3},
    {0x1.886e6037841edp-1, 0x1.1058bc8a07ee1p-2},
    {0x1.767dcf5534862p-1, 0x1.4043057b6ee09p-2},
};
constexpr double ln2 = 0x1.62e42fefa39efp-1;
constexpr double a0 = -0x1.00ea348b88334p-2, a1 = 0x1.5575b0be00b6ap-2,
                 a2 = -0x1.ffffef20a4123p-2;
constexpr uint32_t off = 0x3f330000;

inline bool special(float x) {
  return trig::bits_of(x) - 0x00800000u >= 0x7f800000u - 0x00800000u;
}

inline float core(float x) {
  uint32_t ix = trig::bits_of(x);
  uint32_t tmp = ix - off;
  int i = (tmp >> (23 - 4)) % 16;
  int k = static_cast<int32_t>(tmp) >> 23;
  uint32_t iz = ix - (tmp & 0x1ffu << 23);
  double invc = tab[i].invc;
  double logc = tab[i].logc;
  double z = trig::float_of(iz);

  double r = std::fma(z, invc, -1.0);
  double y0 = std::fma(static_cast<double>(k), ln2, logc);
  double r2 = r * r;
  double y = std::fma(a1, r, a2);
  y = std::fma(a0, r2, y);
  y = std::fma(y, r2, y0 + r);
  return static_cast<float>(y);
}
}

inline void logf_block(const float *in, float *out, int n) {
  uint32_t m = 0;
  for (int j = 0; j < n; ++j)
    m |= logf_impl::special(in[j]);
  if (__builtin_expect(m != 0, 0)) {
    for (int j = 0; j < n; ++j)
      out[j] = logf(in[j]);
    return;
  }
#pragma GCC ivdep
  for (int j = 0; j < n; ++j)
    out[j] = logf_impl::core(in[j]);
}

inline bool any_big(const float *in, int n) {
  uint32_t m = 0;
  for (int j = 0; j < n; ++j)
    m |= trig::abstop12(in[j]) >= trig::top_big;
  return m != 0;
}

inline void sinf_block(const float *in, float *out, int n) {
  if (__builtin_expect(any_big(in, n), 0)) {
    for (int j = 0; j < n; ++j)
      out[j] = sinf(in[j]);
    return;
  }
#pragma GCC ivdep
  for (int j = 0; j < n; ++j) {
    float y = in[j];
    uint32_t f = trig::bits_of(static_cast<float>(trig::sincos_core(y, false)));
    uint32_t m = 0u - static_cast<uint32_t>(trig::abstop12(y) < trig::top_tiny);
    out[j] = trig::float_of((trig::bits_of(y) & m) | (f & ~m));
  }
}

inline void cosf_block(const float *in, float *out, int n) {
  if (__builtin_expect(any_big(in, n), 0)) {
    for (int j = 0; j < n; ++j)
      out[j] = cosf(in[j]);
    return;
  }
#pragma GCC ivdep
  for (int j = 0; j < n; ++j) {
    float y = in[j];
    uint32_t f = trig::bits_of(static_cast<float>(trig::sincos_core(y, true)));
    uint32_t m = 0u - static_cast<uint32_t>(trig::abstop12(y) < trig::top_tiny);
    out[j] = trig::float_of((0x3f800000u & m) | (f & ~m));
  }
}

inline bool blocks_match_libm() {
  constexpr int B = 128;
  float in[B], s[B], c[B], l[B];
  auto check = [&](int n) {
    sinf_block(in, s, n);
    cosf_block(in, c, n);
    logf_block(in, l, n);
    for (int j = 0; j < n; ++j) {
      float rs = sinf(in[j]), rc = cosf(in[j]), rl = logf(in[j]);
      if (std::memcmp(&rs, &s[j], 4) != 0 || std::memcmp(&rc, &c[j], 4) != 0 ||
          std::memcmp(&rl, &l[j], 4) != 0)
        return false;
    }
    return true;
  };
  int n = 0;
  for (uint64_t k = 0; k < 65536; ++k) {
    uint32_t u = static_cast<uint32_t>(k * 65537u);
    std::memcpy(&in[n++], &u, 4);
    if (n == B) {
      if (!check(n))
        return false;
      n = 0;
    }
  }
  for (int k = -130000; k <= 130000; ++k) {
    in[n++] = k * 0.001f;
    if (n == B) {
      if (!check(n))
        return false;
      n = 0;
    }
  }
  return check(n);
}

#endif

inline bool use_fast_math_blocks() {
#if defined(__FMA__)
  static const bool ok = blocks_match_libm();
  return ok;
#else
  return false;
#endif
}

}
}
