#include "fast_math.h"

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

inline float fsel(bool c, float a, float b) {
  uint32_t m = 0u - static_cast<uint32_t>(c);
  return float_of((bits_of(a) & m) | (bits_of(b) & ~m));
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
constexpr double tab_invc[16] = {
    0x1.661ec79f8f3bep+0,
    0x1.571ed4aaf883dp+0,
    0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0,
    0x1.30d190c8864a5p+0,
    0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0,
    0x1.12358f08ae5bap+0,
    0x1.0953f419900a7p+0,
    0x1p+0,
    0x1.e608cfd9a47acp-1,
    0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1,
    0x1.9c2d163a1aa2dp-1,
    0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1};
constexpr double tab_logc[16] = {
    -0x1.57bf7808caadep-2,
    -0x1.2bef0a7c06ddbp-2,
    -0x1.01eae7f513a67p-2,
    -0x1.b31d8a68224e9p-3,
    -0x1.6574f0ac07758p-3,
    -0x1.1aa2bc79c81p-3,
    -0x1.a4e76ce8c0e5ep-4,
    -0x1.1973c5a611cccp-4,
    -0x1.252f438e10c1ep-5,
    0x0p+0,
    0x1.aa5aa5df25984p-5,
    0x1.c5e53aa362eb4p-4,
    0x1.526e57720db08p-3,
    0x1.bc2860d22477p-3,
    0x1.1058bc8a07ee1p-2,
    0x1.4043057b6ee09p-2};
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
  double invc = tab_invc[i];
  double logc = tab_logc[i];
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

void logf_block(const float *in, float *out, int n) {
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

namespace expf_impl {
constexpr uint64_t tab[32] = {
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f,
    0x3fef9301d0125b51, 0x3fef72b83c7d517b, 0x3fef54873168b9aa,
    0x3fef387a6e756238, 0x3fef1e9df51fdee1, 0x3fef06fe0a31b715,
    0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429,
    0x3feea47eb03a5585, 0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74,
    0x3feea11473eb0187, 0x3feea589994cce13, 0x3feeace5422aa0db,
    0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c,
    0x3fef3720dcef9069, 0x3fef5818dcfba487, 0x3fef7c97337b9b5f,
    0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
};
constexpr double N = 32;
constexpr double invln2_scaled = 0x1.71547652b82fep+0 * N;
constexpr double shift = 0x1.8p+52;
constexpr double c0 = 0x1.c6af84b912394p-5 / N / N / N,
                 c1 = 0x1.ebfce50fac4f3p-3 / N / N,
                 c2 = 0x1.62e42ff0c52d6p-1 / N;

inline float core(float x) {
  double xd = x;
  double kd = std::fma(invln2_scaled, xd, shift);
  uint64_t ki;
  std::memcpy(&ki, &kd, sizeof(ki));
  kd -= shift;
  double r = std::fma(invln2_scaled, xd, -kd);
  uint64_t t = tab[ki % 32] + (ki << (52 - 5));
  double sc;
  std::memcpy(&sc, &t, sizeof(sc));
  double y = std::fma(std::fma(c0, r, c1), r * r, std::fma(c2, r, 1.0));
  float res = static_cast<float>(y * sc);

  uint32_t ix = trig::bits_of(x);
  uint32_t abstop = (ix >> 20) & 0x7ff;
  bool finite_big = (abstop >= 0x42b) & (abstop < 0x7f8);
  res = trig::fsel(abstop >= 0x7f8, x + x, res);
  res = trig::fsel(ix == 0xff800000u, 0.0f, res);
  res = trig::fsel(finite_big & (x > 0x1.62e42ep6f), HUGE_VALF, res);
  float tiny = trig::fsel(x < -0x1.9fe368p6f, 0.0f, 0x1p-149f);
  res = trig::fsel(finite_big & (x < -0x1.9d1d9ep6f), tiny, res);
  return res;
}
}

void expf_block(const float *in, float *out, int n) {
#pragma GCC ivdep
  for (int j = 0; j < n; ++j)
    out[j] = expf_impl::core(in[j]);
}

namespace log1pf_impl {
constexpr float ln2_hi = 6.9313812256e-01f, ln2_lo = 9.0580006145e-06f;
constexpr float Lp1 = 6.6666668653e-01f, Lp2 = 4.0000000596e-01f,
                Lp3 = 2.8571429849e-01f, Lp4 = 2.2222198546e-01f,
                Lp5 = 1.8183572590e-01f, Lp6 = 1.5313838422e-01f,
                Lp7 = 1.4798198640e-01f;

inline float core(float x) {
  int32_t hx = static_cast<int32_t>(trig::bits_of(x));
  int32_t ax = hx & 0x7fffffff;

  float u1 = 1.0f + x;
  int32_t hu1 = static_cast<int32_t>(trig::bits_of(u1));
  int32_t k1 = (hu1 >> 23) - 127;
  float c1_pos = 1.0f - (u1 - x);
  float c1_neg = x - (u1 - 1.0f);
  float c1 = trig::fsel(k1 > 0, c1_pos, c1_neg);
  c1 /= u1;
  bool huge = hx >= 0x5a000000;
  int32_t hu = huge ? hx : hu1;
  int32_t k = huge ? (hx >> 23) - 127 : k1;
  float c = trig::fsel(huge, 0.0f, c1);
  hu &= 0x007fffff;
  bool low = hu < 0x3504f7;
  float u = trig::float_of(static_cast<uint32_t>(hu) |
                           (low ? 0x3f800000u : 0x3f000000u));
  k = low ? k : k + 1;
  hu = low ? hu : (0x00800000 - hu) >> 2;
  float f = u - 1.0f;

  bool kzero = (hx < 0x3ed413d7) &
               ((hx > 0) | (hx <= static_cast<int32_t>(0xbe95f61f)));
  f = trig::fsel(kzero, x, f);
  k = kzero ? 0 : k;
  hu = kzero ? 1 : hu;

  float kf = static_cast<float>(k);
  float hfsq = 0.5f * f * f;
  float s = f / (2.0f + f);
  float z = s * s;
  float R = z * (Lp1 + z * (Lp2 + z * (Lp3 + z * (Lp4 + z * (Lp5 + z * (Lp6 + z * Lp7))))));
  float main0 = f - (hfsq - s * (hfsq + R));
  float main1 = kf * ln2_hi - ((hfsq - (s * (hfsq + R) + (kf * ln2_lo + c))) - f);
  float res = trig::fsel(k == 0, main0, main1);

  float Rz = hfsq * (1.0f - static_cast<float>(0.66666666666666666) * f);
  float fz1 = kf * ln2_hi + (c + kf * ln2_lo);
  float nz0 = f - Rz;
  float nz1 = kf * ln2_hi - ((Rz - (kf * ln2_lo + c)) - f);
  float fz = trig::fsel(k == 0, 0.0f, fz1);
  float nz = trig::fsel(k == 0, nz0, nz1);
  res = trig::fsel(hu == 0, trig::fsel(f == 0.0f, fz, nz), res);

  res = trig::fsel(hx >= 0x7f800000, x + x, res);
  float tiny = trig::fsel(ax < 0x24800000, x, x - x * x * 0.5f);
  res = trig::fsel(ax < 0x31000000, tiny, res);
  float below = trig::fsel(x == -1.0f, -HUGE_VALF, (x - x) / (x - x));
  res = trig::fsel((hx < 0) & (ax >= 0x3f800000), below, res);
  return res;
}
}

void log1pf_block(const float *in, float *out, int n) {
#pragma GCC ivdep
  for (int j = 0; j < n; ++j)
    out[j] = log1pf_impl::core(in[j]);
}

static bool any_big(const float *in, int n) {
  uint32_t m = 0;
  for (int j = 0; j < n; ++j)
    m |= trig::abstop12(in[j]) >= trig::top_big;
  return m != 0;
}

void sinf_block(const float *in, float *out, int n) {
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

void cosf_block(const float *in, float *out, int n) {
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

static bool blocks_match_libm() {
  constexpr int B = 128;
  float in[B], s[B], c[B], l[B], e[B], lp[B];
  auto check = [&](int n) {
    sinf_block(in, s, n);
    cosf_block(in, c, n);
    logf_block(in, l, n);
    expf_block(in, e, n);
    log1pf_block(in, lp, n);
    for (int j = 0; j < n; ++j) {
      float r[5] = {sinf(in[j]), cosf(in[j]), logf(in[j]), expf(in[j]),
                    log1pf(in[j])};
      float got[5] = {s[j], c[j], l[j], e[j], lp[j]};
      if (std::memcmp(r, got, sizeof(r)) != 0)
        return false;
    }
    return true;
  };
  int n = 0;
  for (uint64_t k = 0; k < 4096; ++k) {
    uint32_t u = static_cast<uint32_t>(k * 1048573u);
    std::memcpy(&in[n++], &u, 4);
    if (n == B) {
      if (!check(n))
        return false;
      n = 0;
    }
  }
  for (int k = -4160; k <= 4160; ++k) {
    in[n++] = k * 0.03125f;
    if (n == B) {
      if (!check(n))
        return false;
      n = 0;
    }
  }
  return check(n);
}

#endif

bool use_fast_math_blocks() {
#if defined(__FMA__)
  static const bool ok = blocks_match_libm();
  return ok;
#else
  return false;
#endif
}

}
}
