#include "fast_math.h"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

using namespace genetic::detail;

int main() {
#if !defined(__FMA__)
  std::printf("built without FMA: the evaluator uses libm, nothing to check\n");
  return 0;
#else
  constexpr int B = 128;
  constexpr uint64_t kTotal = 1ull << 32, kSpan = 1ull << 20;
  std::atomic<uint64_t> next{0}, bad{0};
  auto work = [&] {
    float in[B], out[B];
    for (;;) {
      uint64_t base = next.fetch_add(kSpan);
      if (base >= kTotal)
        break;
      for (uint64_t b = base; b < base + kSpan; b += B) {
        for (int j = 0; j < B; ++j) {
          uint32_t u = static_cast<uint32_t>(b + j);
          std::memcpy(&in[j], &u, 4);
        }
        for (int fn = 0; fn < 3; ++fn) {
          for (int inplace = 0; inplace < 2; ++inplace) {
            if (inplace)
              std::memcpy(out, in, sizeof(in));
            const float *src = inplace ? out : in;
            if (fn == 0)
              sinf_block(src, out, B);
            else if (fn == 1)
              cosf_block(src, out, B);
            else
              logf_block(src, out, B);
            for (int j = 0; j < B; ++j) {
              float ref = fn == 0 ? sinf(in[j])
                                  : (fn == 1 ? cosf(in[j]) : logf(in[j]));
              if (std::memcmp(&ref, &out[j], 4) != 0)
                bad.fetch_add(1, std::memory_order_relaxed);
            }
          }
        }
      }
    }
  };
  std::vector<std::thread> threads;
  unsigned n = std::max(1u, std::thread::hardware_concurrency());
  for (unsigned i = 0; i < n; ++i)
    threads.emplace_back(work);
  for (auto &t : threads)
    t.join();
  std::printf("%s: %llu mismatching results over 2^32 inputs x {sin,cos,log} x "
              "{out of place, in place}\n",
              bad ? "FAIL" : "OK", static_cast<unsigned long long>(bad.load()));
  std::printf("runtime self-test (use_fast_math_blocks): %d\n",
              use_fast_math_blocks());
  return bad ? 1 : 0;
#endif
}
