#include <custom_distributions.h>
#include <philox_engine.h>

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

static uint64_t lcg_state;
static uint32_t lcg() {
  lcg_state = lcg_state * 6364136223846793005ull + 1442695040888963407ull;
  return static_cast<uint32_t>(lcg_state >> 33);
}

static unsigned long check_philox(unsigned long &checked) {
  unsigned long bad = 0;
  const uint32_t seeds[] = {0,           1,           2025,       0xdead,
                            2025000000u, 0xffffffffu, 0xffffffc0u, 0xfffffff0u};
  for (uint32_t seed : seeds) {
    for (int trial = 0; trial < 3; ++trial) {
      std::vector<uint32_t> ref;
      for (int mode = 0; mode < 2; ++mode) {
        size_t pos = 0;
        lcg_state = seed * 7ull + trial;
        philox_seed(seed);
        if (trial >= 1) {
          global_philox_state.counter[1] = 0xffffffffu;
          global_philox_state.counter[2] = 0xffffffffu;
        }
        for (int op = 0; op < 20000; ++op) {
          uint32_t r = lcg() % 100;
          uint32_t v;
          if (trial == 2 && r < 3) {
            philox_seed(lcg());
            continue;
          } else if (trial == 2 && r < 6) {
            v = philox_random_uint32();
          } else if (trial == 2 && r < 8) {
            philox_skip(lcg() % 5);
            continue;
          } else {
            v = mode == 0 ? philox_random_uint32() : PhiloxEngine::draw();
          }
          if (mode == 0) {
            ref.push_back(v);
          } else {
            bad += ref[pos++] != v;
            ++checked;
          }
        }
      }
    }
  }
  return bad;
}

struct fixed_engine {
  uint32_t v;
  uint32_t operator()() { return v; }
};

static unsigned long long check_fastmod() {
  const std::vector<int> ranges = {1, 3, 7, 455, 512, 16384, 16385, 2147483647};
  std::atomic<unsigned long long> bad{0};
  std::vector<std::thread> threads;
  for (int r : ranges) {
    threads.emplace_back([&bad, r] {
      uniform_int_distribution_custom<int> d(0, r - 1);
      uint32_t range = static_cast<uint32_t>(r);
      uint32_t limit = 0xFFFFFFFFu - (0xFFFFFFFFu % range);
      for (uint64_t x = 0; x < limit; ++x) {
        fixed_engine e{static_cast<uint32_t>(x)};
        if (d(e) != static_cast<int>(x % range))
          bad.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto &t : threads)
    t.join();
  return bad;
}

int main() {
  unsigned long checked = 0;
  unsigned long bad_philox = check_philox(checked);
  std::printf("%s: philox engine, %lu mismatches in %lu draws\n",
              bad_philox ? "FAIL" : "OK", bad_philox, checked);
  unsigned long long bad_mod = check_fastmod();
  std::printf("%s: fastmod, %llu mismatches\n", bad_mod ? "FAIL" : "OK",
              bad_mod);
  return (bad_philox || bad_mod) ? 1 : 0;
}
