#pragma once
#include "program.h"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace genetic {
inline std::array<int, 2> best_program_indices(program *programs, int size) {
  if (size < 2) throw std::invalid_argument("At least two programs are required");
  for (int i = 0; i < size; ++i) {
    if (std::isnan(programs[i].raw_fitness_))
      programs[i].raw_fitness_ = std::numeric_limits<float>::max();
  }
  std::array<int, 2> best = {0, 1};
  if (programs[1].raw_fitness_ < programs[0].raw_fitness_) std::swap(best[0], best[1]);
  for (int i = 2; i < size; ++i) {
    if (programs[i].raw_fitness_ < programs[best[0]].raw_fitness_) {
      best[1] = best[0];
      best[0] = i;
    } else if (programs[i].raw_fitness_ < programs[best[1]].raw_fitness_) {
      best[1] = i;
    }
  }
  return best;
}
}
