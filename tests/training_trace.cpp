#include "genetic.h"
#include "philox_engine.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>

static uint32_t bits(float value) {
  uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
static void hash_word(uint64_t &hash, uint32_t word) {
  for (int byte = 0; byte < 4; ++byte) {
    hash ^= (word >> (8 * byte)) & 255;
    hash *= UINT64_C(1099511628211);
  }
}
static void dump(const genetic::program &program) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (int i = 0; i < program.len; ++i) {
    const auto &node = program.nodes[i];
    hash_word(hash, static_cast<uint32_t>(node.t));
    hash_word(hash, node.t == genetic::node::type::constant
                        ? bits(node.u.val) : static_cast<uint32_t>(node.u.fid));
  }
  std::cout << program.len << ' ' << program.depth << ' '
            << static_cast<int>(program.metric) << ' '
            << static_cast<int>(program.mut_type) << ' ' << std::hex
            << bits(program.raw_fitness_) << ' ' << hash << std::dec << '\n';
}
int main() {
  PhiloxEngine engine(2025);
  std::cout << "rng";
  for (int i = 0; i < 16; ++i) std::cout << ' ' << engine();
  std::cout << '\n';
  constexpr int rows = 23, columns = 3, population = 24;
  float data[rows * columns], labels[rows], weights[rows];
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j)
      data[j * rows + i] = (i - 11) * 0.0625f + j * 0.125f;
    weights[i] = 0.5f + (i % 3) * 0.25f;
  }
  for (int trial = 0; trial < 3; ++trial) {
    genetic::param params;
    params.population_size = population;
    params.generations = 8;
    params.tournament_size = 4;
    params.num_features = columns;
    params.init_depth[0] = 1;
    params.init_depth[1] = 4;
    params.terminalRatio = 0.25f;
    params.random_state = 2025 + trial;
    params.stopping_criteria = -1.0f;
    params.p_crossover = 0.4f;
    params.p_subtree_mutation = 0.15f;
    params.p_hoist_mutation = 0.15f;
    params.p_point_mutation = 0.15f;
    params.p_point_replace = 0.3f;
    params.function_set = {genetic::node::type::add, genetic::node::type::sub,
      genetic::node::type::mul, genetic::node::type::div, genetic::node::type::sin,
      genetic::node::type::abs, genetic::node::type::log};
    params.arity_set.clear();
    params.init_method = static_cast<genetic::init_method_t>(trial);
    params.metric = trial == 0 ? genetic::metric_t::mse
                   : trial == 1 ? genetic::metric_t::rmse : genetic::metric_t::logloss;
    for (int i = 0; i < rows; ++i)
      labels[i] = trial == 2 ? float(i % 2) : data[i] * 0.5f + 0.25f;
    auto owned = std::make_unique<genetic::program[]>(population);
    auto final = owned.get();
    std::vector<std::vector<genetic::program>> history;
    genetic::symFit(data, labels, weights, rows, columns, params, final, history);
    std::cout << "trial " << trial << " epochs " << params.num_epochs << '\n';
    for (const auto &generation : history) {
      std::cout << "generation\n";
      for (const auto &program : generation) dump(program);
    }
    std::cout << "final\n";
    for (int i = 0; i < population; ++i) dump(final[i]);
    float predictions[rows];
    genetic::symRegPredict(data, rows, final, predictions);
    std::cout << "predictions" << std::hex;
    for (float value : predictions) std::cout << ' ' << bits(value);
    std::cout << std::dec << '\n';
  }
}
