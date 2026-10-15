#include "fitness.h"
#include "custom_distributions.h"
#include "genetic.h"
#include "program_selection.h"
#include "src/node_detail.h"
#include "src/reg_stack.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
static bool same(float a, float b) {
  return std::memcmp(&a, &b, sizeof(float)) == 0 || (std::isnan(a) && std::isnan(b));
}
static genetic::program expression(std::initializer_list<genetic::node> nodes) {
  genetic::program result;
  result.len = nodes.size();
  result.nodes = new genetic::node[result.len];
  std::copy(nodes.begin(), nodes.end(), result.nodes);
  result.depth = genetic::get_depth(result);
  return result;
}
static void check_stack_and_ownership() {
  genetic::stack<float, 2> stack;
  require(stack.pop() == 0 && stack.empty(), "Empty stack behavior changed");
  stack.push(2); stack.push(3); stack.push(4);
  require(stack.full() && stack.pop() == 3 && stack.pop() == 2, "Stack ordering/capacity changed");
  using node = genetic::node;
  auto a = expression({node(node::type::sub), node(7.0f), node(2.0f)});
  genetic::program b(a);
  a = a;
  require(a.nodes[1].u.val == 7 && a.nodes != b.nodes, "Copy/self-assignment ownership failed");
  a.nodes[1] = node(9.0f);
  require(b.nodes[1].u.val == 7, "Copy aliases original nodes");
  auto old = b.nodes;
  genetic::program c(std::move(b));
  require(c.nodes == old && b.nodes == nullptr && b.len == 0, "Move construction failed");
  b = std::move(c);
  b = std::move(b);
  require(b.nodes == old && c.nodes == nullptr, "Move assignment failed");
  genetic::program empty;
  genetic::program empty_copy(empty);
  require(empty_copy.nodes == nullptr && empty_copy.len == 0, "Empty copy failed");
}
static void check_execution_and_fitness() {
  using node = genetic::node;
  float data[] = {-50, -33.3f, -18, -1, 0, 0.0009f, 0.001f, 1, 37, 50};
  constexpr int rows = sizeof(data) / sizeof(data[0]);
  float labels[rows], weights[rows], predictions[rows];
  for (int i = 0; i < rows; ++i) {
    labels[i] = i % 2;
    weights[i] = (i % 4) * 0.25f;
  }
  auto subtraction = expression({node(node::type::sub), node(0), node(2.0f)});
  auto pointer = &subtraction;
  genetic::execute(pointer, rows, 1, data, predictions);
  for (int i = 0; i < rows; ++i)
    require(same(predictions[i], data[i] - 2.0f), "Noncommutative operand order changed");
  for (auto type : {node::type::div, node::type::fdim, node::type::atan2}) {
    auto binary = expression({node(type), node(3.0f), node(0)});
    pointer = &binary;
    genetic::execute(pointer, rows, 1, data, predictions);
    for (int i = 0; i < rows; ++i) {
      float in[] = {3.0f, data[i]};
      require(same(predictions[i], genetic::detail::evaluate_node(binary.nodes[0], data, rows, i, in)),
              "Binary protected operation changed");
    }
  }
  for (auto type : {node::type::log, node::type::sqrt, node::type::inv}) {
    auto unary = expression({node(type), node(0)});
    pointer = &unary;
    genetic::execute(pointer, rows, 1, data, predictions);
    for (int i = 0; i < rows; ++i) {
      float in[] = {data[i], 0};
      require(same(predictions[i], genetic::detail::evaluate_node(unary.nodes[0], data, rows, i, in)),
              "Unary protected operation changed");
    }
  }
  genetic::program programs[] = {expression({node(0)}), expression({node(node::type::sin), node(0)})};
  auto programs_pointer = programs;
  float all_predictions[2 * rows];
  genetic::execute(programs_pointer, rows, 2, data, all_predictions);
  for (auto metric : {genetic::metric_t::mse, genetic::metric_t::mae,
                     genetic::metric_t::rmse, genetic::metric_t::logloss,
                     genetic::metric_t::pearson, genetic::metric_t::spearman}) {
    genetic::param params;
    params.metric = metric;
    float reference[2], batched[2], single;
    genetic::compute_metric(rows, 2, labels, all_predictions, weights, reference, params);
    genetic::find_batched_fitness(2, programs, batched, params, rows, data, labels, weights);
    genetic::find_fitness(programs, &single, params, rows, data, labels, weights);
    for (int i = 0; i < 2; ++i) require(same(reference[i], batched[i]), "Batched metric changed rounding");
    require(same(reference[0], single), "Single metric disagrees with batch");
    genetic::set_fitness(programs[0], params, rows, data, labels, weights);
    require(same(reference[0], programs[0].raw_fitness_), "set_fitness declaration/behavior failed");
  }
  float y[] = {0}, yp[] = {0.75f}, w[] = {1}, out[1];
  genetic::meanAbsoluteError(1, 1, y, yp, w, out);
  require(out[0] == 0.75f, "Fractional MAE is truncated");
  float correlated[] = {1, 2, 4}, reversed[] = {4, 3, 1}, unit[] = {1, 1, 1};
  genetic::weightedPearson(3, 1, correlated, correlated, unit, out);
  require(std::abs(out[0] - 1) < 0.00001f, "Pearson identity failed");
  genetic::weightedPearson(3, 1, correlated, reversed, unit, out);
  require(std::abs(out[0] + 1) < 0.00001f, "Pearson reverse failed");
}
static void check_rng_and_selection() {
  struct Fixed { uint32_t operator()() { return 123; } } fixed;
  uniform_int_distribution_custom<int> default_distribution;
  require(default_distribution(fixed) == 123, "Default integer range changed");
  uniform_int_distribution_custom<int> full(INT32_MIN, INT32_MAX);
  require(full(fixed) == INT32_MIN + 123, "Full integer range failed");
  for (int i = 0; i < 100; ++i) {
    philox_seed(i);
    (void)philox_random_int32_range(INT32_MIN, INT32_MAX);
    (void)philox_random_int64_range(INT64_MIN, INT64_MAX);
  }
  genetic::program candidates[6];
  float fitness[] = {3, 1, 1, NAN, INFINITY, -1};
  for (int i = 0; i < 6; ++i) candidates[i].raw_fitness_ = fitness[i];
  auto winners = genetic::best_program_indices(candidates, 6);
  require(winners[0] == 5 && winners[1] == 1, "Winner selection/ties changed");
  for (auto &candidate : candidates) candidate.raw_fitness_ = NAN;
  winners = genetic::best_program_indices(candidates, 6);
  require(winners[0] == 0 && winners[1] == 1, "All-NaN stable selection changed");
}
static void check_metric_rounding() {
  constexpr int rows = 23;
  float x[rows], y[rows], weights[rows], expected, actual;
  auto variable = expression({genetic::node(0)});
  uint32_t random = 1234567;
  for (int trial = 0; trial < 64; ++trial) {
    for (int row = 0; row < rows; ++row) {
      random = random * 1664525u + 1013904223u;
      x[row] = (static_cast<int>(random >> 16) - 32768) / 137.0f;
      random = random * 1664525u + 1013904223u;
      y[row] = (random >> 16) / 193.0f;
      weights[row] = 0.25f + (row % 7) * 0.17f;
    }
    for (auto metric : {genetic::metric_t::mse, genetic::metric_t::mae,
                        genetic::metric_t::rmse, genetic::metric_t::logloss}) {
      genetic::param params;
      params.metric = metric;
      genetic::compute_metric(rows, 1, y, x, weights, &expected, params);
      genetic::find_fitness(&variable, &actual, params, rows, x, y, weights);
      require(same(expected, actual), "Fused metric changed intermediate rounding");
    }
  }
}
static void check_prepared_edges() {
  using node = genetic::node;
  constexpr int rows = 9;
  float data[] = {-0.0f,  0.0f,      -1.0f,    1.0f, 0.0009f,
                  0.001f, -INFINITY, INFINITY, NAN};
  float labels[rows], weights[rows], predictions[rows];
  for (int i = 0; i < rows; ++i) {
    labels[i] = i % 2;
    weights[i] = 0.3f + i * 0.11f;
  }
  auto check = [&](genetic::program &program) {
    auto pointer = &program;
    auto check_samples = [&](int count, float *input, float *target,
                             float *weight) {
      genetic::execute(pointer, count, 1, input, predictions);
      for (auto metric :
           {genetic::metric_t::mse, genetic::metric_t::mae,
            genetic::metric_t::rmse, genetic::metric_t::logloss}) {
        genetic::param params;
        params.metric = metric;
        float expected, actual;
        genetic::compute_metric(count, 1, target, predictions, weight,
                                &expected, params);
        genetic::find_fitness(pointer, &actual, params, count, input, target,
                              weight);
        if (!same(expected, actual)) {
          uint32_t expected_bits, actual_bits;
          std::memcpy(&expected_bits, &expected, sizeof(float));
          std::memcpy(&actual_bits, &actual, sizeof(float));
          std::cerr << "edge opcode="
                    << (program.len ? static_cast<int>(program.nodes[0].t) : -1)
                    << " metric=" << static_cast<int>(metric)
                    << " expected=" << std::hex << expected_bits
                    << " actual=" << actual_bits << std::dec << '\n';
        }
        require(same(expected, actual),
                "Prepared execution changed operator/fallback fitness bits");
      }
    };
    check_samples(6, data, labels, weights);
    check_samples(rows, data, labels, weights);
    for (int i = 0; i < rows; ++i)
      check_samples(1, data + i, labels + i, weights + i);
  };
  for (uint32_t t = static_cast<uint32_t>(node::type::binary_begin);
       t <= static_cast<uint32_t>(node::type::unary_end); ++t) {
    auto type = static_cast<node::type>(t);
    auto program = genetic::detail::arity(type) == 2
                       ? expression({node(type), node(2.0f), node(0)})
                       : expression({node(type), node(0)});
    check(program);
  }
  auto unknown = expression(
      {node(node::type::add), node(static_cast<node::type>(999)), node(3.0f)});
  check(unknown);
  auto underflow = expression({node(node::type::sub), node(4.0f)});
  check(underflow);
  auto leftovers = expression({node(2.0f), node(3.0f)});
  check(leftovers);
  genetic::program empty;
  check(empty);
  genetic::program overflow;
  overflow.len = 32;
  overflow.nodes = new node[overflow.len];
  for (int i = 0; i < overflow.len; ++i)
    overflow.nodes[i] = node(float(i));
  check(overflow);
  genetic::param params;
  params.num_features = 1;
  params.init_depth[0] = 1;
  params.init_depth[1] = 5;
  PhiloxEngine engine(9876);
  for (int i = 0; i < 32; ++i) {
    genetic::program program;
    genetic::build_program(program, params, engine);
    check(program);
  }
}
static void check_reuse_and_history() {
  genetic::param params;
  params.num_features = 1;
  params.population_size = 12;
  params.generations = 4;
  params.init_depth[0] = 1;
  params.init_depth[1] = 3;
  params.metric = genetic::metric_t::mse;
  params.stopping_criteria = -1;
  PhiloxEngine engine(42);
  genetic::program parent, donor, output;
  genetic::build_program(parent, params, engine);
  genetic::build_program(parent, params, engine);
  genetic::build_program(donor, params, engine);
  for (int i = 0; i < 32; ++i) {
    genetic::hoist_mutation(parent, output, params, engine);
    genetic::subtree_mutation(parent, output, params, engine);
    genetic::point_mutation(parent, output, params, engine);
    genetic::crossover(parent, donor, output, params, engine);
  }
  float x[] = {-1, 0, 1, 2}, y[] = {1, 0, 1, 4}, weights[] = {1, 1, 1, 1};
  auto owned = std::make_unique<genetic::program[]>(params.population_size);
  auto final = owned.get();
  std::vector<std::vector<genetic::program>> full, low;
  genetic::symFit(x, y, weights, 4, 1, params, final, full);
  params.low_memory = true;
  genetic::symFit(x, y, weights, 4, 1, params, final, low);
  require(full.size() == 4 && low.size() == 1, "History retention policy changed");
  for (int i = 0; i < params.population_size; ++i) {
    require(full.back()[i].len == low.back()[i].len &&
            same(full.back()[i].raw_fitness_, low.back()[i].raw_fitness_), "Low-memory results changed");
    require(low.back()[i].nodes != final[i].nodes, "History aliases final population");
  }
}
int main() {
  try {
    check_stack_and_ownership();
    check_execution_and_fitness();
    check_rng_and_selection();
    check_metric_rounding();
    check_prepared_edges();
    check_reuse_and_history();
    std::cout << "All correctness checks passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
