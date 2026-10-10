#include "constants.h"
#include "custom_distributions.h"
#include "node_detail.h"
#include "philox_engine.h"
#include "reg_stack.h"
#include "thread_pool.h"
#include <algorithm>
#include <cstdint>
#include <fitness.h>
#include <node.h>
#include <numeric>
#include <program.h>
#include <random>
#include <cstring>
#include <stack>
#include <unordered_map>

namespace genetic {

/**
 * Execution kernel for a single program. We assume that the input data
 * is stored in column major format.
 */
template <int MaxSize = MAX_STACK_SIZE>
void execute_kernel(const program_t d_progs, const float *data, float *y_pred,
                    const uint64_t n_rows, const uint64_t n_progs) {
  for (uint64_t pid = 0; pid < n_progs; ++pid) {
    for (uint64_t row_id = 0; row_id < n_rows; ++row_id) {

      stack<float, MaxSize> eval_stack;
      program_t curr_p = d_progs + pid; // Current program

      int end = curr_p->len - 1;
      node *curr_node = curr_p->nodes + end;

      float res = 0.0f;
      float in[2] = {0.0f, 0.0f};

      while (end >= 0) {
        if (detail::is_nonterminal(curr_node->t)) {
          int ar = detail::arity(curr_node->t);
          in[0] = eval_stack.pop(); // Min arity of function is 1
          if (ar > 1)
            in[1] = eval_stack.pop();
        }
        res = detail::evaluate_node(*curr_node, data, n_rows, row_id, in);
        eval_stack.push(res);
        curr_node--;
        end--;
      }

      // Outputs stored in col-major format
      y_pred[pid * n_rows + row_id] = eval_stack.pop();
    }
  }
}

namespace {

constexpr int kBlock = 128;

bool fits_block_stack(const program &p) {
  int sp = 0;
  for (int i = p.len - 1; i >= 0; --i) {
    int ar = detail::arity(p.nodes[i].t);
    if (sp < ar)
      return false;
    sp -= ar;
    if (sp == MAX_STACK_SIZE)
      return false;
    ++sp;
  }
  return sp >= 1;
}

const float *eval_block(const program &p, const float *data, uint64_t stride,
                        uint64_t r0, int n,
                        float (*buf)[kBlock], const float **slot) {
  int sp = 0;
  for (int i = p.len - 1; i >= 0; --i) {
    const node &nd = p.nodes[i];
    switch (nd.t) {
    case node::type::variable:
      slot[sp++] = data + stride * nd.u.fid + r0;
      continue;
    case node::type::constant: {
      float v = nd.u.val;
      float *o = buf[sp];
      for (int j = 0; j < n; ++j)
        o[j] = v;
      slot[sp++] = o;
      continue;
    }
    default:
      break;
    }

    if (detail::arity(nd.t) == 1) {
      const float *a = slot[sp - 1];
      float *o = buf[sp - 1];
      switch (nd.t) {
      case node::type::abs:
        for (int j = 0; j < n; ++j)
          o[j] = fabsf(a[j]);
        break;
      case node::type::sin:
        for (int j = 0; j < n; ++j)
          o[j] = sinf(a[j]);
        break;
      case node::type::cos:
        for (int j = 0; j < n; ++j)
          o[j] = cosf(a[j]);
        break;
      case node::type::log:
        for (int j = 0; j < n; ++j) {
          float v = fabsf(a[j]);
          o[j] = v < detail::MIN_VAL ? 0.f : logf(v);
        }
        break;
      case node::type::sq:
        for (int j = 0; j < n; ++j)
          o[j] = a[j] * a[j];
        break;
      case node::type::sqrt:
        for (int j = 0; j < n; ++j)
          o[j] = sqrtf(fabsf(a[j]));
        break;
      case node::type::neg:
        for (int j = 0; j < n; ++j)
          o[j] = -a[j];
        break;
      case node::type::cube:
        for (int j = 0; j < n; ++j)
          o[j] = a[j] * a[j] * a[j];
        break;
      default:
        for (int j = 0; j < n; ++j) {
          float in[2] = {a[j], 0.0f};
          o[j] = detail::evaluate_node(nd, data, stride, r0 + j, in);
        }
        break;
      }
      slot[sp - 1] = o;
    } else {
      const float *a = slot[sp - 1];
      const float *b = slot[sp - 2];
      float *o = buf[sp - 2];
      switch (nd.t) {
      case node::type::add:
        for (int j = 0; j < n; ++j)
          o[j] = a[j] + b[j];
        break;
      case node::type::sub:
        for (int j = 0; j < n; ++j)
          o[j] = a[j] - b[j];
        break;
      case node::type::mul:
        for (int j = 0; j < n; ++j)
          o[j] = a[j] * b[j];
        break;
      case node::type::div:
        for (int j = 0; j < n; ++j)
          o[j] = fabsf(b[j]) < detail::MIN_VAL ? 1.0f : a[j] / b[j];
        break;
      case node::type::fdim:
        for (int j = 0; j < n; ++j)
          o[j] = a[j] <= b[j] ? 0.0f : a[j] - b[j];
        break;
      case node::type::gt:
        for (int j = 0; j < n; ++j)
          o[j] = a[j] > b[j] ? 1.0f : 0.0f;
        break;
      default:
        for (int j = 0; j < n; ++j) {
          float in[2] = {a[j], b[j]};
          o[j] = detail::evaluate_node(nd, data, stride, r0 + j, in);
        }
        break;
      }
      slot[sp - 2] = o;
      --sp;
    }
  }
  return slot[sp - 1];
}

struct eval_scratch {
  float buf[MAX_STACK_SIZE][kBlock];
  const float *slot[MAX_STACK_SIZE];
};

void execute_one(const program &p, const float *data, uint64_t n_rows,
                 float *out, eval_scratch &s) {
  if (!fits_block_stack(p)) {
    execute_kernel(const_cast<program_t>(&p), data, out, n_rows, 1);
    return;
  }
  for (uint64_t r0 = 0; r0 < n_rows; r0 += kBlock) {
    int n = static_cast<int>(std::min<uint64_t>(kBlock, n_rows - r0));
    const float *res = eval_block(p, data, n_rows, r0, n, s.buf, s.slot);
    std::copy(res, res + n, out + r0);
  }
}

}

program::program()
    : len(0), depth(0), raw_fitness_(0.0f), metric(metric_t::mse),
      mut_type(mutation_t::none), nodes(nullptr) {}

program::~program() { delete[] nodes; }

program::program(const program &src)
    : len(src.len), depth(src.depth), raw_fitness_(src.raw_fitness_),
      metric(src.metric), mut_type(src.mut_type) {
  nodes = new node[len];
  std::copy(src.nodes, src.nodes + src.len, nodes);
}

program::program(program &&src) noexcept
    : nodes(src.nodes), len(src.len), depth(src.depth),
      raw_fitness_(src.raw_fitness_), metric(src.metric),
      mut_type(src.mut_type) {
  src.nodes = nullptr;
  src.len = 0;
}

program &program::operator=(program &&src) noexcept {
  if (this != &src) {
    delete[] nodes;
    nodes = src.nodes;
    len = src.len;
    depth = src.depth;
    raw_fitness_ = src.raw_fitness_;
    metric = src.metric;
    mut_type = src.mut_type;
    src.nodes = nullptr;
    src.len = 0;
  }
  return *this;
}

program &program::operator=(const program &src) {
  if (this == &src)
    return *this;
  len = src.len;
  depth = src.depth;
  raw_fitness_ = src.raw_fitness_;
  metric = src.metric;
  mut_type = src.mut_type;

  // Copy nodes
  delete[] nodes;
  nodes = new node[len];
  std::copy(src.nodes, src.nodes + src.len, nodes);

  return *this;
}

void compute_metric(int n_rows, int n_progs, const float *y,
                    const float *y_pred, const float *w, float *score,
                    const param &params) {
  // Call appropriate metric function based on metric defined in params
  if (params.metric == metric_t::pearson) {
    weightedPearson(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::spearman) {
    weightedSpearman(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::mae) {
    meanAbsoluteError(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::mse) {
    meanSquareError(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::rmse) {
    rootMeanSquareError(n_rows, n_progs, y, y_pred, w, score);
  } else if (params.metric == metric_t::logloss) {
    logLoss(n_rows, n_progs, y, y_pred, w, score);
  } else {
    // This should not be reachable
  }
}

void execute(const program_t &d_progs, const int n_rows, const int n_progs,
             const float *data, float *y_pred) {
  eval_scratch s;
  for (int pid = 0; pid < n_progs; ++pid) {
    execute_one(d_progs[pid], data, static_cast<uint64_t>(n_rows),
                y_pred + static_cast<uint64_t>(pid) * n_rows, s);
  }
}

void find_fitness(program_t d_prog, float *score, const param &params,
                  const int n_rows, const float *data, const float *y,
                  const float *sample_weights) {

  // Compute predicted values
  std::vector<float> y_pred(n_rows);
  execute(d_prog, n_rows, 1, data, y_pred.data());

  // Compute error
  compute_metric(n_rows, 1, y, y_pred.data(), sample_weights, score, params);
}

void find_batched_fitness(int n_progs, program_t d_progs, float *score,
                          const param &params, const int n_rows,
                          const float *data, const float *y,
                          const float *sample_weights) {

  std::vector<float> y_pred((uint64_t)n_rows * (uint64_t)n_progs);
  execute(d_progs, n_rows, n_progs, data, y_pred.data());

  // Compute error
  compute_metric(n_rows, n_progs, y, y_pred.data(), sample_weights, score,
                 params);
}

void set_fitness(program &h_prog, const param &params, const int n_rows,
                 const float *data, const float *y,
                 const float *sample_weights) {

  std::vector<float> score(1);

  find_fitness(&h_prog, score.data(), params, n_rows, data, y, sample_weights);

  // Update host and device score for program
  h_prog.raw_fitness_ = score[0];
}

namespace {

float fused_score(const program &p, metric_t metric, const float *data,
                  const float *Y, const float *W, uint64_t n_samples,
                  float WS, eval_scratch &s) {
  float N = static_cast<float>(n_samples);
  float out = 0.0f;
  for (uint64_t r0 = 0; r0 < n_samples; r0 += kBlock) {
    int n = static_cast<int>(std::min<uint64_t>(kBlock, n_samples - r0));
    const float *res = eval_block(p, data, n_samples, r0, n, s.buf, s.slot);
    if (metric == metric_t::mse) {
      for (int j = 0; j < n; ++j) {
        uint64_t i = r0 + j;
        float d = res[j] - Y[i];
        float error = N * W[i] * d * d / WS;
        out += error / N;
      }
    } else {
      for (int j = 0; j < n; ++j) {
        uint64_t i = r0 + j;
        float logsig;
        float yp = res[j];
        float y = Y[i];
        float w = W[i];
        if (yp < -33.3)
          logsig = yp;
        else if (yp <= -18)
          logsig = yp - expf(yp);
        else if (yp <= 37)
          logsig = -log1pf(expf(-yp));
        else
          logsig = -expf(-yp);
        float error = ((1 - y) * yp - logsig) * (N * w / WS);
        out += error / N;
      }
    }
  }
  return out;
}

}

namespace {

struct prog_key {
  const node *nodes;
  int len;
  uint64_t hash;
  bool operator==(const prog_key &o) const {
    return len == o.len && hash == o.hash &&
           std::memcmp(nodes, o.nodes, sizeof(node) * len) == 0;
  }
};

struct prog_key_hash {
  size_t operator()(const prog_key &k) const { return k.hash; }
};

prog_key make_key(const program &p) {
  static_assert(sizeof(node) == 8, "node is hashed as 8 raw bytes");
  uint64_t h = 1469598103934665603ull ^ static_cast<uint64_t>(p.len);
  for (int i = 0; i < p.len; ++i) {
    uint64_t w;
    std::memcpy(&w, p.nodes + i, sizeof(w));
    h = (h ^ w) * 1099511628211ull;
    h ^= h >> 29;
  }
  return prog_key{p.nodes, p.len, h};
}

}

void set_batched_fitness(int n_progs, std::vector<program> &h_progs,
                         const param &params, const int n_rows,
                         const float *data, const float *y,
                         const float *sample_weights) {
  set_batched_fitness(n_progs, h_progs, nullptr, params, n_rows, data, y,
                      sample_weights);
}

void set_batched_fitness(int n_progs, std::vector<program> &h_progs,
                         const std::vector<program> *prev_progs,
                         const param &params, const int n_rows,
                         const float *data, const float *y,
                         const float *sample_weights) {
  std::unordered_map<prog_key, int, prog_key_hash> seen;
  seen.reserve(2 * n_progs);
  std::vector<int> source(n_progs);
  std::vector<int> work;
  work.reserve(n_progs);
  if (prev_progs != nullptr) {
    for (int i = 0; i < static_cast<int>(prev_progs->size()); ++i) {
      seen.emplace(make_key((*prev_progs)[i]), -1 - i);
    }
  }
  for (int pid = 0; pid < n_progs; ++pid) {
    auto it = seen.emplace(make_key(h_progs[pid]), pid);
    source[pid] = it.first->second;
    if (it.second || source[pid] == pid) {
      work.push_back(pid);
    }
  }

  const uint64_t rows = static_cast<uint64_t>(n_rows);
  const bool fused =
      params.metric == metric_t::mse || params.metric == metric_t::logloss;

  float WS = 0.0f;
  for (uint64_t i = 0; i < rows; ++i) {
    WS += sample_weights[i];
  }

  std::stable_sort(work.begin(), work.end(), [&](int a, int b) {
    return h_progs[a].len > h_progs[b].len;
  });
  const int chunk = rows >= 4096 ? 1 : 4;
  thread_pool::instance().parallel_for(
      static_cast<int>(work.size()), chunk, [&](int w) {
    const int pid = work[w];
    static thread_local eval_scratch s;
    static thread_local std::vector<float> y_pred;
    program &p = h_progs[pid];
    if (fused && fits_block_stack(p)) {
      p.raw_fitness_ = fused_score(p, params.metric, data, y, sample_weights,
                                   rows, WS, s);
    } else {
      y_pred.resize(rows);
      execute_one(p, data, rows, y_pred.data(), s);
      float score;
      compute_metric(n_rows, 1, y, y_pred.data(), sample_weights, &score,
                     params);
      p.raw_fitness_ = score;
    }
  });

  for (int pid = 0; pid < n_progs; ++pid) {
    int src = source[pid];
    if (src < 0) {
      h_progs[pid].raw_fitness_ = (*prev_progs)[-1 - src].raw_fitness_;
    } else if (src != pid) {
      h_progs[pid].raw_fitness_ = h_progs[src].raw_fitness_;
    }
  }
}

float get_fitness(const program &prog, const param &params) {
  int crit = params.criterion();
  float penalty = params.parsimony_coefficient * prog.len * (2 * crit - 1);
  return (prog.raw_fitness_ - penalty);
}

/**
 * @brief Get a random subtree of the current program nodes (on CPU)
 *
 * @param pnodes  AST represented as a list of nodes
 * @param len     The total number of nodes in the AST
 * @param rng     Random number generator for subtree selection
 * @return A tuple [first,last) which contains the required subtree
 */
std::pair<int, int> get_subtree(node *pnodes, int len, PhiloxEngine &rng) {
  int start, end;
  start = end = 0;

  // Specify RNG
  uniform_real_distribution_custom<float> dist_uniform(0.0f, 1.0f);
  float bound = dist_uniform(rng);

  // Specify subtree start probs acc to Koza's selection approach
  static thread_local std::vector<float> node_probs;
  node_probs.assign(len, 0.1);
  float sum = 0.1 * len;

  for (int i = 0; i < len; ++i) {
    if (pnodes[i].is_nonterminal()) {
      node_probs[i] = 0.9;
      sum += 0.8;
    }
  }

  // Normalize vector
  for (int i = 0; i < len; ++i) {
    node_probs[i] /= sum;
  }

  // Compute cumulative sum
  std::partial_sum(node_probs.begin(), node_probs.end(), node_probs.begin());

  start = std::lower_bound(node_probs.begin(), node_probs.end(), bound) -
          node_probs.begin();
  end = start;

  // Iterate until all function arguments are satisfied in current subtree
  int num_args = 1;
  while (num_args > end - start) {
    node curr;
    curr = pnodes[end];
    if (curr.is_nonterminal())
      num_args += curr.arity();
    ++end;
  }

  return std::make_pair(start, end);
}

int get_depth(const program &p_out) {
  int depth = 0;
  static thread_local std::vector<int> arity_stack;
  arity_stack.clear();
  for (auto i = 0; i < p_out.len; ++i) {
    const node &curr = p_out.nodes[i];

    // Update depth
    int sz = arity_stack.size();
    depth = std::max(depth, sz);

    // Update stack
    if (curr.is_nonterminal()) {
      arity_stack.push_back(curr.arity());
    } else {
      // Only triggered for a depth 0 node
      if (arity_stack.empty())
        break;

      arity_stack.back() -= 1;

      while (arity_stack.back() == 0) {
        arity_stack.pop_back();
        if (arity_stack.empty())
          break;

        arity_stack.back() -= 1;
      }
    }
  }

  return depth;
}

void build_program(program &p_out, const param &params, PhiloxEngine &rng) {
  // Define data structures needed for tree
  std::stack<int, std::vector<int>> arity_stack;
  static thread_local std::vector<node> nodelist;
  nodelist.clear();

  // Specify Distributions with parameters
  uniform_int_distribution_custom<int> dist_function(
      0, params.function_set.size() - 1);
  uniform_int_distribution_custom<int> dist_initDepth(params.init_depth[0],
                                                      params.init_depth[1]);
  uniform_int_distribution_custom<int> dist_terminalChoice(0,
                                                           params.num_features);
  uniform_real_distribution_custom<float> dist_constVal(params.const_range[0],
                                                        params.const_range[1]);
  bernoulli_distribution_custom dist_nodeChoice(params.terminalRatio);
  bernoulli_distribution_custom dist_coinToss(0.5);

  // Initialize nodes
  int max_depth = dist_initDepth(rng);
  node::type func = params.function_set[dist_function(rng)];
  node curr_node(func);
  nodelist.push_back(curr_node);
  arity_stack.push(curr_node.arity());

  init_method_t method = params.init_method;
  if (method == init_method_t::half_and_half) {
    // Choose either grow or full for this tree
    bool choice = dist_coinToss(rng);
    method = choice ? init_method_t::grow : init_method_t::full;
  }

  // Fill tree
  while (!arity_stack.empty()) {
    int depth = arity_stack.size();
    p_out.depth = std::max(depth, p_out.depth);
    bool node_choice = dist_nodeChoice(rng);

    if ((node_choice == false || method == init_method_t::full) &&
        depth < max_depth) {
      // Add a function to node list
      curr_node = node(params.function_set[dist_function(rng)]);
      nodelist.push_back(curr_node);
      arity_stack.push(curr_node.arity());
    } else {
      // Add terminal
      int terminal_choice = dist_terminalChoice(rng);
      if (terminal_choice == params.num_features) {
        // Add constant
        float val = dist_constVal(rng);
        curr_node = node(val);
      } else {
        // Add variable
        int fid = terminal_choice;
        curr_node = node(fid);
      }

      // Modify nodelist
      nodelist.push_back(curr_node);

      // Modify stack
      int e = arity_stack.top();
      arity_stack.pop();
      arity_stack.push(e - 1);
      while (arity_stack.top() == 0) {
        arity_stack.pop();
        if (arity_stack.empty()) {
          break;
        }

        e = arity_stack.top();
        arity_stack.pop();
        arity_stack.push(e - 1);
      }
    }
  }

  // Set new program parameters - need to do a copy as
  // nodelist will be deleted using RAII semantics
  p_out.nodes = new node[nodelist.size()];
  std::copy(nodelist.begin(), nodelist.end(), p_out.nodes);

  p_out.len = nodelist.size();
  p_out.metric = params.metric;
  p_out.raw_fitness_ = 0.0f;
}

void point_mutation(const program &prog, program &p_out, const param &params,
                    PhiloxEngine &rng) {
  // deep-copy program
  p_out = prog;

  // Specify RNGs
  uniform_real_distribution_custom<float> dist_uniform(0.0f, 1.0f);
  uniform_int_distribution_custom<int> dist_terminalChoice(0,
                                                           params.num_features);
  uniform_real_distribution_custom<float> dist_constantVal(
      params.const_range[0], params.const_range[1]);

  // Fill with uniform numbers
  std::vector<float> node_probs(p_out.len);
  std::generate(node_probs.begin(), node_probs.end(),
                [&dist_uniform, &rng] { return dist_uniform(rng); });

  // Mutate nodes
  int len = p_out.len;
  for (int i = 0; i < len; ++i) {
    node curr(prog.nodes[i]);

    if (node_probs[i] < params.p_point_replace) {
      if (curr.is_terminal()) {
        int choice = dist_terminalChoice(rng);

        if (choice == params.num_features) {
          // Add a randomly generated constant
          curr = node(dist_constantVal(rng));
        } else {
          // Add a variable with fid=choice
          curr = node(choice);
        }
      } else if (curr.is_nonterminal()) {
        // Replace current function with another function of the same arity
        int ar = curr.arity();
        // CUML_LOG_DEBUG("Arity is %d, curr function is
        // %d",ar,static_cast<std::underlying_type<node::type>::type>(curr.t));
        std::vector<node::type> fset = params.arity_set.at(ar);
        uniform_int_distribution_custom<int> dist_fset(0, fset.size() - 1);
        int choice = dist_fset(rng);
        curr = node(fset[choice]);
      }

      // Update p_out with updated value
      p_out.nodes[i] = curr;
    }
  }
}

void crossover(const program &prog, const program &donor, program &p_out,
               const param &params, PhiloxEngine &rng) {
  // Get a random subtree of prog to replace
  std::pair<int, int> prog_slice = get_subtree(prog.nodes, prog.len, rng);
  int prog_start = prog_slice.first;
  int prog_end = prog_slice.second;

  // Set metric of output program
  p_out.metric = prog.metric;

  // MAX_STACK_SIZE can only handle tree of depth MAX_STACK_SIZE -
  // max(func_arity=2) + 1 Thus we continuously hoist the donor subtree. Actual
  // indices in donor
  int donor_start = 0;
  int donor_end = donor.len;
  int output_depth = 0;
  int iter = 0;
  do {
    ++iter;
    // Get donor subtree
    std::pair<int, int> donor_slice =
        get_subtree(donor.nodes + donor_start, donor_end - donor_start, rng);

    // Get indices w.r.t current subspace [donor_start,donor_end)
    int donor_substart = donor_slice.first;
    int donor_subend = donor_slice.second;

    // Update relative indices to global indices
    donor_substart += donor_start;
    donor_subend += donor_start;

    // Update to new subspace
    donor_start = donor_substart;
    donor_end = donor_subend;

    // Evolve on current subspace
    p_out.len =
        (prog_start) + (donor_end - donor_start) + (prog.len - prog_end);
    delete[] p_out.nodes;
    p_out.nodes = new node[p_out.len];

    // Copy slices using std::copy
    std::copy(prog.nodes, prog.nodes + prog_start, p_out.nodes);
    std::copy(donor.nodes + donor_start, donor.nodes + donor_end,
              p_out.nodes + prog_start);
    std::copy(prog.nodes + prog_end, prog.nodes + prog.len,
              p_out.nodes + (prog_start) + (donor_end - donor_start));

    output_depth = get_depth(p_out);
  } while (output_depth >= MAX_STACK_SIZE);

  // Set the depth of the final program
  p_out.depth = output_depth;
}

void subtree_mutation(const program &prog, program &p_out, const param &params,
                      PhiloxEngine &rng) {
  // Generate a random program and perform crossover
  program new_program;
  build_program(new_program, params, rng);
  crossover(prog, new_program, p_out, params, rng);
}

void hoist_mutation(const program &prog, program &p_out, const param &params,
                    PhiloxEngine &rng) {
  // Replace program subtree with a random sub-subtree

  std::pair<int, int> prog_slice = get_subtree(prog.nodes, prog.len, rng);
  int prog_start = prog_slice.first;
  int prog_end = prog_slice.second;

  std::pair<int, int> sub_slice =
      get_subtree(prog.nodes + prog_start, prog_end - prog_start, rng);
  int sub_start = sub_slice.first;
  int sub_end = sub_slice.second;

  // Update subtree indices to global indices
  sub_start += prog_start;
  sub_end += prog_start;

  p_out.len = (prog_start) + (sub_end - sub_start) + (prog.len - prog_end);
  delete[] p_out.nodes;
  p_out.nodes = new node[p_out.len];
  p_out.metric = prog.metric;

  // Copy node slices using std::copy
  std::copy(prog.nodes, prog.nodes + prog_start, p_out.nodes);
  std::copy(prog.nodes + sub_start, prog.nodes + sub_end,
            p_out.nodes + prog_start);
  std::copy(prog.nodes + prog_end, prog.nodes + prog.len,
            p_out.nodes + (prog_start) + (sub_end - sub_start));

  // Update depth
  p_out.depth = get_depth(p_out);
}

} // namespace genetic
