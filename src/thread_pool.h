#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace genetic {

class thread_pool {
public:
  static thread_pool &instance() {
    static thread_pool pool;
    return pool;
  }

  int size() const { return static_cast<int>(workers_.size()) + 1; }

  void parallel_for(int n, int chunk, const std::function<void(int)> &f,
                    bool leaf = true) {
    if (n <= 0)
      return;
    chunk = std::max(1, chunk);
    if (workers_.empty() || n <= chunk) {
      for (int i = 0; i < n; ++i)
        f(i);
      return;
    }
    job j;
    j.f = &f;
    j.n = n;
    j.chunk = chunk;
    j.leaf = leaf;

    std::unique_lock<std::mutex> lk(mutex_);
    jobs_.push_back(&j);
    cv_.notify_all();
    lk.unlock();
    run_chunks(j);
    lk.lock();
    retire(j);
    while (j.helpers > 0) {
      job *t = pick( true);
      if (t == nullptr) {
        cv_.wait(lk);
        continue;
      }
      help(*t, lk);
    }
  }

private:
  struct job {
    const std::function<void(int)> *f = nullptr;
    int n = 0, chunk = 1;
    bool leaf = true;
    std::atomic<int> next{0};
    int helpers = 0;
  };

  thread_pool() {
    unsigned hw = std::thread::hardware_concurrency();
    int n_workers = hw > 1 ? static_cast<int>(hw) - 1 : 0;
    for (int i = 0; i < n_workers; ++i)
      workers_.emplace_back([this] { worker_loop(); });
  }

  ~thread_pool() {
    {
      std::lock_guard<std::mutex> lk(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &w : workers_)
      w.join();
  }

  static void run_chunks(job &t) {
    for (;;) {
      int begin = t.next.fetch_add(t.chunk, std::memory_order_relaxed);
      if (begin >= t.n)
        return;
      int end = std::min(t.n, begin + t.chunk);
      for (int i = begin; i < end; ++i)
        (*t.f)(i);
    }
  }

  job *pick(bool leaf_only) {
    for (job *t : jobs_) {
      if (t->next.load(std::memory_order_relaxed) < t->n &&
          (t->leaf || !leaf_only))
        return t;
    }
    return nullptr;
  }

  void retire(job &t) {
    auto it = std::find(jobs_.begin(), jobs_.end(), &t);
    if (it != jobs_.end())
      jobs_.erase(it);
  }

  void help(job &t, std::unique_lock<std::mutex> &lk) {
    ++t.helpers;
    lk.unlock();
    run_chunks(t);
    lk.lock();
    retire(t);
    if (--t.helpers == 0)
      cv_.notify_all();
  }

  void worker_loop() {
    std::unique_lock<std::mutex> lk(mutex_);
    while (!stop_) {
      job *t = pick( false);
      if (t == nullptr) {
        cv_.wait(lk);
        continue;
      }
      help(*t, lk);
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<job *> jobs_;
  bool stop_ = false;
};

}
