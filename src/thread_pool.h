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

  void parallel_for(int n, int chunk, const std::function<void(int)> &f) {
    if (n <= 0)
      return;
    if (in_worker() || workers_.empty() || n <= chunk) {
      for (int i = 0; i < n; ++i)
        f(i);
      return;
    }
    std::lock_guard<std::mutex> caller(call_mutex_);
    {
      std::lock_guard<std::mutex> lk(mutex_);
      job_ = &f;
      n_ = n;
      chunk_ = std::max(1, chunk);
      next_.store(0, std::memory_order_relaxed);
      active_ = static_cast<int>(workers_.size());
      ++generation_;
    }
    wake_.notify_all();
    in_worker() = true;
    run_chunks();
    in_worker() = false;
    std::unique_lock<std::mutex> lk(mutex_);
    done_.wait(lk, [this] { return active_ == 0; });
    job_ = nullptr;
  }

private:
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
    wake_.notify_all();
    for (auto &w : workers_)
      w.join();
  }

  static bool &in_worker() {
    static thread_local bool flag = false;
    return flag;
  }

  void run_chunks() {
    for (;;) {
      int begin = next_.fetch_add(chunk_, std::memory_order_relaxed);
      if (begin >= n_)
        break;
      int end = std::min(n_, begin + chunk_);
      for (int i = begin; i < end; ++i)
        (*job_)(i);
    }
  }

  void worker_loop() {
    in_worker() = true;
    unsigned long seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mutex_);
        wake_.wait(lk, [&] { return stop_ || generation_ != seen; });
        if (stop_)
          return;
        seen = generation_;
      }
      run_chunks();
      std::lock_guard<std::mutex> lk(mutex_);
      if (--active_ == 0)
        done_.notify_one();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex call_mutex_;
  std::mutex mutex_;
  std::condition_variable wake_, done_;
  const std::function<void(int)> *job_ = nullptr;
  int n_ = 0, chunk_ = 1, active_ = 0;
  std::atomic<int> next_{0};
  unsigned long generation_ = 0;
  bool stop_ = false;
};

}
