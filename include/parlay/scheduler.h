
#ifndef PARLAY_SCHEDULER_H_
#define PARLAY_SCHEDULER_H_

#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <atomic>
#include <chrono>         // IWYU pragma: keep
#include <memory>
#include <thread>
#include <type_traits>    // IWYU pragma: keep
#include <utility>
#include <vector>

#include "internal/work_stealing_deque.h"         // IWYU pragma: keep
#include "internal/work_stealing_job.h"
#include "internal/topology.h"
// IWYU pragma: no_include <bits/chrono.h>
// IWYU pragma: no_include <bits/this_thread_sleep.h>



// True if the scheduler should scale the number of awake workers
// proportional to the amount of work to be done. This saves CPU
// time if there is not any parallel work available, but may cause
// some startup lag when more parallelism becomes available.
//
// Default: true
#ifndef PARLAY_ELASTIC_PARALLELISM
#define PARLAY_ELASTIC_PARALLELISM true
#endif


// PARLAY_ELASTIC_STEAL_TIMEOUT sets the number of microseconds
// that a worker will attempt to steal jobs, such that if no
// jobs are successfully stolen, it will go to sleep.
//
// Default: 10000 (10 milliseconds)
#ifndef PARLAY_ELASTIC_STEAL_TIMEOUT
#define PARLAY_ELASTIC_STEAL_TIMEOUT 10000
#endif


#if PARLAY_ELASTIC_PARALLELISM
#include "internal/atomic_wait.h"
#endif

namespace parlay {


template <typename Job>
struct scheduler {

  using worker_id_type = unsigned int;

 private:
  static_assert(std::is_invocable_r_v<void, Job&>);

  struct workerInfo {
    static constexpr worker_id_type UNINITIALIZED = std::numeric_limits<worker_id_type>::max();

    worker_id_type worker_id;
    int my_numa_node;
    scheduler* my_scheduler;

    workerInfo() : worker_id(UNINITIALIZED), my_scheduler(nullptr) {}
    workerInfo(std::size_t worker_id_, int my_numa_node_, scheduler* s) : worker_id(worker_id_), my_numa_node(my_numa_node_), my_scheduler(s) {}
    workerInfo& operator=(const workerInfo&) = delete;
    workerInfo(const workerInfo&) = delete;

    workerInfo& operator=(workerInfo&& w) noexcept {
      if (this != &w) {
        worker_id = std::exchange(w.worker_id, UNINITIALIZED);
        my_numa_node = w.my_numa_node;
        my_scheduler = std::exchange(w.my_scheduler, nullptr);
      }
      return *this;
    }

    workerInfo(workerInfo&& w) noexcept { *this = std::move(w); }
  };

  // After YIELD_FACTOR * P unsuccessful steal attempts, a
  // a worker will sleep briefly for SLEEP_FACTOR * P nanoseconds
  // to give other threads a chance to work and save some cycles.
  constexpr static size_t YIELD_FACTOR = 200;
  constexpr static size_t SLEEP_FACTOR = 200;

  // The length of time that a worker must fail to steal anything
  // before it goes to sleep to save CPU time.
  constexpr static std::chrono::microseconds STEAL_TIMEOUT{PARLAY_ELASTIC_STEAL_TIMEOUT};

  static inline thread_local workerInfo worker_info{};

 public:

  const worker_id_type num_threads;

  // If the current thread is a worker of an existing scheduler, or the thread that spawned
  // a scheduler, return the most recent such scheduler.  Otherwise, returns null.
  static scheduler* get_current_scheduler() {
    return worker_info.my_scheduler;
  }

  explicit scheduler(size_t num_workers)
      : num_threads(num_workers),
        num_deques(num_threads),
        num_awake_workers(num_threads),
        parent_worker_info(std::exchange(worker_info, workerInfo{0, 0,this})),
        deques(num_deques),
        attempts(num_deques),
        spawned_threads(),
        finished_flag(false) {

    // 1. Build topology (executed only once)
    topology.build(num_threads);
    
    // Optional: Print topology info for verification
    topology.print_info();
    root_job_slots = std::make_unique<RootJobSlot[]>(topology.num_numa_nodes);
    // 2. Pin the main thread
    topology.pin_thread(0);
    
    // Initialize worker_info for the main thread
    // Note: We can now look up the NUMA ID directly from the topology
    int main_node = topology.worker_map[0].numa_node;
    worker_info = {0, main_node, this};
    parent_worker_info = std::move(worker_info);
    worker_info = {0, main_node, this};

    // 3. Start other threads
    for (worker_id_type i = 1; i < num_threads; ++i) {
      spawned_threads.emplace_back([&, i]() {
        // Pin child thread
        topology.pin_thread(i);
        
        int my_node = topology.worker_map[i].numa_node;
        worker_info = {i, my_node, this};
        worker(); // Child threads enter the work loop immediately
      });
    }
  }

  ~scheduler() {
    shutdown();
    worker_info = std::move(parent_worker_info);
  }

  // Push onto local stack.
  void spawn(Job* job) {
    int id = worker_id();
    [[maybe_unused]] bool first = deques[id].push_bottom(job);
#if PARLAY_ELASTIC_PARALLELISM
    if (first) wake_up_a_worker();
#endif
  }

  // Wait until the given condition is true.
  //
  // If conservative, this thread will simply busy wait. Otherwise,
  // it will look for work to steal and keep itself occupied. This
  // can deadlock if the stolen work wants a lock held by the code
  // that is waiting, so avoid that.
  template <typename F>
  void wait_until(F&& done, bool conservative = false) {
    // Conservative avoids deadlock if scheduler is used in conjunction
    // with user locks enclosing a wait.
    if (conservative) {
      while (!done())
        std::this_thread::yield();
    }
    // If not conservative, schedule within the wait.
    // Can deadlock if a stolen job uses same lock as encloses the wait.
    else {
      do_work_until(std::forward<F>(done));
    }
  }

  // Pop from local stack.
  Job* get_own_job() {
    auto id = worker_id();
    return deques[id].pop_bottom();
  }

  worker_id_type num_workers() { return num_threads; }
  worker_id_type worker_id() { return worker_info.worker_id; }

  bool finished() const noexcept {
    return finished_flag.load(std::memory_order_acquire);
  }

 private:
  // Align to avoid false sharing.
  struct alignas(128) attempt {
    size_t val;
  };

  public:
  internal::Topology topology;
  struct alignas(64) RootJobSlot {
    std::atomic<Job*> job{nullptr};
  };
  std::unique_ptr<RootJobSlot[]> root_job_slots;
  private:
  int num_deques;
  std::atomic<size_t> num_awake_workers;
  workerInfo parent_worker_info;
  std::vector<internal::Deque<Job>> deques;
  std::vector<attempt> attempts;
  std::vector<std::thread> spawned_threads;
  std::atomic<int> finished_flag;

  std::atomic<size_t> wake_up_counter{0};
  std::atomic<size_t> num_finished_workers{0};

  // Start an individual worker task, stealing work if no local
// work is available. May go to sleep if no work is available
// for a long time, until woken up again when notified that
// new work is available.
void worker() {
#if PARLAY_ELASTIC_PARALLELISM
  wait_for_work();
#endif

  int my_node = worker_info.my_numa_node;

  // 小工具：尝试从本 NUMA mailbox 取一个 job
  auto try_take_mailbox = [&]() -> Job* {
    auto& slot = root_job_slots[my_node].job;
    // 快速路径：绝大多数时间 mailbox 为空，避免 RMW
    if (slot.load(std::memory_order_relaxed) == nullptr) return nullptr;
    // 非空才 exchange（RMW，用 acq_rel）
    return slot.exchange(nullptr, std::memory_order_acq_rel);
  };

  while (!finished()) {
    Job* job = nullptr;

    // =========================================================
    // Step 1: 优先检查 NUMA Root mailbox (Take & Clear)
    // =========================================================
    job = try_take_mailbox();

    // =========================================================
    // Step 2: 原有逻辑：本地 pop + steal
    // 重要：break_early 只用于 finished / 超时等“真正退出”的条件
    // =========================================================
    if (job == nullptr) {
      job = get_job([&]() { return finished(); }, PARLAY_ELASTIC_PARALLELISM);
    }

    if (job) {
      (*job)();
      continue;
    }

#if PARLAY_ELASTIC_PARALLELISM
    if (!finished()) {
      // 睡前再强检查一次 mailbox，避免“刚投递就睡死”的 race
      if (Job* mb = try_take_mailbox()) {
        (*mb)();
      } else {
        wait_for_work();
      }
    }
#endif
  }

  assert(finished());
  num_finished_workers.fetch_add(1);
}

public:
  // Runs tasks until done(), stealing work if necessary.
  //
  // Does not sleep or time out since this can be called
  // by the main thread and by join points, for which sleeping
  // would cause deadlock, and timing out could cause a join
  // point to resume execution before the job it was waiting
  // on has completed.
  template <typename F>
  void do_work_until(F&& done) {
    while (true) {
      Job* job = get_job(done, false);  // timeout MUST BE false
      if (!job) return;
      (*job)();
    }
    assert(done());
  }
private:
  // Find a job, first trying local stack, then random steals.
  //
  // Returns nullptr if break_early() returns true before a job
  // is found, or, if timeout is true and it takes longer than
  // STEAL_TIMEOUT to find a job to steal.
  template <typename F>
  Job* get_job(F&& break_early, bool timeout) {
    if (break_early()) return nullptr;
    Job* job = get_own_job();
    if (job) return job;
    else job = steal_job(std::forward<F>(break_early), timeout);
    return job;
  }
  
  // Find a job with random steals.
  //
  // Returns nullptr if break_early() returns true before a job
  // is found, or, if timeout is true and it takes longer than
  // STEAL_TIMEOUT to find a job to steal.
  // template<typename F>
  // Job* steal_job(F&& break_early, bool timeout) {
  //   size_t id = worker_id();
  //   const auto start_time = std::chrono::steady_clock::now();
  //   do {
  //     // By coupon collector's problem, this should touch all.
  //     for (size_t i = 0; i <= YIELD_FACTOR * num_deques; i++) {
  //       if (break_early()) return nullptr;
  //       Job* job = try_steal(id);
  //       if (job) return job;
  //     }
  //     std::this_thread::sleep_for(std::chrono::nanoseconds(num_deques * 100));
  //   } while (!timeout || std::chrono::steady_clock::now() - start_time < STEAL_TIMEOUT);
  //   return nullptr;
  // }

// 【方案A】在 steal_job 内部周期性抢占检查本 NUMA mailbox。
// 语义：mailbox 一旦有活，直接取走并返回 job，而不是 break_early 退出。
template <typename F>
Job* steal_job(F&& break_early, bool timeout) {
  size_t worker_id = worker_info.worker_id;
  int my_node = worker_info.my_numa_node;

  const auto start_time = std::chrono::steady_clock::now();

  auto& local_range = topology.numa_worker_ranges[my_node];
  size_t local_size = local_range.second - local_range.first;

  // 每个 worker 独有的随机序列基
  size_t my_id_hash = hash(worker_id);

  constexpr size_t LOCAL_TRIAL_FACTOR = 10;
  size_t local_trials = (local_size > 0) ? local_size : 0;
  local_trials *= LOCAL_TRIAL_FACTOR;

  size_t global_trials = num_deques;
  global_trials *= YIELD_FACTOR;

  // -------- mailbox 抢占轮询：低开销 --------
  auto try_take_mailbox = [&]() -> Job* {
    auto& slot = root_job_slots[my_node].job;
    if (slot.load(std::memory_order_relaxed) == nullptr) return nullptr;
    return slot.exchange(nullptr, std::memory_order_acq_rel);
  };

  // 每 64 次 steal attempt 检查一次 mailbox（可根据你 workload 调整）
  constexpr size_t MAILBOX_POLL_MASK = 63;
  size_t poll_ctr = 0;

  auto poll_mailbox = [&]() -> Job* {
    if (((++poll_ctr) & MAILBOX_POLL_MASK) == 0) {
      return try_take_mailbox();
    }
    return nullptr;
  };

  // 一进来先看一次，降低投递->响应延迟
  if (Job* mb = try_take_mailbox()) return mb;

  do {
    // =========================================================
    // Phase 1: Local NUMA Stealing
    // =========================================================
    if (local_size > 1) {
      for (size_t i = 0; i < local_trials; i++) {
        if (break_early()) return nullptr;

        // 抢占：定期检查 mailbox，有就直接返回
        if (Job* mb = poll_mailbox()) return mb;

        size_t step_hash = hash(attempts[worker_id].val);
        attempts[worker_id].val++;

        size_t offset = (my_id_hash + step_hash) % local_size;
        size_t target = local_range.first + offset;

        if (target == worker_id) continue;

        if (Job* job = try_steal(target)) return job;
      }
    }

    // =========================================================
    // Phase 2: Global Stealing
    // =========================================================
    for (size_t i = 0; i < global_trials; i++) {
      if (break_early()) return nullptr;

      if (Job* mb = poll_mailbox()) return mb;

      size_t step_hash = hash(attempts[worker_id].val);
      attempts[worker_id].val++;

      size_t target = (my_id_hash + step_hash) % num_deques;
      if (target == worker_id) continue;

      if (Job* job = try_steal(target)) return job;
    }

    // =========================================================
    // Sleep Strategy: 睡前强检查 mailbox，避免 race
    // =========================================================
    if (Job* mb = try_take_mailbox()) return mb;

    std::this_thread::sleep_for(std::chrono::nanoseconds(num_deques * 100));

  } while (!timeout || std::chrono::steady_clock::now() - start_time < STEAL_TIMEOUT);

  return nullptr;
}

// 必须配合修改 try_steal
  Job* try_steal(size_t target) {
    // 这里的 target 是外面算好的，这里只管偷
    auto [job, empty] = deques[target].pop_top();
#if PARLAY_ELASTIC_PARALLELISM
    if (!empty) wake_up_a_worker();
#endif
    return job;
  }

//   Job* try_steal(size_t id) {
//     // use hashing to get "random" target
//     size_t target = (hash(id) + hash(attempts[id].val)) % num_deques;
//     attempts[id].val++;
//     auto [job, empty] = deques[target].pop_top();
// #if PARLAY_ELASTIC_PARALLELISM
//     if (!empty) wake_up_a_worker();
// #endif
//     return job;
//   }

#if PARLAY_ELASTIC_PARALLELISM
public:
  // Wakes up at least one sleeping worker (more than one
  // worker may be woken up depending on the implementation).
  void wake_up_a_worker() {
    if (num_awake_workers.load(std::memory_order_acquire) < num_threads) {
      wake_up_counter.fetch_add(1);
      parlay::atomic_notify_one(&wake_up_counter);
    }
  }
  
  // Wake up all sleeping workers
  void wake_up_all_workers() {
    if (num_awake_workers.load(std::memory_order_acquire) < num_threads) {
      wake_up_counter.fetch_add(1);
      parlay::atomic_notify_all(&wake_up_counter);
    }
  }
  
  // Wait until notified to wake up
  void wait_for_work() {
    num_awake_workers.fetch_sub(1);
    parlay::atomic_wait(&wake_up_counter, wake_up_counter.load());
    num_awake_workers.fetch_add(1);
  }
private:
#endif

  size_t hash(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return static_cast<size_t>(x);
  }
  
  void shutdown() {
    finished_flag.store(true, std::memory_order_release);
#if PARLAY_ELASTIC_PARALLELISM
    // We must spam wake all workers until they finish in
    // case any of them are just about to fall asleep, since
    // they might therefore miss the flag to finish
    while (num_finished_workers.load() < num_threads - 1) {
      wake_up_all_workers();
      std::this_thread::yield();
    }
#endif
    for (worker_id_type i = 1; i < num_threads; ++i) {
      spawned_threads[i - 1].join();
    }
  }
};


class fork_join_scheduler {
  using Job = WorkStealingJob;
  using scheduler_t = scheduler<Job>;
struct OwningJob : WorkStealingJob {
    std::function<void()> func;
    OwningJob(std::function<void()> f) : func(std::move(f)) {}
    void execute() override { func(); }
  };
 public:
template <typename F>
  static void numa_aware_parfor(scheduler_t& scheduler, size_t start, size_t end, F&& f, size_t granularity = 0) {
    if (end <= start) return;
    
    // 如果只有一个 NUMA 节点，直接回退到普通 parfor
    if (scheduler.topology.num_numa_nodes <= 1) {
        parfor(scheduler, start, end, std::forward<F>(f), granularity);
        return;
    }
  if (granularity == 0) {
      size_t done = get_granularity(start, end, f);
      granularity = std::max(done, (end - start) / static_cast<size_t>(128 * scheduler.num_threads));
      start += done;
    }
    size_t nodes = scheduler.topology.num_numa_nodes;
    size_t chunk_size = (end - start) / nodes;
    
    // 使用 unique_ptr 管理生命周期
    std::vector<std::unique_ptr<OwningJob>> root_jobs;
    root_jobs.reserve(nodes);

    for (size_t i = 0; i < nodes; ++i) {
      size_t c_start = start + i * chunk_size;
      size_t c_end = (i == nodes - 1) ? end : (c_start + chunk_size);
      
      if (c_start >= c_end) {
          root_jobs.push_back(nullptr);
          continue;
      }

      // 构造闭包，内部调用 parfor_
      auto task_lambda = [&scheduler, c_start, c_end, &f, granularity]() {
          parfor_(scheduler, c_start, c_end, f, granularity, false);
      };

      auto job_ptr = std::make_unique<OwningJob>(std::move(task_lambda));
      
      // 投递到信箱
      scheduler.root_job_slots[i].job.store(job_ptr.get(), std::memory_order_release);
      root_jobs.push_back(std::move(job_ptr));
    }

    // 唤醒所有 Worker
    scheduler.wake_up_all_workers();

    // 主线程阻塞等待所有 Root Job 完成
    auto all_done = [&]() {
      for (const auto& job : root_jobs) {
        if (job && !job->finished()) return false;
      }
      return true;
    };
    
    // 主线程参与 Stealing 直到完成
    scheduler.do_work_until(all_done);

    // 清理信箱指针 (防悬垂)
    for (size_t i = 0; i < nodes; ++i) {
      scheduler.root_job_slots[i].job.store(nullptr, std::memory_order_relaxed);
    }
  }
  // Fork two thunks and wait until they both finish.
  template <typename L, typename R>
  static void pardo(scheduler_t& scheduler, L&& left, R&& right, bool conservative = false) {
    auto execute_right = [&]() { std::forward<R>(right)(); };
    auto right_job = make_job(right);
    scheduler.spawn(&right_job);
    std::forward<L>(left)();
    if (const Job* job = scheduler.get_own_job(); job != nullptr) {
      assert(job == &right_job);
      execute_right();
    }
    else {
      auto done = [&]() { return right_job.finished(); };
      scheduler.wait_until(done, conservative);
      assert(right_job.finished());
    }
  }

  template <typename F>
  static void parfor(scheduler_t& scheduler, size_t start, size_t end, F&& f, size_t granularity = 0, bool conservative = false) {
    if (end <= start) return;
    if (granularity == 0) {
      size_t done = get_granularity(start, end, f);
      granularity = std::max(done, (end - start) / static_cast<size_t>(128 * scheduler.num_threads));
      start += done;
    }
    parfor_(scheduler, start, end, f, granularity, conservative);
  }

 private:
  template <typename F>
  static size_t get_granularity(size_t start, size_t end, F& f) {
    size_t done = 0;
    size_t sz = 1;
    unsigned long long int ticks = 0;
    do {
      sz = std::min(sz, end - (start + done));
      auto tstart = std::chrono::steady_clock::now();
      for (size_t i = 0; i < sz; i++) f(start + done + i);
      auto tstop = std::chrono::steady_clock::now();
      ticks = static_cast<unsigned long long int>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(tstop - tstart).count());
      done += sz;
      sz *= 2;
    } while (ticks < 1000 && done < (end - start));
    return done;
  }

  template <typename F>
  static void parfor_(scheduler_t& scheduler, size_t start, size_t end, F& f, size_t granularity, bool conservative) {
    if ((end - start) <= granularity)
      for (size_t i = start; i < end; i++) f(i);
    else {
      size_t n = end - start;
      // Not in middle to avoid clashes on set-associative caches on powers of 2.
      size_t mid = (start + (9 * (n + 1)) / 16);
      pardo(scheduler,
            [&]() { parfor_(scheduler, start, mid, f, granularity, conservative); },
            [&]() { parfor_(scheduler, mid, end, f, granularity, conservative); },
            conservative);
    }
  }

};

}  // namespace parlay

#endif  // PARLAY_SCHEDULER_H_
