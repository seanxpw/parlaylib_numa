#ifndef PARLAY_INTERNAL_TOPOLOGY_H_
#define PARLAY_INTERNAL_TOPOLOGY_H_

#include <vector>
#include <iostream>
#include <mutex>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <sstream>
#include <string>
#include <cerrno>

// Linux specific includes
#include <numa.h>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>

namespace parlay {
namespace internal {

// -------------------------------
// Debug / behavior toggles
// -------------------------------

// 1 = verbose stderr logging in build/pin/print_info
#ifndef PARLAY_TOPOLOGY_VERBOSE
#define PARLAY_TOPOLOGY_VERBOSE 0
#endif

// 1 = call numa_run_on_node(node) inside pin_thread (NOT recommended if you want strict CPU pin)
// If enabled, we call it BEFORE pthread_setaffinity_np so final result is single-cpu pin.
#ifndef PARLAY_TOPOLOGY_USE_NUMA_RUN_ON_NODE
#define PARLAY_TOPOLOGY_USE_NUMA_RUN_ON_NODE 0
#endif

// 1 = set preferred memory node (does not change cpu affinity)
#ifndef PARLAY_TOPOLOGY_USE_NUMA_PREFERRED
#define PARLAY_TOPOLOGY_USE_NUMA_PREFERRED 0
#endif

// How many workers to print in mapping dump (0 = print all)
#ifndef PARLAY_TOPOLOGY_PRINT_WORKERS_LIMIT
#define PARLAY_TOPOLOGY_PRINT_WORKERS_LIMIT 64
#endif

// Simple POD structure to store worker location/affinity
struct WorkerLocation {
  int numa_node;
  int cpu_id;
};

// Topology management class
class Topology {
public:
  int num_numa_nodes;
  std::vector<WorkerLocation> worker_map;                  // worker_id -> (node, cpu)
  std::vector<std::pair<int, int>> numa_worker_ranges;     // node -> [start_worker_id, end_worker_id)

  Topology() : num_numa_nodes(1) {}

  // Build topology mapping: Intersect physical hardware availability with process affinity mask
  void build(size_t num_workers) {
    // 1) Get the set of CPUs the current process/thread is allowed to run on.
    // NOTE: sched_getaffinity(0,...) is per-thread, but typically represents the task's allowed set here.
    cpu_set_t process_mask;
    CPU_ZERO(&process_mask);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &process_mask) != 0) {
#if PARLAY_TOPOLOGY_VERBOSE
      std::cerr << "[topo] sched_getaffinity failed: " << std::strerror(errno)
                << " (fallback: assume all CPUs)\n";
#endif
      for (int i = 0; i < CPU_SETSIZE; i++) CPU_SET(i, &process_mask);
    }

#if PARLAY_TOPOLOGY_VERBOSE
    std::cerr << "[topo] build(num_workers=" << num_workers << ") pid=" << getpid()
              << " initial_allowed=" << affinity_to_string(process_mask) << "\n";
#endif

    if (numa_available() < 0) {
#if PARLAY_TOPOLOGY_VERBOSE
      std::cerr << "[topo] NUMA not available; fallback_no_numa\n";
#endif
      fallback_no_numa(num_workers);
      return;
    }

    int max_node = numa_max_node();
    num_numa_nodes = max_node + 1;

    worker_map.clear();
    worker_map.resize(num_workers);

    numa_worker_ranges.assign(num_numa_nodes, {0, 0});

    struct bitmask* hardware_cpu_mask = numa_allocate_cpumask();
    if (!hardware_cpu_mask) {
#if PARLAY_TOPOLOGY_VERBOSE
      std::cerr << "[topo] numa_allocate_cpumask failed; fallback_no_numa\n";
#endif
      fallback_no_numa(num_workers);
      return;
    }

    size_t current_worker = 0;
    int max_cpus = numa_num_configured_cpus();

#if PARLAY_TOPOLOGY_VERBOSE
    std::cerr << "[topo] numa_max_node=" << max_node
              << " numa_num_configured_cpus=" << max_cpus << "\n";
#endif

    // Iterate over all possible physical NUMA nodes
    for (int node = 0; node < num_numa_nodes; ++node) {
      if (current_worker >= num_workers) break;

      // Get physical CPUs belonging to this Node
      if (numa_node_to_cpus(node, hardware_cpu_mask) != 0) {
#if PARLAY_TOPOLOGY_VERBOSE
        std::cerr << "[topo] numa_node_to_cpus(" << node << ") failed\n";
#endif
        numa_worker_ranges[node] = {0, 0};
        continue;
      }

      int start_id = static_cast<int>(current_worker);
      bool node_has_workers = false;
      int matched_cpus = 0;

      // Iterate over all possible CPUs and assign those that are:
      //   (a) physically on this node
      //   (b) allowed by process/thread affinity mask
      for (int cpu = 0; cpu < max_cpus; ++cpu) {
        if (numa_bitmask_isbitset(hardware_cpu_mask, cpu) &&
            CPU_ISSET(cpu, &process_mask)) {
          matched_cpus++;
          if (current_worker < num_workers) {
            worker_map[current_worker] = {node, cpu};
            current_worker++;
            node_has_workers = true;
          }
        }
      }

      if (node_has_workers) {
        numa_worker_ranges[node] = {start_id, static_cast<int>(current_worker)};
      } else {
        numa_worker_ranges[node] = {0, 0};
      }

#if PARLAY_TOPOLOGY_VERBOSE
      std::cerr << "[topo] node=" << node
                << " matched_cpus=" << matched_cpus
                << " workers_assigned=[" << numa_worker_ranges[node].first
                << "," << numa_worker_ranges[node].second << ")\n";
#endif
    }

    // Fill remaining workers by reusing existing mappings (oversubscription)
    fill_remaining_workers_round_robin(current_worker, num_workers);

    numa_free_cpumask(hardware_cpu_mask);

#if PARLAY_TOPOLOGY_VERBOSE
    // Print a sample of worker_map
    size_t sample = std::min<size_t>(num_workers, 16);
    std::cerr << "[topo] worker_map sample (first " << sample << "):\n";
    for (size_t i = 0; i < sample; ++i) {
      std::cerr << "  worker " << i
                << " cpu=" << worker_map[i].cpu_id
                << " node=" << worker_map[i].numa_node << "\n";
    }
#endif
  }

  // Pin the current thread to the CPU corresponding to the given worker_id
  void pin_thread(size_t worker_id) {
    if (worker_id >= worker_map.size()) return;

    auto loc = worker_map[worker_id];

#if PARLAY_TOPOLOGY_VERBOSE
    std::cerr << "[pin] worker=" << worker_id
              << " target_cpu=" << loc.cpu_id
              << " target_node=" << loc.numa_node
              << " before_allowed=" << current_affinity_string()
              << " cpu_now=" << safe_getcpu()
              << " node_now=" << safe_node_of_cpu(safe_getcpu())
              << "\n";
#endif

#if PARLAY_TOPOLOGY_USE_NUMA_RUN_ON_NODE
    // If enabled: set node-level run mask first, then tighten to one CPU.
    errno = 0;
    int rc_node = numa_run_on_node(loc.numa_node);
#if PARLAY_TOPOLOGY_VERBOSE
    if (rc_node != 0) {
      std::cerr << "[pin] numa_run_on_node(" << loc.numa_node << ") FAILED: "
                << std::strerror(errno) << "\n";
    } else {
      std::cerr << "[pin] numa_run_on_node(" << loc.numa_node << ") OK\n";
    }
#endif
#endif

    // 1) Set CPU affinity to a single CPU
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(loc.cpu_id, &cpuset);

    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
      errno = rc; // pthread_* returns error number directly
#if PARLAY_TOPOLOGY_VERBOSE
      std::cerr << "[pin] pthread_setaffinity_np FAILED: " << std::strerror(errno) << "\n";
#endif
    }

#if PARLAY_TOPOLOGY_USE_NUMA_PREFERRED
    // Prefer local memory allocations on this node (does not override strict membind policies).
    errno = 0;
    numa_set_preferred(loc.numa_node);
#if PARLAY_TOPOLOGY_VERBOSE
    std::cerr << "[pin] numa_set_preferred(" << loc.numa_node << ") (errno=" << errno << ")\n";
#endif
#endif

#if PARLAY_TOPOLOGY_VERBOSE
    std::cerr << "[pin] worker=" << worker_id
              << " after_allowed=" << current_affinity_string()
              << " cpu_now=" << safe_getcpu()
              << " node_now=" << safe_node_of_cpu(safe_getcpu())
              << "\n";
#endif
  }

  // Debugging: Print topology information
  // Includes worker_id -> (node, cpu) mapping
  void print_info() const {
    printf("--- Scheduler Topology Info ---\n");
    printf("Total NUMA Nodes (Physical): %d\n", num_numa_nodes);

    // Print node ranges and a short CPU sample
    for (int n = 0; n < num_numa_nodes; ++n) {
      auto range = numa_worker_ranges[n];
      if (range.first != range.second) {
        printf("  Node %d: Workers [%d, %d)  CPU sample:", n, range.first, range.second);
        int shown = 0;
        for (int w = range.first; w < range.second && shown < 8; ++w, ++shown) {
          printf(" %d", worker_map[w].cpu_id);
        }
        printf("\n");
      } else {
        printf("  Node %d: (No workers / Disabled)\n", n);
      }
    }

    // Print full worker mapping (or limited)
    size_t limit = worker_map.size();
#if PARLAY_TOPOLOGY_PRINT_WORKERS_LIMIT > 0
    limit = std::min<size_t>(limit, PARLAY_TOPOLOGY_PRINT_WORKERS_LIMIT);
#endif
    printf("Worker mapping (worker_id -> node,cpu) [showing %zu / %zu]:\n", limit, worker_map.size());
    for (size_t i = 0; i < limit; ++i) {
      printf("  worker %zu -> node %d, cpu %d\n", i, worker_map[i].numa_node, worker_map[i].cpu_id);
    }
    if (limit < worker_map.size()) {
      printf("  ... (define PARLAY_TOPOLOGY_PRINT_WORKERS_LIMIT=0 to print all)\n");
    }

    printf("-------------------------------\n");
  }

private:
  // Convert an affinity mask to a comma-separated string
  static std::string affinity_to_string(const cpu_set_t& set) {
    std::ostringstream oss;
    bool first = true;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
      if (CPU_ISSET(c, &set)) {
        if (!first) oss << ",";
        oss << c;
        first = false;
      }
    }
    if (first) return "<empty>";
    return oss.str();
  }

  // Current thread affinity string
  static std::string current_affinity_string() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &set) != 0) {
      return std::string("<sched_getaffinity failed: ") + std::strerror(errno) + ">";
    }
    return affinity_to_string(set);
  }

  static int safe_getcpu() {
#ifdef __linux__
    int c = sched_getcpu();
    return (c >= 0) ? c : -1;
#else
    return -1;
#endif
  }

  static int safe_node_of_cpu(int cpu) {
    if (cpu < 0) return -1;
    if (numa_available() < 0) return -1;
    int n = numa_node_of_cpu(cpu);
    return (n >= 0) ? n : -1;
  }

  void fallback_no_numa(size_t num_workers) {
    num_numa_nodes = 1;
    worker_map.clear();
    worker_map.resize(num_workers);
    numa_worker_ranges.clear();
    numa_worker_ranges.push_back({0, (int)num_workers});
    for (size_t i = 0; i < num_workers; ++i) worker_map[i] = {0, 0};
  }

  void fill_remaining_workers_round_robin(size_t current_worker, size_t total_workers) {
    size_t map_idx = 0;
    size_t workers_found = current_worker;

#if PARLAY_TOPOLOGY_VERBOSE
    if (workers_found == 0) {
      std::cerr << "[topo] WARNING: no CPUs matched process_mask; mapping all to {0,0}\n";
    } else if (workers_found < total_workers) {
      std::cerr << "[topo] oversubscription: requested=" << total_workers
                << " mapped_unique=" << workers_found
                << " (reusing mappings round-robin)\n";
    }
#endif

    while (current_worker < total_workers) {
      if (workers_found > 0) {
        worker_map[current_worker] = worker_map[map_idx];

        // Try to extend range statistics if contiguous
        int node = worker_map[current_worker].numa_node;
        if (node >= 0 && node < (int)numa_worker_ranges.size()) {
          if (numa_worker_ranges[node].second == (int)current_worker) {
            numa_worker_ranges[node].second++;
          }
        }

        map_idx = (map_idx + 1) % workers_found;
      } else {
        worker_map[current_worker] = {0, 0};
      }
      current_worker++;
    }
  }
};

} // namespace internal
} // namespace parlay

#endif // PARLAY_INTERNAL_TOPOLOGY_H_
