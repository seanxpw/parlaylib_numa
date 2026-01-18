#ifndef PARLAY_INTERNAL_TOPOLOGY_H_
#define PARLAY_INTERNAL_TOPOLOGY_H_

#include <vector>
#include <iostream>
#include <mutex>
#include <algorithm>
#include <cstring>
#include <cassert>

// Linux specific includes
#include <numa.h>
#include <sched.h>
#include <pthread.h>

namespace parlay {
namespace internal {

// Simple POD structure to store worker location/affinity
struct WorkerLocation {
    int numa_node;
    int cpu_id;
};

// Topology management class
class Topology {
public:
    int num_numa_nodes;
    std::vector<WorkerLocation> worker_map; // worker_id -> (node, cpu)
    std::vector<std::pair<int, int>> numa_worker_ranges; // node -> [start_worker_id, end_worker_id)

    Topology() : num_numa_nodes(1) {}

    // Build topology mapping: Intersect physical hardware availability with process affinity mask
    void build(size_t num_workers) {
        // 1. Get the set of CPUs the current process is allowed to run on 
        // (Handles cases like `numactl --cpubind` or `taskset`)
        cpu_set_t process_mask;
        CPU_ZERO(&process_mask);
        if (sched_getaffinity(0, sizeof(cpu_set_t), &process_mask) < 0) {
            // If failed to get affinity, default to all CPUs enabled
            for (int i = 0; i < CPU_SETSIZE; i++) CPU_SET(i, &process_mask);
        }

        // Check NUMA availability
        if (numa_available() < 0) {
            fallback_no_numa(num_workers);
            return;
        }

        int max_node = numa_max_node();
        worker_map.resize(num_workers);
        
        // num_numa_nodes is set to max physical index + 1
        // Note: some nodes in the ranges vector might remain empty if disabled by OS/numactl
        num_numa_nodes = max_node + 1;
        numa_worker_ranges.assign(num_numa_nodes, {0, 0}); 

        struct bitmask* hardware_cpu_mask = numa_allocate_cpumask();
        size_t current_worker = 0;
        
        // Iterate over all possible physical NUMA nodes
        for (int node = 0; node < num_numa_nodes; ++node) {
            if (current_worker >= num_workers) break;

            // Get physical CPUs belonging to this Node
            numa_node_to_cpus(node, hardware_cpu_mask);
            
            int start_id = static_cast<int>(current_worker);
            bool node_has_workers = false;

            // Iterate over all possible CPUs
            int max_cpus = numa_num_configured_cpus();
            for (int cpu = 0; cpu < max_cpus; ++cpu) {
                // Core check: CPU must belong to the node AND be in the process's allowed mask
                if (numa_bitmask_isbitset(hardware_cpu_mask, cpu) && 
                    CPU_ISSET(cpu, &process_mask)) {
                    
                    if (current_worker < num_workers) {
                        worker_map[current_worker] = {node, cpu};
                        current_worker++;
                        node_has_workers = true;
                    }
                }
            }
            
            // Record the worker range for this node only if it actually got workers assigned
            if (node_has_workers) {
                numa_worker_ranges[node] = {start_id, static_cast<int>(current_worker)};
            } else {
                numa_worker_ranges[node] = {0, 0}; 
            }
        }
        
        // Fallback logic: If requested workers > available physical cores (Over-subscription),
        // fill remaining workers by round-robin reusing existing valid mappings.
        fill_remaining_workers_round_robin(current_worker, num_workers);

        numa_free_cpumask(hardware_cpu_mask);
    }

    // Pin the current thread to the CPU corresponding to the given worker_id
    void pin_thread(size_t worker_id) {
        if (worker_id >= worker_map.size()) return;
        
        auto loc = worker_map[worker_id];
        
        // 1. Set CPU affinity
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(loc.cpu_id, &cpuset);
        
        int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
             // Error handling, usually silent or logged to stderr
        }
        
        // 2. Tell libnuma to prefer memory allocation on this node (optional but recommended)
        // Note: If numactl already forcibly restricted binding (e.g., membind), 
        // this call might fail if it crosses nodes, but that is acceptable.
        numa_run_on_node(loc.numa_node);
    }

    // Debugging: Print topology information
    void print_info() const {
        printf("--- Scheduler Topology Info ---\n");
        printf("Total NUMA Nodes (Physical): %d\n", num_numa_nodes);
        for (int n = 0; n < num_numa_nodes; ++n) {
            auto& range = numa_worker_ranges[n];
            if (range.first != range.second) {
                printf("  Node %d: Workers [%d, %d)\n", n, range.first, range.second);
            } else {
                printf("  Node %d: (No workers / Disabled)\n", n);
            }
        }
        printf("-------------------------------\n");
    }

private:
    void fallback_no_numa(size_t num_workers) {
        num_numa_nodes = 1;
        numa_worker_ranges.push_back({0, (int)num_workers});
        for(size_t i=0; i<num_workers; ++i) worker_map.push_back({0, 0});
    }

    void fill_remaining_workers_round_robin(size_t current_worker, size_t total_workers) {
        size_t map_idx = 0;
        size_t workers_found = current_worker;
        
        while (current_worker < total_workers) {
            if (workers_found > 0) {
                // Reuse previous valid mappings
                worker_map[current_worker] = worker_map[map_idx];
                
                // Fix range statistics (extend range to include these reused workers)
                int node = worker_map[current_worker].numa_node;
                // Only extend if this worker is contiguous to the end of the range (usually true)
                if (numa_worker_ranges[node].second == (int)current_worker) {
                     numa_worker_ranges[node].second++;
                }
                
                map_idx = (map_idx + 1) % workers_found; 
            } else {
                // Extreme case: No available CPUs at all
                worker_map[current_worker] = {0, 0}; 
            }
            current_worker++;
        }
    }
};

} // namespace internal
} // namespace parlay

#endif // PARLAY_INTERNAL_TOPOLOGY_H_