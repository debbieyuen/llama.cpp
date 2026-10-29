#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#if defined(__linux__)
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#endif

#if defined(__linux__)
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
#endif

#if defined(LLAMA_HAVE_LIBURING)
#include <liburing.h>
#endif

#include <sched.h>

#if defined(__linux__)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#endif

namespace {

// kimi-k3-spark patch 0003: LLAMA_EXPERT_STORE_CPUS=<cpu list, e.g. 0-9 or 0,2,4-7>: the store's I/O workers, the
// speculative reader, the fill thread and the CUDA host-bank poller (its own copy of this parser, in ggml-cuda.cu)
// run on these CPUs. The GB10 has 10 X925 and 10 A725 cores and the scheduler is free to put a reader on a little
// core; the Spark's `--threads 8 > 16` result says that costs decode time. Unset: no pinning. Each thread pins
// itself on entry; the set is parsed once and announced by the cache's constructor (store_cpus_log).
static const cpu_set_t * store_cpu_set(int * n_cpus = nullptr) {
    static int n_set = 0;
    static const cpu_set_t * set = [] () -> const cpu_set_t * {
        const char * env = getenv("LLAMA_EXPERT_STORE_CPUS");
        if (env == nullptr || env[0] == 0) {
            return nullptr;
        }
        static cpu_set_t s;
        CPU_ZERO(&s);
        for (const char * p = env; *p; ) {
            char * end = nullptr;
            long a = strtol(p, &end, 10);
            long b = a;
            if (end == p) {
                break;
            }
            p = end;
            if (*p == '-') {
                b = strtol(p + 1, &end, 10);
                p = end;
            }
            for (long c = a; c <= b && c >= 0 && c < CPU_SETSIZE; ++c) {
                CPU_SET((int) c, &s);
                n_set++;
            }
            while (*p == ',' || *p == ' ') {
                p++;
            }
        }
        return n_set > 0 ? &s : nullptr;
    }();
    if (n_cpus != nullptr) {
        *n_cpus = n_set;
    }
    return set;
}

static void store_cpus_log() {
    const char * env = getenv("LLAMA_EXPERT_STORE_CPUS");
    if (env == nullptr || env[0] == 0) {
        return;
    }
    int n = 0;
    if (store_cpu_set(&n) == nullptr) {
        LLAMA_LOG_WARN("%s: LLAMA_EXPERT_STORE_CPUS='%s' names no CPU, ignored\n", __func__, env);
    } else {
        LLAMA_LOG_INFO("%s: expert store threads (I/O, speculative reader, fill, CUDA poller) pinned to %d CPUs (LLAMA_EXPERT_STORE_CPUS=%s)\n", __func__, n, env);
    }
}

static void store_pin_thread() {
    const cpu_set_t * set = store_cpu_set();
    if (set != nullptr && sched_setaffinity(0, sizeof(*set), set) != 0) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            LLAMA_LOG_WARN("%s: sched_setaffinity: %s (LLAMA_EXPERT_STORE_CPUS ignored)\n", __func__, strerror(errno));
        }
    }
}

// LRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty
    int32_t n_filled = 0;            // slots that have held an expert at least once (the cache is full when == n_slots)
    int32_t n_parked = 0;            // empty slots the memory valve has released: they count as full for the scan resistance

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    // decayed use count of every (layer, expert) key, for the HBM tier: the first n_hbm slots of the group
    // live on the GPU's HBM NUMA node and hold the most used experts
    std::vector<uint16_t> uses;  // [n_layer*n_expert]
    int32_t n_hbm = 0;

    // kimi-k3-spark patch 0003: second chance (LLAMA_EXPERT_STORE_SECOND_CHANCE=N, 0 = plain LRU). Every hit on a
    // resident expert raises its slot's count (at most N); the victim search at the head spends a count instead of
    // evicting (count--, slot to the tail) and takes the first slot whose count is 0. A new or one-shot occupant
    // starts at 0. Frequency added to recency: on the Spark's 10 % cache the replay of real decode traces gives
    // 57.5 -> 59.6 % hits with N = 3 (73.4 -> 74.4 % with the speculative prefetch), the same as S3-FIFO, for
    // one byte per slot. Decode output does not depend on which slot holds an expert, so it stays bit-identical.
    std::vector<uint8_t> chance; // [n_slots]
    int32_t chance_max = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    // kimi-k3-spark patch 0003: per-slot state of the speculative reader (nullptr without it): a slot whose record
    // is queued or in flight there is never chosen as a victim, the read would land in a slot given to another expert
    const std::atomic<int8_t> * busy = nullptr;

    // kimi-k3-spark patch 0003: a layer whose slots the victim search must step over (-1: none). The prefill
    // lookahead plans layer il+1 while the GPU computes layer il from its slots, so layer il is skipped.
    int32_t skip_layer = -1;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        n_filled = 0;
        seen.assign(n_expert, 0);
        uses.assign((size_t) n_layer*n_expert, 0);
        chance.assign(n_slots, 0);
    }

    // exchange the experts of slots a and b (the caller exchanges the data); each slot keeps its place in the
    // LRU list, so the expert moved into a takes a's recency
    void exchange(int32_t a, int32_t b) {
        const int32_t ka = key_of[a];
        const int32_t kb = key_of[b];
        key_of[a] = kb;
        key_of[b] = ka;
        std::swap(chance[a], chance[b]);
        if (kb >= 0) {
            slot_map[kb / n_expert][kb % n_expert] = a;
        }
        if (ka >= 0) {
            slot_map[ka / n_expert][ka % n_expert] = b;
        }
    }

    // move slot s to the tail (most recently used)
    void touch(int32_t s) {
        if (s == tail) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    // move slot s to the head (next to be evicted); a demoted slot is one-shot, so it has no second chance either
    void demote(int32_t s) {
        chance[s] = 0;
        if (s == head) {
            return;
        }
        next[prev[s]] = next[s];
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail = prev[s];
        }
        prev[s] = -1;
        next[s] = head;
        prev[head] = s;
        head = s;
    }

    // rebuild the recency list from an explicit order of all the slots (first = least recently used)
    void relink(const std::vector<int32_t> & order) {
        GGML_ASSERT((int32_t) order.size() == n_slots);
        for (int32_t i = 0; i < n_slots; ++i) {
            const int32_t s = order[i];
            prev[s] = i > 0 ? order[i - 1] : -1;
            next[s] = i + 1 < n_slots ? order[i + 1] : -1;
        }
        head = order.front();
        tail = order.back();
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, the misses evict the least recently used experts
    // returns false if the ids select more distinct experts than there are slots
    // promote = false (prefill): hits keep their place and the misses go to the head, so a long prompt that touches
    // every expert of a layer does not evict the working set of the generation that follows (scan resistance)
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit, bool promote = true) {
        fills.clear();
        n_hit = 0;
        // scan resistance only once the cache is full: while it fills, a prefill fill is as good as any other
        const bool scan = !promote && n_filled + n_parked >= n_slots;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // hits go to the tail first, so the head can be evicted below
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                if (!scan) {
                    touch(slots[e]);
                    if (chance[slots[e]] < chance_max) {
                        chance[slots[e]]++;
                    }
                }
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            int32_t s = head;
            if (scan || busy != nullptr || skip_layer >= 0 || chance_max > 0) {
                // the hits of this layer were not moved to the tail (scan): the victim search must step over them, or a
                // miss would evict a hit of the same batch and the lookup would read -1 for it; likewise over the
                // slots the speculative reader is filling, and over the layer the GPU computes during a lookahead.
                // A resident expert with a second chance left spends it (count--, slot to the tail) and the search
                // goes on from the slot after it; the counts only fall, so the search ends at a slot with none.
                while (s >= 0) {
                    if ((scan && key_of[s] >= 0 && key_of[s] / n_expert == il && seen[key_of[s] % n_expert] == seen_gen) ||
                        (busy != nullptr && busy[s].load(std::memory_order_acquire) != 0) ||
                        (skip_layer >= 0 && key_of[s] >= 0 && key_of[s] / n_expert == skip_layer)) {
                        s = next[s];
                        continue;
                    }
                    if (!scan && chance[s] > 0 && key_of[s] >= 0) { // a prefill scan is one-shot: it takes the head as LRU does
                        chance[s]--;
                        const int32_t after = next[s];
                        touch(s);
                        s = after >= 0 ? after : head; // s was the tail: start over at the head
                        continue;
                    }
                    break;
                }
                GGML_ASSERT(s >= 0);
            }
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            } else {
                n_filled++;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            chance[s] = 0;
            touch(s);
            fills.push_back({ e, s });
        }
        if (scan) {
            // a prefill fill into a full cache is one-shot: park it at the head, the next fill evicts it before the
            // working set of the generation
            for (const fill & f : fills) {
                demote(f.slot);
            }
        }
        return true;
    }
};

// kimi-k3-spark patch 0003: the expert store, one record per (layer, expert) holding gate | up | down
// back to back, 4 KiB aligned; format in kimi-k3-spark/tools/repack-experts.py
struct expert_store_entry {
    int32_t layer;
    int32_t types[3];   // ggml types of gate, up, down
    size_t  sizes[3];   // bytes per expert of gate, up, down
    size_t  record_size;
    size_t  base_offset; // record of expert 0
};

static uint32_t store_rd_u32(const uint8_t * p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t store_rd_u64(const uint8_t * p) { uint64_t v; memcpy(&v, p, 8); return v; }

struct expert_store {
    int      fd = -1;
    uint32_t n_expert = 0;
    size_t   max_record = 0;
    std::vector<expert_store_entry> entries;
    std::unordered_map<int32_t, size_t> by_layer;

    ~expert_store() {
#if defined(__linux__)
        if (fd >= 0) {
            close(fd);
        }
#endif
    }

    bool open(const char * path, std::string & err) {
#if defined(__linux__)
        // the records are read with O_DIRECT, the header and index with a plain descriptor
        const int hfd = ::open(path, O_RDONLY | O_CLOEXEC);
        if (hfd < 0) {
            err = std::string("open: ") + strerror(errno);
            return false;
        }
        uint8_t hdr[64];
        if (pread(hfd, hdr, sizeof(hdr), 0) != (ssize_t) sizeof(hdr) || memcmp(hdr, "K3XS", 4) != 0) {
            err = "not an expert store (bad magic)";
            close(hfd);
            return false;
        }
        const uint32_t version      = store_rd_u32(hdr + 4);
        const uint32_t n_records    = store_rd_u32(hdr + 8);
        n_expert                    = store_rd_u32(hdr + 12);
        const uint32_t align        = store_rd_u32(hdr + 16);
        const uint32_t entry_size   = store_rd_u32(hdr + 20);
        const uint64_t index_offset = store_rd_u64(hdr + 24);
        if (version != 1 || entry_size != 64 || align == 0 || n_records == 0 || n_expert == 0) {
            err = "unsupported expert store header";
            close(hfd);
            return false;
        }
        std::vector<uint8_t> index((size_t) n_records*entry_size);
        if (pread(hfd, index.data(), index.size(), index_offset) != (ssize_t) index.size()) {
            err = "short read of the index";
            close(hfd);
            return false;
        }
        close(hfd);
        for (uint32_t i = 0; i < n_records; ++i) {
            const uint8_t * p = index.data() + (size_t) i*entry_size;
            expert_store_entry e;
            e.layer       = (int32_t) store_rd_u32(p + 0);
            e.types[0]    = (int32_t) store_rd_u32(p + 4);
            e.types[1]    = (int32_t) store_rd_u32(p + 8);
            e.types[2]    = (int32_t) store_rd_u32(p + 12);
            e.sizes[0]    = store_rd_u64(p + 16);
            e.sizes[1]    = store_rd_u64(p + 24);
            e.sizes[2]    = store_rd_u64(p + 32);
            e.record_size = store_rd_u64(p + 40);
            e.base_offset = store_rd_u64(p + 48);
            if (e.record_size % align != 0 || e.base_offset % align != 0 || e.sizes[0] + e.sizes[1] + e.sizes[2] > e.record_size) {
                err = "inconsistent index entry for layer " + std::to_string(e.layer);
                return false;
            }
            by_layer[e.layer] = entries.size();
            entries.push_back(e);
            max_record = std::max(max_record, e.record_size);
        }
        fd = ::open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd < 0) {
            err = std::string("open(O_DIRECT): ") + strerror(errno);
            return false;
        }
        return true;
#else
        GGML_UNUSED(path);
        err = "the expert store needs Linux";
        return false;
#endif
    }

    const expert_store_entry * find(int32_t il) const {
        const auto it = by_layer.find(il);
        return it == by_layer.end() ? nullptr : &entries[it->second];
    }
};

// a few threads issuing O_DIRECT preads, enough in flight to keep the NVMe queue full
struct io_pool {
    struct job {
        void * dst;
        size_t len;
        off_t  off;
        void * swap_with = nullptr; // not a read: exchange len bytes of dst and swap_with (off unused)
    };

    // chunk: a record is read as pieces of this many bytes by different threads (0 = whole records). One
    // 6.6 MB O_DIRECT read is latency-bound (~2.9 ms on a GH200 virtio NVMe, ~2.3 GB/s); eight 0.8 MB
    // reads in flight finish in ~1.2 ms, and a decode fill is only one or two records.
    io_pool(int n_threads, size_t chunk) : chunk(chunk) {
        for (int i = 0; i < n_threads; ++i) {
            threads.emplace_back([this] { worker(); });
        }
    }

    ~io_pool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_all();
        for (auto & t : threads) {
            t.join();
        }
    }

    // read all the jobs from fd, blocks until they are done; false if a read failed
    bool run(int fd, const std::vector<job> & in) {
        if (in.empty()) {
            return true;
        }
#if defined(LLAMA_HAVE_LIBURING)
        if (use_uring && in.front().swap_with == nullptr) {
            thread_local uring_reader reader;
            if (reader.ok) {
                return reader.read_all(fd, in, chunk);
            }
        }
#endif
        // one run at a time: the jobs, counters and cur_fd are shared
        struct caller_count {
            std::atomic<int> & n;
            explicit caller_count(std::atomic<int> & n) : n(n) { n.fetch_add(1); }
            ~caller_count() { n.fetch_sub(1); }
        } count(n_callers);
        if (n_callers.load() > 1 && !warned_concurrent.exchange(true)) {
            LLAMA_LOG_WARN("%s: two threads read through the I/O pool at once, serializing them\n", __func__);
        }
        std::lock_guard<std::mutex> run_lock(run_mutex);
        {
            // A worker of the previous run can still be between its last fetch_sub and the fetch_add that ends
            // its work(): resetting next/n_jobs under it hands it a job of this run, which then runs twice, and
            // remaining reaches 0 with one pread still landing in a slot the GPU is about to read (wrong output
            // at temperature 0, seen on the Spark). Wait until every worker has left work() first.
            std::unique_lock<std::mutex> lock(mutex);
            done_cv.wait(lock, [&] { return active == 0; });
            cur_fd = fd;
            jobs.clear();
            for (const job & j : in) {
                // swaps are memory bound, not latency bound: finer pieces put every thread on them
                const size_t piece = j.swap_with != nullptr ? (size_t) 256*1024 : chunk;
                if (piece == 0 || j.len <= piece) {
                    jobs.push_back(j);
                    continue;
                }
                char * dst = (char *) j.dst;
                char * sw  = (char *) j.swap_with;
                size_t len = j.len;
                off_t  off = j.off;
                while (len > 0) {
                    const size_t n = std::min(len, piece);
                    jobs.push_back({ dst, n, off, sw });
                    dst += n;
                    len -= n;
                    off += (off_t) n;
                    if (sw != nullptr) {
                        sw += n;
                    }
                }
            }
            failed.store(false);
            n_jobs.store(jobs.size());
            remaining.store(jobs.size());
            next.store(0);
            gen++;
        }
        cv.notify_all();
        work();
        if (spin_us > 0) {
            // a decode fill is a few reads of ~0.3 ms: the futex wake of the waiting thread costs a good part of
            // that on a VM, so spin for the last pieces before sleeping
            const auto t0 = std::chrono::steady_clock::now();
            while (remaining.load(std::memory_order_acquire) != 0 &&
                   std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count() < spin_us) {
                cpu_relax();
            }
        }
        if (remaining.load(std::memory_order_acquire) != 0) {
            std::unique_lock<std::mutex> lock(mutex);
            done_cv.wait(lock, [&] { return remaining.load() == 0; });
        }
        return !failed.load();
    }

private:
    void worker() {
        store_pin_thread();
        uint64_t seen = 0;
        for (;;) {
            if (spin_us > 0) {
                // LLAMA_EXPERT_STORE_SPIN_US: stay awake this long between runs, so the next fill does not pay the
                // wake-up of a sleeping thread (or of a parked vCPU on a VM); on the GH200 the GPU gap between two
                // layer fills is ~0.85 ms
                const auto t0 = std::chrono::steady_clock::now();
                while (gen.load(std::memory_order_acquire) == seen && !stop &&
                       std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count() < spin_us) {
                    cpu_relax();
                }
            }
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return stop || gen.load() != seen; });
                if (stop) {
                    return;
                }
                seen = gen.load();
                active++;
            }
            work();
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (--active == 0) {
                    done_cv.notify_all();
                }
            }
        }
    }

    void work() {
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= n_jobs.load()) {
                return;
            }
            const job & j = jobs[i];
            if (j.swap_with != nullptr) {
                alignas(64) char tmp[64*1024];
                char * a = (char *) j.dst;
                char * b = (char *) j.swap_with;
                for (size_t o = 0; o < j.len; o += sizeof(tmp)) {
                    const size_t n = std::min(sizeof(tmp), j.len - o);
                    memcpy(tmp, a + o, n);
                    memcpy(a + o, b + o, n);
                    memcpy(b + o, tmp, n);
                }
                if (remaining.fetch_sub(1) == 1) {
                    std::lock_guard<std::mutex> lock(mutex);
                    done_cv.notify_all();
                }
                continue;
            }
            char * dst = (char *) j.dst;
            size_t len = j.len;
            off_t  off = j.off;
            while (len > 0) {
                const ssize_t n = pread(cur_fd, dst, len, off);
                if (n <= 0) {
                    failed.store(true);
                    break;
                }
                dst += n;
                len -= (size_t) n;
                off += n;
            }
            if (remaining.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lock(mutex);
                done_cv.notify_all();
            }
        }
    }

public:
    // reads go through an io_uring of the calling thread instead of the threads (swaps always use the threads)
    bool use_uring = false;
    int  spin_us   = 0; // LLAMA_EXPERT_STORE_SPIN_US: busy-wait this long before sleeping, workers and the caller

    static inline void cpu_relax() {
#if defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
        asm volatile("pause" ::: "memory");
#endif
    }

#if defined(LLAMA_HAVE_LIBURING)
    // The whole fill in flight at once from one submitter. On the Spark's PM9E1 a 7-record decode fill takes 3.9 ms
    // this way against 4.5 ms with 16 threads preading 1 MiB pieces (tools/readbench.cpp); what matters is how much
    // of the fill is queued at once.
    struct uring_reader {
        static constexpr unsigned depth = 64;
        io_uring ring;
        bool     ok = false;

        uring_reader() { ok = io_uring_queue_init(depth, &ring, 0) == 0; }
        ~uring_reader() {
            if (ok) {
                io_uring_queue_exit(&ring);
            }
        }

        // piece: records are submitted as SQEs of this many bytes (0 = whole). On the Spark's NVMe the size does not
        // matter; on the GH200's virtio disk one 6.6 MB request is served serially (1.0 vs 0.58 ms per layer fill),
        // so the records go out as 1 MiB pieces there as with the threads.
        bool read_all(int fd, const std::vector<job> & in, size_t piece) {
            struct pending { char * dst; size_t len; off_t off; };
            std::vector<pending> todo;
            todo.reserve(in.size());
            for (const job & j : in) {
                const size_t step = piece == 0 ? j.len : piece;
                for (size_t o = 0; o < j.len; o += step) {
                    todo.push_back({ (char *) j.dst + o, std::min(step, j.len - o), j.off + (off_t) o });
                }
            }
            size_t next = 0, in_flight = 0;
            std::vector<size_t> retry; // short or interrupted reads, resubmitted with what is left
            bool failed = false;
            while (!failed && (next < todo.size() || !retry.empty() || in_flight > 0)) {
                while (in_flight < depth && (!retry.empty() || next < todo.size())) {
                    io_uring_sqe * sqe = io_uring_get_sqe(&ring);
                    if (sqe == nullptr) {
                        break;
                    }
                    size_t i;
                    if (!retry.empty()) {
                        i = retry.back();
                        retry.pop_back();
                    } else {
                        i = next++;
                    }
                    io_uring_prep_read(sqe, fd, todo[i].dst, (unsigned) todo[i].len, todo[i].off);
                    io_uring_sqe_set_data64(sqe, i);
                    in_flight++;
                }
                if (io_uring_submit_and_wait(&ring, 1) < 0) {
                    failed = true;
                    break;
                }
                io_uring_cqe * cqe;
                unsigned head, seen = 0;
                io_uring_for_each_cqe(&ring, head, cqe) {
                    seen++;
                    const size_t i = (size_t) io_uring_cqe_get_data64(cqe);
                    const int res = cqe->res;
                    if (res == -EAGAIN || res == -EINTR) {
                        retry.push_back(i);
                    } else if (res <= 0) {
                        failed = true;
                    } else {
                        todo[i].dst += res;
                        todo[i].len -= (size_t) res;
                        todo[i].off += res;
                        if (todo[i].len > 0) {
                            retry.push_back(i);
                        }
                    }
                }
                io_uring_cq_advance(&ring, seen);
                in_flight -= seen;
            }
            // a failed fill must not leave reads landing in slots that are reused afterwards
            while (in_flight > 0) {
                io_uring_cqe * cqe;
                if (io_uring_wait_cqe(&ring, &cqe) < 0) {
                    break;
                }
                io_uring_cqe_seen(&ring, cqe);
                in_flight--;
            }
            return !failed;
        }
    };
#endif

private:
    size_t                    chunk = 0;
    std::vector<std::thread>  threads;
    std::mutex                mutex;
    std::mutex                run_mutex;
    std::atomic<int>          n_callers{0};
    std::atomic<bool>         warned_concurrent{false};
    std::condition_variable   cv;
    std::condition_variable   done_cv;
    std::atomic<bool>         stop{false};
    std::atomic<uint64_t>     gen{0};
    int                       active = 0; // workers inside work(), guarded by mutex
    int                       cur_fd = -1;
    std::vector<job>          jobs;
    std::atomic<size_t>       n_jobs{0};
    std::atomic<size_t>       next{0};
    std::atomic<size_t>       remaining{0};
    std::atomic<bool>         failed{false};
};

// kimi-k3-spark patch 0003: remote experts (LLAMA_EXPERT_STORE_REMOTE=host:port[,host:port...]). Each remote box holds
// a static set of records and computes those experts' rows for a single token: request = int32 [17] {k, global ids} +
// the experts' input (n_in f32), reply = k rows of n_out f32, unweighted, in request order (a server may stream them
// one by one). One remote_client per box: one persistent TCP connection (TCP_NODELAY), requests pipelined, replies
// in order, a dedicated thread that sends and busy-polls the socket. The poller fills a slot (only when the slot is
// free and the box is usable) and later consumes it or abandons it (its reply is still drained and dropped).
// Reliability: a broken link (EOF, error, a send that stalls, or a request older than stale_ms) frees every slot in
// flight and reconnects with exponential backoff (non-blocking connect with a timeout, so a box that vanished never
// blocks); a box that misses its deadline trip_after times in a row is skipped for trip_ms (circuit breaker). The
// cache reads anything that did not arrive from its own store, which gives the same rows, so a dead, slow or
// rebooting box only costs speed.
struct remote_client {
    static constexpr int NSLOT = 8;
    static constexpr int KMAX  = 16;
    enum { FREE = 0, QUEUED = 1, SENT = 2, DONE = 3, ABANDONED = 4 };
    struct slot {
        std::atomic<int> state{FREE};
        std::atomic<int> rows_done{0}; // rows of the reply complete so far
        bool             sent = false; // net thread only
        int32_t          hdr[1 + KMAX] = {};
        std::vector<float> x;
        std::vector<float> rows;
        std::chrono::steady_clock::time_point t_post, t_done;
    };
    std::string host;
    int         port = 0;
    int         id = 0;
    int         n_in = 0, n_out = 0;
    int64_t     deadline_us = 3000;      // per box (LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US, one value or one per box)
    int         stale_ms = 1000;         // LLAMA_EXPERT_STORE_REMOTE_STALE_MS: a request unanswered this long drops the link
    int         trip_after = 8;          // LLAMA_EXPERT_STORE_REMOTE_TRIP: consecutive deadline misses that trip the breaker
    int         trip_ms = 2000;          // LLAMA_EXPERT_STORE_REMOTE_TRIP_MS: how long a tripped box is skipped
    slot        slots[NSLOT];
    uint64_t    seq_post = 0;              // poller only
    std::atomic<uint64_t> seq_queued{0};   // published by the poller after a slot is QUEUED
    std::atomic<bool> connected{false};
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> n_connects{0}, n_disconnects{0}, n_stale{0};
    // poller-side state and stats
    int      misses_in_row = 0;
    std::chrono::steady_clock::time_point tripped_until{};
    uint64_t n_trips = 0, n_posts = 0, n_rows = 0, n_hits = 0, n_late = 0, n_late_rows = 0, n_partial = 0, n_nan_rows = 0,
             n_unsent = 0, n_shadow_ontime = 0;
    std::vector<uint32_t> rtt_us = std::vector<uint32_t>(16384, 0); // post -> complete reply, of the replies in time (ring)
    size_t   rtt_next = 0;
    std::thread th;

    bool start(const std::string & hp, int box, int nin, int nout) {
        const size_t c = hp.rfind(':');
        if (c == std::string::npos || c == 0) {
            return false;
        }
        host = hp.substr(0, c);
        port = atoi(hp.c_str() + c + 1);
        sockaddr_in a = {};
        if (port <= 0 || inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) {
            return false;
        }
        id = box;
        n_in = nin;
        n_out = nout;
        for (slot & s : slots) {
            s.x.assign((size_t) n_in, 0.0f);
            s.rows.assign((size_t) KMAX*n_out, 0.0f);
        }
        th = std::thread([this] { main(); });
        return true;
    }
    void shutdown() {
        stop = true;
        if (th.joinable()) {
            th.join();
        }
    }
    ~remote_client() { shutdown(); }

    // poller: the slot the next post would use, if the box is usable now (link up, breaker closed, slot free)
    slot * next_free() {
        if (!connected.load(std::memory_order_acquire)) {
            return nullptr;
        }
        if (misses_in_row >= trip_after) {
            const auto now = std::chrono::steady_clock::now();
            if (tripped_until == std::chrono::steady_clock::time_point{}) {
                tripped_until = now + std::chrono::milliseconds(trip_ms);
                n_trips++;
                LLAMA_LOG_WARN("expert store: remote box %d (%s:%d): %d deadline misses in a row, skipped for %d ms\n", id, host.c_str(), port, misses_in_row, trip_ms);
            }
            if (now < tripped_until) {
                return nullptr;
            }
            // half open: one more try; a hit closes the breaker, a miss re-trips it
            tripped_until = {};
            misses_in_row = trip_after - 1;
        }
        slot & s = slots[seq_post % NSLOT];
        return s.state.load(std::memory_order_acquire) == FREE ? &s : nullptr;
    }
    // poller: hand a filled slot (hdr, x) to the net thread
    void post(slot * s) {
        s->t_post = std::chrono::steady_clock::now();
        s->rows_done.store(0, std::memory_order_relaxed);
        s->state.store(QUEUED, std::memory_order_release);
        seq_post++;
        seq_queued.store(seq_post, std::memory_order_release);
    }
    // poller: give up on a slot (deadline); true if it was done after all
    bool abandon(slot * s) {
        int st = SENT;
        if (s->state.compare_exchange_strong(st, ABANDONED)) {
            return false;
        }
        st = QUEUED;
        if (s->state.compare_exchange_strong(st, ABANDONED)) {
            return false;
        }
        return st == DONE; // DONE (consume it) or FREE (the link broke)
    }

    // non-blocking connect with a timeout: a box that is down or rebooting must not block this thread (or shutdown)
    int try_connect(int timeout_ms) {
        int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
        if (fd < 0) {
            return -1;
        }
        sockaddr_in a = {};
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t) port);
        inet_pton(AF_INET, host.c_str(), &a.sin_addr);
        if (connect(fd, (sockaddr *) &a, sizeof a) != 0) {
            if (errno != EINPROGRESS) {
                close(fd);
                return -1;
            }
            pollfd p = { fd, POLLOUT, 0 };
            int err = 0;
            socklen_t len = sizeof err;
            if (poll(&p, 1, timeout_ms) != 1 || getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) != 0 || err != 0) {
                close(fd);
                return -1;
            }
        }
        // back to blocking for the sends, bounded by a send timeout; the receives use MSG_DONTWAIT
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK);
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        timeval tv = { 0, 200000 };
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        return fd;
    }

    void main() {
        store_pin_thread();
        int fd = -1;
        uint64_t next_send = 0, next_recv = 0;
        size_t got = 0;      // bytes of the head reply received
        int idle = 0;
        int backoff_ms = 50;
        bool warned = false;
        auto drop_link = [&](const char * why) {
            if (fd >= 0) {
                close(fd);
                fd = -1;
            }
            connected.store(false, std::memory_order_release);
            n_disconnects++;
            LLAMA_LOG_WARN("expert store: remote box %d (%s:%d): %s, reconnecting\n", id, host.c_str(), port, why);
            // everything in flight is lost: free the slots (a poller waiting on one sees FREE and reads locally)
            for (slot & s : slots) {
                s.sent = false;
                s.state.store(FREE, std::memory_order_release);
            }
            next_send = next_recv = seq_queued.load(std::memory_order_acquire);
            got = 0;
            backoff_ms = 50;
        };
        while (!stop.load(std::memory_order_relaxed)) {
            if (fd < 0) {
                fd = try_connect(500);
                if (fd < 0) {
                    if (!warned) {
                        LLAMA_LOG_WARN("expert store: remote box %d (%s:%d): not reachable, retrying with backoff (its experts are read locally meanwhile)\n", id, host.c_str(), port);
                        warned = true;
                    }
                    for (int i = 0; i < backoff_ms/10 && !stop.load(); ++i) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    backoff_ms = std::min(backoff_ms*2, 2000);
                    continue;
                }
                warned = false;
                next_send = next_recv = seq_queued.load(std::memory_order_acquire);
                got = 0;
                n_connects++;
                LLAMA_LOG_INFO("expert store: remote box %d: connected to %s:%d\n", id, host.c_str(), port);
                connected.store(true, std::memory_order_release);
            }
            bool work = false;
            // send everything queued, in order
            const uint64_t q = seq_queued.load(std::memory_order_acquire);
            bool broken = false;
            while (next_send < q) {
                slot & s = slots[next_send % NSLOT];
                if (s.state.load(std::memory_order_acquire) == QUEUED) {
                    if (!send_all(fd, s.hdr, sizeof s.hdr) || !send_all(fd, s.x.data(), (size_t) n_in*sizeof(float))) {
                        broken = true;
                        break;
                    }
                    s.sent = true;
                    int st = QUEUED;
                    s.state.compare_exchange_strong(st, SENT); // or it was abandoned meanwhile: still drain its reply
                } // else abandoned before it was sent: nothing to receive
                next_send++;
                work = true;
            }
            if (broken) {
                drop_link("send failed");
                continue;
            }
            // receive the head replies
            while (next_recv < next_send) {
                slot & s = slots[next_recv % NSLOT];
                if (!s.sent) {
                    int st = ABANDONED;
                    s.state.compare_exchange_strong(st, FREE);
                    next_recv++;
                    continue;
                }
                const size_t want = (size_t) s.hdr[0]*n_out*sizeof(float);
                const ssize_t r = recv(fd, (char *) s.rows.data() + got, want - got, MSG_DONTWAIT);
                if (r == 0 || (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) {
                    broken = true;
                    break;
                }
                if (r < 0) {
                    // nothing yet: a server that stopped answering (hung, paused) is dropped after stale_ms
                    if (std::chrono::steady_clock::now() - s.t_post > std::chrono::milliseconds(stale_ms)) {
                        n_stale++;
                        broken = true;
                    }
                    break;
                }
                work = true;
                got += (size_t) r;
                s.rows_done.store((int) (got/((size_t) n_out*sizeof(float))), std::memory_order_release);
                if (got < want) {
                    continue;
                }
                got = 0;
                s.sent = false;
                s.t_done = std::chrono::steady_clock::now();
                int st = SENT;
                if (!s.state.compare_exchange_strong(st, DONE)) {
                    s.state.store(FREE, std::memory_order_release); // abandoned: drop it
                }
                next_recv++;
            }
            if (broken) {
                drop_link("connection lost or server not answering");
                continue;
            }
            if (work || next_recv < next_send) {
                idle = 0;
            } else if (++idle > 200000) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        if (fd >= 0) {
            close(fd);
        }
        connected.store(false);
    }
    static bool send_all(int fd, const void * p, size_t n) {
        const char * c = (const char *) p;
        while (n) {
            const ssize_t r = send(fd, c, n, MSG_NOSIGNAL);
            if (r <= 0) {
                if (r < 0 && errno == EINTR) {
                    continue;
                }
                return false; // includes the send timeout (EAGAIN): a server that stopped reading
            }
            c += r;
            n -= (size_t) r;
        }
        return true;
    }
};

// kimi-k3-spark patch 0003: the speculative reader of the predictive prefetch. The predicted misses of the next layer
// are read here, on a thread of its own with a bounded number of pieces in flight, during the part of a layer in
// which the drive would otherwise sit idle (the expert matmuls, the next layer's attention and router). The real
// fills preempt it: no new record is started until the next prediction is posted, the records already started are
// finished (a half-read record is never left resident) and the ones not started are handed back to the cache, which
// empties their slots. The reader never touches the LRU; per-slot states tell the cache what is in flight.
// The pieces go out through one io_uring ring (Spark: the NVMe takes the whole depth from one submitter) or through
// `depth` pread threads (GH200: the virtio disk gives one blk-mq queue per submitting CPU, so one ring is slow).
struct spec_reader {
    struct rec {
        int32_t ig;
        int32_t slot;
        int32_t key;   // il*n_expert + expert
        char *  dst;
        size_t  len;
        off_t   off;
        std::atomic<int8_t> * state; // 0 none or done, 1 queued, 2 in flight
    };

    // the ring is created by the reader thread itself (like the pool's thread-local rings), not by the caller
    spec_reader(int fd, size_t piece, unsigned depth, bool uring) : fd(fd), piece(piece ? piece : (size_t) 1 << 20), depth(std::max(1u, depth)), uring(uring) {
        thread = std::thread([this] { main(); });
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return started; });
        if (!ok) {
            lock.unlock();
            thread.join();
        }
    }

    ~spec_reader() {
        if (ok) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stop = true;
            }
            cv.notify_all();
            thread.join();
        }
    }

    // queue records to read, in order of priority, and let the reader start them
    void enqueue(const std::vector<rec> & recs) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (const rec & r : recs) {
                r.state->store(1, std::memory_order_release);
                queue.push_back(r);
            }
            preempt = false;
        }
        cv.notify_one();
    }

    // stop starting records (until the next enqueue) and take back the ones not started, their state reset to 0
    void preempt_take(std::vector<rec> & out) {
        std::lock_guard<std::mutex> lock(mutex);
        preempt = true;
        out.assign(queue.begin(), queue.end());
        queue.clear();
        for (const rec & r : out) {
            r.state->store(0, std::memory_order_release);
        }
    }

    bool ok = false;
    std::atomic<uint64_t> n_done{0};

private:
    struct piece_t {
        int    act;
        char * dst;
        size_t len;
        off_t  off;
        int    n_zero = 0;
    };
    struct active_t {
        rec      r;
        size_t   next_off = 0; // bytes submitted so far
        unsigned n_out    = 0; // pieces in flight
        bool     used     = false;
    };
    struct done_t {
        int p;
        int res;
    };

    // thread backend: a worker reads one piece whole (short reads continued here), so a completion is the full
    // length, what was there before the end of the file, or an error
    void worker() {
        store_pin_thread();
        while (true) {
            int p;
            {
                std::unique_lock<std::mutex> lock(wmutex);
                wcv.wait(lock, [&] { return wstop || !work.empty(); });
                if (work.empty()) {
                    return;
                }
                p = work.front();
                work.pop_front();
            }
            const piece_t & pc = pieces[p];
            size_t got = 0;
            int    res = 0;
            while (got < pc.len) {
                const ssize_t n = pread(fd, pc.dst + got, pc.len - got, pc.off + (off_t) got);
                if (n < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    res = -errno;
                    break;
                }
                if (n == 0) {
                    break;
                }
                got += (size_t) n;
            }
            if (res == 0) {
                res = (int) got;
            }
            {
                std::lock_guard<std::mutex> lock(wmutex);
                done.push_back({ p, res });
            }
            dcv.notify_one();
        }
    }

    void main() {
        store_pin_thread();
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (uring) {
#if defined(LLAMA_HAVE_LIBURING)
                ok = io_uring_queue_init(std::max(8u, depth), &ring, 0) == 0;
#else
                ok = false;
#endif
            } else {
                ok = true;
            }
            started = true;
        }
        cv.notify_all();
        if (!ok) {
            return;
        }
        if (!uring) {
            for (unsigned i = 0; i < depth; ++i) {
                workers.emplace_back([this] { worker(); });
            }
        }
        run();
        if (uring) {
#if defined(LLAMA_HAVE_LIBURING)
            io_uring_queue_exit(&ring);
#endif
        } else {
            {
                std::lock_guard<std::mutex> lock(wmutex);
                wstop = true;
            }
            wcv.notify_all();
            for (std::thread & w : workers) {
                w.join();
            }
        }
    }

    void run() {
        std::vector<active_t> act(depth);
        pieces.assign(depth, piece_t{});
        std::vector<int>      free_pieces;
        std::vector<int>      retry; // short or interrupted reads, resubmitted with what is left
        std::vector<done_t>   comp;
        for (int i = (int) depth - 1; i >= 0; --i) {
            free_pieces.push_back(i);
        }
        unsigned in_flight = 0;
        int cur = -1; // the record being submitted
        bool stopping = false;
        while (true) {
            // submit: the retries of started records first, then pieces of the current record, then a new record
            // unless preempted. The piece is chosen before an SQE is taken: an SQE taken and not prepared would go
            // out as a no-op whose completion (res 0, user data 0) corrupts the bookkeeping of piece 0.
            while (in_flight < depth) {
                int p;
                if (!retry.empty()) {
                    p = retry.back();
                    retry.pop_back();
                } else {
                    if (cur < 0 || act[cur].next_off >= act[cur].r.len) {
                        cur = -1;
                        {
                            std::unique_lock<std::mutex> lock(mutex);
                            if (stop) {
                                stopping = true;
                            } else if (!queue.empty() && !preempt) {
                                for (size_t i = 0; i < act.size(); ++i) {
                                    if (!act[i].used) {
                                        cur = (int) i;
                                        break;
                                    }
                                }
                                if (cur >= 0) {
                                    act[cur] = { queue.front(), 0, 0, true };
                                    queue.pop_front();
                                    act[cur].r.state->store(2, std::memory_order_release);
                                }
                            }
                        }
                        if (cur < 0) {
                            break;
                        }
                    }
                    active_t & a = act[cur];
                    p = free_pieces.back();
                    free_pieces.pop_back();
                    const size_t len = std::min(piece, a.r.len - a.next_off);
                    pieces[p] = { cur, a.r.dst + a.next_off, len, a.r.off + (off_t) a.next_off, 0 };
                    a.next_off += len;
                    a.n_out++;
                }
                if (uring) {
#if defined(LLAMA_HAVE_LIBURING)
                    io_uring_sqe * sqe = io_uring_get_sqe(&ring);
                    if (sqe == nullptr) {
                        // cannot happen: the ring has at least depth entries and everything is submitted below
                        io_uring_submit(&ring);
                        sqe = io_uring_get_sqe(&ring);
                        GGML_ASSERT(sqe != nullptr);
                    }
                    io_uring_prep_read(sqe, fd, pieces[p].dst, (unsigned) pieces[p].len, pieces[p].off);
                    io_uring_sqe_set_data64(sqe, (uint64_t) p);
#endif
                } else {
                    {
                        std::lock_guard<std::mutex> lock(wmutex);
                        work.push_back(p);
                    }
                    wcv.notify_one();
                }
                in_flight++;
            }
            if (in_flight == 0) {
                if (stopping) {
                    return;
                }
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return stop || (!queue.empty() && !preempt); });
                if (stop) {
                    return;
                }
                continue;
            }
            comp.clear();
            if (uring) {
#if defined(LLAMA_HAVE_LIBURING)
                if (io_uring_submit_and_wait(&ring, 1) < 0) {
                    GGML_ABORT("expert store: speculative read submit failed: %s", strerror(errno));
                }
                io_uring_cqe * cqe;
                unsigned head, seen = 0;
                io_uring_for_each_cqe(&ring, head, cqe) {
                    seen++;
                    comp.push_back({ (int) io_uring_cqe_get_data64(cqe), cqe->res });
                }
                io_uring_cq_advance(&ring, seen);
#endif
            } else {
                std::unique_lock<std::mutex> lock(wmutex);
                dcv.wait(lock, [&] { return !done.empty(); });
                comp.swap(done);
            }
            for (const done_t & d : comp) {
                const int p = d.p;
                piece_t  & pc = pieces[p];
                active_t & a  = act[pc.act];
                const int res = d.res;
                if (res == -EAGAIN || res == -EINTR || res == 0) {
                    // res == 0: seen once per process on the Spark (kernel 6.17-nvidia), for the first piece of the
                    // first speculative read, with the same read served in full by pread right after; nothing was
                    // written, so the piece is simply submitted again (bounded)
                    if (res == 0 && ++pc.n_zero > 16) {
                        GGML_ABORT("expert store: speculative read of %zu bytes at offset %lld returned 0 bytes 16 times (fd %d, file size %lld)",
                            pc.len, (long long) pc.off, fd, (long long) lseek(fd, 0, SEEK_END));
                    }
                    if (res == 0 && !warned_zero) {
                        warned_zero = true;
                        LLAMA_LOG_WARN("%s: expert store: a speculative read returned 0 bytes (offset %lld, %zu bytes), retrying it\n", __func__, (long long) pc.off, pc.len);
                    }
                    retry.push_back(p);
                } else if (res < 0) {
                    GGML_ABORT("expert store: speculative read failed: %zu bytes at offset %lld: %s", pc.len, (long long) pc.off, strerror(-res));
                } else if ((size_t) res < pc.len) {
                    pc.dst += res;
                    pc.len -= (size_t) res;
                    pc.off += res;
                    retry.push_back(p);
                } else {
                    a.n_out--;
                    free_pieces.push_back(p);
                    if (a.n_out == 0 && a.next_off >= a.r.len) {
                        // every byte of the record landed: resident
                        a.used = false;
                        a.r.state->store(0, std::memory_order_release);
                        n_done.fetch_add(1, std::memory_order_relaxed);
                        if (cur == pc.act) {
                            cur = -1;
                        }
                    }
                }
            }
            in_flight -= (unsigned) comp.size();
        }
    }

    int      fd;
    size_t   piece;
    unsigned depth;
    bool     uring;
    bool     warned_zero = false;
#if defined(LLAMA_HAVE_LIBURING)
    io_uring ring;
#endif
    std::thread thread;
    std::mutex  mutex;
    std::condition_variable cv;
    std::deque<rec> queue;
    bool preempt = false;
    bool stop    = false;
    bool started = false;
    // thread backend
    std::vector<piece_t>     pieces;
    std::vector<std::thread> workers;
    std::mutex               wmutex;
    std::condition_variable  wcv, dcv;
    std::deque<int>          work;
    std::vector<done_t>      done;
    bool                     wstop = false;
};

#if defined(__linux__)
// the NUMA nodes that have CPUs, as a bitmask (0 if unknown). A GH200 exposes the GPU's HBM as a CPU-less NUMA node
// that the kernel hands out to anyone when the host node runs low (before reclaiming its page cache): a cache
// region that spills into it takes the memory the CUDA allocations need. Bind the regions to the CPU nodes.
static unsigned long store_cpu_node_mask() {
    static unsigned long mask = [] {
        unsigned long m = 0;
        FILE * f = fopen("/sys/devices/system/node/has_cpu", "r");
        if (f == nullptr) {
            return m;
        }
        char buf[256] = {0};
        if (fgets(buf, sizeof(buf), f) != nullptr) {
            // "0" or "0-1" or "0,2-3"
            for (char * tok = strtok(buf, ",\n"); tok != nullptr; tok = strtok(nullptr, ",\n")) {
                int lo = -1, hi = -1;
                if (sscanf(tok, "%d-%d", &lo, &hi) < 2) {
                    hi = lo;
                }
                for (int n = lo; n >= 0 && n <= hi && n < (int) (8*sizeof(m)); ++n) {
                    m |= 1UL << n;
                }
            }
        }
        fclose(f);
        return m;
    }();
    return mask;
}

// the NUMA nodes with memory but no CPUs: on a GH200 the GPU's HBM
static unsigned long store_gpu_node_mask() {
    static unsigned long mask = [] {
        unsigned long m = 0;
        FILE * f = fopen("/sys/devices/system/node/has_memory", "r");
        if (f == nullptr) {
            return m;
        }
        char buf[256] = {0};
        if (fgets(buf, sizeof(buf), f) != nullptr) {
            for (char * tok = strtok(buf, ",\n"); tok != nullptr; tok = strtok(nullptr, ",\n")) {
                int lo = -1, hi = -1;
                if (sscanf(tok, "%d-%d", &lo, &hi) < 2) {
                    hi = lo;
                }
                for (int n = lo; n >= 0 && n <= hi && n < (int) (8*sizeof(m)); ++n) {
                    m |= 1UL << n;
                }
            }
        }
        fclose(f);
        return m & ~store_cpu_node_mask();
    }();
    return mask;
}
#endif

// page-aligned, populated host memory for the banks and the staging (O_DIRECT reads land here).
// hbm_bytes > 0: the first hbm_bytes are bound to the GPU's HBM node (GH200), where the GPU reads them at HBM
// speed through the same registered mapping; the allocation fails if that node cannot hold them.
static void * store_region_alloc(size_t size, std::string & err, size_t hbm_bytes = 0) {
#if defined(__linux__)
    void * p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        err = strerror(errno);
        return nullptr;
    }
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    hbm_bytes = std::min(size, (hbm_bytes + page - 1)/page*page);
    if (hbm_bytes > 0) {
        const unsigned long gmask = store_gpu_node_mask();
        if (gmask == 0) {
            munmap(p, size);
            err = "no CPU-less memory node for the HBM tier";
            return nullptr;
        }
        madvise(p, hbm_bytes, MADV_HUGEPAGE);
        if (syscall(SYS_mbind, p, hbm_bytes, MPOL_BIND, &gmask, 8*sizeof(gmask) + 1, 0) != 0 ||
            madvise(p, hbm_bytes, MADV_POPULATE_WRITE) != 0) {
            err = std::string("binding ") + std::to_string(hbm_bytes/1024/1024) + " MiB to the HBM node failed: " + strerror(errno);
            munmap(p, size);
            return nullptr;
        }
    }
    {
        const char * env = getenv("LLAMA_EXPERT_STORE_NUMA_BIND");
        const unsigned long mask = store_cpu_node_mask();
        if (mask != 0 && size > hbm_bytes && !(env != nullptr && env[0] == '0')) {
            if (syscall(SYS_mbind, (char *) p + hbm_bytes, size - hbm_bytes, MPOL_BIND, &mask, 8*sizeof(mask) + 1, 0) != 0) {
                LLAMA_LOG_WARN("%s: mbind of %zu MiB to the CPU nodes (mask 0x%lx) failed: %s\n", __func__, size/1024/1024, mask, strerror(errno));
            }
        }
    }
    // Transparent huge pages only on the HBM part (above), where they speed up the GPU's first reads of an expert
    // (2.6 vs 2.1 TB/s). On the host part they are opt-in (LLAMA_EXPERT_STORE_THP=1): on the Spark, with them, about
    // half of the server starts produced NaN from the first prefill ("GGGG..." at temperature 0, 4 of 7 starts on the
    // thread-pool read path), never without them (0 of 4). Cause not pinned down; the GPU-registered staging that
    // the prefill uploads from is the suspect.
    if (getenv("LLAMA_EXPERT_STORE_THP") != nullptr && atoi(getenv("LLAMA_EXPERT_STORE_THP")) != 0) {
        madvise((char *) p + hbm_bytes, size - hbm_bytes, MADV_HUGEPAGE);
    }
    if (madvise(p, size, MADV_POPULATE_WRITE) != 0) {
        for (size_t off = 0; off < size; off += 4096) {
            ((volatile char *) p)[off] = 0;
        }
    }
    return p;
#else
    GGML_UNUSED(size);
    GGML_UNUSED(hbm_bytes);
    err = "needs Linux";
    return nullptr;
#endif
}

// Lock the process's memory except the model file mappings: the cache bank, the heap, the CUDA runtime's and
// driver's anonymous regions, and the mapped code of the libraries. The model load (the spine streamed through the
// page cache, the bank populated, the GPU buffers allocated) leaves the box with a few GB of headroom, and on the way
// there the kernel swaps out cold anonymous pages and drops the library code it has not seen for a while (measured on
// the Spark after a load: 152 MB of the server in swap, an 86 MB anonymous region almost whole, 44 MB of the heap and
// the code of libcublasLt; with the 8 GiB prompt cache filling up, 5.4 GB swapped out during eight requests). Every
// first touch afterwards is a major fault served behind the expert reads that keep the NVMe queue full, ~10 ms each,
// and a few hundred of them in a row stall a request for seconds. The GGUF mappings are skipped because mlock
// populates: locking a 500 GB model mapping would read it whole. Device mappings (/dev/nvidia*) are skipped too, the
// driver pins those itself. Returns false with err on the first failure (limit or permission).
static bool lock_process_memory(size_t & locked_bytes, size_t & n_locked, size_t & skipped_bytes, std::string & err) {
#if defined(__linux__)
    locked_bytes = n_locked = skipped_bytes = 0;
    struct rlimit rl = {};
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0 && rl.rlim_cur != rl.rlim_max) {
        rl.rlim_cur = rl.rlim_max;
        setrlimit(RLIMIT_MEMLOCK, &rl);
    }
    FILE * f = fopen("/proc/self/maps", "r");
    if (f == nullptr) {
        err = std::string("/proc/self/maps: ") + strerror(errno);
        return false;
    }
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges; // file-backed and small anonymous: locked and populated
    std::vector<std::pair<uintptr_t, uintptr_t>> onfault; // anonymous >= 1 GiB (the bank, the staging): locked as resident
    char line[1024];
    while (fgets(line, sizeof(line), f) != nullptr) {
        unsigned long lo = 0, hi = 0;
        char perms[8] = {};
        int consumed = 0;
        if (sscanf(line, "%lx-%lx %7s %*s %*s %*s %n", &lo, &hi, perms, &consumed) < 3) {
            continue;
        }
        const char * path = line + consumed;
        while (*path == ' ') {
            path++;
        }
        std::string p(path);
        while (!p.empty() && (p.back() == '\n' || p.back() == ' ')) {
            p.pop_back();
        }
        const bool skip = perms[0] != 'r' || p == "[vvar]" || p == "[vdso]" || p == "[vsyscall]" || p == "[vvar_vclock]" ||
            p.rfind("/dev/", 0) == 0 || (p.size() > 5 && p.compare(p.size() - 5, 5, ".gguf") == 0) ||
            p.find(".gguf (deleted)") != std::string::npos || p.find(".experts") != std::string::npos;
        if (skip) {
            skipped_bytes += hi - lo;
            continue;
        }
        if (p.empty() && hi - lo >= (1ull << 30)) {
            onfault.push_back({ lo, hi });
        } else {
            ranges.push_back({ lo, hi });
        }
    }
    fclose(f);
    // mlock populates what is not resident: wanted for the libraries (their code is what gets dropped) and the small
    // anonymous regions (swapped-out pages come back now, not at the first touch); not for the big ones, which are
    // populated already, and where a sparse readable reservation would otherwise be filled with zero pages
    for (const auto & r : ranges) {
        if (mlock((void *) r.first, r.second - r.first) == 0) {
            locked_bytes += r.second - r.first;
            n_locked++;
        } else if (errno == ENOMEM || errno == EPERM) {
            err = strerror(errno);
            return false;
        }
        // EINVAL/EAGAIN: a mapping that cannot be locked (special or partly unmapped meanwhile), not an error
    }
    for (const auto & r : onfault) {
#ifdef MLOCK_ONFAULT
        const int rc = mlock2((void *) r.first, r.second - r.first, MLOCK_ONFAULT);
#else
        const int rc = mlock((void *) r.first, r.second - r.first);
#endif
        if (rc == 0) {
            locked_bytes += r.second - r.first;
            n_locked++;
        } else if (errno == ENOMEM || errno == EPERM) {
            err = strerror(errno);
            return false;
        }
    }
    return true;
#else
    GGML_UNUSED(locked_bytes);
    GGML_UNUSED(n_locked);
    GGML_UNUSED(skipped_bytes);
    err = "needs Linux";
    return false;
#endif
}

static void store_region_free(void * p, size_t size) {
#if defined(__linux__)
    if (p != nullptr) {
        munmap(p, size);
    }
#else
    GGML_UNUSED(p);
    GGML_UNUSED(size);
#endif
}

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;

        // expert store: the banks are strided views into one host region of records
        void * region = nullptr;
        size_t region_size = 0;
        ggml_backend_buffer_ptr region_buf;
        size_t record_size = 0;
        size_t slot_stride = 0;       // bytes between two slots of the region: the record, padded so every part's expert stride is whole blocks
        std::vector<size_t> part_off; // offset of each expert tensor's bytes inside a record
        size_t hbm_budget = 0;        // share of LLAMA_EXPERT_STORE_HBM_MIB; the first lru.n_hbm slots are on the HBM node
        int32_t hbm_min_slot = -1;    // the least used expert in the HBM slots, refreshed every few fills
        uint16_t hbm_min_uses = 0;
        std::unique_ptr<std::atomic<int8_t>[]> spec_state; // [n_slots] what the speculative reader holds, see spec_reader
        ggml_tensor * dummy_const = nullptr; // I32 [1, 1] = n_slots, the zero record's slot, appended to the early ids by the graph
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        // split expert GEMV (see split below): by expert, the slot of each selected expert complete at the early
        // lookup (the rest the zero record); by position j < split_late_max, the slot of the j-th miss (the rest the
        // zero record) and the row it takes in the merge (its position among the selected experts; n_expert_used,
        // the scratch row, for the zero record)
        ggml_tensor * early_map = nullptr;  // I32 [1, n_expert], by expert
        ggml_tensor * late_map  = nullptr;  // I32 [1, n_expert], by position
        ggml_tensor * merge_map = nullptr;  // I32 [1, n_expert], by position
        // remote experts (remote below), only for layers with experts in the remote set
        ggml_tensor * rpost_map   = nullptr; // I32 [1, n_expert]: the map of the "moe_rpost-<il>" post (never read)
        ggml_tensor * remote_x    = nullptr; // F32 [n_in]: the experts' input, copied here by the GPU before the post
        ggml_tensor * remote_rows = nullptr; // F32 [n_out, remote_rows_max]
        ggml_tensor * rmerge_map  = nullptr; // I32 [1, n_expert], by position: merge row of each remote row
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
        ggml_tensor * cached_split = nullptr; // the same view with the zero record (slot n_slots) included, for the split GEMV
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    // largest batch served from the cache; larger ones use the original weights and the scheduler's copy path.
    // LLAMA_EXPERT_STORE_MAX_BATCH raises it where the cache holds most of the experts (prefill from the bank),
    // which needs the GPU to read the bank in place: with the bank on the CPU the scheduler would offload a
    // large-batch matmul by copying the strided bank views, so the knob is clamped to 32 there
    int64_t max_batch     = 32;
    int64_t max_batch_env = 32;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;

    // expert store state
    bool use_store = false;
    expert_store store;
    std::unique_ptr<io_pool> pool;
    std::vector<io_pool::job> jobs;
    void * staging = nullptr;        // n_expert records, large batches are read here and uploaded from here
    size_t staging_size = 0;
    bool   staging_pinned = false;
    ggml_backend_buffer_ptr staging_buf; // the registration of the staging with the GPU, when it has one
    bool   bank_on_gpu = false;      // the cache regions are registered with the GPU and the expert matmuls run there
    ggml_backend_reg_t gpu_reg = nullptr;
    ggml_backend_t gpu_backend = nullptr;
    // the slot maps are registered with the GPU as well: the lookup runs there (no CPU split per layer) and the
    // CUDA backend calls back into the cache with the selected experts right before it, to fill the misses
    bool   slots_on_gpu = false;
    void * slot_map_mem = nullptr;
    size_t slot_map_mem_size = 0;
    // (user data, the lookup's slot map, the kind of lookup, ids, n_ids): see ggml_backend_cuda_host_bank_hook_t
    typedef void (*cuda_host_bank_hook_t)(void *, const ggml_tensor *, int, const int32_t *, int64_t);
    bool    prefill_promote = false; // LLAMA_EXPERT_STORE_PREFILL_PROMOTE=1: prefill fills behave like decode fills in the LRU
    bool    prefetch = false;        // LLAMA_EXPERT_STORE_PREFETCH=1 (needs slots_on_gpu): predicted next-layer experts read ahead
    int32_t last_fill_il = -1;       // the layer of the last real fill: its experts may be in use by the GPU, never evict them
    // LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=1 (CPU split only): after the real fill of layer il of a large batch, the
    // misses of every expert of layer il+1 are read on the fill thread while the GPU computes layer il. A batch of
    // more than ~8 tokens per expert touches nearly all of a layer's experts, so "all of them" is the prediction.
    bool     prefill_lookahead = false;
    int32_t  la_il = -1;             // layer whose lookahead reads are in flight on the fill thread (-1: none)
    size_t   la_min_ids = 0;         // smallest batch (in ids) that posts a lookahead: 8 ids per expert by default
    std::vector<int32_t> la_all;     // 0..n_expert-1
    uint64_t n_la_posts = 0, n_la_records = 0, t_la_wait_us = 0, n_la_waits = 0;
    uint64_t n_pf_fills = 0, n_pf_records = 0, t_pf_io_us = 0, n_pf_hits = 0, n_pf_lookups = 0;
    std::chrono::steady_clock::time_point t_pf_first, t_pf_last;
    std::vector<int32_t> last_fill_ids;
    uint64_t n_prefetch_calls = 0, n_prefetch_reads = 0, n_prefetch_used = 0;
    bool     self_prefetch = false;  // LLAMA_EXPERT_STORE_SELF_PREFETCH=1 (needs slots_on_gpu): the layer's real experts posted early
    uint64_t n_self_calls = 0, n_self_reads = 0, t_self_io_us = 0;
    // predictive prefetch v2: the predicted misses go to the speculative reader (spec_reader above) instead of the
    // I/O pool, so they never delay a real fill and take at most spec_depth pieces of the drive at once
    std::unique_ptr<spec_reader> spec;
    std::vector<spec_reader::rec> spec_recs;
    bool     spec_uring  = true;     // LLAMA_EXPERT_STORE_PREFETCH_URING (default: as the pool): one ring, else `depth` pread threads
    unsigned spec_depth  = 8;        // LLAMA_EXPERT_STORE_PREFETCH_DEPTH: pieces (of the pool's chunk) in flight at most
    int      spec_max    = 16;       // LLAMA_EXPERT_STORE_PREFETCH_MAX: predicted misses read per post at most
    bool     spec_demote = false;    // LLAMA_EXPERT_STORE_PREFETCH_DEMOTE=1: predictions the real fill did not use go to the eviction end
    int      prefetch_input = 0;     // LLAMA_EXPERT_STORE_PREFETCH_INPUT: what the next layer's router is applied to (see build_moe_cache_prefetch)
    std::vector<std::vector<int32_t>> spec_ids; // [n_layer] the experts of the last prediction of each layer
    uint64_t n_spec_posts = 0, n_spec_reclaimed = 0, n_spec_adopted = 0, n_spec_waits = 0, t_spec_wait_us = 0;
    uint64_t t_fill_plan_us = 0, t_fill_io_us = 0; // of the last fill_real
    size_t   pool_chunk = 0;

    // split expert GEMV (LLAMA_EXPERT_STORE_SPLIT=1, needs the slot maps on the GPU; single-token decode only): the
    // graph posts the layer's selected experts as an early lookup ("moe_early-<il>", on the early map), which plans
    // the fill, publishes the slots of the experts already complete in the bank (the misses point at the group's
    // zero record, slot n_slots) and starts the reads on the fill thread, so the GPU runs the matmuls of the resident
    // experts while the misses are read. The real lookup ("ffn_moe_slots", on the late map) waits for the reads and
    // publishes the misses' slots (the hits now point at the zero record). The graph merges the two outputs by rows
    // (merge map: the position of each miss, or n_expert_used for a hit, a scratch row), so every expert's row is
    // computed exactly once, from the same slot, by the same kernel: bit-identical with the single lookup.
    bool     split_req = false;      // the env asked for it
    bool     split     = false;      // and the configuration allows it
    std::unordered_map<const ggml_tensor *, int32_t> map_kind; // early/late/merge map -> 1/2/3 (slot maps are absent)
    int32_t  split_il  = -1;         // layer of the early lookup whose reads are in flight (-1: none)
    int      split_late_max = 8;     // LLAMA_EXPERT_STORE_SPLIT_LATE: rows of the late matmul; more misses than this make the early lookup wait
    std::vector<int32_t> split_ids;  // the selected experts of the early lookup, in position order
    ggml_tensor * split_iota = nullptr; // I32 [1, n_expert_used] = 0, 1, ...: the ids of the late and merge lookups (position-indexed maps)
    uint64_t n_split_calls = 0, n_split_fallback = 0, n_split_sync = 0, n_split_sync_records = 0, n_split_early_hits = 0, n_split_late = 0, t_split_plan_us = 0, t_split_wait_us = 0;
    std::vector<uint8_t> split_miss; // [n_expert] scratch of fill_early: 1 if the expert is read by this fill

    // remote experts (LLAMA_EXPERT_STORE_REMOTE=host:port[,host:port...] + LLAMA_EXPERT_STORE_REMOTE_SET=<one file of
    // "layer expert" lines per box, comma separated in the same order>; needs the split GEMV): static sets of experts
    // held by other boxes. At the early lookup, the selected experts of a usable box's set that are not resident are
    // left out of the plan (no slot, no read; early map: the zero record) and become the layer's remote candidates,
    // grouped by box. After the early post the graph copies the experts' input into the layer's remote input and
    // posts "moe_rpost-<il>": the poller sends input + ids to every box involved. The real lookup waits for each box's
    // reply up to its deadline (from its post), places the rows that arrived (partial replies included, NaN rows
    // refused) in the layer's remote rows with their merge positions, and reads every missing expert locally into
    // the late matmul's free rows instead. The rows are the same either way.
    std::vector<std::unique_ptr<remote_client>> remotes;
    std::vector<uint8_t> remote_box;   // [n_layer*n_expert]: 0 = local, else box index + 1
    bool     remote_shadow = false;    // LLAMA_EXPERT_STORE_REMOTE_SHADOW=1: send and time, but read every expert locally
    int      remote_rows_max = 8;      // LLAMA_EXPERT_STORE_REMOTE_ROWS: rows per layer at most (all boxes together)
    int32_t  remote_il = -1;           // layer of the early lookup whose remote candidates are pending
    std::vector<int32_t> remote_ids;   // its candidates (experts), grouped by box, their positions among the selected
    std::vector<int32_t> remote_pos;   // experts, and per box the first candidate, the count and the request sent
    std::vector<int32_t> remote_off, remote_cnt;
    std::vector<remote_client::slot *> remote_slot;
    std::vector<uint8_t> remote_usable;   // scratch of fill_early: box usable now
    std::vector<int32_t> remote_split_box, remote_fill_at; // scratch of fill_early
    std::vector<int32_t> remote_local; // scratch of fill_early: the selected experts that stay local
    std::vector<int32_t> remote_pf;    // scratch of prefetch_decode: the predictions minus the remote ones
    std::vector<int32_t> remote_miss_ids, remote_miss_pos; // scratch of remote_resolve: the rows that did not arrive
    size_t   split_late_used = 0;      // late rows the early lookup filled
    uint64_t n_remote_layers = 0, t_remote_wait_us = 0, n_remote_fallback_layers = 0;
    int      remote_dma_fd = -1;      // /dev/cpu_dma_latency, held while the remote experts are on
    bool remote_on() const {
        return !remotes.empty();
    }
    // the box that would serve expert x of layer il now, or -1 (local: not in a set, or the box is not usable)
    int remote_box_of(int32_t il, int32_t x) {
        if (!remote_on()) {
            return -1;
        }
        const int b = (int) remote_box[(size_t) il*store.n_expert + x] - 1;
        return b >= 0 && remotes[b]->connected.load(std::memory_order_acquire) ? b : -1;
    }
    struct fill_thread {
        std::thread              th;
        std::mutex               m;
        std::condition_variable  cv;
        std::vector<io_pool::job> jobs;
        int                      fd = -1;
        std::atomic<int>         state{0}; // 0 idle, 1 queued, 2 running
        std::atomic<bool>        failed{false};
        bool                     stop = false;
        io_pool *                pool = nullptr;

        void start_thread(io_pool * p) {
            pool = p;
            th = std::thread([this] { main(); });
        }
        void main() {
            store_pin_thread();
            for (;;) {
                {
                    std::unique_lock<std::mutex> lock(m);
                    cv.wait(lock, [&] { return stop || state.load() == 1; });
                    if (stop) {
                        return;
                    }
                    state.store(2);
                }
                const bool ok = pool->run(fd, jobs);
                if (!ok) {
                    failed.store(true);
                }
                state.store(0, std::memory_order_release);
            }
        }
        void submit(int fd_, const std::vector<io_pool::job> & in) {
            GGML_ASSERT(state.load() == 0);
            {
                std::lock_guard<std::mutex> lock(m);
                fd   = fd_;
                jobs = in;
                state.store(1);
            }
            cv.notify_one();
        }
        // true if a submitted fill has not finished
        bool busy() const {
            return state.load(std::memory_order_acquire) != 0;
        }
        void wait() {
            while (busy()) {
                io_pool::cpu_relax();
            }
        }
        ~fill_thread() {
            if (th.joinable()) {
                wait();
                {
                    std::lock_guard<std::mutex> lock(m);
                    stop = true;
                }
                cv.notify_all();
                th.join();
            }
        }
    } filler;

    bool spec_ok() const {
        return spec != nullptr;
    }
    // HBM tier (LLAMA_EXPERT_STORE_HBM_MIB): lookups served from it, promotions into it
    size_t   hbm_size = 0;
    int      hbm_decay = 128;      // LLAMA_EXPERT_STORE_HBM_DECAY: tokens between halvings of the use counts
    int      hbm_max_promote = 2;  // LLAMA_EXPERT_STORE_HBM_PROMOTE: exchanges per layer fill at most
    // LLAMA_EXPERT_STORE_SNAPSHOT=<file>: the cache contents (which experts, their recency and use counts, not the
    // bytes) are written there at exit and every LLAMA_EXPERT_STORE_SNAPSHOT_EVERY seconds (default 300, 0 = exit
    // only), and read back at the next start, so the cache is warm from the first token instead of after minutes of
    // misses (GH200: ~6 min at 6 instead of 13 tok/s after every restart; the refill reads ~410 GB in ~1 min)
    std::string snapshot_path;
    int         snapshot_every_s = 300;
    std::chrono::steady_clock::time_point t_snapshot_last{};
    // LLAMA_EXPERT_STORE_MLOCK (default 1): lock the process memory except the model files at the first fill, when
    // the model, the context and the GPU buffers are all in place (see lock_process_memory)
    bool mlock_process = true;
    bool mlock_done    = false;
    std::mutex mlock_mutex;

    void lock_memory_once() {
        if (!mlock_process || mlock_done) {
            return;
        }
        std::lock_guard<std::mutex> lock(mlock_mutex);
        if (mlock_done) {
            return;
        }
        mlock_done = true;
        const auto t0 = std::chrono::steady_clock::now();
        size_t locked = 0, n = 0, skipped = 0;
        std::string err;
        const bool ok = lock_process_memory(locked, n, skipped, err);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ok) {
            LLAMA_LOG_INFO("%s: expert store: locked %.0f MiB of process memory (%zu mappings, %.0f ms), the model file mappings (%.0f MiB) stay pageable\n",
                __func__, locked/1024.0/1024.0, n, ms, skipped/1024.0/1024.0);
        } else {
            LLAMA_LOG_WARN("%s: expert store: could not lock the process memory (%s after %.0f MiB): raise RLIMIT_MEMLOCK or add CAP_IPC_LOCK; cold pages of the cache and the libraries can be paged out and every first touch then waits behind the expert reads\n",
                __func__, err.c_str(), locked/1024.0/1024.0);
        }
        mlock_ok = ok;
    }
    bool mlock_ok = false;

    // --- memory pressure valve (kimi-k3-spark patch 0003) ---
    //
    // The cache is sized by hand to what the box has left after the model, and that guess is only ever checked with
    // short prompts: a long prompt grows the CUDA compute buffers by several GB (on the GH200 a 30k-token prompt
    // added ~10 GB on the HBM node), the driver and the server allocate a little more over time, and when the host
    // node reaches its min watermark the kernel's answer is SIGKILL, not reclaim (the bank is anonymous and locked).
    // The valve keeps a reserve per memory domain instead: a thread samples the host node's MemAvailable and, with an
    // HBM tier, the HBM node's free memory every LLAMA_EXPERT_STORE_VALVE_INTERVAL_MS (default 500) and wakes early on
    // a PSI trigger (/proc/pressure/memory "some 150000 2000000": tasks stalled on reclaim for 150 ms within 2 s,
    // the kernel's only warning before the OOM killer). Under the reserve it releases the least recently used
    // resident slots of that domain in one batch sized to the deficit plus a hysteresis band: munlock and
    // MADV_DONTNEED give their pages back to the kernel, the mapping stays, the experts become misses. Released slots
    // are parked (never chosen by the plan) until the domain has been above the reserve plus two bands for
    // LLAMA_EXPERT_STORE_VALVE_REGROW_S seconds, then they are re-armed a chunk at a time. Lossless: a record is only
    // ever dropped whole, never changed. Slots the GPU may be reading (the last layers filled) and slots the
    // speculative reader is writing are never touched. LLAMA_EXPERT_STORE_VALVE=0 turns it off.
    bool     valve_on          = true;
    size_t   valve_reserve     = (size_t) 4096 << 20; // LLAMA_EXPERT_STORE_RESERVE_MIB: host MemAvailable to keep
    size_t   valve_hbm_reserve = (size_t) 6144 << 20; // LLAMA_EXPERT_STORE_HBM_RESERVE_MIB: free memory on the HBM node to keep
    int      valve_floor_pct   = 50;                  // LLAMA_EXPERT_STORE_VALVE_FLOOR: never release more than this share of a domain's slots
    int      valve_interval_ms = 500;                 // LLAMA_EXPERT_STORE_VALVE_INTERVAL_MS
    int      valve_regrow_s    = 10;                  // LLAMA_EXPERT_STORE_VALVE_REGROW_S: time above reserve + 2 bands before re-arming
    int      valve_regrow_chunk = 256;                // slots re-armed per sample at most
    std::thread       valve_thread;
    std::atomic<bool> valve_stop{false};
    std::recursive_mutex lru_mutex;                   // the fill hooks hold it for a whole fill, the valve for a release
    int      valve_psi_fd = -1;
    std::vector<std::vector<int32_t>> valve_released_host; // [group] released slots of the host domain
    std::vector<std::vector<int32_t>> valve_released_hbm;  // [group] released slots of the HBM domain
    size_t   valve_released_bytes_host = 0, valve_released_bytes_hbm = 0;
    uint64_t valve_n_releases = 0, valve_n_rearms = 0, valve_n_psi = 0;
    bool     valve_broken = false;                    // a release did not give its pages back: no more releases
    size_t   valve_lowest_host = SIZE_MAX, valve_lowest_hbm = SIZE_MAX;
    size_t   valve_last_host = 0, valve_last_hbm = 0;
    std::chrono::steady_clock::time_point valve_calm_host{}, valve_calm_hbm{};
    int32_t  protected_il[4] = { -1, -1, -1, -1 };    // layers whose slots the GPU may be reading right now
    int      protected_pos = 0;
    uint64_t cg_oom_kill = 0, cg_high = 0;            // cgroup memory.events seen so far
    unsigned long valve_hbm_mask = 0;

    void protect_layer(int32_t il) {
        for (int32_t p : protected_il) {
            if (p == il) {
                return;
            }
        }
        protected_il[protected_pos] = il;
        protected_pos = (protected_pos + 1) % 4;
    }
    bool is_protected(int32_t il) const {
        for (int32_t p : protected_il) {
            if (p == il) {
                return true;
            }
        }
        return false;
    }

    static size_t meminfo_field(const char * path, const char * field) {
        FILE * f = fopen(path, "r");
        if (f == nullptr) {
            return SIZE_MAX;
        }
        char line[256];
        size_t res = SIZE_MAX;
        const size_t n = strlen(field);
        while (fgets(line, sizeof(line), f) != nullptr) {
            const char * p = line;
            if (strncmp(p, "Node ", 5) == 0) { // "Node 1 MemFree:   123 kB"
                p += 5;
                while (*p >= '0' && *p <= '9') {
                    p++;
                }
                while (*p == ' ') {
                    p++;
                }
            }
            if (strncmp(p, field, n) == 0 && p[n] == ':') {
                unsigned long kb = 0;
                if (sscanf(p + n + 1, " %lu", &kb) == 1) {
                    res = (size_t) kb*1024;
                }
                break;
            }
        }
        fclose(f);
        return res;
    }

    // free memory on NUMA node n as the kernel sees it (what its allocations on that node can still take)
    static size_t node_free(int n) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", n);
        return meminfo_field(path, "MemFree");
    }

    int hbm_node() const {
        for (int n = 0; n < (int) (8*sizeof(valve_hbm_mask)); ++n) {
            if (valve_hbm_mask & (1UL << n)) {
                return n;
            }
        }
        return -1;
    }

    // MemAvailable of the CPU nodes: the system figure minus the HBM node's free memory, which host allocations are
    // never given (the HBM node is CPU-less and the bank's host part is bound away from it)
    size_t host_available() const {
        size_t avail = meminfo_field("/proc/meminfo", "MemAvailable");
        if (avail == SIZE_MAX) {
            return SIZE_MAX;
        }
        const int h = hbm_node();
        if (h >= 0) {
            const size_t hf = node_free(h);
            if (hf != SIZE_MAX) {
                avail = avail > hf ? avail - hf : 0;
            }
        }
        return avail;
    }

    void valve_start() {
        if (!valve_on || !use_store) {
            return;
        }
        bool any = false;
        for (group & g : groups) {
            if (g.region != nullptr && g.n_slots > 0) {
                any = true;
                if (!g.spec_state) {
                    g.spec_state.reset(new std::atomic<int8_t>[g.n_slots]);
                    for (int32_t s = 0; s < g.n_slots; ++s) {
                        g.spec_state[s].store(0);
                    }
                    g.lru.busy = g.spec_state.get();
                }
            }
        }
        if (!any) {
            return;
        }
        valve_released_host.assign(groups.size(), {});
        valve_released_hbm.assign(groups.size(), {});
        bool hbm = false;
        for (const group & g : groups) {
            hbm = hbm || g.lru.n_hbm > 0;
        }
        valve_hbm_mask = hbm ? store_gpu_node_mask() : 0;
        if (host_available() == SIZE_MAX) {
            LLAMA_LOG_WARN("%s: expert store: memory valve: /proc/meminfo unreadable, off\n", __func__);
            return;
        }
        // PSI trigger: wakes the sampler when tasks have stalled on memory for 150 ms within 2 s
        valve_psi_fd = open("/proc/pressure/memory", O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (valve_psi_fd >= 0) {
            const char * trig = "some 150000 2000000"; // a 2 s window: unprivileged triggers need a multiple of 2 s
            if (write(valve_psi_fd, trig, strlen(trig)) < 0) {
                close(valve_psi_fd);
                valve_psi_fd = -1;
            }
        }
        valve_stop.store(false);
        valve_thread = std::thread([this] { valve_loop(); });
        LLAMA_LOG_INFO("%s: expert store: memory valve on: host reserve %zu MiB%s, floor %d%% of the slots, sampled every %d ms%s, released slots re-armed after %d s of calm\n", __func__,
            valve_reserve >> 20, hbm_node() >= 0 ? (", HBM node reserve " + std::to_string(valve_hbm_reserve >> 20) + " MiB").c_str() : "",
            valve_floor_pct, valve_interval_ms, valve_psi_fd >= 0 ? " and on a PSI memory trigger" : " (no PSI trigger: /proc/pressure/memory not writable)", valve_regrow_s);
    }

    void valve_stop_thread() {
        if (valve_thread.joinable()) {
            valve_stop.store(true);
            valve_thread.join();
        }
        if (valve_psi_fd >= 0) {
            close(valve_psi_fd);
            valve_psi_fd = -1;
        }
    }

    void valve_loop() {
        while (!valve_stop.load()) {
            bool psi = false;
            if (valve_psi_fd >= 0) {
                struct pollfd pfd = { valve_psi_fd, POLLPRI, 0 };
                const int rc = poll(&pfd, 1, valve_interval_ms);
                if (rc > 0 && (pfd.revents & POLLERR)) {
                    close(valve_psi_fd); // the trigger is gone (the fs went away), fall back to the timer
                    valve_psi_fd = -1;
                } else if (rc > 0 && (pfd.revents & POLLPRI)) {
                    psi = true;
                    valve_n_psi++;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(valve_interval_ms));
            }
            if (valve_stop.load()) {
                break;
            }
            valve_sample(psi);
        }
    }

    void valve_sample(bool psi) {
        const auto now = std::chrono::steady_clock::now();
        const size_t host = host_available();
        if (host == SIZE_MAX) {
            return;
        }
        valve_last_host = host;
        valve_lowest_host = std::min(valve_lowest_host, host);
        const int hn = hbm_node();
        size_t hbm = SIZE_MAX;
        if (hn >= 0) {
            hbm = node_free(hn);
            if (hbm != SIZE_MAX) {
                valve_last_hbm = hbm;
                valve_lowest_hbm = std::min(valve_lowest_hbm, hbm);
            }
        }
        // cgroup v2 memory events of the container, if any: a kill or a memory.high breach is worth a log line
        {
            FILE * f = fopen("/sys/fs/cgroup/memory.events", "r");
            if (f != nullptr) {
                char line[128];
                while (fgets(line, sizeof(line), f) != nullptr) {
                    unsigned long long v = 0;
                    if (sscanf(line, "oom_kill %llu", &v) == 1 && v != cg_oom_kill) {
                        LLAMA_LOG_WARN("%s: expert store: memory valve: the cgroup reports %llu OOM kills\n", __func__, v);
                        cg_oom_kill = v;
                    } else if (sscanf(line, "high %llu", &v) == 1 && v != cg_high) {
                        LLAMA_LOG_WARN("%s: expert store: memory valve: the cgroup crossed memory.high %llu times\n", __func__, v);
                        cg_high = v;
                    }
                }
                fclose(f);
            }
        }
        const size_t band = valve_reserve/4;
        if (host < valve_reserve || (psi && host < valve_reserve + band)) {
            valve_calm_host = {};
            valve_release(false, valve_reserve + band - host, host, psi ? "PSI trigger" : "sample");
        } else if (host >= valve_reserve + 2*band) {
            if (valve_calm_host.time_since_epoch().count() == 0) {
                valve_calm_host = now;
            } else if (std::chrono::duration_cast<std::chrono::seconds>(now - valve_calm_host).count() >= valve_regrow_s) {
                valve_rearm(false, host - (valve_reserve + 2*band));
            }
        } else {
            valve_calm_host = {};
        }
        if (hbm != SIZE_MAX) {
            const size_t hband = valve_hbm_reserve/4;
            if (hbm < valve_hbm_reserve) {
                valve_calm_hbm = {};
                valve_release(true, valve_hbm_reserve + hband - hbm, hbm, psi ? "PSI trigger" : "sample");
            } else if (hbm >= valve_hbm_reserve + 2*hband) {
                if (valve_calm_hbm.time_since_epoch().count() == 0) {
                    valve_calm_hbm = now;
                } else if (std::chrono::duration_cast<std::chrono::seconds>(now - valve_calm_hbm).count() >= valve_regrow_s) {
                    valve_rearm(true, hbm - (valve_hbm_reserve + 2*hband));
                }
            } else {
                valve_calm_hbm = {};
            }
        }
        // released slots drift from the tail as fills touch other slots past them: keep them behind everything
        std::lock_guard<std::recursive_mutex> lock(lru_mutex);
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            for (int32_t s : valve_released_host[ig]) {
                groups[ig].lru.touch(s);
            }
            for (int32_t s : valve_released_hbm[ig]) {
                groups[ig].lru.touch(s);
            }
        }
    }

    // the whole pages inside a run of slots: madvise needs a page-aligned start, and rounding outwards would drop
    // the first or last bytes of a neighbouring slot that is still resident (on the GH200's 64 KiB pages a slot
    // boundary is almost never page aligned). The partial pages at the two ends stay resident.
    bool slots_page_range(const group & g, int32_t first, int32_t last, char *& p, size_t & len) const {
        const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
        const uintptr_t lo = (uintptr_t) g.region + (size_t) first*g.slot_stride;
        const uintptr_t hi = (uintptr_t) g.region + (size_t) (last + 1)*g.slot_stride;
        const uintptr_t a = (lo + page - 1)/page*page;
        const uintptr_t b = hi/page*page;
        if (b <= a) {
            return false;
        }
        p = (char *) a;
        len = b - a;
        return true;
    }

    // give the pages of a run of slots back to the kernel; returns the bytes given back
    size_t slots_drop_pages(group & g, int32_t first, int32_t last) {
        char * p = nullptr;
        size_t len = 0;
        if (!slots_page_range(g, first, last, p, len)) {
            return 0;
        }
        if (mlock_ok && munlock(p, len) != 0) { // MADV_DONTNEED is refused on a locked range
            LLAMA_LOG_WARN("%s: expert store: memory valve: munlock of slots %d-%d failed: %s\n", __func__, first, last, strerror(errno));
            return 0;
        }
        if (madvise(p, len, MADV_DONTNEED) != 0) {
            LLAMA_LOG_WARN("%s: expert store: memory valve: MADV_DONTNEED of slots %d-%d failed: %s\n", __func__, first, last, strerror(errno));
            return 0;
        }
        return len;
    }

    void slots_take_pages(group & g, int32_t first, int32_t last) {
        char * p = nullptr;
        size_t len = 0;
        if (!mlock_ok || !slots_page_range(g, first, last, p, len)) {
            return;
        }
#ifdef MLOCK_ONFAULT
        mlock2(p, len, MLOCK_ONFAULT);
#endif
    }

    // release the least recently used slots of a domain until `bytes` are freed (one batch, under the LRU lock)
    void valve_release(bool hbm, size_t bytes, size_t avail, const char * why) {
        std::lock_guard<std::recursive_mutex> lock(lru_mutex);
        if (valve_broken) {
            return;
        }
        size_t freed = 0;   // slot bytes taken out of the cache
        size_t dropped = 0; // bytes actually given back to the kernel (whole pages only)
        int n_slots_released = 0;
        std::vector<int32_t> victims;
        for (size_t ig = 0; ig < groups.size() && freed < bytes; ++ig) {
            group & g = groups[ig];
            if (g.region == nullptr || g.n_slots == 0) {
                continue;
            }
            moe_cache_lru & lru = g.lru;
            // slots 0..n_slots-1 only: the split GEMV's zero record (slot n_slots) is never released, and the page
            // range of the last slot is rounded down, so its pages are never touched either
            const int32_t lo = hbm ? 0 : lru.n_hbm;
            const int32_t hi = hbm ? lru.n_hbm : g.n_slots;
            if (hi <= lo) {
                continue;
            }
            std::vector<int32_t> & released = hbm ? valve_released_hbm[ig] : valve_released_host[ig];
            const size_t keep_min = (size_t) (hi - lo)*valve_floor_pct/100;
            victims.clear();
            for (int32_t s = lru.head; s >= 0 && freed < bytes; s = lru.next[s]) {
                if (s < lo || s >= hi) {
                    continue;
                }
                if ((size_t) (hi - lo) - released.size() - victims.size() <= keep_min) {
                    break;
                }
                if (g.spec_state[s].load(std::memory_order_acquire) != 0) {
                    continue; // the speculative reader's, or already released
                }
                const int32_t k = lru.key_of[s];
                if (k >= 0 && is_protected(k / lru.n_expert)) {
                    continue;
                }
                victims.push_back(s);
                freed += g.slot_stride;
            }
            for (int32_t s : victims) {
                const int32_t k = lru.key_of[s];
                if (k >= 0) {
                    lru.slot_map[k / lru.n_expert][k % lru.n_expert] = -1;
                    lru.key_of[s] = -1;
                    lru.n_filled--;
                    if (prefetch) {
                        prefetched[k] = 0;
                    }
                }
                g.spec_state[s].store(3, std::memory_order_release); // parked: never a victim of the plan
                lru.n_parked++;
                lru.touch(s);
                released.push_back(s);
            }
            // drop the pages run by run
            std::sort(victims.begin(), victims.end());
            for (size_t i = 0; i < victims.size();) {
                size_t j = i;
                while (j + 1 < victims.size() && victims[j + 1] == victims[j] + 1) {
                    j++;
                }
                dropped += slots_drop_pages(g, victims[i], victims[j]);
                i = j + 1;
            }
            n_slots_released += (int) victims.size();
            if (g.hbm_min_slot >= 0 && hbm) {
                hbm_refresh_min(g);
            }
        }
        if (hbm) {
            valve_released_bytes_hbm += freed;
        } else {
            valve_released_bytes_host += freed;
        }
        if (n_slots_released > 0 && dropped < freed/2) {
            // the pages did not go back: releasing more would only empty the cache. Stop, keep serving.
            valve_broken = true;
            LLAMA_LOG_ERROR("%s: expert store: memory valve: %d slots (%.2f GB) released but only %.2f GB given back to the kernel, the valve stops releasing\n", __func__,
                n_slots_released, freed/1e9, dropped/1e9);
        }
        if (n_slots_released > 0) {
            valve_n_releases++;
            LLAMA_LOG_WARN("%s: expert store: memory valve: %s %.2f GB under the %.2f GB reserve (%s): released %d slots (%.2f GB), %.2f GB of the %s cache released in all\n", __func__,
                hbm ? "HBM node free" : "host MemAvailable", avail/1e9, (hbm ? valve_hbm_reserve : valve_reserve)/1e9, why,
                n_slots_released, freed/1e9, (hbm ? valve_released_bytes_hbm : valve_released_bytes_host)/1e9, hbm ? "HBM" : "host");
        } else if (bytes > 0) {
            static uint64_t warned = 0;
            if (warned++ % 64 == 0) {
                LLAMA_LOG_WARN("%s: expert store: memory valve: %s %.2f GB under the reserve but nothing left to release (floor %d%%, protected or busy slots)\n", __func__,
                    hbm ? "HBM node free" : "host MemAvailable", avail/1e9, valve_floor_pct);
            }
        }
    }

    // re-arm released slots of a domain, at most a chunk and at most `room` bytes worth, so a regrowth never
    // takes the domain back under the reserve
    void valve_rearm(bool hbm, size_t room) {
        std::lock_guard<std::recursive_mutex> lock(lru_mutex);
        int n = 0;
        size_t bytes = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            std::vector<int32_t> & released = hbm ? valve_released_hbm[ig] : valve_released_host[ig];
            std::vector<int32_t> back;
            while (!released.empty() && n < valve_regrow_chunk && bytes + g.slot_stride <= room) {
                const int32_t s = released.back();
                released.pop_back();
                back.push_back(s);
                g.spec_state[s].store(0, std::memory_order_release);
                g.lru.n_parked--;
                g.lru.demote(s); // empty and usable: next to be taken
                n++;
                bytes += g.slot_stride;
            }
            std::sort(back.begin(), back.end());
            for (size_t i = 0; i < back.size();) {
                size_t j = i;
                while (j + 1 < back.size() && back[j + 1] == back[j] + 1) {
                    j++;
                }
                slots_take_pages(g, back[i], back[j]);
                i = j + 1;
            }
        }
        if (n > 0) {
            valve_n_rearms++;
            if (hbm) {
                valve_released_bytes_hbm -= std::min(valve_released_bytes_hbm, bytes);
            } else {
                valve_released_bytes_host -= std::min(valve_released_bytes_host, bytes);
            }
            size_t left = 0;
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                left += (hbm ? valve_released_hbm[ig] : valve_released_host[ig]).size();
            }
            LLAMA_LOG_INFO("%s: expert store: memory valve: %s calm for %d s: %d slots (%.2f GB) re-armed, %zu still released\n", __func__,
                hbm ? "HBM node" : "host", valve_regrow_s, n, bytes/1e9, left);
        }
    }

    void valve_log(const char * func) const {
        if (!valve_thread.joinable()) {
            return;
        }
        size_t rel_host = 0, rel_hbm = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            rel_host += valve_released_host[ig].size();
            rel_hbm  += valve_released_hbm[ig].size();
        }
        LLAMA_LOG_INFO("%s: expert store: memory valve: host MemAvailable %.2f GB (lowest %.2f, reserve %.2f), %zu host slots released (%.2f GB)%s; %llu releases, %llu re-arms, %llu PSI wake-ups\n", func,
            valve_last_host/1e9, valve_lowest_host == SIZE_MAX ? 0.0 : valve_lowest_host/1e9, valve_reserve/1e9, rel_host, valve_released_bytes_host/1e9,
            hbm_node() >= 0 ? (", HBM node free " + std::to_string(valve_last_hbm/1000000) + " MB (lowest " + std::to_string(valve_lowest_hbm == SIZE_MAX ? 0 : valve_lowest_hbm/1000000) + "), " + std::to_string(rel_hbm) + " HBM slots released").c_str() : "",
            (unsigned long long) valve_n_releases, (unsigned long long) valve_n_rearms, (unsigned long long) valve_n_psi);
    }
    uint64_t n_hbm_fills = 0;
    uint64_t n_hbm_lookups = 0, n_lookups = 0, n_promotions = 0;
    uint64_t t_promote_us = 0;
    std::vector<uint8_t> prefetched; // [n_layer*n_expert] read by a prefetch and not yet used by a real fill
    struct {
        int32_t il = -1;
        const ggml_tensor * ids = nullptr;
    } prefill;
    stats  stats_store_decode;  // fills from the store for small batches
    size_t store_prefill_bytes = 0;
    size_t n_decode_fills = 0;
    bool   warned_ids = false;
    // where a decode step spends its time, per layer fill: LRU planning, store reads, and the gap since the
    // previous fill ended (the layer's matmuls, the spine, scheduler copies); gaps over a second are idle time
    uint64_t t_dec_plan_us = 0;
    uint64_t t_dec_io_us   = 0;
    uint64_t t_dec_gap_us  = 0;
    size_t   n_dec_gaps    = 0;
    std::chrono::steady_clock::time_point t_dec_last{};

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        const char * store_path = getenv("LLAMA_EXPERT_STORE");
        use_store = store_path != nullptr && store_path[0] != '\0';
        if (use_store) {
            // the cache lives in host RAM and the CPU backend runs the expert matmuls from it
            std::string err;
            if (!store.open(store_path, err)) {
                throw std::runtime_error(std::string("expert store ") + store_path + ": " + err);
            }
            for (size_t i = 0; i < backends.size(); ++i) {
                const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU && devices.empty()) {
                    auto & d = devices.emplace_back();
                    d.backend = backends[i];
                    d.buft    = ggml_backend_cpu_buffer_type();
                }
                if ((dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) && gpu_reg == nullptr) {
                    gpu_reg     = ggml_backend_dev_backend_reg(ggml_backend_get_device(backends[i]));
                    gpu_backend = backends[i];
                }
            }
            if (devices.empty()) {
                throw std::runtime_error("expert store requires the CPU backend");
            }
            const char * env_threads = getenv("LLAMA_EXPERT_STORE_THREADS");
            const int n_io = std::max(1, env_threads ? atoi(env_threads) : 16);
            // LLAMA_EXPERT_STORE_CHUNK_KIB: records are read as pieces of this size by different threads (0 = whole)
            prefill_promote = getenv("LLAMA_EXPERT_STORE_PREFILL_PROMOTE") != nullptr && getenv("LLAMA_EXPERT_STORE_PREFILL_PROMOTE")[0] == '1';
            const char * env_chunk = getenv("LLAMA_EXPERT_STORE_CHUNK_KIB");
            size_t chunk = env_chunk ? (size_t) std::max(0, atoi(env_chunk))*1024 : (size_t) 1024*1024;
            chunk -= chunk % 4096; // O_DIRECT alignment
            pool = std::make_unique<io_pool>(n_io, chunk);
            pool_chunk = chunk;
            if (getenv("LLAMA_EXPERT_STORE_SPIN_US") != nullptr) {
                pool->spin_us = std::max(0, atoi(getenv("LLAMA_EXPERT_STORE_SPIN_US")));
            }
#if defined(LLAMA_HAVE_LIBURING)
            // LLAMA_EXPERT_STORE_URING=0: read with the threads as before
            pool->use_uring = getenv("LLAMA_EXPERT_STORE_URING") == nullptr || atoi(getenv("LLAMA_EXPERT_STORE_URING")) != 0;
#endif
            const char * env_hbm = getenv("LLAMA_EXPERT_STORE_HBM_MIB");
            hbm_size = env_hbm != nullptr ? (size_t) std::max(0, atoi(env_hbm))*1024*1024 : 0;
            if (getenv("LLAMA_EXPERT_STORE_HBM_DECAY") != nullptr) {
                hbm_decay = std::max(1, atoi(getenv("LLAMA_EXPERT_STORE_HBM_DECAY")));
            }
            if (getenv("LLAMA_EXPERT_STORE_HBM_PROMOTE") != nullptr) {
                hbm_max_promote = std::max(0, atoi(getenv("LLAMA_EXPERT_STORE_HBM_PROMOTE")));
            }
            if (getenv("LLAMA_EXPERT_STORE_SNAPSHOT") != nullptr) {
                snapshot_path = getenv("LLAMA_EXPERT_STORE_SNAPSHOT");
            }
            if (getenv("LLAMA_EXPERT_STORE_SNAPSHOT_EVERY") != nullptr) {
                snapshot_every_s = std::max(0, atoi(getenv("LLAMA_EXPERT_STORE_SNAPSHOT_EVERY")));
            }
            if (getenv("LLAMA_EXPERT_STORE_MLOCK") != nullptr) {
                mlock_process = atoi(getenv("LLAMA_EXPERT_STORE_MLOCK")) != 0;
            }
            if (getenv("LLAMA_EXPERT_STORE_VALVE") != nullptr) {
                valve_on = atoi(getenv("LLAMA_EXPERT_STORE_VALVE")) != 0;
            }
            if (getenv("LLAMA_EXPERT_STORE_RESERVE_MIB") != nullptr) {
                valve_reserve = (size_t) std::max(0, atoi(getenv("LLAMA_EXPERT_STORE_RESERVE_MIB"))) << 20;
            }
            if (getenv("LLAMA_EXPERT_STORE_HBM_RESERVE_MIB") != nullptr) {
                valve_hbm_reserve = (size_t) std::max(0, atoi(getenv("LLAMA_EXPERT_STORE_HBM_RESERVE_MIB"))) << 20;
            }
            if (getenv("LLAMA_EXPERT_STORE_VALVE_FLOOR") != nullptr) {
                valve_floor_pct = std::min(100, std::max(0, atoi(getenv("LLAMA_EXPERT_STORE_VALVE_FLOOR"))));
            }
            if (getenv("LLAMA_EXPERT_STORE_VALVE_INTERVAL_MS") != nullptr) {
                valve_interval_ms = std::max(50, atoi(getenv("LLAMA_EXPERT_STORE_VALVE_INTERVAL_MS")));
            }
            if (getenv("LLAMA_EXPERT_STORE_VALVE_REGROW_S") != nullptr) {
                valve_regrow_s = std::max(1, atoi(getenv("LLAMA_EXPERT_STORE_VALVE_REGROW_S")));
            }
            const char * env_batch = getenv("LLAMA_EXPERT_STORE_MAX_BATCH");
            if (env_batch != nullptr && atoi(env_batch) > 0) {
                max_batch_env = atoi(env_batch);
            }
            if (pool->use_uring) {
                LLAMA_LOG_INFO("%s: expert store %s: %zu layers x %u experts, records up to %zu bytes, reads of %zu KiB through io_uring\n",
                    __func__, store_path, store.entries.size(), store.n_expert, store.max_record, chunk/1024);
            } else {
                LLAMA_LOG_INFO("%s: expert store %s: %zu layers x %u experts, records up to %zu bytes, %d I/O threads, reads of %zu KiB\n",
                    __func__, store_path, store.entries.size(), store.n_expert, store.max_record, n_io, chunk/1024);
            }
        } else {
            for (size_t i = 0; i < backends.size(); ++i) {
                const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    auto & d = devices.emplace_back();
                    d.backend = backends[i];
                    d.buft    = bufts[i];
                }
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            int32_t id = -1;
            if (use_store) {
                // every stored layer is cached on the one CPU device, whatever device runs the rest of the layer
                if (store.find((int32_t) il) == nullptr) {
                    continue;
                }
                if (model.layers[il].ffn_gate_up_exps != nullptr || experts.size() != 3) {
                    LLAMA_LOG_WARN("%s: layer %zu has %zu expert tensors, the expert store needs gate, up, down; not cached\n", __func__, il, experts.size());
                    continue;
                }
                id = 0;
            } else {
                const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
                if (it_dev == devices.end()) {
                    continue;
                }
                id = (int32_t) (it_dev - devices.begin());
            }
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // expert store: the records must match the tensors of every layer of the group exactly
        if (use_store) {
            for (group & g : groups) {
                const expert_store_entry * e0 = store.find(g.layers[0]);
                GGML_ASSERT(e0 != nullptr);
                g.record_size = e0->record_size;
                size_t off = 0;
                for (size_t ip = 0; ip < g.ref.size(); ++ip) {
                    if (e0->sizes[ip] != g.ref[ip]->nb[2] || e0->types[ip] != (int32_t) g.ref[ip]->type) {
                        throw std::runtime_error(format("expert store: layer %d tensor %s does not match its record (%zu vs %zu bytes per expert, type %d vs %d)",
                            g.layers[0], g.ref[ip]->name, (size_t) g.ref[ip]->nb[2], e0->sizes[ip], (int) g.ref[ip]->type, e0->types[ip]));
                    }
                    g.part_off.push_back(off);
                    off += e0->sizes[ip];
                }
                // the GPU matmul kernels address an expert as (slot * nb[2] / type_size) blocks, so the slot stride has to be
                // a multiple of every part's block size as well as of the O_DIRECT alignment; 4096 and 50 (iq1_s) need no
                // padding, 66 (iq2_xxs) pads the mixed layers' records by up to 30 %
                g.slot_stride = GGML_PAD(g.record_size, (size_t) 4096);
                while (std::any_of(g.ref.begin(), g.ref.end(), [&](const ggml_tensor * t) { return g.slot_stride % ggml_type_size(t->type) != 0; })) {
                    g.slot_stride += 4096;
                }
                for (int32_t il : g.layers) {
                    const expert_store_entry * e = store.find(il);
                    if (e->record_size != g.record_size || memcmp(e->sizes, e0->sizes, sizeof(e->sizes)) != 0 || memcmp(e->types, e0->types, sizeof(e->types)) != 0) {
                        throw std::runtime_error(format("expert store: layer %d has a different record layout than layer %d of its group", il, g.layers[0]));
                    }
                }
            }
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            if (use_store) {
                return GGML_PAD(g.slot_stride*(size_t) (n_slots + 1), (size_t) 4096);
            }
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (use_store) {
                // the one CPU cache device is not a model device (no tensor split entry): it gets the whole budget
                d.split = 1.0;
            } else if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        split_req = use_store && getenv("LLAMA_EXPERT_STORE_SPLIT") != nullptr && getenv("LLAMA_EXPERT_STORE_SPLIT")[0] == '1';
        // remote experts: the sets are read here (one file per box), the connections start once the maps exist
        std::vector<std::string> remote_hosts;
        if (getenv("LLAMA_EXPERT_STORE_REMOTE") != nullptr && getenv("LLAMA_EXPERT_STORE_REMOTE")[0] != 0) {
            auto split_list = [](const char * v) {
                std::vector<std::string> out;
                std::string cur;
                for (const char * c = v; ; ++c) {
                    if (*c == ',' || *c == 0) {
                        if (!cur.empty()) {
                            out.push_back(cur);
                        }
                        cur.clear();
                        if (*c == 0) {
                            break;
                        }
                    } else if (*c != ' ') {
                        cur += *c;
                    }
                }
                return out;
            };
            remote_hosts = split_list(getenv("LLAMA_EXPERT_STORE_REMOTE"));
            const std::vector<std::string> sets = getenv("LLAMA_EXPERT_STORE_REMOTE_SET") ? split_list(getenv("LLAMA_EXPERT_STORE_REMOTE_SET")) : std::vector<std::string>();
            if (!split_req || store.n_expert == 0 || sets.size() != remote_hosts.size() || remote_hosts.size() > 250) {
                LLAMA_LOG_WARN("%s: expert store: LLAMA_EXPERT_STORE_REMOTE needs LLAMA_EXPERT_STORE_SPLIT=1 and one LLAMA_EXPERT_STORE_REMOTE_SET file per box, off\n", __func__);
                remote_hosts.clear();
            }
            size_t n_set = 0;
            for (size_t bx = 0; bx < remote_hosts.size(); ++bx) {
                FILE * f = fopen(sets[bx].c_str(), "r");
                if (f == nullptr) {
                    LLAMA_LOG_WARN("%s: expert store: cannot read the remote set %s, remote experts off\n", __func__, sets[bx].c_str());
                    remote_hosts.clear();
                    remote_box.clear();
                    break;
                }
                if (remote_box.empty()) {
                    remote_box.assign(model.layers.size()*(size_t) store.n_expert, 0);
                }
                int a = 0, b = 0;
                while (fscanf(f, "%d %d", &a, &b) == 2) {
                    if (a >= 0 && (size_t) a < model.layers.size() && b >= 0 && (uint32_t) b < store.n_expert && !remote_box[(size_t) a*store.n_expert + b]) {
                        remote_box[(size_t) a*store.n_expert + b] = (uint8_t) (bx + 1); // an expert listed twice stays with the first box
                        n_set++;
                    }
                }
                fclose(f);
            }
            if (n_set == 0) {
                remote_box.clear();
                remote_hosts.clear();
            }
        }
        auto layer_has_remote = [&](int32_t il) {
            if (remote_box.empty()) {
                return false;
            }
            for (uint32_t x = 0; x < store.n_expert; ++x) {
                if (remote_box[(size_t) il*store.n_expert + x]) {
                    return true;
                }
            }
            return false;
        };
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            g.hbm_budget = use_store ? (size_t) ((double) hbm_size*g.host_bytes/d.host_bytes) : 0;
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes) + g.hbm_budget;
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            g.lru.n_hbm = g.slot_stride > 0 ? std::min<int32_t>(g.n_slots, (int32_t) (g.hbm_budget/g.slot_stride)) : 0;
            // the second init below (store open, snapshot) keeps chance_max: init() does not touch it
            g.lru.chance_max = getenv("LLAMA_EXPERT_STORE_SECOND_CHANCE") != nullptr ? std::min(255, std::max(0, atoi(getenv("LLAMA_EXPERT_STORE_SECOND_CHANCE")))) : 0;
            if (&g == &groups.front() && g.lru.chance_max > 0) {
                LLAMA_LOG_INFO("%s: MoE cache eviction: LRU with %d second chances per hit (LLAMA_EXPERT_STORE_SECOND_CHANCE)\n", __func__, g.lru.chance_max);
            }
            if (&g == &groups.front()) {
                store_cpus_log();
            }
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            if (split_req && use_store) {
                n_tensors[g.id] += g.ref.size()*g.layers.size(); // the views with the zero record included
            }
            n_tensors_host  += g.layers.size();
            if (split_req && use_store) {
                n_tensors_host += 3*g.layers.size() + 2; // early, late and merge maps, the zero record's slot, the iota
                for (int32_t il : g.layers) {
                    n_tensors_host += layer_has_remote(il) ? 4 : 0; // remote post map, input, rows, merge map
                }
            }
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            if (use_store) {
                // one region of records; each expert tensor's bank is a view with the record as its expert stride
                g.region_size = alloc_size(g, g.n_slots);
                if (!model.hparams.no_alloc) {
                    std::string err;
                    g.region = store_region_alloc(g.region_size, err, (size_t) g.lru.n_hbm*g.slot_stride);
                    if (g.region == nullptr) {
                        throw std::runtime_error("expert store: failed to allocate " + std::to_string(g.region_size/1024/1024) + " MiB of host cache: " + err);
                    }
                    // on an integrated GPU that can read registered host memory in place (GB10), the bank is handed
                    // to the GPU backend as a "host bank" buffer: the small-batch expert matmuls then run there,
                    // straight from the cache the CPU fills from the store. LLAMA_EXPERT_STORE_GPU=0 keeps them on the CPU.
                    ggml_backend_buffer_t region_buf = nullptr;
                    const char * env_gpu = getenv("LLAMA_EXPERT_STORE_GPU");
                    if (gpu_reg != nullptr && !(env_gpu != nullptr && env_gpu[0] == '0')) {
                        auto * bank_from_ptr = (ggml_backend_buffer_t (*)(void *, size_t)) ggml_backend_reg_get_proc_address(gpu_reg, "ggml_backend_cuda_host_bank_buffer_from_ptr");
                        if (bank_from_ptr != nullptr) {
                            region_buf = bank_from_ptr(g.region, g.region_size);
                        }
                    }
                    bank_on_gpu = region_buf != nullptr;
                    if (region_buf == nullptr) {
                        region_buf = ggml_backend_cpu_buffer_from_ptr(g.region, g.region_size);
                    }
                    g.region_buf.reset(region_buf);
                    ggml_backend_buffer_set_usage(g.region_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                }
                for (size_t ip = 0; ip < g.ref.size(); ++ip) {
                    const ggml_tensor * t = g.ref[ip];
                    ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                    bank->nb[2] = g.slot_stride;
                    bank->nb[3] = g.slot_stride*(size_t) (g.n_slots + 1);
                    if (g.region != nullptr) {
                        bank->data   = (char *) g.region + g.part_off[ip];
                        bank->buffer = g.region_buf.get();
                    }
                    ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                    g.banks.push_back(bank);
                }
                devices[g.id].buf_size += g.region_size;
            } else {
                for (const ggml_tensor * t : g.ref) {
                    ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                    GGML_ASSERT(bank->nb[2] == t->nb[2]);
                    ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                    g.banks.push_back(bank);
                }
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    if (use_store && g.region != nullptr) {
                        ggml_backend_view_init(cached);
                    }
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                    if (split_req && use_store && g.region != nullptr) {
                        ggml_tensor * cached_split = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + 1, bank->nb[1], bank->nb[2], 0);
                        ggml_format_name(cached_split, "moe_cache_split.%s", l.experts[ip]->name);
                        ggml_backend_view_init(cached_split);
                        bindings[l.experts[ip]].cached_split = cached_split;
                    }
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (split_req && use_store) {
                    ggml_tensor ** maps[3] = { &l.early_map, &l.late_map, &l.merge_map };
                    const char * names[3] = { "early", "late", "merge" };
                    for (int k = 0; k < 3; ++k) {
                        *maps[k] = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                        ggml_format_name(*maps[k], "moe_cache.%s_map-%d", names[k], il);
                        layer_of[*maps[k]] = il;
                        map_kind[*maps[k]] = k + 1;
                        buf_host_size += GGML_PAD(ggml_nbytes(*maps[k]), alignment_host);
                    }
                    if (g.dummy_const == nullptr) {
                        g.dummy_const = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, 1);
                        ggml_format_name(g.dummy_const, "moe_cache.dummy_slot-%zu", ig);
                        buf_host_size += GGML_PAD(ggml_nbytes(g.dummy_const), alignment_host);
                    }
                    if (split_iota == nullptr) {
                        split_iota = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, n_expert_used);
                        ggml_format_name(split_iota, "moe_cache.split_iota");
                        buf_host_size += GGML_PAD(ggml_nbytes(split_iota), alignment_host);
                    }
                    if (layer_has_remote(il)) {
                        int64_t n_in = 0, n_out = 0;
                        for (const ggml_tensor * t : l.experts) {
                            if (strstr(t->name, "down") != nullptr) {
                                n_out = t->ne[1];
                            } else {
                                n_in = t->ne[0];
                            }
                        }
                        GGML_ASSERT(n_in > 0 && n_out > 0);
                        remote_rows_max = getenv("LLAMA_EXPERT_STORE_REMOTE_ROWS") != nullptr ?
                            std::min(std::max(1, atoi(getenv("LLAMA_EXPERT_STORE_REMOTE_ROWS"))), std::min(n_expert_used, (int32_t) remote_client::KMAX)) :
                            std::min(remote_rows_max, n_expert_used);
                        l.rpost_map   = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                        l.remote_x    = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_F32, n_in);
                        l.remote_rows = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_F32, n_out, remote_rows_max);
                        l.rmerge_map  = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                        ggml_format_name(l.rpost_map,   "moe_cache.rpost_map-%d", il);
                        ggml_format_name(l.remote_x,    "moe_cache.remote_x-%d", il);
                        ggml_format_name(l.remote_rows, "moe_cache.remote_rows-%d", il);
                        ggml_format_name(l.rmerge_map,  "moe_cache.rmerge_map-%d", il);
                        layer_of[l.rpost_map] = il;
                        map_kind[l.rpost_map] = 6;
                        for (ggml_tensor * t : { l.rpost_map, l.remote_x, l.remote_rows, l.rmerge_map }) {
                            buf_host_size += GGML_PAD(ggml_nbytes(t), alignment_host);
                        }
                    }
                }
            }
            if (!use_store) {
                devices[g.id].buf_size += alloc_size(g, g.n_slots);
            }
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx || use_store) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx || use_store) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            // with the banks on the GPU, put the slot maps there too (host memory registered with it): the lookup then
            // runs on the GPU, the whole decode graph is one split, and the CUDA backend calls the hook below with the
            // selected experts before each lookup so the misses are in the bank when the matmul runs
            auto * set_cuda_hook = (bank_on_gpu && gpu_reg != nullptr && !(getenv("LLAMA_EXPERT_STORE_GPU_SLOTS") != nullptr && getenv("LLAMA_EXPERT_STORE_GPU_SLOTS")[0] == '0'))
                ? (void (*)(cuda_host_bank_hook_t, void *)) ggml_backend_reg_get_proc_address(gpu_reg, "ggml_backend_cuda_set_host_bank_hook") : nullptr;
            auto * slots_from_ptr = set_cuda_hook != nullptr
                ? (ggml_backend_buffer_t (*)(void *, size_t)) ggml_backend_reg_get_proc_address(gpu_reg, "ggml_backend_cuda_host_bank_buffer_from_ptr") : nullptr;
            if (slots_from_ptr != nullptr) {
                std::string err;
                slot_map_mem_size = buf_host_size;
                slot_map_mem = store_region_alloc(slot_map_mem_size, err);
                if (slot_map_mem == nullptr) {
                    throw std::runtime_error("expert store: failed to allocate the slot maps: " + err);
                }
                memset(slot_map_mem, 0xff, slot_map_mem_size); // all slots are -1
                buf_host.reset(slots_from_ptr(slot_map_mem, slot_map_mem_size));
                if (!buf_host) {
                    throw std::runtime_error("expert store: failed to register the slot maps with the GPU");
                }
                size_t off = 0;
                for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                    if (ggml_backend_tensor_alloc(buf_host.get(), t, (char *) slot_map_mem + off) != GGML_STATUS_SUCCESS) {
                        throw std::runtime_error("expert store: failed to place a slot map");
                    }
                    off += GGML_PAD(ggml_nbytes(t), alignment_host);
                }
                slots_on_gpu = true;
            } else {
                buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
                if (!buf_host) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
                buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
            }
            if (split_req) {
                // the split needs the lookups on the GPU (the early post and its wait are the host-bank hook's) and
                // no HBM tier (an exchange would move a record the early matmul is reading)
                bool hbm = false;
                for (const group & g : groups) {
                    hbm = hbm || g.lru.n_hbm > 0;
                }
                split = slots_on_gpu && !hbm;
                if (!split) {
                    LLAMA_LOG_WARN("%s: expert store: LLAMA_EXPERT_STORE_SPLIT=1 needs the slot maps on the GPU and no HBM tier, off\n", __func__);
                }
            }
            if (split_req) {
                // every map starts at the zero record (no expert is complete), the merge takes every row from the
                // late output; the zero record itself (slot n_slots, inside the region) is cleared
                for (group & g : groups) {
                    if (g.n_slots == 0) {
                        continue;
                    }
                    std::vector<int32_t> v((size_t) g.ref[0]->ne[2], g.n_slots);
                    for (int32_t il : g.layers) {
                        const layer & l = layers[il];
                        ggml_backend_tensor_set(l.early_map, v.data(), 0, ggml_nbytes(l.early_map));
                        ggml_backend_tensor_set(l.late_map,  v.data(), 0, ggml_nbytes(l.late_map));
                        std::vector<int32_t> m(v.size(), n_expert_used);
                        ggml_backend_tensor_set(l.merge_map, m.data(), 0, ggml_nbytes(l.merge_map));
                    }
                    const int32_t d = g.n_slots;
                    ggml_backend_tensor_set(g.dummy_const, &d, 0, sizeof(d));
                    if (g.region != nullptr) {
                        memset((char *) g.region + (size_t) g.n_slots*g.slot_stride, 0, g.record_size);
                    }
                }
                if (split_iota != nullptr) {
                    std::vector<int32_t> io((size_t) n_expert_used);
                    for (int32_t i = 0; i < n_expert_used; ++i) {
                        io[i] = i;
                    }
                    ggml_backend_tensor_set(split_iota, io.data(), 0, ggml_nbytes(split_iota));
                }
                split_miss.assign((size_t) store.n_expert, 0);
                if (getenv("LLAMA_EXPERT_STORE_SPLIT_LATE") != nullptr) {
                    split_late_max = std::min(std::max(1, atoi(getenv("LLAMA_EXPERT_STORE_SPLIT_LATE"))), n_expert_used);
                }
                // remote experts: every remote merge position is the scratch row, the rows are zero
                int64_t r_in = 0, r_out = 0;
                for (const layer & l : layers) {
                    if (l.rpost_map == nullptr) {
                        continue;
                    }
                    std::vector<int32_t> m((size_t) l.rmerge_map->ne[1], n_expert_used);
                    ggml_backend_tensor_set(l.rmerge_map, m.data(), 0, ggml_nbytes(l.rmerge_map));
                    memset(l.remote_rows->data, 0, ggml_nbytes(l.remote_rows));
                    memset(l.remote_x->data, 0, ggml_nbytes(l.remote_x));
                    r_in = l.remote_x->ne[0];
                    r_out = l.remote_rows->ne[0];
                }
                if (split && !remote_box.empty() && r_in > 0) {
                    remote_shadow = getenv("LLAMA_EXPERT_STORE_REMOTE_SHADOW") != nullptr && getenv("LLAMA_EXPERT_STORE_REMOTE_SHADOW")[0] == '1';
                    // deadline: one value for every box or one per box
                    std::vector<int64_t> dl;
                    if (const char * v = getenv("LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US")) {
                        for (const char * c = v; *c; ) {
                            char * end = nullptr;
                            const long d = strtol(c, &end, 10);
                            if (end == c) {
                                break;
                            }
                            dl.push_back(std::max(0L, d));
                            c = *end == ',' ? end + 1 : end;
                        }
                    }
                    auto env_int = [](const char * k, int def) { return getenv(k) != nullptr ? atoi(getenv(k)) : def; };
                    std::vector<size_t> n_per(remote_hosts.size(), 0);
                    for (uint8_t v : remote_box) {
                        if (v) {
                            n_per[v - 1]++;
                        }
                    }
                    for (size_t bx = 0; bx < remote_hosts.size(); ++bx) {
                        auto rc = std::make_unique<remote_client>();
                        rc->deadline_us = dl.empty() ? 3000 : dl[std::min(bx, dl.size() - 1)];
                        rc->stale_ms    = std::max(50, env_int("LLAMA_EXPERT_STORE_REMOTE_STALE_MS", 1000));
                        rc->trip_after  = std::max(1, env_int("LLAMA_EXPERT_STORE_REMOTE_TRIP", 8));
                        rc->trip_ms     = std::max(0, env_int("LLAMA_EXPERT_STORE_REMOTE_TRIP_MS", 2000));
                        if (!rc->start(remote_hosts[bx], (int) bx, (int) r_in, (int) r_out)) {
                            LLAMA_LOG_WARN("%s: expert store: remote box %zu: '%s' is not ipv4:port, remote experts off\n", __func__, bx, remote_hosts[bx].c_str());
                            remotes.clear();
                            break;
                        }
                        LLAMA_LOG_INFO("%s: expert store: remote box %zu: %zu experts at %s, deadline %lld us\n", __func__,
                            bx, n_per[bx], remote_hosts[bx].c_str(), (long long) rc->deadline_us);
                        remotes.push_back(std::move(rc));
                    }
                    if (remote_on()) {
                        // the round trip is short enough that the Spark's idle states matter: its network IRQ cores
                        // sleep in a deep state (~430 us exit) between requests. Hold /dev/cpu_dma_latency while the
                        // remote experts are on (PM QoS: released when the fd closes, i.e. at exit).
                        // LLAMA_EXPERT_STORE_REMOTE_DMA_LATENCY=<us> (default 0), -1 leaves the idle states alone.
                        const int dma_us = getenv("LLAMA_EXPERT_STORE_REMOTE_DMA_LATENCY") != nullptr ? atoi(getenv("LLAMA_EXPERT_STORE_REMOTE_DMA_LATENCY")) : 0;
                        if (dma_us >= 0) {
                            remote_dma_fd = open("/dev/cpu_dma_latency", O_WRONLY | O_CLOEXEC);
                            const int32_t v = dma_us;
                            if (remote_dma_fd < 0 || write(remote_dma_fd, &v, sizeof v) != (ssize_t) sizeof v) {
                                LLAMA_LOG_WARN("%s: expert store: remote experts: cannot hold /dev/cpu_dma_latency (%s): the round trips "
                                    "pay the CPUs' idle exit latency (needs root or a udev rule giving the server's user write access; "
                                    "in a container, the device mapped in)\n", __func__, strerror(errno));
                                if (remote_dma_fd >= 0) {
                                    close(remote_dma_fd);
                                }
                                remote_dma_fd = -1;
                            } else {
                                LLAMA_LOG_INFO("%s: expert store: remote experts: holding /dev/cpu_dma_latency = %d us\n", __func__, dma_us);
                            }
                        }
                        remote_off.assign(remotes.size(), 0);
                        remote_cnt.assign(remotes.size(), 0);
                        remote_slot.assign(remotes.size(), nullptr);
                        remote_usable.assign(remotes.size(), 0);
                        remote_split_box.assign((size_t) n_expert_used*4, -1);
                        LLAMA_LOG_INFO("%s: expert store: remote experts: %zu boxes, up to %d rows per layer%s\n", __func__,
                            remotes.size(), remote_rows_max, remote_shadow ? ", SHADOW (every expert is still read locally)" : "");
                    }
                } else if (!remote_box.empty()) {
                    LLAMA_LOG_WARN("%s: expert store: remote experts need the split GEMV, off\n", __func__);
                }
            }

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        if (use_store && !model.hparams.no_alloc) {
            // the slot maps are read by the CPU graph, so nothing is ever copied: the fills happen in the CPU graph hook
            ggml_backend_dev_t cpu_dev = ggml_backend_get_device(devices[0].backend);
            auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
            auto * set_hook = (void (*)(ggml_cpu_graph_hook_t, void *)) ggml_backend_reg_get_proc_address(reg, "ggml_cpu_set_graph_hook");
            if (set_hook == nullptr) {
                throw std::runtime_error("expert store: the CPU backend has no graph hook (ggml-cpu is missing patch 0003)");
            }
            set_hook(&impl::graph_hook, this);
            if (slots_on_gpu) {
                auto * set_cuda_hook2 = (void (*)(cuda_host_bank_hook_t, void *)) ggml_backend_reg_get_proc_address(gpu_reg, "ggml_backend_cuda_set_host_bank_hook");
                set_cuda_hook2(&impl::cuda_hook, this);
                const char * env_spf = getenv("LLAMA_EXPERT_STORE_SELF_PREFETCH");
                self_prefetch = env_spf != nullptr && env_spf[0] == '1';
                if (self_prefetch) {
                    LLAMA_LOG_INFO("%s: expert store: self-prefetch on (a layer's misses are read while its shared expert computes)\n", __func__);
                }
                if (split) {
                    filler.start_thread(pool.get());
                    LLAMA_LOG_INFO("%s: expert store: split expert GEMV on (the resident experts' matmuls run while a layer's misses are read; the late matmul has %d rows; the zero record is slot %d of each group)\n", __func__, split_late_max, groups[0].n_slots);
                }
                LLAMA_LOG_INFO("%s: expert store: slot maps registered with the GPU, the lookups run there and the fills in its host hook (LLAMA_EXPERT_STORE_GPU_SLOTS=0 for the CPU split)\n", __func__);
            } else if (getenv("LLAMA_EXPERT_STORE_SELF_PREFETCH") != nullptr && getenv("LLAMA_EXPERT_STORE_SELF_PREFETCH")[0] == '1') {
                LLAMA_LOG_WARN("%s: expert store: self-prefetch needs the slot maps on the GPU (the CPU split fills before it runs), off\n", __func__);
            }
            // predictive prefetch: with the slot maps on the GPU the prediction is posted through the host-bank hook
            // once the real lookup is served; with the CPU split both lookups are in the same CPU graph and the hook
            // hands the prediction to the speculative reader right after the real fill (the same drive-idle window)
            const char * env_pf = getenv("LLAMA_EXPERT_STORE_PREFETCH");
            prefetch = env_pf != nullptr && env_pf[0] == '1';
            if (prefetch) {
                prefetched.assign(model.layers.size()*(size_t) store.n_expert, 0);
                spec_ids.assign(model.layers.size(), {});
                if (getenv("LLAMA_EXPERT_STORE_PREFETCH_DEPTH") != nullptr) {
                    spec_depth = (unsigned) std::max(1, atoi(getenv("LLAMA_EXPERT_STORE_PREFETCH_DEPTH")));
                }
                if (getenv("LLAMA_EXPERT_STORE_PREFETCH_MAX") != nullptr) {
                    spec_max = std::max(1, atoi(getenv("LLAMA_EXPERT_STORE_PREFETCH_MAX")));
                }
                spec_demote    = getenv("LLAMA_EXPERT_STORE_PREFETCH_DEMOTE") != nullptr && getenv("LLAMA_EXPERT_STORE_PREFETCH_DEMOTE")[0] == '1';
                prefetch_input = getenv("LLAMA_EXPERT_STORE_PREFETCH_INPUT") != nullptr ? atoi(getenv("LLAMA_EXPERT_STORE_PREFETCH_INPUT")) : 0;
                spec_uring     = pool->use_uring;
                if (getenv("LLAMA_EXPERT_STORE_PREFETCH_URING") != nullptr) {
                    spec_uring = atoi(getenv("LLAMA_EXPERT_STORE_PREFETCH_URING")) != 0;
                }
#if !defined(LLAMA_HAVE_LIBURING)
                spec_uring = false;
#endif
                spec = std::make_unique<spec_reader>(store.fd, pool_chunk, spec_depth, spec_uring);
                if (!spec->ok) {
                    spec.reset();
                }
                if (spec_ok()) {
                    for (group & g : groups) {
                        if (g.n_slots > 0) {
                            g.spec_state.reset(new std::atomic<int8_t>[g.n_slots]);
                            for (int32_t s = 0; s < g.n_slots; ++s) {
                                g.spec_state[s].store(0);
                            }
                            g.lru.busy = g.spec_state.get();
                        }
                    }
                    LLAMA_LOG_INFO("%s: expert store: predictive prefetch on (the next layer's predicted misses are read by a side reader, %u pieces of %zu KiB in flight at most through %s, %d records per post at most, unused predictions %s, router input %d, posted by the %s)\n", __func__,
                        spec_depth, pool_chunk/1024, spec_uring ? "one io_uring ring" : "pread threads", spec_max, spec_demote ? "demoted" : "kept", prefetch_input,
                        slots_on_gpu ? "GPU's host hook" : "CPU graph hook");
                } else {
                    prefetch = false;
                    LLAMA_LOG_WARN("%s: expert store: predictive prefetch: the speculative reader could not start, off\n", __func__);
                }
            }

            // prefill lookahead: the fill thread reads layer il+1's misses while the GPU computes layer il of a batch.
            // CPU split only: there the next layer's lookup cannot run before the GPU has finished the current one,
            // so the slots of the layer before it are free to be evicted; the split GEMV owns the fill thread.
            if (getenv("LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD") != nullptr && atoi(getenv("LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD")) != 0) {
                if (slots_on_gpu || split) {
                    LLAMA_LOG_WARN("%s: expert store: prefill lookahead needs the CPU split (LLAMA_EXPERT_STORE_GPU_SLOTS=0) and no split GEMV, off\n", __func__);
                } else {
                    prefill_lookahead = true;
                    la_min_ids = (size_t) 8*store.n_expert;
                    if (getenv("LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD_MIN") != nullptr) {
                        la_min_ids = (size_t) std::max(1, atoi(getenv("LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD_MIN")));
                    }
                    la_all.resize(store.n_expert);
                    for (uint32_t x = 0; x < store.n_expert; ++x) {
                        la_all[x] = (int32_t) x;
                    }
                    filler.start_thread(pool.get());
                    LLAMA_LOG_INFO("%s: expert store: prefill lookahead on (batches of %zu ids or more: the next layer's misses are read while the GPU computes the current one)\n", __func__, la_min_ids);
                }
            }

            // large batches over the cache limit: all the experts of a layer are read into a staging, then uploaded.
            // Allocated on first use (ensure_staging): with the bank on the GPU and LLAMA_EXPERT_STORE_MAX_BATCH at the
            // ubatch size it is never needed, and its 6.7 GB are better spent on cache slots.
            staging_size = store.max_record*(size_t) store.n_expert;
            if (max_batch_env > 32 && !bank_on_gpu) {
                LLAMA_LOG_WARN("%s: expert store: LLAMA_EXPERT_STORE_MAX_BATCH=%" PRId64 " needs the bank on the GPU, using 32\n", __func__, max_batch_env);
            }
            max_batch = bank_on_gpu ? max_batch_env : std::min<int64_t>(max_batch_env, 32);
            // with the default cache limit every prompt goes through the staging: allocate it now, at load, rather
            // than mid-run on the first prompt, when the box may have no room left for 6.7 GB (on the Spark that
            // allocation pushed the node into reclaim thrash for 40 minutes). Lazy only when the cache serves the
            // ubatch (LLAMA_EXPERT_STORE_MAX_BATCH raised) and the staging is a fallback.
            if (max_batch <= 32 && !ensure_staging()) {
                throw std::runtime_error("expert store: failed to allocate the prefill staging");
            }
            if (!snapshot_path.empty()) {
                snapshot_restore();
            }
            valve_start();
            LLAMA_LOG_INFO("%s: expert store: host cache below is filled from the store, expert matmuls run on the %s, batches up to %" PRId64 " tokens from the cache (larger ones through a %.2f MiB staging, allocated when one comes)\n",
                __func__, bank_on_gpu ? "GPU from the registered cache" : "CPU", max_batch, staging_size/1024.0/1024.0);
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%), %d of them on the HBM node\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]), g.lru.n_hbm);
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        // the slot-map lookup is pinned to this backend by the graph
        return slots_on_gpu ? gpu_backend : devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        log_stats();
        if (remote_dma_fd >= 0) {
            close(remote_dma_fd);
        }
        valve_stop_thread(); // no release runs from here on
        if (use_store && !devices.empty()) {
            ggml_backend_dev_t cpu_dev = ggml_backend_get_device(devices[0].backend);
            auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
            auto * set_hook = (void (*)(ggml_cpu_graph_hook_t, void *)) ggml_backend_reg_get_proc_address(reg, "ggml_cpu_set_graph_hook");
            if (set_hook != nullptr) {
                set_hook(nullptr, nullptr);
            }
            if (slots_on_gpu) {
                auto * set_cuda_hook = (void (*)(cuda_host_bank_hook_t, void *)) ggml_backend_reg_get_proc_address(gpu_reg, "ggml_backend_cuda_set_host_bank_hook");
                if (set_cuda_hook != nullptr) {
                    set_cuda_hook(nullptr, nullptr); // joins the poller: no fill runs from here on
                }
            }
            snapshot_save("exit");
            if (slots_on_gpu) {
                buf_host.reset(); // unregisters
                store_region_free(slot_map_mem, slot_map_mem_size);
            }
            staging_buf.reset();
            store_region_free(staging, staging_size);
            for (group & g : groups) {
                g.region_buf.reset();
                store_region_free(g.region, g.region_size);
            }
        }
    }

    // --- expert store ---

    static void graph_hook(ggml_cgraph * cgraph, void * ud) {
        static_cast<impl *>(ud)->on_cpu_graph(cgraph);
    }

    // the CUDA backend, right before a slot-map lookup, with the selected experts read back from the device. With the
    // asynchronous hook this runs on the backend's poller thread when the GPU reaches the lookup, which can be after
    // the graph that launched it was rebuilt for the next ubatch, so the backend passes the slot map and the kind of
    // lookup by value rather than the lookup node (see ggml_cuda_bank_mailbox)
    static void cuda_hook(void * ud, const ggml_tensor * slot_map, int kind, const int32_t * ids, int64_t n_ids) {
        impl * self = static_cast<impl *>(ud);
        const auto it = self->layer_of.find(slot_map);
        if (it == self->layer_of.end()) {
            if (kind == 0) {
                // a real lookup the GPU is waiting for: without its fill it would read -1 as an expert slot
                GGML_ABORT("expert store: a bank lookup refers to an unknown slot map (%p, %" PRId64 " ids)", (const void *) slot_map, n_ids);
            }
            return;
        }
        // the memory valve releases slots under the same lock, and never those of the last layers looked up: the
        // split GEMV's early lookup leaves its misses' reads in flight on the fill thread until the late lookup
        std::lock_guard<std::recursive_mutex> lock(self->lru_mutex);
        self->protect_layer(it->second);
        if (!self->map_kind.empty()) {
            // the split GEMV: the early lookup (early map) plans and starts the reads, the real one (late map) waits
            const auto mk = self->map_kind.find(slot_map);
            if (mk != self->map_kind.end()) {
                if (mk->second == 1) {
                    self->fill_early(it->second, ids, (size_t) n_ids);
                } else if (mk->second == 2) {
                    self->fill_late(it->second, ids, (size_t) n_ids);
                } else if (mk->second == 6) {
                    self->remote_post(it->second);
                }
                return; // the merge map (3) needs no fill: the real lookup wrote it
            }
        }
        switch (kind) {
            case 2:  self->fill_self(it->second, ids, (size_t) n_ids);       break; // moe_prefetch_self-*
            case 1:  self->prefetch_decode(it->second, ids, (size_t) n_ids); break; // moe_prefetch-*
            default: self->fill_decode(it->second, ids, (size_t) n_ids);     break;
        }
    }

    // before the CPU graph runs: every slot-map lookup in it gets its layer's selected experts fetched into the bank
    void on_cpu_graph(ggml_cgraph * cgraph) {
        const int n_nodes = ggml_graph_n_nodes(cgraph);
        std::lock_guard<std::recursive_mutex> lock(lru_mutex);
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(cgraph, i);
            if (node->op != GGML_OP_GET_ROWS) {
                continue;
            }
            const auto it = layer_of.find(node->src[0]);
            if (it == layer_of.end()) {
                continue;
            }
            protect_layer(it->second);
            const ggml_tensor * sel  = node->src[1];
            const ggml_tensor * base = sel;
            while (base->view_src != nullptr) {
                base = base->view_src;
            }
            // the ids are an input of this graph (copied in by the scheduler), so they are ready
            if (base->op != GGML_OP_NONE || sel->data == nullptr || sel->type != GGML_TYPE_I32 || !ggml_is_contiguous(sel)) {
                if (!warned_ids) {
                    LLAMA_LOG_WARN("%s: the selected experts of layer %d are not an input of the CPU graph, the expert store cannot fill them\n", __func__, it->second);
                    warned_ids = true;
                }
                continue;
            }
            if (strncmp(node->name, "moe_prefetch_self-", 18) == 0) {
                continue; // never built for the CPU split (self-prefetch needs the GPU hook)
            }
            if (strncmp(node->name, "moe_prefetch-", 13) == 0) {
                // the prediction for the next layer, behind this layer's real lookup in the same CPU graph: the real
                // fill above has run, the drive is idle until the next CPU graph, the speculative reader takes it
                prefetch_decode(it->second, (const int32_t *) sel->data, (size_t) ggml_nelements(sel));
                continue;
            }
            fill_decode(it->second, (const int32_t *) sel->data, (size_t) ggml_nelements(sel));
        }
    }

    // the prefill lookahead's reads must be complete before any other fill plans the LRU or the GPU reads a slot
    void la_finish() {
        if (la_il < 0) {
            return;
        }
        if (filler.busy()) {
            const auto t0 = std::chrono::steady_clock::now();
            filler.wait();
            t_la_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            n_la_waits++;
        }
        if (filler.failed.load()) {
            GGML_ABORT("expert store: prefill lookahead read failed for layer %d: %s", la_il, strerror(errno));
        }
        la_il = -1;
    }

    // after the real fill of layer il of a large batch: plan every expert of layer il+1 (scan resistant, like any
    // prefill fill, and never evicting layer il, which the GPU computes now) and read the misses on the fill thread.
    // The real fill of il+1 finds them as hits once la_finish has waited for the reads.
    void la_post(int32_t il) {
        const int32_t nl = il + 1;
        if (nl >= (int32_t) layers.size() || layers[nl].ig < 0) {
            return;
        }
        const expert_store_entry * e = store.find(nl);
        group & g = groups[layers[nl].ig];
        if (e == nullptr || g.region == nullptr) {
            return;
        }
        size_t n_hit = 0;
        g.lru.skip_layer = il;
        const bool ok = g.lru.plan(nl, la_all.data(), la_all.size(), fills, n_hit, false);
        g.lru.skip_layer = -1;
        if (!ok || fills.empty()) {
            return;
        }
        protect_layer(nl); // the memory valve leaves the slots in flight alone
        jobs.clear();
        for (const auto & f : fills) {
            jobs.push_back({ (char *) g.region + (size_t) f.slot*g.slot_stride, g.record_size, (off_t) (e->base_offset + (size_t) f.expert*g.record_size) });
        }
        filler.submit(store.fd, jobs);
        la_il = nl;
        n_la_posts++;
        n_la_records += fills.size();
    }

    void fill_decode(int32_t il, const int32_t * ids_data, size_t n_ids) {
        lock_memory_once();
        la_finish();
        layer & l = layers[il];
        group & g = groups[l.ig];
        const expert_store_entry * e = store.find(il);
        GGML_ASSERT(e != nullptr && g.region != nullptr);

        const auto t0 = std::chrono::steady_clock::now();
        if (t_dec_last.time_since_epoch().count() != 0) {
            const uint64_t gap = std::chrono::duration_cast<std::chrono::microseconds>(t0 - t_dec_last).count();
            if (gap < 1000000) {
                t_dec_gap_us += gap;
                n_dec_gaps++;
            }
        }
        // LLAMA_EXPERT_STORE_TRACE=<file>: append every fill as int32 [layer, n_ids, ids...] (cache policy studies)
        static FILE * trace = [] {
            const char * p = getenv("LLAMA_EXPERT_STORE_TRACE");
            return p != nullptr ? fopen(p, "wb") : nullptr;
        }();
        if (trace != nullptr) {
            const int32_t hdr[2] = { il, (int32_t) n_ids };
            fwrite(hdr, sizeof(hdr), 1, trace);
            fwrite(ids_data, sizeof(int32_t), n_ids, trace);
            static uint32_t n_traced = 0;
            if (++n_traced % 1024 == 0) {
                fflush(trace);
            }
        }
        size_t n_hit = 0;
        // a batch of more than 32 tokens is prefill: its fills must not evict the generation's working set
        const bool promote = prefill_promote || n_expert_used <= 0 || n_ids <= (size_t) 32*n_expert_used;
        fill_real(il, ids_data, n_ids, promote, n_hit);
        if (g.lru.n_hbm > 0 && promote && !prefetch) {
            hbm_tier(il, g);
        }
        const auto t2 = std::chrono::steady_clock::now();
        last_fill_il = il;
        t_dec_plan_us += t_fill_plan_us;
        t_dec_io_us   += t_fill_io_us;
        GGML_UNUSED(e);
        t_dec_last = t2;
        stats & st = n_ids <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += fills.size()*g.record_size;
        stats_store_decode.hits   += n_hit;
        stats_store_decode.misses += fills.size();
        stats_store_decode.bytes  += fills.size()*g.record_size;
        if (!promote) {
            // prefill batches through the bank: one line per 4 batches of fills (a ubatch is one fill per MoE layer),
            // with what the lookahead read ahead and how long the fills waited for it
            if (n_pf_fills == 0) {
                t_pf_first = t0;
            }
            n_pf_fills++;
            n_pf_records += fills.size();
            n_pf_hits    += n_hit;
            n_pf_lookups += n_hit + fills.size();
            t_pf_io_us   += t_fill_io_us;
            t_pf_last     = t2;
            if (n_pf_fills % (4*std::max<size_t>(1, store.by_layer.size())) == 0) {
                const double wall = std::chrono::duration_cast<std::chrono::microseconds>(t_pf_last - t_pf_first).count()/1e6;
                LLAMA_LOG_INFO("%s: expert store: prefill: %llu batch fills in %.1f s, hit rate %.1f%%, %.1f records read per fill in the fill (%.1f ms), %.1f by the lookahead per post (%llu posts, %.1f ms waited per post)\n", __func__,
                    (unsigned long long) n_pf_fills, wall, n_pf_lookups ? 100.0*n_pf_hits/n_pf_lookups : 0.0,
                    (double) n_pf_records/n_pf_fills, t_pf_io_us/1e3/n_pf_fills,
                    n_la_posts ? (double) n_la_records/n_la_posts : 0.0, (unsigned long long) n_la_posts,
                    n_la_posts ? t_la_wait_us/1e3/n_la_posts : 0.0);
            }
            if (prefill_lookahead && n_ids >= la_min_ids) {
                la_post(il);
            }
        }
        if (snapshot_every_s > 0 && !snapshot_path.empty() &&
                std::chrono::duration_cast<std::chrono::seconds>(t2 - t_snapshot_last).count() >= snapshot_every_s) {
            t_snapshot_last = t2;
            snapshot_save("periodic");
        }
        if (++n_decode_fills % 4096 == 0) {
            const size_t n = stats_store_decode.hits + stats_store_decode.misses;
            LLAMA_LOG_INFO("%s: expert store: %zu lookups, hit rate %.1f%%, %.1f GB read for decode, %.1f GB for prefill\n", __func__,
                n, n ? 100.0*stats_store_decode.hits/n : 0.0, stats_store_decode.bytes/1e9, store_prefill_bytes/1e9);
            log_decode_timing(__func__);
        }
    }

    // HBM tier, after a decode fill of layer il: count the uses of its selected experts (lru.uniq, left by the plan),
    // and move the most used ones that sit in host slots into the HBM slots, in exchange for the least used expert
    // there. Safe here: the GPU is waiting for this fill, nothing reads the bank until the slot map is published.
    void hbm_tier(int32_t il, group & g) {
        moe_cache_lru & lru = g.lru;
        const size_t base = (size_t) il*lru.n_expert;
        for (int32_t e : lru.uniq) {
            uint16_t & u = lru.uses[base + e];
            if (u < UINT16_MAX) {
                u++;
            }
        }
        // halve every count every hbm_decay tokens (one fill per layer and token): the tier follows the
        // working set of the current conversation without forgetting it in a few tokens
        if (++n_hbm_fills % ((uint64_t) hbm_decay*layers.size()) == 0) {
            for (group & gg : groups) {
                for (uint16_t & u : gg.lru.uses) {
                    u >>= 1;
                }
                gg.hbm_min_slot = -1;
            }
        }
        if (g.hbm_min_slot < 0 || n_hbm_fills % 16 == 0) {
            hbm_refresh_min(g);
        }
        const auto t0 = std::chrono::steady_clock::now();
        int n_promoted = 0;
        for (int32_t e : lru.uniq) {
            const int32_t s = lru.slot_map[il][e];
            if (s < lru.n_hbm || n_promoted >= hbm_max_promote || g.hbm_min_slot < 0) {
                continue;
            }
            const uint16_t u = lru.uses[base + e];
            if (u < g.hbm_min_uses + std::max<int>(2, g.hbm_min_uses/4)) {
                continue;
            }
            const int32_t h = g.hbm_min_slot;
            if (g.spec_state && g.spec_state[h].load(std::memory_order_acquire) != 0) {
                continue; // the speculative reader is writing it
            }
            jobs.clear();
            jobs.push_back({ (char *) g.region + (size_t) s*g.slot_stride, g.record_size, 0, (char *) g.region + (size_t) h*g.slot_stride });
            pool->run(store.fd, jobs);
            lru.exchange(s, h);
            n_promoted++;
            n_promotions++;
            hbm_refresh_min(g);
        }
        if (n_promoted > 0) {
            t_promote_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        }
        for (int32_t e : lru.uniq) {
            n_lookups++;
            n_hbm_lookups += lru.slot_map[il][e] < lru.n_hbm;
        }
    }

    void hbm_refresh_min(group & g) {
        const moe_cache_lru & lru = g.lru;
        g.hbm_min_slot = -1;
        g.hbm_min_uses = UINT16_MAX;
        for (int32_t h = 0; h < lru.n_hbm; ++h) {
            const int32_t k = lru.key_of[h];
            // an empty HBM slot is filled by the next miss that evicts it, never by an exchange (n_filled)
            if (k >= 0 && lru.uses[k] < g.hbm_min_uses) {
                g.hbm_min_uses = lru.uses[k];
                g.hbm_min_slot = h;
            }
        }
    }

    // the real reads of layer il's selected experts, for the real lookup and the self-prefetch post: the speculative
    // reader is preempted and its unstarted records handed back, the LRU is planned, the misses are read through the
    // pool, then the predicted records of this layer still in flight are waited for (never re-read, never half read)
    // plan a fill: give every selected expert a slot (fill_plan), read the misses and adopt the records the
    // speculative reader still writes (fill_io). fill_real does both; the split GEMV runs the reads on the fill
    // thread between its early and real lookup instead
    void fill_plan(int32_t il, const int32_t * ids_data, size_t n_ids, bool promote, size_t & n_hit) {
        layer & l = layers[il];
        group & g = groups[l.ig];
        const expert_store_entry * e = store.find(il);
        GGML_ASSERT(e != nullptr && g.region != nullptr);
        spec_reclaim();
        if (!g.lru.plan(il, ids_data, n_ids, fills, n_hit, promote)) {
            GGML_ABORT("the expert store cache is too small for the experts selected in layer %d", il);
        }
        // every selected expert must now have a slot: the lookup that follows reads the slot map on the GPU and a
        // -1 or a stray index would be used as an expert offset
        for (size_t i = 0; i < n_ids; ++i) {
            const int32_t s = g.lru.slot_map[il][ids_data[i]];
            if (s < 0 || s >= g.n_slots) {
                GGML_ABORT("expert store: layer %d expert %d has slot %d after the plan (n_slots %d, %zu ids, promote %d)", il, ids_data[i], s, g.n_slots, n_ids, (int) promote);
            }
        }
        if (prefetch) {
            // hits that a prefetch read: the prediction paid off
            for (size_t i = 0; i < n_ids; ++i) {
                uint8_t & m = prefetched[(size_t) il*store.n_expert + ids_data[i]];
                if (m) {
                    m = 0;
                    n_prefetch_used++;
                }
            }
        }
        jobs.clear();
        for (const auto & f : fills) {
            jobs.push_back({ (char *) g.region + (size_t) f.slot*g.slot_stride, g.record_size, (off_t) (e->base_offset + (size_t) f.expert*g.record_size) });
        }
    }

    // after the reads of a plan: adopt the speculative records among the selected experts, park the unused
    // predictions of the layer if asked
    void fill_adopt(int32_t il) {
        group & g = groups[layers[il].ig];
        spec_adopt(il, g);
        if (prefetch && spec_demote) {
            // predictions of this layer that were not selected: one-shot, park them at the eviction end
            for (int32_t x : spec_ids[il]) {
                uint8_t & m = prefetched[(size_t) il*store.n_expert + x];
                const int32_t s = g.lru.slot_map[il][x];
                if (m && s >= 0 && g.lru.key_of[s] == (int32_t) (il*(int32_t) store.n_expert + x) && g.spec_state[s].load(std::memory_order_acquire) == 0) {
                    g.lru.demote(s);
                }
                m = 0;
            }
            spec_ids[il].clear();
        }
    }

    void fill_real(int32_t il, const int32_t * ids_data, size_t n_ids, bool promote, size_t & n_hit) {
        const auto t0 = std::chrono::steady_clock::now();
        fill_plan(il, ids_data, n_ids, promote, n_hit);
        const auto t1 = std::chrono::steady_clock::now();
        if (!pool->run(store.fd, jobs)) {
            GGML_ABORT("expert store: read failed for layer %d: %s", il, strerror(errno));
        }
        fill_adopt(il);
        const auto t2 = std::chrono::steady_clock::now();
        t_fill_plan_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        t_fill_io_us   = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    }

    // the early lookup of the split GEMV ("moe_early-<il>", single-token decode): plan the fill, publish which
    // selected experts are complete in the bank (early map, by expert: their slot, the others the zero record), the
    // misses' slots and merge rows (late and merge maps, by position; the GPU reads them only after the real lookup),
    // start the reads on the fill thread and return, so the GPU runs the resident experts' matmuls while the misses
    // are read. Experts the speculative reader is still writing count as misses and are adopted by the real lookup.
    // More misses than the late matmul has rows: the reads run here, synchronously, and every expert is complete.
    void fill_early(int32_t il, const int32_t * ids_data, size_t n_ids) {
        lock_memory_once();
        layer & l = layers[il];
        group & g = groups[l.ig];
        GGML_ASSERT(split && n_ids == (size_t) n_expert_used);
        const auto t0 = std::chrono::steady_clock::now();
        if (split_il >= 0) {
            // an early lookup without its real one (should not happen): finish its reads first
            filler.wait();
            fill_adopt(split_il);
            split_il = -1;
            n_split_fallback++;
        }
        // remote experts: the selected experts of a usable box's set that are not in the bank go to that box instead
        // of the plan, as many as the late matmul can still take if the replies miss their deadlines (local misses
        // first: those are read anyway), best router scores first, grouped by box
        remote_il = -1;
        remote_ids.clear();
        remote_pos.clear();
        const int32_t * plan_ids = ids_data;
        size_t n_plan = n_ids;
        if (remote_on() && l.rpost_map != nullptr) {
            int32_t * rm = (int32_t *) l.rmerge_map->data;
            for (int r = 0; r < remote_rows_max; ++r) {
                rm[r] = n_expert_used; // the scratch row (the GPU finished this layer's previous token long ago)
            }
            const size_t nb = remotes.size();
            bool any = false;
            for (size_t bx = 0; bx < nb; ++bx) {
                remote_cnt[bx] = 0;
                remote_slot[bx] = nullptr;
                remote_usable[bx] = remotes[bx]->next_free() != nullptr;
                any = any || remote_usable[bx];
            }
            if (any) {
                auto box_of = [&](int32_t x) -> int {
                    if (g.lru.slot_map[il][x] >= 0) {
                        return -1;
                    }
                    const int bx = (int) remote_box[(size_t) il*store.n_expert + x] - 1;
                    return bx >= 0 && remote_usable[bx] ? bx : -1;
                };
                size_t n_local_miss = 0;
                for (size_t i = 0; i < n_ids; ++i) {
                    const int32_t x = ids_data[i];
                    const int32_t sl = g.lru.slot_map[il][x];
                    if (box_of(x) >= 0) {
                        continue;
                    }
                    if (sl < 0 || (spec_ok() && g.spec_state[sl].load(std::memory_order_acquire) != 0)) {
                        n_local_miss++;
                    }
                }
                const size_t cap = std::min((size_t) remote_rows_max, n_local_miss < (size_t) split_late_max ? (size_t) split_late_max - n_local_miss : 0);
                remote_local.clear();
                size_t taken = 0;
                for (size_t i = 0; i < n_ids; ++i) {
                    const int32_t x = ids_data[i];
                    const int bx = taken < cap ? box_of(x) : -1;
                    if (bx >= 0) {
                        remote_split_box[i] = bx;
                        remote_cnt[bx]++;
                        taken++;
                    } else {
                        remote_split_box[i] = -1;
                        remote_local.push_back(x);
                    }
                }
                if (taken > 0) {
                    // grouped by box: box bx's candidates are remote_ids[remote_off[bx] .. + remote_cnt[bx]]
                    int32_t off = 0;
                    for (size_t bx = 0; bx < nb; ++bx) {
                        remote_off[bx] = off;
                        off += remote_cnt[bx];
                    }
                    remote_ids.assign(taken, -1);
                    remote_pos.assign(taken, -1);
                    std::vector<int32_t> & fill_at = remote_fill_at;
                    fill_at.assign(remote_off.begin(), remote_off.end());
                    for (size_t i = 0; i < n_ids; ++i) {
                        const int bx = remote_split_box[i];
                        if (bx >= 0) {
                            remote_ids[fill_at[bx]] = ids_data[i];
                            remote_pos[fill_at[bx]] = (int32_t) i;
                            fill_at[bx]++;
                        }
                    }
                    remote_il = il;
                    n_remote_layers++;
                    if (!remote_shadow) {
                        plan_ids = remote_local.data();
                        n_plan = remote_local.size();
                    }
                }
            }
        }
        size_t n_hit = 0;
        fill_plan(il, plan_ids, n_plan, true, n_hit);
        split_ids.assign(plan_ids, plan_ids + n_plan);
        // misses: the records read now, and the speculative reader's records still in flight (adopted at the real lookup)
        for (const auto & f : fills) {
            split_miss[f.expert] = 1;
        }
        const size_t n_remote = remote_shadow ? 0 : remote_ids.size();
        for (size_t r = 0; r < n_remote; ++r) {
            split_miss[remote_ids[r]] = 2; // remote: no slot, the zero record at the early lookup, not in the late matmul
        }
        const size_t late_cap = (size_t) split_late_max - n_remote;
        if (spec_ok()) {
            for (size_t i = 0; i < n_ids; ++i) {
                const int32_t sl = g.lru.slot_map[il][ids_data[i]];
                if (sl >= 0 && split_miss[ids_data[i]] != 2 && g.spec_state[sl].load(std::memory_order_acquire) != 0) {
                    split_miss[ids_data[i]] = 1;
                }
            }
        }
        size_t n_miss = 0;
        for (size_t i = 0; i < n_ids; ++i) {
            n_miss += split_miss[ids_data[i]] == 1;
        }
        if (n_miss > late_cap) {
            // too many for the late matmul: first wait for the speculative reader's records (nearly done, usually)
            n_split_sync++;
            if (spec_ok()) {
                spec_adopt(il, g);
                for (size_t i = 0; i < n_ids; ++i) {
                    const int32_t x = ids_data[i];
                    if (split_miss[x] == 1 && g.lru.slot_map[il][x] >= 0 && g.spec_state[g.lru.slot_map[il][x]].load(std::memory_order_acquire) == 0) {
                        // complete now, unless it is one of this fill's own reads
                        bool own = false;
                        for (const auto & f : fills) {
                            own = own || f.expert == x;
                        }
                        if (!own) {
                            split_miss[x] = 0;
                            n_miss--;
                        }
                    }
                }
            }
            // then read the excess misses here, synchronously; the rest go to the fill thread
            if (n_miss > late_cap) {
                const size_t n_sync = n_miss - late_cap;
                GGML_ASSERT(n_sync <= jobs.size());
                std::vector<io_pool::job> sync_jobs(jobs.begin(), jobs.begin() + n_sync);
                if (!pool->run(store.fd, sync_jobs)) {
                    GGML_ABORT("expert store: read failed for layer %d: %s", il, strerror(errno));
                }
                for (size_t k = 0; k < n_sync; ++k) {
                    split_miss[fills[k].expert] = 0;
                }
                jobs.erase(jobs.begin(), jobs.begin() + n_sync);
                n_split_sync_records += n_sync;
                n_miss = late_cap;
            }
        }
        const bool sync = false;
        int32_t * early = (int32_t *) l.early_map->data;
        int32_t * late  = (int32_t *) l.late_map->data;
        int32_t * merge = (int32_t *) l.merge_map->data;
        size_t j = 0;
        for (size_t i = 0; i < n_ids; ++i) {
            const int32_t x = ids_data[i];
            const int32_t s = g.lru.slot_map[il][x];
            if (split_miss[x] == 2) {
                early[x]  = g.n_slots; // remote
            } else if (split_miss[x]) {
                early[x]  = g.n_slots;
                late[j]   = s;
                merge[j]  = (int32_t) i;
                j++;
            } else {
                early[x]  = s;
            }
            split_miss[x] = 0;
        }
        split_late_used = j;
        for (; j < (size_t) split_late_max; ++j) {
            late[j]  = g.n_slots;
            merge[j] = n_expert_used;
        }
        n_split_early_hits += n_ids - n_miss - n_remote;
        n_split_late += n_miss;
        n_split_calls++;
        if (!sync && !jobs.empty()) {
            filler.submit(store.fd, jobs);
        }
        split_il = il;
        t_split_plan_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        // the poller publishes the maps (fence + done flag) when this returns
    }

    // the real lookup of the split GEMV ("ffn_moe_slots" on the late map, ids = the positions): wait for the early
    // lookup's reads and adopt the speculative records; the maps were written by the early lookup
    void fill_late(int32_t il, const int32_t * ids_data, size_t n_ids) {
        layer & l = layers[il];
        group & g = groups[l.ig];
        GGML_ASSERT(split);
        if (split_il != il) {
            GGML_ABORT("expert store: the split's real lookup of layer %d came without its early lookup (%d)", il, split_il);
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (filler.busy()) {
            filler.wait();
            if (filler.failed.load()) {
                GGML_ABORT("expert store: read failed for layer %d", il);
            }
            fill_adopt(il);
        } else if (!jobs.empty() && n_split_sync == 0) {
            // nothing in flight and nothing was read synchronously: the fill thread finished before we looked
            fill_adopt(il);
        } else {
            fill_adopt(il);
        }
        split_il = -1;
        const size_t n_hit = split_ids.size() - fills.size();
        const auto t2 = std::chrono::steady_clock::now();
        t_split_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();
        last_fill_il = il;
        t_dec_last = t2;
        stats_small.hits   += n_hit;
        stats_small.misses += fills.size();
        stats_small.bytes  += fills.size()*g.record_size;
        stats_store_decode.hits   += n_hit;
        stats_store_decode.misses += fills.size();
        stats_store_decode.bytes  += fills.size()*g.record_size;
        if (remote_il == il) {
            remote_resolve(il, l, g);
        }
        remote_il = -1;
        if (snapshot_every_s > 0 && !snapshot_path.empty() &&
                std::chrono::duration_cast<std::chrono::seconds>(t2 - t_snapshot_last).count() >= snapshot_every_s) {
            t_snapshot_last = t2;
            snapshot_save("periodic");
        }
        if (++n_decode_fills % 4096 == 0) {
            const size_t n = stats_store_decode.hits + stats_store_decode.misses;
            LLAMA_LOG_INFO("%s: expert store: %zu lookups, hit rate %.1f%%, %.1f GB read for decode, %.1f GB for prefill\n", __func__,
                n, n ? 100.0*stats_store_decode.hits/n : 0.0, stats_store_decode.bytes/1e9, store_prefill_bytes/1e9);
            log_decode_timing(__func__);
        }
        GGML_UNUSED(ids_data);
        GGML_UNUSED(n_ids);
    }

    // remote experts, "moe_rpost-<il>": the GPU has copied the experts' input into the layer's remote input; send it
    // with each box's candidates (the slots fill_early found free are still free: only this thread posts)
    void remote_post(int32_t il) {
        if (!remote_on() || remote_il != il || remote_ids.empty()) {
            return;
        }
        const layer & l = layers[il];
        std::atomic_thread_fence(std::memory_order_acquire);
        for (size_t bx = 0; bx < remotes.size(); ++bx) {
            if (remote_cnt[bx] == 0 || remote_slot[bx] != nullptr) {
                continue;
            }
            remote_client & rc = *remotes[bx];
            remote_client::slot * s = rc.next_free();
            if (s == nullptr) {
                rc.n_unsent++; // the link dropped since the early lookup: the real lookup reads them
                continue;
            }
            const int k = remote_cnt[bx];
            s->hdr[0] = k;
            for (int r = 0; r < remote_client::KMAX; ++r) {
                s->hdr[1 + r] = r < k ? il*(int32_t) store.n_expert + remote_ids[remote_off[bx] + r] : 0;
            }
            memcpy(s->x.data(), l.remote_x->data, ggml_nbytes(l.remote_x));
            rc.post(s);
            remote_slot[bx] = s;
            rc.n_posts++;
            rc.n_rows += k;
        }
    }

    // a reply row is used only if it is all numbers: a server answers NaN rows for ids it does not hold
    static bool remote_row_ok(const float * r, size_t n) {
        uint32_t bad = 0;
        for (size_t i = 0; i < n; ++i) {
            bad |= (uint32_t) (r[i] != r[i]);
        }
        return bad == 0;
    }

    // remote experts, at the real lookup (after the local reads): wait for each box's reply until its deadline, place
    // the rows that arrived (remote rows + merge positions), then read every missing expert here into the late
    // matmul's free rows
    void remote_resolve(int32_t il, layer & l, group & g) {
        const auto t0 = std::chrono::steady_clock::now();
        const size_t row = (size_t) l.remote_rows->ne[0];
        int32_t * rm = (int32_t *) l.rmerge_map->data;
        remote_miss_ids.clear();
        remote_miss_pos.clear();
        for (size_t bx = 0; bx < remotes.size(); ++bx) {
            const int k = remote_cnt[bx];
            if (k == 0) {
                continue;
            }
            remote_client & rc = *remotes[bx];
            remote_client::slot * s = remote_slot[bx];
            remote_slot[bx] = nullptr;
            const int off = remote_off[bx];
            int n_ok = 0; // rows of this box placed
            if (s != nullptr) {
                const auto deadline = s->t_post + std::chrono::microseconds(rc.deadline_us);
                int st = s->state.load(std::memory_order_acquire);
                if (!remote_shadow) {
                    while ((st == remote_client::QUEUED || st == remote_client::SENT) && std::chrono::steady_clock::now() < deadline) {
                        st = s->state.load(std::memory_order_acquire);
                    }
                }
                // the rows complete now: all of them, or the first ones of a reply still streaming in (the net thread
                // only writes past them from here on)
                bool done = st == remote_client::DONE;
                int n_rows = done ? k : s->rows_done.load(std::memory_order_acquire);
                if (!done) {
                    if (rc.abandon(s)) {
                        done = true;
                        n_rows = k;
                    } else if (s->state.load(std::memory_order_acquire) == remote_client::FREE) {
                        n_rows = 0; // the link broke and the slot was reset: read them all locally
                    }
                }
                n_rows = std::min(n_rows, k);
                if (done) {
                    const auto rtt = std::chrono::duration_cast<std::chrono::microseconds>(s->t_done - s->t_post).count();
                    rc.rtt_us[rc.rtt_next++ % rc.rtt_us.size()] = (uint32_t) std::max<int64_t>(0, rtt);
                }
                if (remote_shadow) {
                    rc.n_shadow_ontime += done;
                } else {
                    for (int r = 0; r < n_rows; ++r) {
                        const float * src = s->rows.data() + (size_t) r*row;
                        if (!remote_row_ok(src, row)) {
                            rc.n_nan_rows++;
                            break; // this row and the rest are read locally
                        }
                        memcpy((float *) l.remote_rows->data + (size_t) (off + r)*row, src, row*sizeof(float));
                        rm[off + r] = remote_pos[off + r];
                        n_ok++;
                    }
                    rc.n_hits += n_ok;
                    rc.n_partial += !done && n_ok > 0;
                }
                if (done) {
                    s->state.store(remote_client::FREE, std::memory_order_release);
                }
            }
            if (remote_shadow) {
                continue;
            }
            if (n_ok == k) {
                rc.misses_in_row = 0;
            } else {
                rc.misses_in_row++;
                rc.n_late++;
                rc.n_late_rows += k - n_ok;
                for (int r = n_ok; r < k; ++r) {
                    remote_miss_ids.push_back(remote_ids[off + r]);
                    remote_miss_pos.push_back(remote_pos[off + r]);
                }
            }
        }
        if (!remote_miss_ids.empty()) {
            // missed the deadline, never sent, a partial or a refused reply: the same experts from the store, into the
            // late matmul's free rows (fill_early kept local misses + remote candidates within its rows)
            size_t n_hit = 0;
            fill_plan(il, remote_miss_ids.data(), remote_miss_ids.size(), true, n_hit);
            if (!jobs.empty() && !pool->run(store.fd, jobs)) {
                GGML_ABORT("expert store: read failed for layer %d: %s", il, strerror(errno));
            }
            fill_adopt(il);
            int32_t * late  = (int32_t *) l.late_map->data;
            int32_t * merge = (int32_t *) l.merge_map->data;
            size_t j = split_late_used;
            for (size_t r = 0; r < remote_miss_ids.size(); ++r) {
                GGML_ASSERT(j < (size_t) split_late_max);
                late[j]  = g.lru.slot_map[il][remote_miss_ids[r]];
                merge[j] = remote_miss_pos[r];
                j++;
            }
            stats_small.misses        += fills.size();
            stats_small.hits          += n_hit;
            stats_small.bytes         += fills.size()*g.record_size;
            stats_store_decode.misses += fills.size();
            stats_store_decode.hits   += n_hit;
            stats_store_decode.bytes  += fills.size()*g.record_size;
            n_remote_fallback_layers++;
        }
        t_remote_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    }

    void remote_log(const char * func) const {
        if (!remote_on()) {
            return;
        }
        LLAMA_LOG_INFO("%s: expert store: remote experts%s: %llu layers with candidates, %llu of them read some locally; wait at the real lookup %.3f ms per layer with candidates\n", func,
            remote_shadow ? " (SHADOW)" : "", (unsigned long long) n_remote_layers, (unsigned long long) n_remote_fallback_layers,
            n_remote_layers ? t_remote_wait_us/1e3/n_remote_layers : 0.0);
        for (const auto & p : remotes) {
            const remote_client & rc = *p;
            std::vector<uint32_t> v(rc.rtt_us.begin(), rc.rtt_us.begin() + std::min(rc.rtt_next, rc.rtt_us.size()));
            std::sort(v.begin(), v.end());
            auto pct = [&](double q) { return v.empty() ? 0.0 : v[(size_t) (q*(v.size() - 1))]/1e3; };
            LLAMA_LOG_INFO("%s: expert store: remote box %d (%s:%d, %s): %llu posts, %.2f rows per post, %llu rows placed, %llu requests short (%llu rows read locally, %llu partial replies, %llu NaN rows), %llu unsent, %llu breaker trips%s; RTT p50 %.2f p90 %.2f p99 %.2f ms (%zu replies); %llu connects, %llu drops (%llu stale)\n", func,
                rc.id, rc.host.c_str(), rc.port, rc.connected.load() ? "up" : "DOWN",
                (unsigned long long) rc.n_posts, rc.n_posts ? (double) rc.n_rows/rc.n_posts : 0.0, (unsigned long long) rc.n_hits,
                (unsigned long long) rc.n_late, (unsigned long long) rc.n_late_rows, (unsigned long long) rc.n_partial, (unsigned long long) rc.n_nan_rows,
                (unsigned long long) rc.n_unsent, (unsigned long long) rc.n_trips,
                remote_shadow ? (", " + std::to_string(rc.n_shadow_ontime) + " replies by the real lookup").c_str() : "",
                pct(0.5), pct(0.9), pct(0.99), v.size(),
                (unsigned long long) rc.n_connects.load(), (unsigned long long) rc.n_disconnects.load(), (unsigned long long) rc.n_stale.load());
        }
    }

    // the self-prefetch post: layer il's real experts, read now while the GPU computes the shared expert; the real
    // lookup that follows plans the same ids and finds them resident
    void fill_self(int32_t il, const int32_t * ids_data, size_t n_ids) {
        la_finish();
        if (!self_prefetch || n_ids > (size_t) 4*n_expert_used) {
            return;
        }
        const expert_store_entry * e = store.find(il);
        if (e == nullptr || groups[layers[il].ig].region == nullptr) {
            return;
        }
        size_t n_hit = 0;
        fill_real(il, ids_data, n_ids, true, n_hit);
        n_self_calls++;
        n_self_reads += fills.size();
        t_self_io_us += t_fill_io_us;
    }

    // stop the speculative reader from starting records and take back the ones it has not started: their slots
    // are emptied and go to the eviction end, the next plan reuses them first
    void spec_reclaim() {
        if (!spec) {
            return;
        }
        spec->preempt_take(spec_recs);
        for (const spec_reader::rec & r : spec_recs) {
            moe_cache_lru & lru = groups[r.ig].lru;
            if (lru.key_of[r.slot] != r.key) {
                continue;
            }
            lru.slot_map[r.key / lru.n_expert][r.key % lru.n_expert] = -1;
            lru.key_of[r.slot] = -1;
            lru.n_filled--;
            lru.demote(r.slot);
            prefetched[r.key] = 0;
            n_spec_reclaimed++;
        }
        spec_recs.clear();
    }

    // wait for the selected experts (lru.uniq, left by the plan) whose records the speculative reader is still writing
    void spec_adopt(int32_t il, group & g) {
        if (!spec_ok()) {
            return;
        }
        bool waited = false;
        std::chrono::steady_clock::time_point t0;
        for (int32_t x : g.lru.uniq) {
            const int32_t s = g.lru.slot_map[il][x];
            std::atomic<int8_t> & st = g.spec_state[s];
            if (st.load(std::memory_order_acquire) == 0) {
                continue;
            }
            if (!waited) {
                waited = true;
                t0 = std::chrono::steady_clock::now();
            }
            n_spec_adopted++;
            while (st.load(std::memory_order_acquire) != 0) {
                io_pool::cpu_relax();
            }
        }
        if (waited) {
            n_spec_waits++;
            t_spec_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        }
    }

    // predicted experts of layer il (the router of il applied to the previous layer): plan them into the LRU on this
    // thread (which owns it) and hand the misses to the speculative reader, best router scores first. Returns at once:
    // the poller must be free for the real posts. The previous layer's experts were promoted to the LRU tail by its
    // real fill and are far from the eviction end, so the reads cannot land in a slot the GPU is reading.
    void prefetch_decode(int32_t il, const int32_t * ids_data, size_t n_ids) {
        la_finish();
        if (!prefetch || !spec_ok() || n_ids > (size_t) 4*n_expert_used) {
            return;
        }
        layer & l = layers[il];
        group & g = groups[l.ig];
        const expert_store_entry * e = store.find(il);
        if (e == nullptr || g.region == nullptr) {
            return;
        }
        spec_reclaim(); // a prediction not started by now is stale
        if (remote_on() && !remote_shadow && l.rpost_map != nullptr) {
            // the remote experts are not read ahead: saving the drive those reads is the point
            remote_pf.clear();
            for (size_t i = 0; i < n_ids; ++i) {
                if (!(g.lru.slot_map[il][ids_data[i]] < 0 && remote_box_of(il, ids_data[i]) >= 0)) {
                    remote_pf.push_back(ids_data[i]);
                }
            }
            ids_data = remote_pf.data();
            n_ids = remote_pf.size();
        }
        size_t n_hit = 0;
        if (!g.lru.plan(il, ids_data, n_ids, fills, n_hit, true)) {
            return;
        }
        n_prefetch_calls++;
        if (fills.empty()) {
            return;
        }
        // the ids come best first per token: read the misses in that order, and only the first spec_max
        auto rank = [&](int32_t x) {
            for (size_t i = 0; i < n_ids; ++i) {
                if (ids_data[i] == x) {
                    return i;
                }
            }
            return n_ids;
        };
        std::stable_sort(fills.begin(), fills.end(), [&](const moe_cache_lru::fill & a, const moe_cache_lru::fill & b) { return rank(a.expert) < rank(b.expert); });
        while (fills.size() > (size_t) spec_max) {
            const moe_cache_lru::fill & f = fills.back();
            g.lru.slot_map[il][f.expert] = -1;
            g.lru.key_of[f.slot] = -1;
            g.lru.n_filled--;
            g.lru.demote(f.slot);
            fills.pop_back();
        }
        spec_recs.clear();
        std::vector<int32_t> & sid = spec_ids[il];
        if (!spec_demote) {
            sid.clear();
        }
        for (const auto & f : fills) {
            const int32_t key = il*store.n_expert + f.expert;
            spec_recs.push_back({ l.ig, f.slot, key, (char *) g.region + (size_t) f.slot*g.slot_stride, g.record_size,
                (off_t) (e->base_offset + (size_t) f.expert*g.record_size), &g.spec_state[f.slot] });
            prefetched[key] = 1;
            sid.push_back(f.expert);
        }
        spec->enqueue(spec_recs);
        spec_recs.clear();
        n_spec_posts++;
        n_prefetch_reads += fills.size();
    }

    // --- cache snapshot: which experts are resident, their recency and use counts (the bytes are re-read from the store) ---
    //
    // file: "K3SN" u32 version, u64 store size, u32 n_expert, u32 n_layer, u32 n_groups, then per group
    // u32 n_layers u64 record_size u32 n_entries and n_entries x (i32 key, u16 uses) in recency order (least recent first)

    void snapshot_save(const char * why) {
        if (snapshot_path.empty() || !use_store || groups.empty() || groups[0].region == nullptr) {
            return;
        }
        struct stat st = {};
        if (fstat(store.fd, &st) != 0) {
            return;
        }
        std::vector<uint8_t> buf;
        auto put = [&](const void * p, size_t n) { buf.insert(buf.end(), (const uint8_t *) p, (const uint8_t *) p + n); };
        auto put32 = [&](uint32_t v) { put(&v, 4); };
        auto put64 = [&](uint64_t v) { put(&v, 8); };
        put("K3SN", 4);
        put32(1);
        put64((uint64_t) st.st_size);
        put32(store.n_expert);
        put32((uint32_t) layers.size());
        put32((uint32_t) groups.size());
        size_t n_total = 0;
        for (const group & g : groups) {
            const moe_cache_lru & lru = g.lru;
            put32((uint32_t) g.layers.size());
            put64((uint64_t) g.record_size);
            std::vector<int32_t> keys;
            for (int32_t s = lru.head; s >= 0; s = lru.next[s]) {
                if (lru.key_of[s] >= 0) {
                    keys.push_back(lru.key_of[s]);
                }
            }
            put32((uint32_t) keys.size());
            for (int32_t k : keys) {
                put(&k, 4);
                const uint16_t u = lru.uses[k];
                put(&u, 2);
            }
            n_total += keys.size();
        }
        const std::string tmp = snapshot_path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (f == nullptr || fwrite(buf.data(), 1, buf.size(), f) != buf.size() || fflush(f) != 0 || fsync(fileno(f)) != 0 || fclose(f) != 0 ||
                rename(tmp.c_str(), snapshot_path.c_str()) != 0) {
            LLAMA_LOG_WARN("%s: expert store: writing the cache snapshot %s failed: %s\n", __func__, snapshot_path.c_str(), strerror(errno));
            if (f != nullptr) {
                fclose(f);
            }
            unlink(tmp.c_str());
            return;
        }
        LLAMA_LOG_INFO("%s: expert store: cache snapshot (%s): %zu resident experts, %.1f KB to %s\n", __func__, why, n_total, buf.size()/1e3, snapshot_path.c_str());
    }

    void snapshot_restore() {
        std::vector<uint8_t> buf;
        {
            FILE * f = fopen(snapshot_path.c_str(), "rb");
            if (f == nullptr) {
                LLAMA_LOG_INFO("%s: expert store: no cache snapshot at %s, the cache starts cold\n", __func__, snapshot_path.c_str());
                return;
            }
            uint8_t chunk[65536];
            size_t n;
            while ((n = fread(chunk, 1, sizeof(chunk), f)) > 0) {
                buf.insert(buf.end(), chunk, chunk + n);
            }
            fclose(f);
        }
        size_t pos = 0;
        auto need = [&](size_t n) { if (pos + n > buf.size()) { throw std::runtime_error("truncated"); } };
        auto get32 = [&]() { need(4); uint32_t v; memcpy(&v, buf.data() + pos, 4); pos += 4; return v; };
        auto get64 = [&]() { need(8); uint64_t v; memcpy(&v, buf.data() + pos, 8); pos += 8; return v; };
        struct stat st = {};
        fstat(store.fd, &st);
        try {
            need(4);
            if (memcmp(buf.data(), "K3SN", 4) != 0) {
                throw std::runtime_error("bad magic");
            }
            pos = 4;
            if (get32() != 1) {
                throw std::runtime_error("unknown version");
            }
            if (get64() != (uint64_t) st.st_size || get32() != store.n_expert || get32() != layers.size() || get32() != groups.size()) {
                throw std::runtime_error("made for another store or model");
            }
            const auto t0 = std::chrono::steady_clock::now();
            size_t n_restored = 0, n_bytes = 0;
            for (size_t gi = 0; gi < groups.size(); ++gi) {
                group & g = groups[gi];
                moe_cache_lru & lru = g.lru;
                const uint32_t n_layers = get32();
                const uint64_t rec = get64();
                uint32_t n_entries = get32();
                if (n_layers != g.layers.size() || rec != g.record_size) {
                    throw std::runtime_error("group layout differs");
                }
                struct entry { int32_t key; uint16_t uses; };
                std::vector<entry> ent;
                ent.reserve(n_entries);
                for (uint32_t i = 0; i < n_entries; ++i) {
                    need(6);
                    entry e;
                    memcpy(&e.key, buf.data() + pos, 4);
                    memcpy(&e.uses, buf.data() + pos + 4, 2);
                    pos += 6;
                    const int32_t il = e.key / (int32_t) store.n_expert, ex = e.key % (int32_t) store.n_expert;
                    if (e.key < 0 || e.key >= (int32_t) (layers.size()*store.n_expert) || il >= (int32_t) layers.size() || layers[il].ig != (int32_t) gi || lru.slot_map[il] == nullptr || lru.slot_map[il][ex] >= 0) {
                        continue; // not this group's, or a duplicate
                    }
                    lru.slot_map[il][ex] = 0; // marks it seen, the real slot is set below
                    ent.push_back(e);
                }
                for (const entry & e : ent) {
                    lru.slot_map[e.key / (int32_t) store.n_expert][e.key % (int32_t) store.n_expert] = -1;
                }
                if (ent.size() > (size_t) g.n_slots) {
                    ent.erase(ent.begin(), ent.end() - g.n_slots); // keep the most recent ones
                }
                if (g.region == nullptr || ent.empty()) {
                    continue;
                }
                // the most used experts take the HBM slots (0 .. n_hbm-1), the rest fill the host slots; both in
                // recency order, so the list below reproduces it
                std::vector<int32_t> by_use(ent.size());
                for (size_t i = 0; i < ent.size(); ++i) {
                    by_use[i] = (int32_t) i;
                }
                std::stable_sort(by_use.begin(), by_use.end(), [&](int32_t a, int32_t b) { return ent[a].uses > ent[b].uses; });
                std::vector<char> hot(ent.size(), 0);
                for (size_t i = 0; i < ent.size() && (int32_t) i < lru.n_hbm; ++i) {
                    hot[by_use[i]] = 1;
                }
                std::vector<int32_t> slot_of(ent.size(), -1);
                int32_t next_hbm = 0, next_host = lru.n_hbm;
                for (size_t i = 0; i < ent.size(); ++i) {
                    slot_of[i] = hot[i] ? next_hbm++ : next_host++;
                    if (slot_of[i] >= g.n_slots) { // fewer HBM entries than HBM slots: host overflow goes into the free HBM slots
                        slot_of[i] = next_hbm++;
                    }
                }
                std::vector<int32_t> order;
                order.reserve(g.n_slots);
                std::vector<char> used(g.n_slots, 0);
                for (size_t i = 0; i < ent.size(); ++i) {
                    used[slot_of[i]] = 1;
                }
                for (int32_t s = 0; s < g.n_slots; ++s) { // empty slots first: they are taken before anything is evicted
                    if (!used[s]) {
                        order.push_back(s);
                    }
                }
                jobs.clear();
                for (size_t i = 0; i < ent.size(); ++i) {
                    const entry & e = ent[i];
                    const int32_t s = slot_of[i];
                    const int32_t il = e.key / (int32_t) store.n_expert, ex = e.key % (int32_t) store.n_expert;
                    lru.key_of[s] = e.key;
                    lru.slot_map[il][ex] = s;
                    lru.uses[e.key] = e.uses;
                    order.push_back(s);
                    const expert_store_entry * se = store.find(il);
                    GGML_ASSERT(se != nullptr);
                    jobs.push_back({ (char *) g.region + (size_t) s*g.slot_stride, g.record_size, (off_t) (se->base_offset + (size_t) ex*g.record_size) });
                }
                lru.n_filled = (int32_t) ent.size();
                lru.relink(order);
                // read the records in batches, so the log shows progress on a 400 GB refill
                const size_t batch = 1024;
                std::vector<io_pool::job> part;
                for (size_t i = 0; i < jobs.size(); i += batch) {
                    part.assign(jobs.begin() + i, jobs.begin() + std::min(jobs.size(), i + batch));
                    if (!pool->run(store.fd, part)) {
                        throw std::runtime_error(std::string("reading the records failed: ") + strerror(errno));
                    }
                    n_bytes += part.size()*g.record_size;
                    if ((i/batch) % 16 == 15) {
                        LLAMA_LOG_INFO("%s: expert store: cache snapshot: %zu of %zu records of group %zu read, %.1f GB so far\n", __func__,
                            std::min(jobs.size(), i + batch), jobs.size(), gi, n_bytes/1e9);
                    }
                }
                if (lru.n_hbm > 0) {
                    hbm_refresh_min(g);
                }
                n_restored += ent.size();
            }
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            LLAMA_LOG_INFO("%s: expert store: cache snapshot restored: %zu experts, %.1f GB read in %.1f s (%.1f GB/s)\n", __func__,
                n_restored, n_bytes/1e9, sec, sec > 0 ? n_bytes/1e9/sec : 0.0);
            t_snapshot_last = std::chrono::steady_clock::now();
        } catch (const std::exception & e) {
            LLAMA_LOG_WARN("%s: expert store: cache snapshot %s not used (%s), the cache starts cold\n", __func__, snapshot_path.c_str(), e.what());
            // a slot placed whose record was not read would be served as an expert: start every group cold
            for (group & g : groups) {
                const int32_t n_hbm = g.lru.n_hbm;
                for (int32_t il : g.layers) {
                    if (g.lru.slot_map[il] != nullptr) {
                        std::fill(g.lru.slot_map[il], g.lru.slot_map[il] + store.n_expert, -1);
                    }
                }
                auto maps = g.lru.slot_map;
                g.lru.init((int32_t) layers.size(), (int32_t) store.n_expert, g.n_slots);
                g.lru.slot_map = maps;
                g.lru.n_hbm = n_hbm;
                g.lru.busy = g.spec_state.get();
                g.hbm_min_slot = -1;
            }
        }
    }

    void log_decode_timing(const char * func) const {
        if (n_decode_fills == 0) {
            return;
        }
        const double n = (double) n_decode_fills;
        LLAMA_LOG_INFO("%s: expert store: per layer fill: plan %.2f ms, reads %.2f ms (%.1f records, %.2f GB/s), between fills %.2f ms\n", func,
            t_dec_plan_us/n/1e3, t_dec_io_us/n/1e3, (double) stats_store_decode.misses/n,
            t_dec_io_us ? stats_store_decode.bytes/(t_dec_io_us/1e6)/1e9 : 0.0,
            n_dec_gaps ? t_dec_gap_us/(double) n_dec_gaps/1e3 : 0.0);
        valve_log(func);
        if (n_lookups > 0) {
            LLAMA_LOG_INFO("%s: expert store: HBM tier: %.1f%% of decode lookups served from it, %llu promotions (%.3f ms each)\n", func,
                100.0*n_hbm_lookups/n_lookups, (unsigned long long) n_promotions, n_promotions ? t_promote_us/1e3/n_promotions : 0.0);
        }
        if (split && n_split_calls > 0) {
            LLAMA_LOG_INFO("%s: expert store: split GEMV: %llu early lookups, %.1f of %d experts complete at the early lookup, %.2f read under the early matmul (%d rows), %llu lookups with more misses (%.2f records read at the early lookup), plan %.3f ms, wait at the real lookup %.2f ms, %llu early lookups without a real one\n", func,
                (unsigned long long) n_split_calls, (double) n_split_early_hits/n_split_calls, n_expert_used, (double) n_split_late/n_split_calls, split_late_max,
                (unsigned long long) n_split_sync, (double) n_split_sync_records/n_split_calls, t_split_plan_us/1e3/n_split_calls, t_split_wait_us/1e3/n_split_calls, (unsigned long long) n_split_fallback);
        }
        remote_log(func);
        if (self_prefetch && n_self_calls > 0) {
            LLAMA_LOG_INFO("%s: expert store: self-prefetch: %llu posts, %.1f records and %.2f ms of reads per post\n", func,
                (unsigned long long) n_self_calls, (double) n_self_reads/n_self_calls, t_self_io_us/1e3/n_self_calls);
        }
        if (prefetch) {
            LLAMA_LOG_INFO("%s: expert store: prefetch: %llu predicted misses read, %llu of them used by the real fill (%.1f%%)\n", func,
                (unsigned long long) n_prefetch_reads, (unsigned long long) n_prefetch_used,
                n_prefetch_reads ? 100.0*n_prefetch_used/n_prefetch_reads : 0.0);
            LLAMA_LOG_INFO("%s: expert store: prefetch: %llu posts, %.2f records queued per post, %llu handed back unread, %llu adopted in flight (%.3f ms per wait, %llu waits)\n", func,
                (unsigned long long) n_spec_posts, n_spec_posts ? (double) (n_prefetch_reads)/n_spec_posts : 0.0,
                (unsigned long long) n_spec_reclaimed, (unsigned long long) n_spec_adopted,
                n_spec_waits ? t_spec_wait_us/1e3/n_spec_waits : 0.0, (unsigned long long) n_spec_waits);
        }
    }

    // the staging for batches over the cache limit, allocated (and registered with the GPU) the first time one comes
    bool ensure_staging() {
        if (staging != nullptr) {
            return true;
        }
        std::string err;
        staging = store_region_alloc(staging_size, err);
        if (staging == nullptr) {
            LLAMA_LOG_ERROR("%s: expert store: failed to allocate %zu MiB of staging: %s\n", __func__, staging_size/1024/1024, err.c_str());
            return false;
        }
        if (gpu_reg != nullptr) {
            // registered the same way as the cache regions (upstream's ggml_backend_register_host_buffer is gated
            // on GGML_CUDA_REGISTER_HOST and registers read-only, which the GB10 refused for this region)
            auto * bank_from_ptr = (ggml_backend_buffer_t (*)(void *, size_t)) ggml_backend_reg_get_proc_address(gpu_reg, "ggml_backend_cuda_host_bank_buffer_from_ptr");
            if (bank_from_ptr != nullptr) {
                staging_buf.reset(bank_from_ptr(staging, staging_size));
            }
            staging_pinned = staging_buf != nullptr;
        }
        LLAMA_LOG_INFO("%s: expert store: %.2f MiB of prefill staging allocated (%s)\n", __func__, staging_size/1024.0/1024.0, staging_pinned ? "pinned" : "pageable");
        return true;
    }

    // large batch: the scheduler offloads the matmul and asks for a copy of w's used experts; read them from the
    // store into the staging (once per layer and ids) and upload this tensor's part of every record
    bool fill_prefill(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, const ggml_tensor * ids, const std::vector<bool> & used) {
        if (!use_store || !ensure_staging()) {
            return false;
        }
        lock_memory_once();
        la_finish();
        std::lock_guard<std::recursive_mutex> lock(lru_mutex);
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return false;
        }
        const binding & b = it->second;
        group & g = groups[layers[b.il].ig];
        const expert_store_entry * e = store.find(b.il);
        GGML_ASSERT(e != nullptr);
        const size_t  rec      = g.record_size;
        const int64_t n_expert = (int64_t) store.n_expert;
        GGML_ASSERT((int64_t) used.size() == n_expert);

        if (prefill.il != b.il || prefill.ids != ids) {
            // the uploads of the previous layer may still be reading the staging
            ggml_backend_synchronize(backend);
            jobs.clear();
            for (int64_t x = 0; x < n_expert; ++x) {
                if (used[x]) {
                    jobs.push_back({ (char *) staging + (size_t) x*rec, rec, (off_t) (e->base_offset + (size_t) x*rec) });
                }
            }
            if (!pool->run(store.fd, jobs)) {
                GGML_ABORT("expert store: read failed for layer %d: %s", b.il, strerror(errno));
            }
            store_prefill_bytes += jobs.size()*rec;
            prefill.il  = b.il;
            prefill.ids = ids;
        }

        const size_t moff  = g.part_off[b.ip];
        const size_t msize = w->nb[2];
        static const std::vector<uint8_t> zeros(512, 0);
        for (int64_t x = 0; x < n_expert; ++x) {
            if (!used[x]) {
                continue;
            }
            ggml_backend_tensor_set_async(backend, dst, (const char *) staging + (size_t) x*rec + moff, (size_t) x*msize, msize);
            // CUDA MMQ reads a little past the last expert, keep that from being NaN
            if (x + 1 < n_expert && !used[x + 1]) {
                ggml_backend_tensor_set_async(backend, dst, zeros.data(), (size_t) (x + 1)*msize, std::min<size_t>(msize, zeros.size()));
            }
        }
        return true;
    }

    void prefill_reset() {
        prefill.il  = -1;
        prefill.ids = nullptr;
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    ggml_tensor * get_experts_split(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached_split : nullptr;
    }

    // the maps of the split GEMV for a single-token decode of layer il (kind 1 early, 2 late, 3 merge, 4 the group's
    // zero-record slot), nullptr if the split does not apply
    ggml_tensor * get_split_map(int32_t il, int64_t n_tokens, int kind) const {
        if (!split || n_tokens != 1 || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];
        switch (kind) {
            case 1: return l.early_map;
            case 2: return l.late_map;
            case 3: return l.merge_map;
            case 4: return groups[l.ig].dummy_const;
            case 5: return split_iota;
            case 6: return remote_on() ? l.rpost_map   : nullptr;
            case 7: return remote_on() ? l.remote_x    : nullptr;
            case 8: return remote_on() ? l.remote_rows : nullptr;
            case 9: return remote_on() ? l.rmerge_map  : nullptr;
            default: return nullptr;
        }
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        if (use_store) {
            // large batches are served by fill_prefill
            return 0;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        std::lock_guard<std::recursive_mutex> lock(lru_mutex);
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        if (use_store) {
            GGML_ABORT("expert store: the slot map of layer %d was copied to a device, it must be read by the CPU graph", il);
        }

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // the experts live in host memory, usually a memory-mapped file: advise the kernel about
        // the missing experts of every matrix of the layer before the uploads start, so their
        // pages are read with large asynchronous I/Os instead of being faulted in one page at a
        // time by the copies
        if (!fills.empty()) {
            llama_memory_ranges mr;
            for (const ggml_tensor * w : l.experts) {
                if (w->buffer != nullptr && !ggml_backend_buffer_is_host(w->buffer)) {
                    continue;
                }
                for (const auto & f : fills) {
                    mr.push_back({ (const uint8_t *) w->data + f.expert*w->nb[2], w->nb[2] });
                }
            }
            llama_prefetch(std::move(mr));
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        if (use_store) {
            LLAMA_LOG_INFO("llama_moe_cache: expert store: %.2f GB read for decode, %.2f GB read for prefill\n",
                stats_store_decode.bytes/1e9, store_prefill_bytes/1e9);
            log_decode_timing("llama_moe_cache");
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
    pimpl(new impl(model, backends, bufts, size)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::prefetch_enabled() const {
    return pimpl->prefetch;
}

bool llama_moe_cache::split_enabled() const {
    return pimpl->split;
}

int llama_moe_cache::split_late_rows() const {
    return pimpl->split_late_max;
}

ggml_tensor * llama_moe_cache::get_split_map(int32_t il, int64_t n_tokens, int kind) const {
    return pimpl->get_split_map(il, n_tokens, kind);
}

ggml_tensor * llama_moe_cache::get_experts_split(const ggml_tensor * w) const {
    return pimpl->get_experts_split(w);
}

bool llama_moe_cache::self_prefetch_enabled() const {
    return pimpl->self_prefetch;
}

bool llama_moe_cache::lookups_on_gpu() const {
    return pimpl->slots_on_gpu;
}

int llama_moe_cache::prefetch_input() const {
    return pimpl->prefetch_input;
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

bool llama_moe_cache::fill_prefill(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, const ggml_tensor * ids, const std::vector<bool> & used) {
    return pimpl->fill_prefill(backend, w, dst, ids, used);
}

void llama_moe_cache::prefill_reset() {
    pimpl->prefill_reset();
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
