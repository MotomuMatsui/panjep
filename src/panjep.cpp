#include "panjep.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

#ifdef _OPENMP
#  include <omp.h>
#endif

// ARM NEON – available on Apple Silicon and any ARMv7/AArch64 with NEON.
// The compiler uses NEON automatically with -march=native; we add explicit
// intrinsics only for the hot inner loop where the branch on active_[j] would
// otherwise block auto-vectorisation.
#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif

namespace panjep {

static constexpr float kInactiveR = -1e30f;

// ============================================================
// DistMatrix
// ============================================================

DistMatrix::DistMatrix(int n)
    : n_(n),
      data_(static_cast<std::size_t>(n) * (n - 1) / 2, 0.0f)
{}

float& DistMatrix::at(int i, int j) noexcept { return data_[tri_idx(i, j)]; }

float DistMatrix::get(int i, int j) const noexcept {
    if (i == j) return 0.0f;
    return data_[tri_idx(i, j)];
}

// ============================================================
// NJSolver – construction
// ============================================================

NJSolver::NJSolver(int n)
    : n_leaves_(n),
      max_nodes_(2 * n - 1),
      n_active_(n),
      next_id_(n),
      dist_(2 * n - 1),
      R_       (2 * n - 1, 0.0),
      R_masked_(2 * n - 1, 0.0f),
      active_  (2 * n - 1, false),
      names_   (2 * n - 1),
      tree_    (2 * n - 1)
{
    for (int i = 0; i < n; i++) active_[i] = true;
}

void NJSolver::set_distance(int i, int j, float d) { dist_.at(i, j) = d; }
void NJSolver::set_name    (int i, const std::string& s) { names_[i] = s; }

// ============================================================
// Initialisation  –  O(n²/p)
// ============================================================

void NJSolver::init_row_sums() {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n_leaves_; i++) {
        // Lower-triangle half  (sequential, cache-friendly)
        const float* row = dist_.row_ptr(i);
        double s = 0.0;
        for (int j = 0; j < i; j++) s += row[j];

        // Upper-triangle half  (column i, strided)
        for (int j = i + 1; j < n_leaves_; j++) s += dist_.get(i, j);

        R_[i]        = s;
        R_masked_[i] = static_cast<float>(s);
    }
}

// ============================================================
// find_min_q  –  O(n²/p) parallel lower-triangle scan
//
// Key ideas:
//  1. Scan only (i,j) with i > j  →  each pair visited exactly once.
//     Row i accesses dist_.row_ptr(i)[0..i-1] which is contiguous in memory.
//  2. R_masked_[j] = −1e30f for inactive j  →  makes Q(i,inactive_j) ≈ +∞,
//     so inactive nodes are never selected without a branch.
//  3. ARM NEON processes 4 float pairs per cycle in the inner loop.
//  4. Row-level skip: if −R[i] − R_max ≥ thread_best_Q, the minimum Q from
//     this row cannot improve the running best  →  skip the whole row.
// ============================================================

std::pair<int,int> NJSolver::find_min_q() const {
    const float n2f  = static_cast<float>(n_active_ - 2);

    // R_max over active nodes (for row-level lower bound).
    // R_masked_[inactive] = NEG_INF_MASK ≪ any real R, so max ignores them.
    float R_max = *std::max_element(R_masked_.begin(), R_masked_.end());

    int   g_i = -1, g_j = -1;
    float g_Q = std::numeric_limits<float>::infinity();

#ifdef _OPENMP
    #pragma omp parallel
    {
        int   l_i = -1, l_j = -1;
        float l_Q = std::numeric_limits<float>::infinity();

        // dynamic,8: early outer rows prune heavily; let threads steal work.
        #pragma omp for schedule(dynamic, 8) nowait
        for (int i = 0; i < max_nodes_; i++) {
            if (!active_[i]) continue;

            const float Rif = static_cast<float>(R_[i]);

            // Row-level lower bound (d ≥ 0  →  Q_row_min ≥ −R[i]−R_max).
            if (-Rif - R_max >= l_Q) continue;

            const float* row = dist_.row_ptr(i);   // d(i,0)…d(i,i−1)
            const float* Rm  = R_masked_.data();
            int j = 0;

#if defined(__ARM_NEON)
            // ---- NEON inner loop: 4 pairs per iteration ----
            float32x4_t q_best = vdupq_n_f32(l_Q);
            int32x4_t   j_best = vdupq_n_s32(-1);
            const int32x4_t j_step = vdupq_n_s32(4);
            int32_t j_init[4] = {0, 1, 2, 3};
            int32x4_t j_cur = vld1q_s32(j_init);

            for (; j + 3 < i; j += 4) {
                float32x4_t d_v  = vld1q_f32(row + j);
                float32x4_t rm_v = vld1q_f32(Rm  + j);
                // Q = n2 * d - Ri - Rm[j]   (Rm[j]=−1e30 for inactive → Q≈+∞)
                float32x4_t q_v  = vsubq_f32(
                                       vsubq_f32(vmulq_n_f32(d_v, n2f),
                                                 vdupq_n_f32(Rif)),
                                       rm_v);
                uint32x4_t  lt   = vcltq_f32(q_v, q_best);
                q_best = vbslq_f32(lt, q_v, q_best);
                j_best = vbslq_s32(vreinterpretq_s32_u32(lt), j_cur, j_best);
                j_cur  = vaddq_s32(j_cur, j_step);
            }

            // Horizontal reduction: extract best (q, j) from 4 NEON lanes.
            float   q_arr[4]; vst1q_f32(q_arr, q_best);
            int32_t j_arr[4]; vst1q_s32(j_arr, j_best);
            for (int k = 0; k < 4; k++) {
                int jj = j_arr[k];
                // jj = -1 means no update in this lane; active_ guards stale k.
                if (jj >= 0 && active_[jj] && q_arr[k] < l_Q) {
                    l_Q = q_arr[k];  l_i = i;  l_j = jj;
                }
            }
#endif
            // ---- Scalar tail (also the full inner loop on non-NEON) ----
            for (; j < i; j++) {
                // R_masked_ makes this branch-free: inactive j → very large Q.
                float Q = n2f * row[j] - Rif - Rm[j];
                if (Q < l_Q) { l_Q = Q;  l_i = i;  l_j = j; }
            }
        }

        #pragma omp critical
        if (l_Q < g_Q) { g_Q = l_Q;  g_i = l_i;  g_j = l_j; }
    }

#else  // ---- Single-threaded path ----
    float l_Q = std::numeric_limits<float>::infinity();
    for (int i = 0; i < max_nodes_; i++) {
        if (!active_[i]) continue;
        const float Rif = static_cast<float>(R_[i]);
        if (-Rif - R_max >= l_Q) continue;

        const float* row = dist_.row_ptr(i);
        const float* Rm  = R_masked_.data();
        for (int j = 0; j < i; j++) {
            float Q = n2f * row[j] - Rif - Rm[j];
            if (Q < l_Q) { l_Q = Q;  g_i = i;  g_j = j; }
        }
    }
    g_Q = l_Q;
#endif

    return {g_i, g_j};
}

// ============================================================
// do_merge  –  join nodes i and j into new internal node k
//
// Operations: O(n_active) distance computations, O(n_active) R updates.
// No sorted-row insertion  →  each merge is O(n), not O(n²).
// The inner loop is parallelised when n_active is large enough to amortise
// the OpenMP fork/join overhead.
// ============================================================

void NJSolver::do_merge(int i, int j) {
    const int    k   = next_id_++;
    const double dij = static_cast<double>(dist_.get(i, j));
    const double n2  = (n_active_ > 2) ? static_cast<double>(n_active_ - 2) : 1.0;

    // Standard NJ branch-length formulae
    const double bi = 0.5 * dij + (R_[i] - R_[j]) / (2.0 * n2);
    const double bj = dij - bi;

    tree_[k].children = {{i, bi}, {j, bj}};
    tree_[i].parent   = k;
    tree_[j].parent   = k;

    // Compute d(k,m) = (d(i,m)+d(j,m)−d(i,j))/2  for every active m ≠ i,j.
    // Also update R[m] incrementally and synchronise R_masked_[m].
    // k is NOT yet active here, so the loop cannot accidentally include k.
    const float dij_f = static_cast<float>(dij);
    double R_k = 0.0;

    // Parallelise when the workload justifies the fork overhead (~256 items).
    const bool par = (n_active_ >= 256);
    if (par) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(+:R_k)
#endif
        for (int m = 0; m < max_nodes_; m++) {
            if (!active_[m] || m == i || m == j) continue;
            const float dm_i = dist_.get(i, m);
            const float dm_j = dist_.get(j, m);
            const float dm_k = 0.5f * (dm_i + dm_j - dij_f);
            dist_.at(k, m)   = dm_k;
            const double d   = static_cast<double>(dm_k);
            R_[m]       += d - static_cast<double>(dm_i) - static_cast<double>(dm_j);
            R_masked_[m] = static_cast<float>(R_[m]);
            R_k          += d;
        }
    } else {
        for (int m = 0; m < max_nodes_; m++) {
            if (!active_[m] || m == i || m == j) continue;
            const float dm_i = dist_.get(i, m);
            const float dm_j = dist_.get(j, m);
            const float dm_k = 0.5f * (dm_i + dm_j - dij_f);
            dist_.at(k, m)   = dm_k;
            const double d   = static_cast<double>(dm_k);
            R_[m]       += d - static_cast<double>(dm_i) - static_cast<double>(dm_j);
            R_masked_[m] = static_cast<float>(R_[m]);
            R_k          += d;
        }
    }

    R_[k]        = R_k;
    R_masked_[k] = static_cast<float>(R_k);

    // Activate k, deactivate i and j.
    // Setting R_masked_ to -1e30f makes inactive nodes invisible to find_min_q.
    active_[k]   = true;
    active_[i]   = false;
    active_[j]   = false;
    R_masked_[i] = kInactiveR;
    R_masked_[j] = kInactiveR;
    n_active_--;
}

// ============================================================
// run  –  main NJ loop
// ============================================================

void NJSolver::run_nj_() {
    // Assumes n_leaves_ >= 3.
    init_row_sums();

#ifdef NJ_PROFILE
    long long t_minq = 0, t_merge = 0;
    using Clk = std::chrono::steady_clock;
#endif
    while (n_active_ > 2) {
#ifdef NJ_PROFILE
        auto t0 = Clk::now();
#endif
        auto [pi, pj] = find_min_q();
#ifdef NJ_PROFILE
        auto t1 = Clk::now();
#endif
        do_merge(pi, pj);
#ifdef NJ_PROFILE
        auto t2 = Clk::now();
        t_minq  += std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
        t_merge += std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
#endif
    }
#ifdef NJ_PROFILE
    std::cerr << "[profile] find_min_q: " << t_minq/1000  << " ms\n";
    std::cerr << "[profile] do_merge:   " << t_merge/1000 << " ms\n";
#endif

    last_a_ = last_b_ = -1;
    for (int x = 0; x < max_nodes_; x++) {
        if (!active_[x]) continue;
        if (last_a_ < 0) last_a_ = x; else { last_b_ = x; break; }
    }
    assert(last_a_ >= 0 && last_b_ >= 0);
}

std::string NJSolver::run() {
    if (n_leaves_ == 1) return names_[0] + ";";
    if (n_leaves_ == 2) {
        const double d = dist_.get(0, 1);
        return "(" + names_[0] + ":" + fmt_len(d * 0.5)
             + "," + names_[1] + ":" + fmt_len(d * 0.5) + ");";
    }
    run_nj_();
    const double d_ab = dist_.get(last_a_, last_b_);
    return "(" + newick(last_a_) + ":" + fmt_len(d_ab * 0.5)
         + "," + newick(last_b_) + ":" + fmt_len(d_ab * 0.5) + ");";
}

// ============================================================
// Newick serialisation
// ============================================================

std::string NJSolver::newick(int v) const {
    const auto& node = tree_[v];
    if (node.children.empty()) {
        const std::string& name = names_[v];
        if (name.find_first_of(" (),:;[]") != std::string::npos)
            return "'" + name + "'";
        return name;
    }
    std::string s = "(";
    for (std::size_t ci = 0; ci < node.children.size(); ci++) {
        if (ci > 0) s += ',';
        auto [child, len] = node.children[ci];
        s += newick(child) + ':' + fmt_len(len);
    }
    return s + ")";
}

std::string NJSolver::fmt_len(double d) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.10g", d);
    return buf;
}

// ============================================================
// EP support: leaf-set computation, bipartition keys, EP Newick
// ============================================================

// Bottom-up fill of leaf_sets_[v] for all internal nodes.
// Internal nodes are created in order (n_leaves_ … next_id_-1) so earlier
// children always have smaller indices → ascending order is bottom-up.
void NJSolver::compute_leaf_sets_() {
    leaf_sets_.assign(max_nodes_, {});
    for (int i = 0; i < n_leaves_; i++) leaf_sets_[i] = {i};
    for (int v = n_leaves_; v < next_id_; v++) {
        for (auto& [child, unused] : tree_[v].children) {
            leaf_sets_[v].insert(leaf_sets_[v].end(),
                                 leaf_sets_[child].begin(),
                                 leaf_sets_[child].end());
        }
        std::sort(leaf_sets_[v].begin(), leaf_sets_[v].end());
    }
}

// Canonical bipartition key for node v: always the side that contains leaf 0.
// Returns "" for trivial bipartitions (one side has fewer than 2 leaves).
std::string NJSolver::bip_key_(int v) const {
    const auto& lset = leaf_sets_[v];
    const int n_in  = static_cast<int>(lset.size());
    const int n_out = n_leaves_ - n_in;
    if (n_in < 2 || n_out < 2) return {};

    const bool zero_in_lset = (!lset.empty() && lset[0] == 0);
    std::string key;
    if (zero_in_lset) {
        for (int l : lset) { key += std::to_string(l); key += '|'; }
    } else {
        // Use complement (which contains leaf 0).
        int j = 0;
        for (int i = 0; i < n_leaves_; i++) {
            if (j < n_in && lset[j] == i) { ++j; }
            else { key += std::to_string(i); key += '|'; }
        }
    }
    return key;
}

// Newick string for node v with EP support values appended after each ')'.
// Leaf nodes and trivial bipartitions get no label.
std::string NJSolver::newick_ep_(int v,
    const std::unordered_map<std::string,double>& sbk) const
{
    const auto& node = tree_[v];
    if (node.children.empty()) {
        const std::string& name = names_[v];
        if (name.find_first_of(" (),:;[]") != std::string::npos)
            return "'" + name + "'";
        return name;
    }
    std::string s = "(";
    for (std::size_t ci = 0; ci < node.children.size(); ci++) {
        if (ci > 0) s += ',';
        auto [child, len] = node.children[ci];
        s += newick_ep_(child, sbk) + ':' + fmt_len(len);
    }
    s += ')';
    const std::string key = bip_key_(v);
    if (!key.empty()) {
        auto it = sbk.find(key);
        if (it != sbk.end()) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << it->second;
            s += oss.str();
        }
    }
    return s;
}

// ============================================================
// PHYLIP I/O  –  supports lower-triangular and full-matrix formats
// ============================================================

// Fast float parser: no locale, no errno, no NaN/Inf.
// Handles optional sign, integer part, fractional part, optional exponent.
// Advances *p past the parsed token.
static float fast_f32(const char*& p) noexcept {
    while (*p == ' ' || *p == '\t') ++p;

    bool neg = false;
    if      (*p == '-') { neg = true; ++p; }
    else if (*p == '+') {              ++p; }

    uint32_t int_part = 0;
    while ((unsigned)(*p - '0') < 10u)
        int_part = int_part * 10u + (unsigned)(*p++ - '0');

    uint32_t frac = 0;
    int      frac_digits = 0;
    if (*p == '.') {
        ++p;
        while ((unsigned)(*p - '0') < 10u) {
            frac = frac * 10u + (unsigned)(*p++ - '0');
            ++frac_digits;
        }
    }

    // Accumulate in double to preserve 6-digit precision before float cast.
    static constexpr double kPow10Neg[] = {
        1.0,   1e-1,  1e-2,  1e-3,  1e-4,  1e-5,  1e-6,  1e-7,
        1e-8,  1e-9,  1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15,
        1e-16, 1e-17
    };
    constexpr int kMaxFracDigits = (int)(sizeof kPow10Neg / sizeof *kPow10Neg) - 1;
    if (frac_digits > kMaxFracDigits) frac_digits = kMaxFracDigits;
    double val = (double)int_part + (double)frac * kPow10Neg[frac_digits];

    if (*p == 'e' || *p == 'E') {
        ++p;
        bool eneg = false;
        if      (*p == '-') { eneg = true; ++p; }
        else if (*p == '+') {              ++p; }
        int exp = 0;
        while ((unsigned)(*p - '0') < 10u)
            exp = exp * 10 + (int)(*p++ - '0');
        static constexpr double kPow10[] = {
            1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
            1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
            1e16, 1e17, 1e18, 1e19
        };
        constexpr int kMaxExp = (int)(sizeof kPow10 / sizeof *kPow10) - 1;
        if (exp > kMaxExp) exp = kMaxExp;
        val = eneg ? val / kPow10[exp] : val * kPow10[exp];
    }

    return (float)(neg ? -val : val);
}

NJSolver NJSolver::from_phylip(std::istream& in) {
    int n = 0;
    {
        std::string line;
        std::getline(in, line);
        std::istringstream ss(line);
        if (!(ss >> n) || n <= 0)
            throw std::runtime_error("Invalid PHYLIP header");
    }

    NJSolver solver(n);
    std::vector<std::string>        names(n);
    std::vector<std::vector<float>> rows(n);

    for (int i = 0; i < n; i++) {
        std::string line;
        while (line.empty())
            if (!std::getline(in, line))
                throw std::runtime_error("Unexpected EOF at taxon " + std::to_string(i));
        std::istringstream ss(line);
        ss >> names[i];
        float v;
        while (ss >> v) rows[i].push_back(v);
    }

    // Detect format: lower-triangular → row i has i values; full → n values.
    const bool lower = (n < 2) || rows[0].empty() || static_cast<int>(rows[1].size()) == 1;

    for (int i = 0; i < n; i++) {
        solver.set_name(i, names[i]);
        if (lower) {
            for (int j = 0; j < static_cast<int>(rows[i].size()) && j < i; j++)
                solver.set_distance(i, j, rows[i][j]);
        } else {
            if (static_cast<int>(rows[i].size()) < n)
                throw std::runtime_error("Row " + std::to_string(i) + " too short");
            for (int j = 0; j < i; j++)
                solver.set_distance(i, j, rows[i][j]);
        }
    }
    return solver;
}

NJSolver NJSolver::from_phylip(const std::string& path) {
    // Bulk-read the entire file to avoid per-character stream overhead.
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) throw std::runtime_error("Cannot open: " + path);
    std::fseek(fp, 0, SEEK_END);
    const long fsz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::vector<char> buf(static_cast<std::size_t>(fsz) + 2);
    const std::size_t nr = std::fread(buf.data(), 1, static_cast<std::size_t>(fsz), fp);
    std::fclose(fp);
    buf[nr] = buf[nr + 1] = '\0';

    // Parse header.
    const char* p = buf.data();
    while (*p == ' ' || *p == '\t' || *p == '\r') ++p;
    int n = 0;
    while (*p >= '0' && *p <= '9') n = n * 10 + (*p++ - '0');
    while (*p && *p != '\n') ++p;
    if (*p == '\n') ++p;
    if (n <= 0) throw std::runtime_error("Invalid PHYLIP header");

    NJSolver solver(n);

    // Sequential pass: parse taxon names and record per-row float start pointers.
    std::vector<const char*> float_start(n);
    for (int i = 0; i < n; i++) {
        while (*p == '\n' || *p == '\r') ++p;               // skip blank lines
        const char* t = p;
        while (*t && *t != ' ' && *t != '\t' && *t != '\n' && *t != '\r') ++t;
        solver.set_name(i, std::string(p, t));
        while (*t == ' ' || *t == '\t') ++t;
        float_start[i] = t;
        p = t;
        while (*p && *p != '\n') ++p;
        if (*p == '\n') ++p;
    }

    // Detect format: lower-triangular (row 0 carries 0 floats) vs full matrix.
    int row0_floats = 0;
    {
        const char* r = float_start[0];
        while (*r && *r != '\n' && *r != '\r') {
            while (*r == ' ' || *r == '\t') ++r;
            if (!*r || *r == '\n' || *r == '\r') break;
            while (*r && *r != ' ' && *r != '\t' && *r != '\n' && *r != '\r') ++r;
            ++row0_floats;
        }
    }
    const bool lower = (row0_floats == 0);

    // Parallel float parsing.  Row i writes dist_.data_[lo_idx(i)..lo_idx(i)+i-1]
    // – non-overlapping segments – so no synchronisation is required.
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int i = 0; i < n; i++) {
        const char* r = float_start[i];
        const int ncols = lower ? i : n;
        for (int j = 0; j < ncols; j++) {
            const float v = fast_f32(r);
            if (j < i)
                solver.set_distance(i, j, v);
        }
    }

    return solver;
}

// ============================================================
// FASTA I/O + MMseqs2 / blastn pipeline
// ============================================================

namespace {   // helpers local to this translation unit

// POSIX exec-based command runner; immune to shell injection.
// args[0] is searched in PATH. stdout/stderr are suppressed.
bool run_safe(std::vector<std::string> args) {
    std::vector<const char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& a : args) argv.push_back(a.c_str());
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        int fd = open("/dev/null", O_WRONLY);
        if (fd >= 0) { dup2(fd, STDOUT_FILENO); dup2(fd, STDERR_FILENO); close(fd); }
        execvp(argv[0], const_cast<char**>(argv.data()));
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

// RAII guard: removes a temporary directory tree on destruction.
struct ScopedTempDir {
    std::string path;
    explicit ScopedTempDir(std::string p) : path(std::move(p)) {}
    ~ScopedTempDir() { if (!path.empty()) run_safe({"rm", "-rf", path}); }
    ScopedTempDir(const ScopedTempDir&) = delete;
    ScopedTempDir& operator=(const ScopedTempDir&) = delete;
};

struct FastaSeq { std::string name; std::string seq; };

// Bulk-read a FASTA file.  Name = first whitespace-delimited token on header.
std::vector<FastaSeq> parse_fasta(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) throw std::runtime_error("Cannot open FASTA: " + path);
    std::fseek(fp, 0, SEEK_END);
    const long fsz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::vector<char> buf(static_cast<std::size_t>(fsz) + 2);
    const std::size_t nr = std::fread(buf.data(), 1, static_cast<std::size_t>(fsz), fp);
    std::fclose(fp);
    buf[nr] = buf[nr + 1] = '\0';

    std::vector<FastaSeq> result;
    const char* p = buf.data();
    while (*p) {
        while (*p == '\n' || *p == '\r' || *p == ' ' || *p == '\t') ++p;
        if (!*p) break;
        if (*p != '>') throw std::runtime_error("Not a FASTA file (expected '>')");
        ++p;
        const char* name_end = p;
        while (*name_end && *name_end != ' ' && *name_end != '\t' &&
               *name_end != '\n' && *name_end != '\r') ++name_end;
        std::string name(p, name_end);
        while (*p && *p != '\n') ++p;
        if (*p == '\n') ++p;

        std::string seq;
        while (*p && *p != '>') {
            const char* line = p;
            while (*p && *p != '\n' && *p != '\r') ++p;
            seq.append(line, p);              // append whole line (no spaces expected)
            while (*p == '\n' || *p == '\r') ++p;
        }
        if (!name.empty() && !seq.empty())
            result.push_back({std::move(name), std::move(seq)});
    }
    if (result.size() < 2)
        throw std::runtime_error("FASTA file must contain at least 2 sequences");
    return result;
}

// Returns true when >90% of sampled non-gap characters are nucleotides.
bool detect_nucleotide(const std::vector<FastaSeq>& seqs) {
    long long total = 0, nucl = 0;
    for (auto& fs : seqs) {
        for (unsigned char c : fs.seq) {
            if (c == '-' || c == '.') continue;
            ++total;
            const unsigned char lc = c | 0x20u;
            if (lc=='a'||lc=='c'||lc=='g'||lc=='t'||lc=='n'||lc=='u'||
                lc=='w'||lc=='s'||lc=='m'||lc=='k'||lc=='r'||lc=='y'||
                lc=='b'||lc=='d'||lc=='h'||lc=='v') ++nucl;
        }
        if (total > 50000) break;
    }
    return total > 0 && (nucl * 100 / total) >= 90;
}

// Run MMseqs2 all-vs-all search.  Writes 5-column tabular output to
// workdir/result.m8 with columns: query, target, qaln, taln, bits.
// `-a 1` is required during search so convertalis can emit the alignment.
// Uses execvp (not std::system) so file paths are never interpreted by a shell.
void run_mmseqs2(const std::string& fasta, const std::string& wd,
                 int threads, float sens, bool nucl) {
    if (!run_safe({"mmseqs", "version"}))
        throw std::runtime_error("mmseqs not found in PATH – install MMseqs2 first");

    const std::string db  = wd + "/db";
    const std::string res = wd + "/result";
    const std::string tmp = wd + "/tmp";
    const std::string m8  = wd + "/result.m8";

    // 1. createdb
    {
        std::vector<std::string> cmd = {"mmseqs", "createdb", fasta, db};
        if (nucl) { cmd.push_back("--dbtype"); cmd.push_back("2"); }
        if (!run_safe(cmd)) throw std::runtime_error("MMseqs2 createdb failed");
    }

    // 2. search (self vs self).  -a 1 keeps the alignment for convertalis.
    {
        char sbuf[16]; std::snprintf(sbuf, sizeof sbuf, "%.1f", sens);
        std::vector<std::string> cmd = {
            "mmseqs", "search", db, db, res, tmp,
            "--threads", std::to_string(threads), "-e", "10", "-s", sbuf,
            "-a", "1"
        };
        if (nucl) { cmd.push_back("--search-type"); cmd.push_back("3"); }
        if (!run_safe(cmd)) throw std::runtime_error("MMseqs2 search failed");
    }

    // 3. convertalis → 5-col TSV: query, target, qaln, taln, bits
    if (!run_safe({"mmseqs", "convertalis", db, db, res, m8,
                   "--format-output", "query,target,qaln,taln,bits"}))
        throw std::runtime_error("MMseqs2 convertalis failed");
}

// Run NCBI blastn all-vs-all search.  Writes 5-col tabular output to
// workdir/result.m8: qseqid, sseqid, qseq, sseq, bitscore — the same shape
// parse_aln_m8 expects from MMseqs2.
// Task selection (megablast / dc-megablast / blastn / blastn-short) is mapped
// from MMseqs2's sensitivity scale so the -s flag keeps a coherent meaning:
//   sens < 3       → megablast      (fastest, closely-related sequences)
//   3 ≤ sens < 5   → dc-megablast   (moderately diverged)
//   5 ≤ sens < 7   → blastn         (default sensitivity)
//   sens ≥ 7       → blastn-short   (most sensitive, short/diverged queries)
void run_blastn(const std::string& fasta, const std::string& wd,
                int threads, float sens) {
    if (!run_safe({"blastn", "-version"}))
        throw std::runtime_error("blastn not found in PATH – install NCBI BLAST+ first");
    if (!run_safe({"makeblastdb", "-version"}))
        throw std::runtime_error("makeblastdb not found in PATH – install NCBI BLAST+ first");

    const std::string db = wd + "/db";
    const std::string m8 = wd + "/result.m8";

    // 1. makeblastdb
    if (!run_safe({"makeblastdb", "-in", fasta, "-dbtype", "nucl",
                   "-out", db, "-parse_seqids"}))
        throw std::runtime_error("makeblastdb failed");

    // 2. blastn self vs self.  outfmt 6 with explicit columns: aligned query
    // and subject sequences (qseq/sseq) replace the default %identity layout.
    const char* task = (sens < 3.0f) ? "megablast"
                     : (sens < 5.0f) ? "dc-megablast"
                     : (sens < 7.0f) ? "blastn"
                                     : "blastn-short";
    std::vector<std::string> cmd = {
        "blastn",
        "-task",        task,
        "-query",       fasta,
        "-db",          db,
        "-out",         m8,
        "-outfmt",      "6 qseqid sseqid qseq sseq bitscore",
        "-evalue",      "10",
        "-num_threads", std::to_string(threads),
    };
    if (!run_safe(cmd)) throw std::runtime_error("blastn search failed");
}

// Per-pair pairwise alignment, keyed by (i,j) row-major.  Each entry stores
// the highest-bitscore HSP's aligned query and target strings (equal length,
// gap chars '-' or '.' allowed).  Empty strings → no HSP for that pair.
struct AlnTable {
    std::vector<std::string> qaln;   // size n*n
    std::vector<std::string> taln;   // size n*n
    std::vector<double>      bits;   // size n*n; for keep-best dedup
};

// Parse the 5-column (query, target, qaln, taln, bits) TSV emitted by
// run_mmseqs2 / run_blastn.  Multiple HSPs for the same (q,t) pair: keep the
// alignment of the max-bitscore HSP.
AlnTable parse_aln_m8(const std::string& path,
                      const std::unordered_map<std::string,int>& idx,
                      int n) {
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) throw std::runtime_error("Cannot open search output: " + path);
    std::fseek(fp, 0, SEEK_END);
    const long fsz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::vector<char> buf(static_cast<std::size_t>(fsz) + 2);
    const std::size_t nr = std::fread(buf.data(), 1, static_cast<std::size_t>(fsz), fp);
    std::fclose(fp);
    buf[nr] = buf[nr + 1] = '\0';

    const std::size_t nn = static_cast<std::size_t>(n) * n;
    AlnTable A;
    A.qaln.assign(nn, std::string());
    A.taln.assign(nn, std::string());
    A.bits.assign(nn, -std::numeric_limits<double>::infinity());

    const char* p = buf.data();
    while (*p) {
        while (*p == '\n' || *p == '\r') ++p;
        if (!*p) break;
        if (*p == '#') { while (*p && *p != '\n') ++p; continue; }

        // Scan 5 tab-separated fields.
        const char* col[5] = {};
        const char* end_[5] = {};
        int nc = 0;
        const char* q = p;
        while (*q && *q != '\n' && nc < 5) {
            col[nc] = q;
            while (*q && *q != '\t' && *q != '\n') ++q;
            end_[nc] = q;
            ++nc;
            if (*q == '\t') ++q;
        }
        while (*p && *p != '\n') ++p;
        if (*p == '\n') ++p;
        if (nc < 5) continue;

        const auto qi_it = idx.find(std::string(col[0], end_[0]));
        const auto ti_it = idx.find(std::string(col[1], end_[1]));
        if (qi_it == idx.end() || ti_it == idx.end()) continue;

        char* endp;
        const double bsc = std::strtod(col[4], &endp);
        const std::size_t qi = static_cast<std::size_t>(qi_it->second);
        const std::size_t ti = static_cast<std::size_t>(ti_it->second);
        const std::size_t k  = qi * static_cast<std::size_t>(n) + ti;
        if (bsc > A.bits[k]) {
            A.bits[k] = bsc;
            A.qaln[k].assign(col[2], end_[2]);
            A.taln[k].assign(col[3], end_[3]);
        }
    }
    return A;
}

// ============================================================
// Evolutionary-distance kernels (Sonnhammer & Hollich ScoreDist;
// Poisson correction).  Ported from mi6 (Matsui & Iwasaki, 2018).
// ============================================================

// BLOSUM62 (24×24, half-bit units).  Order: A R N D C Q E G H I L K M F P S
// T W Y V B Z X *.  Used by ScoreDist for protein distance estimation.
static constexpr int kBlosum62[24][24] = {
    { 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4},
    {-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4},
    {-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4},
    {-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4},
    { 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4},
    {-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4},
    {-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4},
    { 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4},
    {-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4},
    {-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4},
    {-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4},
    {-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4},
    {-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4},
    {-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4},
    {-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4},
    { 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4},
    { 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4},
    {-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4},
    {-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4},
    { 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4},
    {-2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4},
    {-1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4},
    { 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4},
    {-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1}
};

// Sentinel for non-protein characters (gaps, X, ambiguous, lowercase noise).
constexpr int kAaInvalid = -1;

// 256-entry lookup: ASCII byte → BLOSUM62 row (0..23) or kAaInvalid for gap /
// unknown.  Both upper- and lower-case letters map to the same index.  Built
// as a runtime-static const because std::array::operator[] is not usable in a
// C++17 constexpr writer (it gains the property in C++20).
static const std::array<int, 256> kS2P = []() {
    std::array<int, 256> t{};
    for (auto& v : t) v = kAaInvalid;
    const char* aa = "ARNDCQEGHILKMFPSTWYVBZX";   // 23 letters → indices 0..22
    for (int i = 0; aa[i]; ++i) {
        t[static_cast<unsigned char>(aa[i])]               = i;
        t[static_cast<unsigned char>(aa[i] | 0x20)]        = i;   // lower case
    }
    return t;
}();

// 256-entry lookup for 4-letter nucleotide alphabet (ACGT + U→T).  Other
// IUPAC ambiguity codes (R/Y/N/...) and gaps map to kAaInvalid so they are
// excluded from the Poisson identity count.
static const std::array<int, 256> kN2P = []() {
    std::array<int, 256> t{};
    for (auto& v : t) v = kAaInvalid;
    t['A'] = t['a'] = 0;
    t['C'] = t['c'] = 1;
    t['G'] = t['g'] = 2;
    t['T'] = t['t'] = 3;
    t['U'] = t['u'] = 3;
    return t;
}();

// Saturation cap shared by both methods.  Pairs with no alignment, or with
// p ≥ (n−1)/n (Poisson-undefined regime), or ScoreDist od ≤ 0, are clamped
// to this value before being handed to NJ.
static constexpr double kMaxDist = 5.0;

// ScoreDist (Sonnhammer & Hollich 2005, BMC Bioinformatics).
// Protein only.  Gap–any and any–gap columns are excluded.
double scoredist(const std::string& A, const std::string& B) {
    const std::size_t L = std::min(A.size(), B.size());
    long long sc = 0, scA = 0, scB = 0;
    long long len = 0;
    for (std::size_t p = 0; p < L; ++p) {
        const int pa = kS2P[(unsigned char)A[p]];
        const int pb = kS2P[(unsigned char)B[p]];
        if (pa < 0 || pb < 0) continue;
        sc  += kBlosum62[pa][pb];
        scA += kBlosum62[pa][pa];
        scB += kBlosum62[pb][pb];
        ++len;
    }
    if (len == 0) return kMaxDist;

    const double scMAX = 0.5 * (scA + scB);
    const double scR   = -0.5209 * static_cast<double>(len);   // BLOSUM62 expected score per column
    const double od    = (static_cast<double>(sc) - scR) / (scMAX - scR);
    if (!(od > 1e-8)) return kMaxDist;
    if (od >= 1.0)    return 0.0;
    const double d = -std::log(od);
    return d > kMaxDist ? kMaxDist : d;
}

// Poisson correction (Zuckerkandl & Pauling).  Works for both alphabets:
// nc=20 for protein, nc=4 for nucleotide.
double poisson_dist(const std::string& A, const std::string& B, int nc) {
    const std::size_t L = std::min(A.size(), B.size());
    const auto& tab = (nc == 4) ? kN2P : kS2P;
    long long total = 0, matches = 0;
    for (std::size_t p = 0; p < L; ++p) {
        const int pa = tab[(unsigned char)A[p]];
        const int pb = tab[(unsigned char)B[p]];
        if (pa < 0 || pb < 0) continue;
        ++total;
        if (pa == pb) ++matches;
    }
    if (total == 0) return kMaxDist;

    const double s   = 1.0 - static_cast<double>(matches) / static_cast<double>(total);
    const double lim = static_cast<double>(nc - 1) / static_cast<double>(nc);
    if (s >= lim) return kMaxDist;        // Poisson formula diverges past saturation.
    const double d = -lim * std::log(1.0 - s / lim);
    return d > kMaxDist ? kMaxDist : d;
}

// ============================================================
// FastME distance kernels (Lefort, Desper & Gascuel 2015).
// All functions operate on the per-pair qaln/taln strings produced by
// MMseqs2/blastn.  Gap and ambiguous columns (kS2P/kN2P sentinel) are
// excluded pairwise.  Saturated/undefined regimes return kMaxDist.
// ============================================================

// 20-aa lookup (ACDEFGHIKLMNPQRSTVWY → 0..19).  Used for F81 protein where we
// need the standard 20 letters; B/Z/X/J/U/O/*/gaps map to kAaInvalid so they
// are dropped from both the Hamming count and the stationary frequency.
static const std::array<int, 256> kAA20 = []() {
    std::array<int, 256> t{};
    for (auto& v : t) v = kAaInvalid;
    const char* aa = "ACDEFGHIKLMNPQRSTVWY";
    for (int i = 0; aa[i]; ++i) {
        t[static_cast<unsigned char>(aa[i])]        = i;
        t[static_cast<unsigned char>(aa[i] | 0x20)] = i;
    }
    return t;
}();

// Hamming dissimilarity (b = mismatch fraction over non-gap columns).
// Returns (b, numS).  numS == 0 → sentinel = -1; caller must clamp.
struct HammingResult { double b; long long numS; };

HammingResult hamming(const std::string& A, const std::string& B,
                      const std::array<int,256>& tab) {
    const std::size_t L = std::min(A.size(), B.size());
    long long total = 0, mism = 0;
    for (std::size_t p = 0; p < L; ++p) {
        const int pa = tab[(unsigned char)A[p]];
        const int pb = tab[(unsigned char)B[p]];
        if (pa < 0 || pb < 0) continue;
        ++total;
        if (pa != pb) ++mism;
    }
    return { total ? static_cast<double>(mism) / static_cast<double>(total) : 0.0,
             total };
}

// Build the 4×4 DNA pair-substitution matrix P from aligned columns.
// P[a][b] = count(s1=a, s2=b) / numS, so sum_b P[a][b] = freq of a in s1.
// Returns numS (non-gap columns shared by both); 0 means no signal.
long long dna_pmatrix(const std::string& A, const std::string& B,
                      double P[4][4]) {
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) P[i][j] = 0.0;
    const std::size_t L = std::min(A.size(), B.size());
    long long numS = 0;
    for (std::size_t p = 0; p < L; ++p) {
        const int pa = kN2P[(unsigned char)A[p]];
        const int pb = kN2P[(unsigned char)B[p]];
        if (pa < 0 || pb < 0) continue;
        P[pa][pb] += 1.0;
        ++numS;
    }
    if (numS == 0) return 0;
    const double inv = 1.0 / static_cast<double>(numS);
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) P[i][j] *= inv;
    return numS;
}

// DNA: index convention A=0, C=1, G=2, T=3 (matches kN2P and FastME utils.h).
constexpr int IA = 0, IC = 1, IG = 2, IT = 3;

double dna_transition_rate(const double P[4][4]) {
    return P[IA][IG] + P[IG][IA] + P[IC][IT] + P[IT][IC];
}

double dna_transversion_rate(const double P[4][4]) {
    return P[IA][IC] + P[IA][IT] + P[IG][IC] + P[IG][IT]
         + P[IC][IA] + P[IC][IG] + P[IT][IA] + P[IT][IG];
}

// 4×4 determinant via 6 2×2 minors of rows {0,1} × 6 minors of rows {2,3}.
// ~35 flops vs ~63 for the 3×3-cofactor expansion, with no nested calls.
double det4(const double M[4][4]) noexcept {
    const double s0 = M[0][0]*M[1][1] - M[0][1]*M[1][0];
    const double s1 = M[0][0]*M[1][2] - M[0][2]*M[1][0];
    const double s2 = M[0][0]*M[1][3] - M[0][3]*M[1][0];
    const double s3 = M[0][1]*M[1][2] - M[0][2]*M[1][1];
    const double s4 = M[0][1]*M[1][3] - M[0][3]*M[1][1];
    const double s5 = M[0][2]*M[1][3] - M[0][3]*M[1][2];
    const double c5 = M[2][2]*M[3][3] - M[2][3]*M[3][2];
    const double c4 = M[2][1]*M[3][3] - M[2][3]*M[3][1];
    const double c3 = M[2][1]*M[3][2] - M[2][2]*M[3][1];
    const double c2 = M[2][0]*M[3][3] - M[2][3]*M[3][0];
    const double c1 = M[2][0]*M[3][2] - M[2][2]*M[3][0];
    const double c0 = M[2][0]*M[3][1] - M[2][1]*M[3][0];
    return s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;
}

// ---- Closed-form distance kernels (FastME distance.c, gamma=off) ----

double calc_jc69(double b) {
    if (b <= 1e-12) return 0.0;
    const double loc = 1.0 - 4.0 * b / 3.0;
    if (loc <= 0.0) return kMaxDist;
    const double d = -0.75 * std::log(loc);
    return d > kMaxDist ? kMaxDist : d;
}

double calc_k2p(double a, double b) {
    if (a <= 1e-12 && b <= 1e-12) return 0.0;
    const double loc1 = 1.0 - 2.0 * a - b;
    const double loc2 = 1.0 - 2.0 * b;
    if (loc1 <= 0.0 || loc2 <= 0.0) return kMaxDist;
    const double d = -0.5 * std::log(loc1) - 0.25 * std::log(loc2);
    return d > kMaxDist ? kMaxDist : d;
}

// Generic F81-style "Poisson with empirical pi" estimator.
// loc = 1 - sum(pi^2); b = mismatch fraction.
double calc_f81(double loc, double b) {
    if (b <= 1e-12) return 0.0;
    if (loc <= 1e-12) return kMaxDist;
    const double y = 1.0 - b / loc;
    if (y <= 0.0) return kMaxDist;
    const double d = -loc * std::log(y);
    return d > kMaxDist ? kMaxDist : d;
}

double calc_f84(double a, double b, double Mloc, double Nloc, double Ploc) {
    if (a <= 1e-12 && b <= 1e-12) return 0.0;
    if (Mloc <= 0.0 || Ploc <= 0.0) return kMaxDist;
    const double loc1 = 1.0 - a / (2.0 * Mloc) - b * (Mloc - Nloc) / (2.0 * Mloc * Ploc);
    const double loc2 = 1.0 - b / (2.0 * Ploc);
    if (loc1 <= 0.0 || loc2 <= 0.0) return kMaxDist;
    const double d = -2.0 * Mloc * std::log(loc1)
                   - 2.0 * (Nloc + Ploc - Mloc) * std::log(loc2);
    return d > kMaxDist ? kMaxDist : d;
}

double calc_tn93(double aR, double aY, double b,
                 double PR, double PY, double PAPG, double PCPT) {
    if (aR <= 1e-12 && aY <= 1e-12 && b <= 1e-12) return 0.0;
    if (PAPG <= 0.0 || PCPT <= 0.0 || PR <= 0.0 || PY <= 0.0) return kMaxDist;
    const double loc1 = 1.0 - (PR  * aR) / (2.0 * PAPG) - b / (2.0 * PR);
    const double loc2 = 1.0 - (PY  * aY) / (2.0 * PCPT) - b / (2.0 * PY);
    const double loc3 = 1.0 - b / (2.0 * PR * PY);
    if (loc1 <= 0.0 || loc2 <= 0.0 || loc3 <= 0.0) return kMaxDist;
    const double d = -2.0 * PAPG / PR * std::log(loc1)
                   - 2.0 * PCPT / PY * std::log(loc2)
                   - 2.0 * (PR * PY - PAPG * PY / PR - PCPT * PR / PY) * std::log(loc3);
    return d > kMaxDist ? kMaxDist : d;
}

double calc_rysym(double b) {
    if (b <= 1e-12) return 0.0;
    const double loc = 1.0 - 2.0 * b;
    if (loc <= 0.0) return kMaxDist;
    const double d = -0.5 * std::log(loc);
    return d > kMaxDist ? kMaxDist : d;
}

double calc_ry(double PR, double PY, double b) {
    const double Z = 1.0 - PR * PR - PY * PY;
    if (Z <= 0.0) return kMaxDist;
    if (b / Z >= 1.0) return kMaxDist;
    const double d = -Z * std::log(1.0 - b / Z);
    return d > kMaxDist ? kMaxDist : d;
}

// LogDet (paralinear) distance.  log_pi_sum_k = (1/8) · Σ_i log π_k[i] is
// pre-computed per-sequence so the per-pair work drops from 8 logs to 2 adds.
// Sentinel −INF in either argument means "some π_i = 0" → return saturation.
double calc_logdet(const double P[4][4],
                   double log_pi_sum_i, double log_pi_sum_j) noexcept {
    if (std::isinf(log_pi_sum_i) || std::isinf(log_pi_sum_j)) return kMaxDist;
    const double dP = det4(P);
    if (dP <= 0.0) return kMaxDist;
    double d = -0.5 * std::log(dP) + log_pi_sum_i + log_pi_sum_j;
    if (d < 0.0) return 0.0;
    return d > kMaxDist ? kMaxDist : d;
}

// ---- Stationary-frequency estimators (over raw input residue composition) ----

std::array<double, 4> dna_stationary(const std::vector<FastaSeq>& seqs) {
    long long cnt[4] = {0, 0, 0, 0};
    long long total = 0;
    for (const auto& fs : seqs) {
        for (unsigned char c : fs.seq) {
            const int x = kN2P[c];
            if (x < 0) continue;
            ++cnt[x]; ++total;
        }
    }
    std::array<double, 4> pi{};
    if (total == 0) { pi.fill(0.25); return pi; }
    for (int i = 0; i < 4; ++i) pi[i] = static_cast<double>(cnt[i]) / total;
    return pi;
}

// Per-sequence (1/8) · Σ_i log π_i, used by calc_logdet.  Sentinel −INF means
// "π_i = 0 for some i" → caller returns kMaxDist.  Loop is OpenMP-parallel
// because for n ≳ 10^4 the sequential pass becomes visible (~10s of ms).
std::vector<double>
dna_log_pi_sum_per_seq(const std::vector<FastaSeq>& seqs) {
    constexpr double kInvalid = -std::numeric_limits<double>::infinity();
    std::vector<double> out(seqs.size(), kInvalid);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t s = 0; s < seqs.size(); ++s) {
        long long cnt[4] = {0, 0, 0, 0};
        long long total = 0;
        for (unsigned char c : seqs[s].seq) {
            const int x = kN2P[c];
            if (x < 0) continue;
            ++cnt[x]; ++total;
        }
        if (total == 0) continue;            // out[s] stays kInvalid
        bool ok = true;
        double sum = 0.0;
        for (int i = 0; i < 4; ++i) {
            if (cnt[i] == 0) { ok = false; break; }
            sum += std::log(static_cast<double>(cnt[i]) /
                            static_cast<double>(total));
        }
        if (ok) out[s] = sum / 8.0;
    }
    return out;
}

std::array<double, 20> aa_stationary(const std::vector<FastaSeq>& seqs) {
    long long cnt[20] = {0};
    long long total = 0;
    for (const auto& fs : seqs) {
        for (unsigned char c : fs.seq) {
            const int x = kAA20[c];
            if (x < 0) continue;
            ++cnt[x]; ++total;
        }
    }
    std::array<double, 20> pi{};
    if (total == 0) { pi.fill(1.0 / 20.0); return pi; }
    for (int i = 0; i < 20; ++i) pi[i] = static_cast<double>(cnt[i]) / total;
    return pi;
}

// ---- DistContext: precomputed model-specific constants -----------------------

struct DistContext {
    DistMethod method;
    bool       nucl;
    // F81 (DNA & protein): loc = 1 − Σ pi^2.
    double f81_loc_dna = 0.75;     // uniform default → F81 ≡ JC69
    double f81_loc_aa  = 0.95;     // uniform default → F81 ≡ Poisson(nc=20)
    // F84
    double f84_M = 0.0, f84_N = 0.0, f84_P = 0.0;
    // TN93 / RY
    double tn_PR = 0.5, tn_PY = 0.5;
    double tn_PAPG = 0.0, tn_PCPT = 0.0;
    // LogDet: per-sequence (1/8)·Σ log π_i (kInvalid = −INF if any π_i = 0).
    std::vector<double> per_seq_log_pi_sum;
};

DistContext build_dist_context(DistMethod m, bool nucl,
                               const std::vector<FastaSeq>& seqs) {
    DistContext c;
    // Resolve Auto early so downstream logic only sees concrete methods.
    if (m == DistMethod::Auto)
        m = nucl ? DistMethod::Poisson : DistMethod::ScoreDist;
    c.method = m;
    c.nucl   = nucl;

    const bool needs_dna_pi =
        nucl && (m == DistMethod::F81  || m == DistMethod::F84  ||
                 m == DistMethod::TN93 || m == DistMethod::RY);
    const bool needs_aa_pi  = !nucl && (m == DistMethod::F81);

    if (needs_dna_pi) {
        const auto pi = dna_stationary(seqs);
        c.f81_loc_dna = 1.0 - pi[IA]*pi[IA] - pi[IC]*pi[IC]
                            - pi[IG]*pi[IG] - pi[IT]*pi[IT];

        const double PAG = pi[IA] * pi[IG];
        const double PCT = pi[IC] * pi[IT];
        const double PR  = pi[IA] + pi[IG];
        const double PY  = pi[IC] + pi[IT];
        c.tn_PR   = PR;
        c.tn_PY   = PY;
        c.tn_PAPG = PAG;
        c.tn_PCPT = PCT;
        if (PR > 0.0 && PY > 0.0) {
            c.f84_M = PAG / PR + PCT / PY;
            c.f84_N = PAG + PCT;
            c.f84_P = PR  * PY;
        }
    }
    if (needs_aa_pi) {
        const auto pi = aa_stationary(seqs);
        double s = 0.0;
        for (double v : pi) s += v * v;
        c.f81_loc_aa = 1.0 - s;
    }
    if (nucl && m == DistMethod::LogDet)
        c.per_seq_log_pi_sum = dna_log_pi_sum_per_seq(seqs);

    return c;
}

// ---- Per-pair dispatcher ----------------------------------------------------

double pair_distance(const std::string& A, const std::string& B,
                     int i, int j, const DistContext& c) {
    if (A.empty() || B.empty()) return kMaxDist;

    const DistMethod m    = c.method;
    const bool       nucl = c.nucl;

    // -- alphabet-agnostic / Hamming-only paths ---------------------------------
    if (m == DistMethod::ScoreDist) {
        if (nucl) return poisson_dist(A, B, 4);   // ScoreDist is protein-only.
        return scoredist(A, B);
    }
    if (m == DistMethod::Poisson) {
        return poisson_dist(A, B, nucl ? 4 : 20);
    }
    if (m == DistMethod::PDist) {
        const auto h = hamming(A, B, nucl ? kN2P : kS2P);
        if (h.numS == 0) return kMaxDist;
        return h.b > kMaxDist ? kMaxDist : h.b;
    }
    if (m == DistMethod::JC69) {
        const auto h = hamming(A, B, nucl ? kN2P : kS2P);
        if (h.numS == 0) return kMaxDist;
        return calc_jc69(h.b);
    }
    if (m == DistMethod::F81) {
        // For protein use the 20-aa table so kAA20-based stationary lines up
        // with Hamming counts; for DNA use kN2P.
        const auto h = hamming(A, B, nucl ? kN2P : kAA20);
        if (h.numS == 0) return kMaxDist;
        return calc_f81(nucl ? c.f81_loc_dna : c.f81_loc_aa, h.b);
    }

    // -- DNA-only paths needing the 4×4 P matrix --------------------------------
    if (!nucl) {
        // K2P / F84 / TN93 / RY / RYSym / LogDet are DNA-only.  Fall back to
        // protein Poisson(nc=20) so the run still produces a tree.
        return poisson_dist(A, B, 20);
    }
    double P[4][4];
    const long long numS = dna_pmatrix(A, B, P);
    if (numS == 0) return kMaxDist;
    const double a = dna_transition_rate(P);
    const double b = dna_transversion_rate(P);

    switch (m) {
        case DistMethod::K2P:   return calc_k2p(a, b);
        case DistMethod::F84:   return calc_f84(a, b, c.f84_M, c.f84_N, c.f84_P);
        case DistMethod::TN93: {
            const double aR = P[IA][IG] + P[IG][IA];
            const double aY = P[IC][IT] + P[IT][IC];
            return calc_tn93(aR, aY, b, c.tn_PR, c.tn_PY, c.tn_PAPG, c.tn_PCPT);
        }
        case DistMethod::RY:    return calc_ry(c.tn_PR, c.tn_PY, b);
        case DistMethod::RYSym: return calc_rysym(b);
        case DistMethod::LogDet: {
            if (i < 0 || j < 0 ||
                static_cast<std::size_t>(i) >= c.per_seq_log_pi_sum.size() ||
                static_cast<std::size_t>(j) >= c.per_seq_log_pi_sum.size())
                return kMaxDist;
            return calc_logdet(P, c.per_seq_log_pi_sum[i],
                                  c.per_seq_log_pi_sum[j]);
        }
        default: break;
    }
    return kMaxDist;
}

// ---------------------------------------------------------------------------
// GEV (Generalised Extreme Value) perturbation kernel.
// Ported from gs (Matsui & Tanabe): perturbs similarity w ∈ (0,1] using a
// GEV distribution parameterised by the original value.
//   x   – uniform random in (0,1)
//   mu  – original similarity score
// ---------------------------------------------------------------------------
double gev(double x, double mu) noexcept {
    const double theta = mu * (1.0 - mu) / 3.5;
    const double gamma = std::exp(-3.0 * mu) - 1.0;
    double result;
    if (std::abs(gamma) < 1e-10) {               // Gumbel limit (mu ≈ 0)
        result = mu - theta * std::log(-std::log(x));
    } else {
        result = mu + (std::pow(-std::log(x), -gamma) - 1.0) * theta / gamma;
    }
    return (result > 1.0) ? 1.0 : (result < 0.0) ? 0.0 : result;
}

// Shared helper: FASTA → (MMseqs2 | blastn) → n×n symmetric distance matrix D.
// D[i*n+j] is the average of the per-direction alignment distances; pairs
// without a hit collapse to kMaxDist.  Diagonal is forced to zero.
struct FastaData { std::vector<FastaSeq> seqs; std::vector<double> D; bool nucl; };

FastaData load_fasta_distance(const std::string& path, int threads,
                              float sensitivity, DistMethod method) {
    auto seqs = parse_fasta(path);
    const int n = static_cast<int>(seqs.size());

    std::unordered_map<std::string,int> name_idx;
    name_idx.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; i++) name_idx[seqs[i].name] = i;

    const bool nucl = detect_nucleotide(seqs);

    const DistContext ctx = build_dist_context(method, nucl, seqs);

    auto label = [&](DistMethod m, bool nu) -> const char* {
        switch (m) {
            case DistMethod::ScoreDist: return "ScoreDist (BLOSUM62)";
            case DistMethod::Poisson:   return nu ? "Poisson(nc=4)" : "Poisson(nc=20)";
            case DistMethod::PDist:     return "p-distance";
            case DistMethod::JC69:      return "JC69";
            case DistMethod::K2P:       return "K2P";
            case DistMethod::F81:       return nu ? "F81 (DNA)" : "F81 (protein)";
            case DistMethod::F84:       return "F84";
            case DistMethod::TN93:      return "TN93";
            case DistMethod::LogDet:    return "LogDet";
            case DistMethod::RY:        return "RY (transversion)";
            case DistMethod::RYSym:     return "RY-symmetric";
            default:                    return "?";
        }
    };
    const char* method_name = (method == DistMethod::Auto)
        ? (nucl ? "Poisson(nc=4) [auto]" : "ScoreDist [auto]")
        : label(ctx.method, nucl);
    std::cerr << "[panjep] " << n << " "
              << (nucl ? "nucleotide" : "protein") << " sequences; "
              << "distance = " << method_name << "\n";

    char tmpl[] = "/tmp/panjep_XXXXXX";
    if (!mkdtemp(tmpl)) throw std::runtime_error("mkdtemp failed");
    ScopedTempDir guard(tmpl);   // cleaned up on any exit path

    if (nucl) {
        std::cerr << "[panjep] Running blastn (threads=" << threads
                  << ", sensitivity=" << sensitivity << ")...\n";
        run_blastn(path, guard.path, threads, sensitivity);
    } else {
        std::cerr << "[panjep] Running MMseqs2 (threads=" << threads
                  << ", sensitivity=" << sensitivity << ")...\n";
        run_mmseqs2(path, guard.path, threads, sensitivity, nucl);
    }

    std::cerr << "[panjep] Computing pairwise distances from alignments...\n";
    const auto A = parse_aln_m8(guard.path + "/result.m8", name_idx, n);

    const std::size_t nn = static_cast<std::size_t>(n);
    std::vector<double> D(nn * nn, kMaxDist);
    for (std::size_t i = 0; i < nn; ++i) D[i * nn + i] = 0.0;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const std::size_t ij = static_cast<std::size_t>(i) * nn + j;
            const std::size_t ji = static_cast<std::size_t>(j) * nn + i;
            const std::string& q_ij = A.qaln[ij];
            const std::string& t_ij = A.taln[ij];
            const std::string& q_ji = A.qaln[ji];
            const std::string& t_ji = A.taln[ji];

            const bool have_ij = !q_ij.empty();
            const bool have_ji = !q_ji.empty();
            double d;
            if (have_ij && have_ji) {
                const double dij = pair_distance(q_ij, t_ij, i, j, ctx);
                const double dji = pair_distance(q_ji, t_ji, j, i, ctx);
                d = 0.5 * (dij + dji);
            } else if (have_ij) {
                d = pair_distance(q_ij, t_ij, i, j, ctx);
            } else if (have_ji) {
                d = pair_distance(q_ji, t_ji, j, i, ctx);
            } else {
                d = kMaxDist;
            }
            D[ij] = d;
            D[ji] = d;
        }
    }
    return {std::move(seqs), std::move(D), nucl};
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// NJSolver::from_fasta
// ---------------------------------------------------------------------------
NJSolver NJSolver::from_fasta(const std::string& path, int threads,
                              float sensitivity, DistMethod method) {
    if (threads <= 0) {
#ifdef _OPENMP
        threads = omp_get_max_threads();
#else
        threads = 1;
#endif
    }

    auto data    = load_fasta_distance(path, threads, sensitivity, method);
    auto& seqs   = data.seqs;
    const auto& D = data.D;
    const int n  = static_cast<int>(seqs.size());
    const std::size_t nn = static_cast<std::size_t>(n);

    NJSolver solver(n);
    for (int i = 0; i < n; i++) solver.set_name(i, seqs[i].name);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            solver.set_distance(i, j, static_cast<float>(
                D[static_cast<std::size_t>(i) * nn + j]));
    return solver;
}

// ============================================================
// NJSolver::run_fasta_ep  –  full FASTA → (MMseqs2|blastn) → NJ + EP pipeline
// ============================================================

std::string NJSolver::run_fasta_ep(const std::string& path,
                                    int threads, float sensitivity,
                                    int n_ep, DistMethod method)
{
    if (threads <= 0) {
#ifdef _OPENMP
        threads = omp_get_max_threads();
#else
        threads = 1;
#endif
    }

    // ── 1. Parse FASTA, run MMseqs2/blastn, build distance matrix D ──────────
    auto data     = load_fasta_distance(path, threads, sensitivity, method);
    auto& seqs    = data.seqs;
    const auto& D = data.D;
    const int  n  = static_cast<int>(seqs.size());
    const std::size_t nn = static_cast<std::size_t>(n);

    // ── 2. Helper: fill a fresh NJSolver from a (possibly perturbed) D ───────
    auto make_solver = [&](auto dist_fn) {
        NJSolver sol(n);
        for (int i = 0; i < n; i++) sol.set_name(i, seqs[i].name);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++)
                sol.set_distance(i, j, dist_fn(i, j));
        return sol;
    };

    // ── 3. Build and run original NJ tree ────────────────────────────────────
    auto orig = make_solver([&](int i, int j) {
        return static_cast<float>(D[static_cast<std::size_t>(i) * nn + j]);
    });

    if (n <= 2) return orig.run();

    orig.run_nj_();
    orig.compute_leaf_sets_();

    // ── 4. Collect original non-trivial bipartition keys ─────────────────────
    std::vector<std::string> bip_keys;
    {
        std::unordered_set<std::string> seen;
        for (int v = n; v < orig.next_id_; v++) {
            std::string key = orig.bip_key_(v);
            if (!key.empty() && seen.insert(key).second)
                bip_keys.push_back(key);
        }
    }
    const int n_bips = static_cast<int>(bip_keys.size());

    if (n_ep <= 0 || n_bips == 0) {
        const double d_ab = orig.dist_.get(orig.last_a_, orig.last_b_);
        return "(" + orig.newick(orig.last_a_) + ":" + fmt_len(d_ab * 0.5)
             + "," + orig.newick(orig.last_b_) + ":" + fmt_len(d_ab * 0.5) + ");";
    }

    // ── 5. EP iterations ──────────────────────────────────────────────────────
    // Distance perturbation: GEV operates on similarity-like values in (0,1],
    // so we round-trip via sim = exp(−d).  Perturbed similarity is mapped back
    // to a distance with d_ep = −log(sim_ep), clamped to [0, kMaxDist].
    //
    // sim is independent of EP iteration, so precompute once into a lower-tri
    // buffer (n*(n−1)/2 doubles) and reuse — saves (n_ep − 1)·n·(n−1)/2 calls
    // to std::exp(), the dominant transcendental in the inner loop.
    std::vector<double> sim_orig(static_cast<std::size_t>(n) * (n - 1) / 2);
    auto tri_lo = [](int ii, int jj) noexcept -> std::size_t {
        // Caller guarantees ii > jj.
        return static_cast<std::size_t>(ii) * (ii - 1) / 2 + jj;
    };
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 1; i < n; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * (i - 1) / 2;
        const std::size_t row  = static_cast<std::size_t>(i) * nn;
        for (int j = 0; j < i; ++j)
            sim_orig[base + j] = std::exp(-D[row + j]);
    }

    std::cerr << "[panjep] Running " << n_ep << " EP iterations"
              << " (" << n_bips << " internal branches)...\n";

    std::vector<int> counts(n_bips, 0);

    // Seed each iteration's RNG from `iter` itself (not thread id), so the
    // perturbed distance matrix — and thus the bipartition counts — are
    // independent of OpenMP scheduling and thread count.  mt19937_64 seeding
    // is ~ns and runs n_ep times: negligible overhead, full reproducibility.
    constexpr uint64_t kEpSeedBase   = 42ULL;
    constexpr uint64_t kEpSeedStride = 6364136223846793005ULL;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int iter = 0; iter < n_ep; iter++) {
        std::mt19937_64 rng(kEpSeedBase +
                            static_cast<uint64_t>(iter) * kEpSeedStride);
        std::uniform_real_distribution<double> uni(1e-10, 1.0 - 1e-10);

        auto ep_sol = make_solver([&](int i, int j) -> float {
            const double sim = sim_orig[tri_lo(i, j)];
            const double sim_ep = gev(uni(rng), sim);
            double d_ep;
            if (sim_ep <= 1e-10) d_ep = kMaxDist;
            else                 d_ep = -std::log(sim_ep);
            if (d_ep < 0.0)      d_ep = 0.0;
            if (d_ep > kMaxDist) d_ep = kMaxDist;
            return static_cast<float>(d_ep);
        });

        ep_sol.run_nj_();
        ep_sol.compute_leaf_sets_();

        std::unordered_set<std::string> ep_bips;
        for (int v = n; v < ep_sol.next_id_; v++) {
            std::string key = ep_sol.bip_key_(v);
            if (!key.empty()) ep_bips.insert(std::move(key));
        }

        for (int bi = 0; bi < n_bips; bi++) {
            if (ep_bips.count(bip_keys[bi])) {
#ifdef _OPENMP
                #pragma omp atomic
#endif
                counts[bi]++;
            }
        }
    }

    // ── 6. Build support map and output Newick ────────────────────────────────
    std::unordered_map<std::string,double> sbk;
    sbk.reserve(static_cast<std::size_t>(n_bips));
    for (int bi = 0; bi < n_bips; bi++)
        sbk[bip_keys[bi]] = static_cast<double>(counts[bi]) / n_ep;

    const double d_ab = orig.dist_.get(orig.last_a_, orig.last_b_);
    return "(" + orig.newick_ep_(orig.last_a_, sbk) + ":" + fmt_len(d_ab * 0.5)
         + "," + orig.newick_ep_(orig.last_b_, sbk) + ":" + fmt_len(d_ab * 0.5) + ");";
}

} // namespace panjep
