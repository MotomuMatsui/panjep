#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace panjep {

// ---------------------------------------------------------------------------
// Distance method selector for FASTA pipeline.
// Methods marked DNA are only meaningful for nucleotide input; protein input
// silently falls back to a sensible default (Poisson(nc=20)) for those.
//
//   Auto      → ScoreDist for protein, Poisson(nc=4) for nucleotide
//   ScoreDist → BLOSUM62-based (Sonnhammer & Hollich 2005); protein only
//   Poisson   → −((n−1)/n)·ln(1 − n/(n−1)·p) with uniform stationary
//               (nc=20 protein, nc=4 nucleotide)
//   PDist     → raw Hamming distance, no correction
//   JC69      → Jukes-Cantor 1969: −(3/4)·ln(1 − (4/3)·p)
//   K2P       → Kimura 2-parameter (DNA)
//   F81       → Felsenstein 1981 with empirical stationary
//               (Poisson generalised to non-uniform pi; protein & DNA)
//   F84       → Felsenstein 1984 (DNA)
//   TN93      → Tamura-Nei 1993 (DNA)
//   LogDet    → Lockhart et al. 1994 paralinear/log-det (DNA)
//   RY        → Transversion-only with empirical stationary (DNA)
//   RYSym     → Transversion-only symmetric: −0.5·ln(1 − 2·b) (DNA)
//
// All kernels are ported from FastME 2.1.6.4 (Lefort, Desper & Gascuel 2015).
// They are applied per-pair on the qaln/taln columns produced by mmseqs2
// (--format-output query,target,qaln,taln,bits) or blastn
// (-outfmt "6 qseqid sseqid qseq sseq bitscore").  Stationary frequencies for
// F81 / F84 / TN93 / RY are computed from the raw input sequences (residue
// composition over the whole FASTA), and per-sequence frequencies for LogDet.
// ---------------------------------------------------------------------------
enum class DistMethod {
    Auto, ScoreDist, Poisson,
    PDist, JC69, K2P, F81, F84, TN93, LogDet, RY, RYSym
};

// ---------------------------------------------------------------------------
// DistMatrix
// Lower-triangular distance matrix: element (i,j), i>j, at i*(i-1)/2 + j.
// Float precision to halve memory and improve cache utilisation.
// ---------------------------------------------------------------------------
class DistMatrix {
public:
    explicit DistMatrix(int n);

    int size() const noexcept { return n_; }

    float& at(int i, int j) noexcept;
    float  get(int i, int j) const noexcept;

    // Pointer to the contiguous block d(i,0)…d(i,i−1) in the lower triangle.
    const float* row_ptr(int i) const noexcept { return data_.data() + lo_idx(i); }
    float*       row_ptr(int i) noexcept       { return data_.data() + lo_idx(i); }

private:
    int n_;
    std::vector<float> data_;   // size n*(n-1)/2

    static std::size_t tri_idx(int i, int j) noexcept {
        if (i < j) { int t = i; i = j; j = t; }
        return static_cast<std::size_t>(i)*(i-1)/2 + static_cast<std::size_t>(j);
    }
    static std::size_t lo_idx(int i) noexcept {
        return static_cast<std::size_t>(i)*(i-1)/2;
    }
    friend class NJSolver;
};

// ---------------------------------------------------------------------------
// NJSolver  –  Neighbour Joining with parallel O(n²/p) search + ARM NEON SIMD
//
// Algorithm: Saitou & Nei (1987) / Studier & Keppler (1988)
//
// Optimisations:
//   • No sorted-row maintenance: do_merge is O(n) per merge, not O(n²)
//   • OpenMP over outer i-loop of find_min_q (lower-triangle, each pair once)
//   • ARM NEON: process 4 float comparisons per cycle in the inner j-loop
//   • R-masked sentinel (-∞) allows branch-free inactive-node masking in SIMD
//   • Row-level lower-bound check: skip row i when −R[i]−R_max ≥ thread best
//   • Parallel do_merge with OpenMP reduction for the new row-sum
//
// Complexity: O(n² log n) init, O(n³/p) search, O(n²) merge  →  O(n³/p) total
// Memory:     O(n²) for the distance matrix + O(n) bookkeeping
// ---------------------------------------------------------------------------
class NJSolver {
public:
    explicit NJSolver(int n_taxa);

    void set_distance(int i, int j, float d);
    void set_name    (int i, const std::string& name);

    // Run NJ and return the resulting tree in Newick format.
    std::string run();

    static NJSolver from_phylip(std::istream& in);
    static NJSolver from_phylip(const std::string& path);

    // Build NJSolver from a FASTA file.  Auto-detects protein vs nucleotide:
    // protein → MMseqs2 all-vs-all, nucleotide → NCBI blastn all-vs-all.
    // Per-pair pairwise alignment (qaln/taln) is converted to an evolutionary
    // distance using ScoreDist (protein) or Poisson correction (nucleotide).
    // threads=0 → use all available cores.  sensitivity in [1.0, 7.5].
    static NJSolver from_fasta(const std::string& path,
                                int        threads     = 0,
                                float      sensitivity = 7.5f,
                                DistMethod method      = DistMethod::Auto);

    // Full FASTA → (MMseqs2 or blastn) → NJ + EP pipeline.
    // Returns a Newick string with EP support values on internal branches.
    // n_ep=0 disables EP (returns plain NJ Newick).  threads=0 → all cores.
    static std::string run_fasta_ep(const std::string& path,
                                     int        threads     = 0,
                                     float      sensitivity = 7.5f,
                                     int        n_ep        = 100,
                                     DistMethod method      = DistMethod::Auto);

private:
    int n_leaves_;
    int max_nodes_;   // 2*n_leaves_ - 1

    int n_active_;
    int next_id_;     // next internal-node ID (starts at n_leaves_)

    DistMatrix           dist_;
    std::vector<double>  R_;         // row sums (double for accumulation accuracy)
    std::vector<float>   R_masked_;  // float copy; −1e30f for inactive nodes (SIMD mask)
    std::vector<bool>    active_;
    std::vector<std::string> names_;

    struct TreeNode {
        int parent = -1;
        std::vector<std::pair<int, double>> children;
    };
    std::vector<TreeNode> tree_;

    // Set by run_nj_(): the last two active nodes joined in the final edge.
    int last_a_ = -1;
    int last_b_ = -1;

    // Set by compute_leaf_sets_(): sorted leaf indices reachable from each node.
    std::vector<std::vector<int>> leaf_sets_;

    void               init_row_sums();
    std::pair<int,int> find_min_q()  const;
    void               do_merge(int i, int j);

    // Core NJ loop: runs all merges, sets last_a_/last_b_.  Called by run().
    void run_nj_();

    // Populate leaf_sets_ bottom-up.  Call after run_nj_().
    void compute_leaf_sets_();

    // Canonical bipartition key for node v (side containing leaf 0).
    // Returns "" for trivial bipartitions (one side has < 2 leaves).
    std::string bip_key_(int v) const;

    // Newick with EP support values appended after each internal node's ')'.
    // support_by_key: canonical bipartition key → EP score in [0,1].
    std::string newick_ep_(int v,
                            const std::unordered_map<std::string,double>& sbk) const;

    std::string newick(int v) const;
    static std::string fmt_len(double d);
};

} // namespace panjep
