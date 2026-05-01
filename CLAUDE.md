# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`panjep` is a C++17 parallel Neighbour-Joining (NJ) phylogenetic tree builder. The single binary accepts either a PHYLIP distance matrix or a FASTA sequence file (format auto-detected from the first non-whitespace character in `main.cpp`). FASTA inputs trigger an all-vs-all homology search — MMseqs2 for protein, NCBI `blastn` for nucleotide (auto-detected by scanning ≤50k chars for nucleotide alphabet at >90%) — and the per-pair pairwise alignments returned by the search tool are converted to evolutionary distances via either ScoreDist (BLOSUM62; protein) or Poisson correction (nc=20 protein / nc=4 nucleotide). NJ runs on that distance matrix and optional EP iterations attach bipartition support values.

## Common commands

```bash
make                          # release build (auto-detects clang++ on macOS, g++ on Linux)
make OMP=0                    # single-threaded build
make CXX=g++                  # force compiler
make test                     # quick correctness on test/ inputs (PHYLIP + FASTA)
make bench                    # synthetic n=500/1000/2000 benchmarks via tools/gen_test
make clean
make install PREFIX=~/.local
```

Single test runs (the `panjep` binary auto-detects format):
```bash
./panjep -v test/sample5.phy                  # PHYLIP
./panjep -v test/sample5.faa                  # protein FASTA → MMseqs2 → ScoreDist
./panjep -v test/sample_nucl.fna              # nucleotide FASTA → blastn → Poisson(nc=4)
./panjep -t 1 -e 0 test/sample5.faa           # 1 thread, no EP support values
./panjep -d poisson test/sample5.faa          # force Poisson(nc=20) for protein
./panjep -d scoredist test/sample5.faa        # force ScoreDist (default for protein)
```

Address/UB-sanitizer build (used for memory debugging — referenced in `.claude/settings.local.json`):
```bash
clang++ -std=c++17 -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -g \
        -o panjep_asan src/main.cpp src/panjep.cpp
```

External tools required at runtime for FASTA inputs only — they are invoked via `execvp`, so they must be on `PATH`:
- Protein FASTA: `mmseqs` (MMseqs2)
- Nucleotide FASTA: `blastn`, `makeblastdb` (NCBI BLAST+)

## Architecture

Three source files under `src/`:

- **`panjep.hpp`** — public API: `DistMatrix` (lower-triangular `float` storage, index `i*(i-1)/2 + j` for `i>j`) and `NJSolver`. `NJSolver` exposes `from_phylip`, `from_fasta`, `run`, and the full pipeline `run_fasta_ep`.
- **`panjep.cpp`** — all implementation. Roughly four sections, in order: (1) NJ core (`init_row_sums` / `find_min_q` / `do_merge` / `run_nj_`), (2) Newick + EP support helpers, (3) PHYLIP parser, (4) FASTA pipeline + MMseqs2/blastn wrappers + EP iterations.
- **`main.cpp`** — argv parsing, format detection, dispatch to `from_phylip().run()` or `run_fasta_ep()`, timing.

### NJ hot-path invariants (read before touching `find_min_q` / `do_merge`)

- The matrix is allocated for `2n−1` nodes upfront (NJ creates `n−1` internal nodes); `next_id_` allocates internal node IDs `n .. 2n−2`. `do_merge` writes `dist_.at(k, m)` for the new node `k`, never resizes.
- `R_` is `double` for accumulation accuracy, `R_masked_` is `float` for the SIMD inner loop. **Inactive nodes are masked by setting `R_masked_[x] = kInactiveR (-1e30f)`**; this makes `Q = n2*d - R[i] - Rm[j]` blow up for inactive `j` so the inner loop stays branch-free. Both arrays must be kept in sync — every place that toggles `active_[x]` must also update `R_masked_[x]`.
- `find_min_q` only scans `i > j` (each pair once) and uses a row-level lower bound `−R[i] − R_max ≥ best_Q` to skip whole rows. Don't break the lower-triangle walking pattern.
- An ARM NEON inner loop processes 4 floats per cycle (`__ARM_NEON`), with a scalar tail; non-ARM targets fall through to the scalar loop. Keep both branches in sync if you change the Q formula.
- `do_merge` is O(n_active) and has a serial vs OpenMP-parallel switch at `n_active_ >= 256` to amortize fork/join overhead. The new row's R is built with `reduction(+:R_k)` — preserve this if you parallelize further.

### FASTA → alignment → distance → EP pipeline

`load_fasta_distance` (anonymous namespace in `panjep.cpp`) is the shared entry point used by both `from_fasta` and `run_fasta_ep`. Sequence runs:

1. `parse_fasta` (bulk-read, name = first whitespace-delimited token after `>`).
2. `detect_nucleotide` (≥90% nucleotide alphabet on a 50k-char sample) selects the search tool *and* the default distance method.
3. `mkdtemp` under `/tmp/panjep_XXXXXX`, wrapped in `ScopedTempDir` (RAII; `rm -rf` on any exit path).
4. `run_mmseqs2` *or* `run_blastn` — both use `run_safe`, which is `fork`+`execvp` (no shell, immune to injection). The tools are configured to emit *aligned sequences*, not just bitscores: mmseqs is invoked as `search ... -a 1` followed by `convertalis ... --format-output query,target,qaln,taln,bits`; blastn uses `-outfmt "6 qseqid sseqid qseq sseq bitscore"`. Sensitivity (`-s`) maps to blastn `-task` thresholds: `<3 megablast`, `<5 dc-megablast`, `<7 blastn`, else `blastn-short`.
5. `parse_aln_m8` reads the 5-col TSV; for duplicate `(query,target)` HSPs, the alignment of the max-bitscore HSP wins.
6. For each pair `i<j`, `pair_distance(qaln, taln, method, nucl)` computes a distance via `scoredist` (BLOSUM62) or `poisson_dist` (nc=20 / nc=4). When both directions `i→j` and `j→i` have hits, distances are averaged; missing pairs collapse to `kMaxDist = 5.0`. Diagonal is forced to 0. The pairwise loop is OpenMP-parallel.

`DistMethod::Auto` (CLI: `-d auto`, the default) picks ScoreDist for protein and Poisson(nc=4) for nucleotide. `-d scoredist` falls back to Poisson(nc=4) if the input is detected as nucleotide (BLOSUM62 doesn't apply).

EP (`run_fasta_ep`) perturbs distances directly. The GEV kernel is defined on similarity-like values in `(0,1]`, so each iteration converts `d → sim = exp(−d)`, applies `gev(uni, sim)`, and maps back via `d_ep = −log(sim_ep)` clamped to `[0, kMaxDist]`. This matches the formulation in the mi6 reference. NJ runs on the perturbed matrix and bipartitions are counted; canonicalization via `bip_key_` (always the side containing leaf 0; trivial bips `< 2` taxa per side return `""`). Counts/`n_ep` become support values rendered into Newick by `newick_ep_`. EP is OpenMP-parallel over iterations with per-thread RNG seeded `42 + tid * 6364136223846793005`.

### Distance kernels

- `scoredist(A, B)` — Sonnhammer & Hollich 2005. Iterates `min(|A|,|B|)` columns, skipping any position where either char maps to `kAaInvalid` via the `kS2P` table (gap, X, ambiguous). Accumulates `sc`, `scA = sum BLOSUM62[a][a]`, `scB = sum BLOSUM62[b][b]`, and `len`. Then `od = (sc − scR) / (scMAX − scR)` with `scR = −0.5209·len` (BLOSUM62 expected per-column score), and `d = −log(od)`, capped at `kMaxDist`.
- `poisson_dist(A, B, nc)` — counts identical non-gap columns to get `s = 1 − matches/total`. If `s ≥ (nc−1)/nc` the formula is undefined and the saturation cap is returned; otherwise `d = −((nc−1)/nc)·log(1 − s·nc/(nc−1))`. `kS2P` is used for protein, `kN2P` for nucleotide; both lookup tables are `static const std::array<int,256>` initialized via an immediately-invoked lambda (operator[]-write into `std::array` only became `constexpr` in C++20, so a constexpr-init function won't compile under `-std=c++17`).

## Build-system notes

- macOS: the Makefile **forcibly rewrites `CXX` away from conda-forge's `arm64-apple-darwin*-c++` wrapper** because it injects `-isystem $CONDA_PREFIX/include`, which loads conda's libc++ headers while the binary still links the system `libc++.dylib` — the resulting `std::string` ABI mismatch causes runtime malloc aborts. Don't undo that block (`Makefile:30-37`) without re-testing in an activated conda env.
- macOS OpenMP: the Makefile probes `brew --prefix libomp`, then `/opt/homebrew/opt/libomp`, then `/usr/local/opt/libomp`, then falls back to `-fopenmp` if the compiler accepts it. If none work, it warns and builds single-threaded rather than failing.
- Architecture flags: `arm64` gets `-march=armv8-a -flax-vector-conversions`; everything else gets `-march=native`.
- `NJ_PROFILE` (compile-time define) enables `find_min_q` / `do_merge` timing in `run_nj_`.

## Test data

- `test/sample5.phy`, `test/wiki5.phy` — small lower-triangular PHYLIP matrices (the wiki5 input is the canonical 5-taxon NJ example).
- `test/sample5.faa` — 5-protein FASTA (exercises MMseqs2 path).
- `test/sample_nucl.fna` — nucleotide FASTA (exercises blastn path).
- `tools/gen_test.cpp` builds `gen_test`, which emits a random PHYLIP matrix from 1-D leaf positions + noise (additive, satisfies triangle inequality). Used by `make bench`.
