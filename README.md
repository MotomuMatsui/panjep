# panjep

A fast, parallel Neighbour-Joining (NJ) phylogenetic tree builder written in C++17.

`panjep` accepts either a PHYLIP distance matrix or a FASTA sequence file. For FASTA input it runs an all-vs-all homology search (MMseqs2 for protein, NCBI BLAST+ `blastn` for nucleotide), converts the per-pair alignments into evolutionary distances, builds an NJ tree, and optionally attaches bipartition support values via Edge Perturbation (EP). The format is auto-detected; a single binary handles every path.

## Features

- **Single binary, two inputs.** PHYLIP distance matrix or FASTA — auto-detected from the first character.
- **Parallel NJ core.** OpenMP-parallel `Q`-matrix scan with an ARM NEON inner loop and a row-level lower-bound that skips whole rows.
- **Lower-triangular `float` storage.** Minimal memory footprint; matrix is sized for `2n−1` nodes upfront and never resizes.
- **FASTA pipeline.** Auto-selects MMseqs2 (protein) or `blastn` (nucleotide) by alphabet detection. Tools are invoked via `fork`+`execvp` (no shell), so input filenames are injection-safe.
- **Many distance methods.** ScoreDist (BLOSUM62), Poisson, p-distance, JC69, K2P, F81, F84, TN93, log-det, RY (transversion-only).
- **EP branch support.** GEV-perturbed distances over `n` iterations; bipartition frequencies are written into the Newick output.

## Build

```bash
make                  # release build (auto-detects clang++ on macOS, g++ on Linux)
make OMP=0            # single-threaded
make CXX=g++          # force a compiler
make test             # quick correctness on test/ inputs
make bench            # synthetic n=500/1000/2000 benchmarks
make install PREFIX=~/.local
```

Requirements:

- C++17 compiler (clang++ or g++).
- OpenMP (optional but recommended). On macOS, `brew install libomp` — the Makefile will discover it.
- For FASTA input only:
  - Protein → [`mmseqs`](https://github.com/soedinglab/MMseqs2) on `PATH`.
  - Nucleotide → `blastn` and `makeblastdb` ([NCBI BLAST+](https://blast.ncbi.nlm.nih.gov/)) on `PATH`.

## Usage

```
panjep [options] <input>

  -t N        OpenMP threads (default: all available)
  -s S        Search sensitivity 1.0–7.5 (FASTA only)
  -d METHOD   Distance method: scoredist | poisson | pdist | jc69 | k2p
              | f81 | f84 | tn93 | logdet | ry | rysym | auto  (default: auto)
  -e N        EP iterations for branch support (default: 100; 0 disables)
  -v          Print timing / statistics to stderr
  -h          Show help

Output: Newick tree on stdout.
```

`-d auto` (the default) selects ScoreDist for protein and Poisson(nc=4) for nucleotide. For FASTA input, internal-node support values (`0.00`–`1.00`) are appended after each `)` in the Newick output.

### Examples

```bash
./panjep -v test/sample5.phy                  # PHYLIP distance matrix
./panjep -v test/sample5.faa                  # protein FASTA → MMseqs2 → ScoreDist
./panjep -v test/sample_nucl.fna              # nucleotide FASTA → blastn → Poisson(nc=4)
./panjep -t 1 -e 0 test/sample5.faa           # single-threaded, no EP support values
./panjep -d poisson test/sample5.faa          # force Poisson(nc=20) on protein
```

## How it works

For FASTA input, `panjep` runs the following pipeline:

1. **Parse FASTA** and detect protein vs nucleotide by sampling up to 50 kB and checking whether ≥90% of characters lie in the nucleotide alphabet.
2. **All-vs-all search** in a private `mkdtemp` directory:
   - Protein: `mmseqs search ... -a 1` then `mmseqs convertalis ... --format-output query,target,qaln,taln,bits`.
   - Nucleotide: `blastn -outfmt "6 qseqid sseqid qseq sseq bitscore"`. Sensitivity `-s` maps to `-task` (megablast → blastn-short).
3. **Pairwise distances** from the aligned sequences (OpenMP-parallel). When both directions `i→j` and `j→i` have hits, distances are averaged; missing pairs collapse to a saturation cap of 5.0.
4. **Neighbour-Joining.** `O(n³)` core with vectorised `Q`-min scan; `do_merge` switches between serial and parallel reductions at `n_active ≥ 256`.
5. **EP support (optional).** Each iteration converts `d → exp(−d)`, applies a GEV perturbation, maps back via `−log(·)`, and re-runs NJ. Bipartition frequencies (canonicalised on the side containing leaf 0) become support values.

### Distance kernels

- **ScoreDist** (Sonnhammer & Hollich, 2005). BLOSUM62 with `scR = −0.5209·len`; positions with gap/X/ambiguous chars are skipped.
- **Poisson correction.** `d = −((nc−1)/nc) · log(1 − s · nc/(nc−1))`, with `nc = 20` for protein and `nc = 4` for nucleotide; saturated pairs return the cap.
- **DNA-specific.** JC69, K2P, F81, F84, TN93, log-det, RY/RYsym.

## Repository layout

```
src/
  panjep.hpp     public API: DistMatrix, NJSolver
  panjep.cpp     NJ core, Newick + EP helpers, PHYLIP parser, FASTA pipeline
  main.cpp       argv parsing, format detection, dispatch, timing
tools/
  gen_test.cpp   random PHYLIP matrix generator (used by `make bench`)
test/            small PHYLIP and FASTA inputs for `make test`
Makefile
```

## License

Released under the [GNU General Public License v3.0 or later](https://www.gnu.org/licenses/gpl-3.0.html). See the `LICENSE` file for the full text.
