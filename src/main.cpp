#include "panjep.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#ifdef _OPENMP
#  include <omp.h>
#endif

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options] <input>\n"
        << "\n"
        << "  input       PHYLIP distance matrix  OR  FASTA sequence file\n"
        << "              (format is auto-detected from the first character)\n"
        << "\n"
        << "  -t N        OpenMP threads for NJ and homology search (default: all available)\n"
        << "  -s S        Search sensitivity [1.0–7.5] (default: 7.5, FASTA only)\n"
        << "                protein input  → MMseqs2 -s flag\n"
        << "                nucleotide input → blastn -task: <3 megablast,\n"
        << "                                  3–5 dc-megablast, 5–7 blastn,\n"
        << "                                  ≥7 blastn-short\n"
        << "  -d METHOD   Evolutionary-distance method (FASTA only).  One of:\n"
        << "                scoredist  BLOSUM62-based ScoreDist (protein)\n"
        << "                poisson    Poisson correction, uniform pi\n"
        << "                           (nc=20 protein, nc=4 nucleotide)\n"
        << "                pdist      raw p-distance (no correction)\n"
        << "                jc69       Jukes-Cantor 1969\n"
        << "                k2p        Kimura 2-parameter (DNA)\n"
        << "                f81        Felsenstein 1981, empirical pi\n"
        << "                f84        Felsenstein 1984 (DNA)\n"
        << "                tn93       Tamura-Nei 1993 (DNA)\n"
        << "                logdet     log-det / paralinear (DNA)\n"
        << "                ry         transversion-only with empirical pi (DNA)\n"
        << "                rysym      transversion-only symmetric (DNA)\n"
        << "                auto       ScoreDist for protein, Poisson(nc=4) for\n"
        << "                           nucleotide [default]\n"
        << "  -e N        EP iterations for branch support (default: 100, FASTA only;\n"
        << "              set 0 to disable EP)\n"
        << "  -v          Print timing / statistics to stderr\n"
        << "  -h          Show this help\n"
        << "\n"
        << "Output: Newick tree on stdout.\n"
        << "        For FASTA input, internal node support values (0.00–1.00)\n"
        << "        are appended after each ')' using the EP method.\n";
}

static bool is_fasta(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) return false;
    int c;
    while ((c = std::fgetc(fp)) != EOF &&
           (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {}
    std::fclose(fp);
    return c == '>';
}

int main(int argc, char** argv) {
    std::string input_path;
    int   n_threads   = 0;      // 0 = use all available
    float sensitivity = 7.5f;
    int   n_ep        = 100;    // EP iterations (FASTA only); 0 = disable
    bool  verbose     = false;
    panjep::DistMethod dmethod = panjep::DistMethod::Auto;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else if (arg == "-v") { verbose = true; }
        else if (arg == "-t" && i + 1 < argc) { n_threads   = std::atoi(argv[++i]); }
        else if (arg == "-s" && i + 1 < argc) { sensitivity = std::stof(argv[++i]); }
        else if (arg == "-e" && i + 1 < argc) { n_ep        = std::atoi(argv[++i]); }
        else if (arg == "-d" && i + 1 < argc) {
            std::string m = argv[++i];
            if      (m == "scoredist") dmethod = panjep::DistMethod::ScoreDist;
            else if (m == "poisson")   dmethod = panjep::DistMethod::Poisson;
            else if (m == "pdist")     dmethod = panjep::DistMethod::PDist;
            else if (m == "jc69")      dmethod = panjep::DistMethod::JC69;
            else if (m == "k2p")       dmethod = panjep::DistMethod::K2P;
            else if (m == "f81")       dmethod = panjep::DistMethod::F81;
            else if (m == "f84")       dmethod = panjep::DistMethod::F84;
            else if (m == "tn93")      dmethod = panjep::DistMethod::TN93;
            else if (m == "logdet")    dmethod = panjep::DistMethod::LogDet;
            else if (m == "ry")        dmethod = panjep::DistMethod::RY;
            else if (m == "rysym")     dmethod = panjep::DistMethod::RYSym;
            else if (m == "auto")      dmethod = panjep::DistMethod::Auto;
            else {
                std::cerr << "Unknown distance method: " << m
                          << " (expected scoredist | poisson | pdist | jc69 |"
                             " k2p | f81 | f84 | tn93 | logdet | ry | rysym | auto)\n";
                return 1;
            }
        }
        else if (arg[0] != '-') { input_path = arg; }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    if (input_path.empty()) {
        std::cerr << "Error: no input file specified.\n";
        usage(argv[0]);
        return 1;
    }

#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    if (verbose)
        std::cerr << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    if (verbose)
        std::cerr << "OpenMP not available; running single-threaded.\n";
    (void)n_threads;
#endif

    const bool fasta = is_fasta(input_path);

    try {
        auto t0 = std::chrono::steady_clock::now();

        std::string tree;
        if (fasta) {
            tree = panjep::NJSolver::run_fasta_ep(
                input_path, n_threads, sensitivity, n_ep, dmethod);
        } else {
            auto solver = panjep::NJSolver::from_phylip(input_path);
            auto t1     = std::chrono::steady_clock::now();
            tree        = solver.run();
            auto t2     = std::chrono::steady_clock::now();
            std::cout << tree << "\n";
            if (verbose) {
                auto ms = [](auto a, auto b) {
                    return std::chrono::duration_cast<
                               std::chrono::milliseconds>(b - a).count();
                };
                std::cerr << "I/O:  " << ms(t0, t1) << " ms\n";
                std::cerr << "NJ:   " << ms(t1, t2) << " ms\n";
                std::cerr << "Total:" << ms(t0, t2) << " ms\n";
            }
            return 0;
        }

        auto t2 = std::chrono::steady_clock::now();
        std::cout << tree << "\n";

        if (verbose) {
            auto ms = [](auto a, auto b) {
                return std::chrono::duration_cast<
                           std::chrono::milliseconds>(b - a).count();
            };
            std::cerr << "Search+NJ+EP: " << ms(t0, t2) << " ms\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
