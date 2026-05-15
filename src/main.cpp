#include "panjep.hpp"

#include <algorithm>
#include <cctype>
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
        << "                lg|wag|jtt|dayhoff|dcmut|mtrev|rtrev|cprev|vt|\n"
        << "                hivb|hivw|flu   empirical AA models (ML branch length\n"
        << "                           under Q matrix; protein-only)\n"
        << "                auto       ScoreDist for protein, Poisson(nc=4) for\n"
        << "                           nucleotide [default]\n"
        << "  -p MODEL    FastME-style alias for protein models.  Accepts the\n"
        << "              same single-letter and full-name shortcuts as fastme -p\n"
        << "              (e.g. -p LG, -p L, -p WAG, -p HIVB, -p F81, -p P-DIST).\n"
        << "  -e N        EP iterations for branch support (default: 100, FASTA only;\n"
        << "              set 0 to disable EP)\n"
        << "  -v          Print timing / statistics to stderr\n"
        << "  -h          Show this help\n"
        << "\n"
        << "Output: Newick tree on stdout.\n"
        << "        For FASTA input, internal node support values (0.00–1.00)\n"
        << "        are appended after each ')' using the EP method.\n";
}

// Parse a fastme -p style protein model token.  Accepts the same set as
// fastme's testP / getModel_PROTEIN (interface_utilities.c), case-insensitive,
// including the one-letter shortcuts:
//   B → HIVB, C → CPREV, D → DCMUT, F → F81LIKE, H → DAYHOFF, I → HIVW,
//   J → JTT,  L → LG,    M → MTREV, P → PDIST,   R → RTREV,   U → FLU,
//   V → VT,   W → WAG
// plus full names (LG / VT / JTT / WAG / FLU / F81 / F81LIKE / F81-LIKE /
// HIVB / HIVW / CPREV / DCMUT / MTREV / RTREV / PDIST / P-DIST / DAYHOFF).
// Returns true with *out set on success; false on unknown token.
static bool parse_protein_model(const std::string& in, panjep::DistMethod& out) {
    std::string s = in;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    if (s.size() == 1) {
        switch (s[0]) {
            case 'B': out = panjep::DistMethod::HIVb;    return true;
            case 'C': out = panjep::DistMethod::CpREV;   return true;
            case 'D': out = panjep::DistMethod::DCMut;   return true;
            case 'F': out = panjep::DistMethod::F81;     return true;
            case 'H': out = panjep::DistMethod::Dayhoff; return true;
            case 'I': out = panjep::DistMethod::HIVw;    return true;
            case 'J': out = panjep::DistMethod::JTT;     return true;
            case 'L': out = panjep::DistMethod::LG;      return true;
            case 'M': out = panjep::DistMethod::MtREV;   return true;
            case 'P': out = panjep::DistMethod::PDist;   return true;
            case 'R': out = panjep::DistMethod::RtREV;   return true;
            case 'U': out = panjep::DistMethod::FLU;     return true;
            case 'V': out = panjep::DistMethod::VT;      return true;
            case 'W': out = panjep::DistMethod::WAG;     return true;
            default:  return false;
        }
    }
    if (s == "LG")                                        { out = panjep::DistMethod::LG;      return true; }
    if (s == "VT")                                        { out = panjep::DistMethod::VT;      return true; }
    if (s == "JTT")                                       { out = panjep::DistMethod::JTT;     return true; }
    if (s == "WAG")                                       { out = panjep::DistMethod::WAG;     return true; }
    if (s == "FLU")                                       { out = panjep::DistMethod::FLU;     return true; }
    if (s == "F81" || s == "F81LIKE" || s == "F81-LIKE")  { out = panjep::DistMethod::F81;     return true; }
    if (s == "HIVB")                                      { out = panjep::DistMethod::HIVb;    return true; }
    if (s == "HIVW")                                      { out = panjep::DistMethod::HIVw;    return true; }
    if (s == "CPREV")                                     { out = panjep::DistMethod::CpREV;   return true; }
    if (s == "DCMUT")                                     { out = panjep::DistMethod::DCMut;   return true; }
    if (s == "MTREV")                                     { out = panjep::DistMethod::MtREV;   return true; }
    if (s == "RTREV")                                     { out = panjep::DistMethod::RtREV;   return true; }
    if (s == "PDIST" || s == "P-DIST")                    { out = panjep::DistMethod::PDist;   return true; }
    if (s == "DAYHOFF")                                   { out = panjep::DistMethod::Dayhoff; return true; }
    return false;
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
            std::string mlc = m;
            std::transform(mlc.begin(), mlc.end(), mlc.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            if      (mlc == "scoredist") dmethod = panjep::DistMethod::ScoreDist;
            else if (mlc == "poisson")   dmethod = panjep::DistMethod::Poisson;
            else if (mlc == "pdist")     dmethod = panjep::DistMethod::PDist;
            else if (mlc == "jc69")      dmethod = panjep::DistMethod::JC69;
            else if (mlc == "k2p")       dmethod = panjep::DistMethod::K2P;
            else if (mlc == "f81")       dmethod = panjep::DistMethod::F81;
            else if (mlc == "f84")       dmethod = panjep::DistMethod::F84;
            else if (mlc == "tn93")      dmethod = panjep::DistMethod::TN93;
            else if (mlc == "logdet")    dmethod = panjep::DistMethod::LogDet;
            else if (mlc == "ry")        dmethod = panjep::DistMethod::RY;
            else if (mlc == "rysym")     dmethod = panjep::DistMethod::RYSym;
            else if (mlc == "auto")      dmethod = panjep::DistMethod::Auto;
            else if (mlc == "lg")        dmethod = panjep::DistMethod::LG;
            else if (mlc == "wag")       dmethod = panjep::DistMethod::WAG;
            else if (mlc == "jtt")       dmethod = panjep::DistMethod::JTT;
            else if (mlc == "dayhoff")   dmethod = panjep::DistMethod::Dayhoff;
            else if (mlc == "dcmut")     dmethod = panjep::DistMethod::DCMut;
            else if (mlc == "mtrev")     dmethod = panjep::DistMethod::MtREV;
            else if (mlc == "rtrev")     dmethod = panjep::DistMethod::RtREV;
            else if (mlc == "cprev")     dmethod = panjep::DistMethod::CpREV;
            else if (mlc == "vt")        dmethod = panjep::DistMethod::VT;
            else if (mlc == "hivb")      dmethod = panjep::DistMethod::HIVb;
            else if (mlc == "hivw")      dmethod = panjep::DistMethod::HIVw;
            else if (mlc == "flu")       dmethod = panjep::DistMethod::FLU;
            else {
                std::cerr << "Unknown distance method: " << m << "\n";
                return 1;
            }
        }
        else if (arg == "-p" && i + 1 < argc) {
            // FastME-compatible protein-model selector.  -p LG, -p L, -p WAG, …
            std::string m = argv[++i];
            if (!parse_protein_model(m, dmethod)) {
                std::cerr << "-p option: '" << m
                          << "' invalid evolutionary model.\n"
                          << "Expected one of: LG, WAG, JTT, Dayhoff, DCMut, "
                             "MtREV, RtREV, CpREV, VT, HIVb, HIVw, FLU, "
                             "F81, PDist (or single-letter aliases).\n";
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
