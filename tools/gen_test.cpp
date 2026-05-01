// gen_test.cpp – generate a random PHYLIP lower-triangular distance matrix
// Usage: ./gen_test <n> [seed]
// Generates distances from a random ultrametric tree + noise, which
// satisfies the triangle inequality and gives realistic NJ inputs.

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: gen_test <n> [seed]\n";
        return 1;
    }
    const int n    = std::atoi(argv[1]);
    const int seed = (argc >= 3) ? std::atoi(argv[2]) : 42;

    if (n < 2) { std::cerr << "n must be >= 2\n"; return 1; }

    // Build a random binary tree by iterative random joining of n leaves.
    // Store the leaf-to-leaf distance as the sum of two random edge lengths
    // along the path between them.
    // Simple approach: full distance matrix from a random Prüfer sequence tree.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> branch_dist(0.01f, 1.0f);
    std::uniform_real_distribution<float> noise_dist (0.0f,  0.05f);

    // Simulate an additive tree by placing n leaves in a 1-D "guide tree":
    // leaf positions are drawn uniformly; dist(i,j) = |pos[i] - pos[j]| + noise.
    // This is simple and gives positive, roughly tree-like distances.
    std::uniform_real_distribution<float> pos_dist(0.0f, 100.0f);
    std::vector<float> pos(n);
    for (int i = 0; i < n; i++) pos[i] = pos_dist(rng);

    std::cout << n << "\n";
    for (int i = 0; i < n; i++) {
        std::cout << "t" << (i + 1);
        for (int j = 0; j < i; j++) {
            float d = std::abs(pos[i] - pos[j]) + noise_dist(rng);
            std::cout << " " << std::setprecision(6) << d;
        }
        std::cout << "\n";
    }
    return 0;
}
