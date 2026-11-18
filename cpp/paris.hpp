// One PARIS iteration, the native counterpart of prase/paris/python.py.
//
// Graph indexes are built in Python (prase.paris.state.GraphIndex) and passed
// in as borrowed arrays. Terms are entities followed by literals; predicate p
// has inverse p + P. See the Python module for the algorithm.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace prase::paris {

using Index = std::int64_t;

struct GraphIndex {
    Index num_entities = 0;
    Index num_terms = 0;
    Index num_predicates = 0;
    const double* functionality = nullptr;
    const Index* source_ptr = nullptr;   // num_entities + 1 offsets into source_* (facts by tail)
    const Index* source_pred = nullptr;
    const Index* source_head = nullptr;
    const Index* target_tail = nullptr;  // facts sorted by (head, predicate, tail)
    const Index* group_start = nullptr;  // one (head, predicate) group per entry
    const Index* group_end = nullptr;
    const Index* group_pred = nullptr;
    const Index* head_groups = nullptr;  // num_terms + 1 offsets into group_*
    const Index* pair_keys = nullptr;    // sorted head * num_entities + tail
    const Index* pair_pred = nullptr;
    std::size_t num_pairs = 0;
};

struct Settings {
    double theta;
    double delta;
    double epsilon;
    double smoothing;
    double embedding_weight;
    int threads;
};

// Sparse P(a subset-of b), keyed by a * width + b, with sorted keys.
struct RelationTable {
    std::vector<Index> keys;
    std::vector<double> values;
    Index width = 0;

    double lookup(Index a, Index b) const;
};

struct State {
    Index* left_match;
    double* left_prob;
    Index* right_match;
    double* right_prob;
    const bool* fixed_left;
    const bool* fixed_right;
    const Index* rejected;  // sorted keys left * right_entities + right
    std::size_t num_rejected;
    std::vector<Index> fixed_left_relations;
    std::vector<Index> fixed_right_relations;
};

struct Vectors {
    const float* left = nullptr;
    const float* right = nullptr;
    Index dim = 0;
};

struct IterationResult {
    RelationTable left_relations;
    RelationTable right_relations;
    std::map<std::string, Index> stats;
};

class Engine {
public:
    Engine(GraphIndex left, GraphIndex right, Settings settings);

    // Run one iteration, updating the counterpart arrays in State in place.
    IterationResult iterate(State& state, const RelationTable& left_relations,
                            const RelationTable& right_relations, bool warm, const Vectors* vectors) const;

private:
    GraphIndex left_;
    GraphIndex right_;
    Settings settings_;
};

}  // namespace prase::paris
