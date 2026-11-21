#include "paris.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <unordered_map>
#include <utility>

namespace prase::paris {

double RelationTable::lookup(Index a, Index b) const {
    const Index key = a * width + b;
    const auto it = std::lower_bound(keys.begin(), keys.end(), key);
    return it != keys.end() && *it == key ? values[static_cast<std::size_t>(it - keys.begin())] : 0.0;
}

namespace {

struct Update {
    Index entity;
    Index counterpart;
    double probability;
};

// What one thread computes over a contiguous range of source entities.
// The relation statistics of one range of entities, as (key, weight) terms in fact order. The terms of
// all ranges are summed in order afterwards, so every thread count gives the same sums as one thread
// and as the NumPy engine.
struct Partial {
    std::vector<std::pair<Index, double>> norm;
    std::vector<std::pair<Index, double>> support;
    std::vector<Update> updates;
    Index evidence = 0;
};

double cosine(const Vectors& vectors, Index left, Index right) {
    const float* a = vectors.left + left * vectors.dim;
    const float* b = vectors.right + right * vectors.dim;
    double dot = 0.0, left_norm = 0.0, right_norm = 0.0;
    for (Index d = 0; d < vectors.dim; ++d) {
        const double x = a[d], y = b[d];
        dot += x * y;
        left_norm += x * x;
        right_norm += y * y;
    }
    const double norms = std::sqrt(left_norm * right_norm);
    return norms > 0.0 ? dot / norms : 0.0;
}

bool rejected(const State& state, Index key) {
    return std::binary_search(state.rejected, state.rejected + state.num_rejected, key);
}

struct Pass {
    const GraphIndex& source;
    const GraphIndex& target;
    const Index* match;
    const double* prob;
    const RelationTable& forward;
    const RelationTable& backward;
    const Settings& settings;
    bool warm;
    const State* state;      // Set for a pass that updates entities.
    const Vectors* vectors;  // Set when embedding similarity is used.

    void run(Index first, Index last, Partial& out) const {
        std::unordered_map<Index, double> products;
        const Index* pairs_end = target.pair_keys + target.num_pairs;
        for (Index x = first; x < last; ++x) {
            products.clear();
            const Index x_match = match[x];
            const double x_prob = prob[x];
            for (Index fact = source.source_ptr[x]; fact < source.source_ptr[x + 1]; ++fact) {
                const Index head = source.source_head[fact];
                const Index head_match = match[head];
                const double head_prob = prob[head];
                // A neighbor counts as evidence when its counterpart has a positive probability of at least theta.
                if (head_match < 0 || !(head_prob > 0.0) || head_prob < settings.theta) {
                    continue;
                }
                const Index q = source.source_pred[fact];
                if (x_match >= 0) {
                    const double weight = head_prob * x_prob;
                    out.norm.emplace_back(q, weight);
                    if (x_prob > 0.0) {
                        const Index key = head_match * target.num_entities + x_match;
                        for (const Index* it = std::lower_bound(target.pair_keys, pairs_end, key);
                             it != pairs_end && *it == key; ++it) {
                            const Index predicate = target.pair_pred[it - target.pair_keys];
                            out.support.emplace_back(q * target.num_predicates + predicate, weight);
                        }
                    }
                }
                if (state == nullptr) {
                    continue;
                }
                for (Index group = target.head_groups[head_match]; group < target.head_groups[head_match + 1];
                     ++group) {
                    const Index q2 = target.group_pred[group];
                    double sub = forward.lookup(q, q2) / settings.epsilon;
                    double sup = backward.lookup(q2, q) / settings.epsilon;
                    if (sub < settings.theta && sup < settings.theta) {
                        if (!warm) {
                            continue;
                        }
                        sub = sup = settings.theta;
                    }
                    // PARIS Eq. 13: each inclusion carries the functionality of its superrelation.
                    const double factor =
                        (1.0 - head_prob * sub * (target.functionality[q2] / settings.epsilon)) *
                        (1.0 - head_prob * sup * (source.functionality[q] / settings.epsilon));
                    if (!(1.0 - factor > settings.delta)) {
                        continue;
                    }
                    ++out.evidence;
                    for (Index k = target.group_start[group]; k < target.group_end[group]; ++k) {
                        auto [entry, inserted] = products.try_emplace(target.target_tail[k], 1.0);
                        entry->second *= factor;
                    }
                }
            }
            if (state != nullptr) {
                choose(x, x_prob, products, out);
            }
        }
    }

    void choose(Index x, double x_prob, const std::unordered_map<Index, double>& products, Partial& out) const {
        if (state->fixed_left[x]) {
            return;
        }
        bool considered = false;
        Index best_right = -1;
        double best = -1.0;
        const double weight = settings.embedding_weight;
        for (const auto& [y, product] : products) {
            if (state->fixed_right[y] && state->right_match[y] != x) {
                continue;
            }
            if (state->num_rejected && rejected(*state, x * target.num_entities + y)) {
                continue;
            }
            double score = 1.0 - product;
            if (vectors != nullptr) {
                score = (1.0 - weight) * score + weight * cosine(*vectors, x, y);
            }
            score = std::clamp(score, 0.0, 1.0);
            considered = true;
            if (score > best || (score == best && y < best_right)) {
                best = score;
                best_right = y;
            }
        }
        if (considered && best >= x_prob) {
            out.updates.push_back({x, best_right, best});
        } else if (!considered && x_prob <= 0.0) {
            out.updates.push_back({x, -1, 0.0});
        }
    }
};

// Split [0, count) into contiguous ranges with about equal numbers of facts.
std::vector<std::pair<Index, Index>> ranges(const GraphIndex& graph, int threads) {
    std::vector<std::pair<Index, Index>> result;
    const Index count = graph.num_entities;
    const Index facts = graph.source_ptr[count];
    Index start = 0;
    for (int part = 1; part <= threads && start < count; ++part) {
        Index stop = count;
        if (part < threads) {
            const Index goal = facts * part / threads;
            stop = std::upper_bound(graph.source_ptr, graph.source_ptr + count + 1, goal) - graph.source_ptr - 1;
            stop = std::clamp(stop, start + 1, count);
        }
        result.emplace_back(start, stop);
        start = stop;
    }
    return result;
}

struct PassResult {
    RelationTable relations;
    std::vector<Update> updates;
    Index evidence = 0;
};

PassResult run_pass(const Pass& pass, const std::vector<Index>& fixed, int threads) {
    const auto parts = ranges(pass.source, std::max(1, threads));
    std::vector<Partial> partials(parts.size());
    if (parts.size() <= 1) {
        for (std::size_t i = 0; i < parts.size(); ++i) {
            pass.run(parts[i].first, parts[i].second, partials[i]);
        }
    } else {
        std::vector<std::thread> workers;
        workers.reserve(parts.size());
        for (std::size_t i = 0; i < parts.size(); ++i) {
            workers.emplace_back([&, i] { pass.run(parts[i].first, parts[i].second, partials[i]); });
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }

    PassResult result;
    std::vector<double> norm(static_cast<std::size_t>(pass.source.num_predicates), 0.0);
    std::unordered_map<Index, double> support;
    for (const auto& partial : partials) {
        for (const auto& [q, weight] : partial.norm) {
            norm[static_cast<std::size_t>(q)] += weight;
        }
        for (const auto& [key, weight] : partial.support) {
            support[key] += weight;
        }
        result.updates.insert(result.updates.end(), partial.updates.begin(), partial.updates.end());
        result.evidence += partial.evidence;
    }
    for (const Index key : fixed) {
        support.try_emplace(key, 0.0);
    }
    auto& table = result.relations;
    table.width = pass.target.num_predicates;
    table.keys.reserve(support.size());
    for (const auto& [key, value] : support) {
        table.keys.push_back(key);
    }
    std::sort(table.keys.begin(), table.keys.end());
    table.values.reserve(table.keys.size());
    auto fixed_key = fixed.begin();
    for (const Index key : table.keys) {
        if (fixed_key != fixed.end() && key == *fixed_key) {
            table.values.push_back(1.0);
            ++fixed_key;
        } else {
            const double scale = pass.settings.smoothing + norm[static_cast<std::size_t>(key / table.width)];
            table.values.push_back(scale > 0.0 ? support[key] / scale : 0.0);
        }
    }
    return result;
}

// Each right entity keeps the most probable left entity that points at it.
void one_to_one(State& state, Index left_entities, Index right_entities) {
    std::vector<Index> winner(static_cast<std::size_t>(right_entities), -1);
    std::vector<double> best(static_cast<std::size_t>(right_entities), 0.0);
    for (Index l = 0; l < left_entities; ++l) {
        const Index r = state.left_match[l];
        // Only a positive probability claims a counterpart.
        if (r < 0 || !(state.left_prob[l] > 0.0)) {
            continue;
        }
        if (winner[r] < 0 || state.left_prob[l] > best[r]) {
            winner[r] = l;
            best[r] = state.left_prob[l];
        }
    }
    // Rebuilt from the current left matches, so a right entity never keeps a left
    // entity that has moved to another counterpart.
    for (Index r = 0; r < right_entities; ++r) {
        state.right_match[r] = winner[r];
        state.right_prob[r] = winner[r] >= 0 ? best[r] : 0.0;
    }
    for (Index l = 0; l < left_entities; ++l) {
        const Index r = state.left_match[l];
        if (r >= 0 && state.right_match[r] != l) {
            state.left_match[l] = -1;
            state.left_prob[l] = 0.0;
        }
    }
}

}  // namespace

Engine::Engine(GraphIndex left, GraphIndex right, Settings settings)
    : left_(left), right_(right), settings_(settings) {}

IterationResult Engine::iterate(State& state, const RelationTable& left_relations,
                                const RelationTable& right_relations, bool warm, const Vectors* vectors) const {
    IterationResult result;
    const Pass forward{left_, right_, state.left_match, state.left_prob, left_relations, right_relations,
                       settings_, warm, &state, warm ? nullptr : vectors};
    PassResult left = run_pass(forward, state.fixed_left_relations, settings_.threads);
    for (const auto& update : left.updates) {
        state.left_match[update.entity] = update.counterpart;
        state.left_prob[update.entity] = update.probability;
    }
    result.left_relations = std::move(left.relations);
    one_to_one(state, left_.num_entities, right_.num_entities);

    const Pass backward{right_, left_, state.right_match, state.right_prob, right_relations,
                        result.left_relations, settings_, warm, nullptr, nullptr};
    result.right_relations = run_pass(backward, state.fixed_right_relations, settings_.threads).relations;

    Index matched = 0;
    for (Index l = 0; l < left_.num_entities; ++l) {
        matched += state.left_match[l] >= 0;
    }
    result.stats["evidence"] = left.evidence;
    result.stats["relation_pairs"] = static_cast<Index>(result.left_relations.keys.size());
    result.stats["matched"] = matched;
    return result;
}

}  // namespace prase::paris
