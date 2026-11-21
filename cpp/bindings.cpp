// Python bindings for the native PARIS iteration (prase._paris).
//
// The engine borrows the index arrays of two prase.paris.state.GraphIndex
// objects and keeps references to them. State arrays are updated in place,
// so they must already have the exact dtype and be C-contiguous.
#include "paris.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace prase::paris;

namespace {

template <typename T>
T* data(const py::array& array, const char* name, bool writable = false) {
    if (!py::isinstance<py::array_t<T>>(array) || !(array.flags() & py::array::c_style)) {
        throw std::invalid_argument(std::string(name) + " must be a C-contiguous array of the expected type");
    }
    if (writable && !array.writeable()) {
        throw std::invalid_argument(std::string(name) + " must be writable");
    }
    return static_cast<T*>(const_cast<void*>(array.data()));
}

struct Graph {
    GraphIndex index;
    std::vector<py::array> owned;  // Keeps the borrowed arrays alive.
};

Graph graph(const py::dict& arrays) {
    Graph graph;
    auto array = [&](const char* name) {
        py::array value = arrays[name].cast<py::array>();
        graph.owned.push_back(value);
        return value;
    };
    GraphIndex& g = graph.index;
    g.num_entities = arrays["num_entities"].cast<Index>();
    g.num_terms = arrays["num_terms"].cast<Index>();
    g.num_predicates = arrays["num_predicates"].cast<Index>();
    g.functionality = data<double>(array("functionality"), "functionality");
    g.source_ptr = data<Index>(array("source_ptr"), "source_ptr");
    g.source_pred = data<Index>(array("source_pred"), "source_pred");
    g.source_head = data<Index>(array("source_head"), "source_head");
    g.target_tail = data<Index>(array("target_tail"), "target_tail");
    g.group_start = data<Index>(array("group_start"), "group_start");
    g.group_end = data<Index>(array("group_end"), "group_end");
    g.group_pred = data<Index>(array("group_pred"), "group_pred");
    g.head_groups = data<Index>(array("head_groups"), "head_groups");
    py::array pairs = array("pair_keys");
    g.pair_keys = data<Index>(pairs, "pair_keys");
    g.num_pairs = static_cast<std::size_t>(pairs.size());
    g.pair_pred = data<Index>(array("pair_pred"), "pair_pred");
    return graph;
}

RelationTable table(const py::array& keys, const py::array& values, Index width) {
    RelationTable result;
    const Index* k = data<Index>(keys, "relation keys");
    const double* v = data<double>(values, "relation values");
    result.keys.assign(k, k + keys.size());
    result.values.assign(v, v + values.size());
    result.width = width;
    return result;
}

class PyEngine {
public:
    PyEngine(const py::dict& left, const py::dict& right, double theta, double delta, double epsilon,
             double smoothing, double embedding_weight, int threads)
        : left_(graph(left)), right_(graph(right)),
          engine_(left_.index, right_.index, Settings{theta, delta, epsilon, smoothing, embedding_weight, threads}) {}

    py::tuple iterate(py::array left_match, py::array left_prob, py::array right_match, py::array right_prob,
                      py::array left_keys, py::array left_values, py::array right_keys, py::array right_values,
                      py::array fixed_left_relations, py::array fixed_right_relations,
                      py::array fixed_left, py::array fixed_right, py::array rejected, bool warm,
                      py::object left_vectors, py::object right_vectors) {
        const Index* fixed_left_keys = data<Index>(fixed_left_relations, "fixed_left_relations");
        const Index* fixed_right_keys = data<Index>(fixed_right_relations, "fixed_right_relations");
        State state{
            data<Index>(left_match, "left_match", true), data<double>(left_prob, "left_prob", true),
            data<Index>(right_match, "right_match", true), data<double>(right_prob, "right_prob", true),
            data<bool>(fixed_left, "fixed_left"), data<bool>(fixed_right, "fixed_right"),
            data<Index>(rejected, "rejected"), static_cast<std::size_t>(rejected.size()),
            {fixed_left_keys, fixed_left_keys + fixed_left_relations.size()},
            {fixed_right_keys, fixed_right_keys + fixed_right_relations.size()},
        };
        const RelationTable left_table = table(left_keys, left_values, right_.index.num_predicates);
        const RelationTable right_table = table(right_keys, right_values, left_.index.num_predicates);
        Vectors vectors;
        py::array left_array, right_array;
        const bool use_vectors = !left_vectors.is_none() && !right_vectors.is_none();
        if (use_vectors) {
            left_array = left_vectors.cast<py::array>();
            right_array = right_vectors.cast<py::array>();
            if (left_array.ndim() != 2 || right_array.ndim() != 2 || left_array.shape(1) != right_array.shape(1)) {
                throw std::invalid_argument("embedding matrices must be 2-D with equal widths");
            }
            // The engine reads one row per entity, by entity index.
            if (left_array.shape(0) != left_.index.num_entities || right_array.shape(0) != right_.index.num_entities) {
                throw std::invalid_argument("embedding matrices need one row per entity of each graph");
            }
            vectors = {data<float>(left_array, "left vectors"), data<float>(right_array, "right vectors"),
                       static_cast<Index>(left_array.shape(1))};
        }
        IterationResult result;
        {
            py::gil_scoped_release release;
            result = engine_.iterate(state, left_table, right_table, warm, use_vectors ? &vectors : nullptr);
        }
        auto keys = [](const RelationTable& t) { return py::array_t<Index>(t.keys.size(), t.keys.data()); };
        auto values = [](const RelationTable& t) { return py::array_t<double>(t.values.size(), t.values.data()); };
        return py::make_tuple(keys(result.left_relations), values(result.left_relations),
                              keys(result.right_relations), values(result.right_relations), result.stats);
    }

private:
    Graph left_;
    Graph right_;
    Engine engine_;
};

}  // namespace

PYBIND11_MODULE(_paris, module) {
    module.doc() = "Native PARIS iteration for prase.paris";
    module.attr("API_VERSION") = 7;
    py::class_<PyEngine>(module, "Engine")
        .def(py::init<const py::dict&, const py::dict&, double, double, double, double, double, int>())
        .def("iterate", &PyEngine::iterate);
}
