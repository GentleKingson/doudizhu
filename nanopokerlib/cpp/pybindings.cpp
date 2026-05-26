#define NANOPOKERLIB_IMPLEMENTATION

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <sstream>

#include "core/bindings.hpp"

#include "ext/ext_bindings.hpp"

namespace py = pybind11;
using namespace nanopokerlib;

// Forward declaration for the isolated Dou Dizhu translation unit.
#if defined(NANOPOKERLIB_ENABLE_DOUDIZHU) && NANOPOKERLIB_ENABLE_DOUDIZHU
void init_tabletop_module(py::module_& m);
#endif

PYBIND11_MODULE(_nanopokerlib, m) {
    m.doc() = "Python bindings for nanopokerlib games"; // Set main module docstring
    bind_numpy_wrappers(m);
    detail::bind_constant(m, "PLAYER_CHANCE_TRAINER", PLAYER_CHANCE_TRAINER);
    detail::bind_constant(m, "PLAYER_CHANCE_ENVIRONMENT", PLAYER_CHANCE_ENVIRONMENT);
    detail::bind_constant(m, "CHANCE_TRAINER", PLAYER_CHANCE_TRAINER);
    detail::bind_constant(m, "CHANCE_ENVIRONMENT", PLAYER_CHANCE_ENVIRONMENT);

    auto m_utils = m.def_submodule("utils", "Utility helpers");
    m_utils.def(
        "append_prefix",
        [](const std::vector<std::string>& histories, const std::string& prefix_in) {
            std::string prefix = prefix_in;
            if (!prefix.empty() && prefix.back() != ':') { prefix.push_back(':'); }
            std::vector<std::string> out;
            out.reserve(histories.size());
            for (const auto& history : histories) {
                if (history.find(':') != std::string::npos || prefix.empty()) {
                    out.push_back(history);
                } else {
                    out.push_back(prefix + history);
                }
            }
            return out;
        },
        py::arg("histories"), py::arg("prefix"), "Prefix each history string unless it already contains a player tag."
    );

#if defined(NANOPOKERLIB_ENABLE_DOUDIZHU) && NANOPOKERLIB_ENABLE_DOUDIZHU
    init_tabletop_module(m);
#endif

    auto m_ext = m.def_submodule("ext", "Extra/experimental game modules.");
    init_extra_bindings(m_ext);
}
