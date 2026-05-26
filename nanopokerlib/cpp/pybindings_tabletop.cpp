// Split TU: Dou Dizhu variants
#define NANOPOKERLIB_IMPLEMENTATION

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "games/doudizhu/doudizhu_bindings.hpp"

namespace py = pybind11;
using namespace nanopokerlib;

void init_tabletop_module(py::module_& m) {
    auto m_doudizhu = m.def_submodule("doudizhu", "DouDiZhu with fixed landlord seat and multi-step tokenized actions.");
    init_doudizhu_bindings<doudizhu::DouDiZhu>(m_doudizhu);

    auto m_doudizhu_small = m.def_submodule("doudizhu_small", "DouDiZhu with a reduced 9TJQKA2BR deck (30 cards).");
    init_doudizhu_bindings<doudizhu::DouDiZhuSmall>(m_doudizhu_small);

    auto m_doudizhu_tiny = m.def_submodule("doudizhu_tiny", "DouDiZhu with a tiny QKA2BR deck (18 cards).");
    init_doudizhu_bindings<doudizhu::DouDiZhuTiny>(m_doudizhu_tiny);

}
