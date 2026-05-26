#pragma once

#include <pybind11/pybind11.h>

namespace nanopokerlib {
namespace py = pybind11;

inline void init_ext_bindings(py::module_& m_ext) {
    (void)m_ext;
}

inline void init_extra_bindings(py::module_& m_ext) { init_ext_bindings(m_ext); }
} // namespace nanopokerlib
