#pragma once

#include <cstdint>

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace nanopokerlib {

struct MatchStateView {
    py::array_t<float> rewards;
    bool is_terminal{};
    py::array_t<float> channels;
};

struct MatchStateBatch {
    py::array_t<float> rewards;
    py::array_t<std::uint8_t> is_terminal;
    py::array_t<float> channels;
};

} // namespace nanopokerlib
