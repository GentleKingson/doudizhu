#pragma once

#include <cstdint>

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace nanopokerlib {

struct AvailableActions {
    py::array_t<int> tokens;
    py::array_t<float> weights;
    py::array_t<std::uint8_t> is_commit;
    int player_id{};
};

} // namespace nanopokerlib
