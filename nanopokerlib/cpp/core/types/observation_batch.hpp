#pragma once

#include <pybind11/numpy.h>

#include "constants.hpp"

namespace py = pybind11;

namespace nanopokerlib {

struct ObservationBatch {
    py::array_t<int> tokens;
    py::array_t<ChannelValue> channels;
};

} // namespace nanopokerlib
