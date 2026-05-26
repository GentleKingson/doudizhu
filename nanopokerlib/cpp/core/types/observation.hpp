#pragma once

#include <array>
#include <vector>

#include "constants.hpp"

namespace nanopokerlib {

template<typename Game>
struct Observation {
    std::array<std::vector<int>, Game::NUM_PLAYERS> tokens;
    std::array<std::vector<ChannelValue>, Game::NUM_PLAYERS> channels;
};

} // namespace nanopokerlib
