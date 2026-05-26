#pragma once

#include <array>
#include <vector>

#include "action_proto.hpp"
#include "observation.hpp"

namespace nanopokerlib {

template<typename Game>
struct StepResult {
    std::array<float, Game::NUM_PLAYERS> rewards;
    int is_terminal{};
    Observation<Game> observations;
    int player_id{};
    std::vector<ActionProto<Game>> available_actions;
};

} // namespace nanopokerlib
