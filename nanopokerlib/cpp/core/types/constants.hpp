#pragma once

#include <cstdint>
#include <string>

namespace nanopokerlib {

using ChannelValue = std::uint8_t;

struct NullChannels {
    static std::string decode_channel(int idx) { return "CH(" + std::to_string(idx) + ")"; }
    static std::string decode_channel_index(int idx) { return decode_channel(idx); }
};

inline constexpr int PAD_TOKEN = 0;
inline constexpr int INIT_TOKEN = 1;
inline constexpr int PLAYER_CHANCE_TRAINER = -1;
inline constexpr int PLAYER_CHANCE_ENVIRONMENT = -2;

inline constexpr bool is_chance_player(int player) {
    return player == PLAYER_CHANCE_TRAINER || player == PLAYER_CHANCE_ENVIRONMENT;
}

enum class HistoryBoundary : int {
    ENDS_WITH_STATE = 0,
    ENDS_WITH_ACTION = 1,
};

} // namespace nanopokerlib
