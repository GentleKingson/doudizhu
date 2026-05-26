#pragma once

#include "games/doudizhu/doudizhu_core.hpp"

#ifdef NANOPOKERLIB_IMPLEMENTATION
namespace nanopokerlib::doudizhu {

template<typename Config>
std::ostream& operator<<(std::ostream& os, const DouDiZhuGame<Config>& state) {
    const auto counts_to_string = [](const typename DouDiZhuGame<Config>::HandCounts& counts) {
        std::string out;
        out.reserve(DouDiZhuGame<Config>::NUM_CARDS);
        for (int r = 0; r < DouDiZhuGame<Config>::NUM_RANKS; ++r) {
            for (int c = 0; c < counts[r]; ++c) {
                out.push_back(DouDiZhuGame<Config>::RANKS[r]);
            }
        }
        if (out.empty()) out = "-";
        return out;
    };

    os << DouDiZhuGame<Config>::GAME_IDENTIFIER << " State:\n"
       << "  Phase=" << (state.phase_ == DouDiZhuGame<Config>::Phase::DEALING ? "DEAL" : "PLAY") << " player=" << state.player_
       << " leader=" << state.current_leader_ << " bombs=" << state.bomb_count_ << " passes=" << state.consecutive_passes_ << " remaining=["
       << state.cards_remaining_[0] << "," << state.cards_remaining_[1] << "," << state.cards_remaining_[2] << "]\n";

    os << "  Hands:\n"
       << "    L : " << counts_to_string(state.hand_counts_[0]) << "\n"
       << "    P1: " << counts_to_string(state.hand_counts_[1]) << "\n"
       << "    P2: " << counts_to_string(state.hand_counts_[2]) << "\n";

    os << "  Discards:\n"
       << "    L : " << counts_to_string(state.discard_counts_[0]) << "\n"
       << "    P1: " << counts_to_string(state.discard_counts_[1]) << "\n"
       << "    P2: " << counts_to_string(state.discard_counts_[2]) << "\n";
    return os;
}

} // namespace nanopokerlib::doudizhu
#endif
