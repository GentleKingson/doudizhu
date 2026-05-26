#pragma once

#include "core/semantic_string.hpp"
#include "games/doudizhu/doudizhu_core.hpp"

#ifdef NANOPOKERLIB_IMPLEMENTATION
namespace nanopokerlib::doudizhu {

template<typename Config>
std::string DouDiZhuGame<Config>::append_history_segment(const std::string& history_str, const std::string& segment) const {
    if (segment.empty()) {
        throw std::runtime_error("append_history_segment requires a non-empty segment.");
    }
    if (history_str.empty()) {
        return segment;
    }
    return history_str + "|" + segment;
}

template<typename Config>
std::string DouDiZhuGame<Config>::opening_root_history_str() const {
    if (phase_ != Phase::PLAYING) {
        throw std::runtime_error("opening_root_history_str is only valid during the playing phase.");
    }

    const auto player_tag = [](int player) {
        if (player == 0) return std::string("L");
        if (player == 1) return std::string("P1");
        if (player == 2) return std::string("P2");
        throw std::runtime_error("opening_root_history_str: player index out of range.");
    };
    const auto counts_to_cards = [](const HandCounts& counts) {
        std::string out;
        for (int r = 0; r < NUM_RANKS; ++r) {
            for (int c = 0; c < counts[static_cast<size_t>(r)]; ++c) {
                out.push_back(static_cast<char>(RANKS[static_cast<size_t>(r)]));
            }
        }
        return out;
    };
    const auto count_cards = [](const HandCounts& counts) {
        int total = 0;
        for (int r = 0; r < NUM_RANKS; ++r) {
            total += counts[static_cast<size_t>(r)];
        }
        return total;
    };

    std::array<HandCounts, NUM_PLAYERS> full_hands{};
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        for (int r = 0; r < NUM_RANKS; ++r) {
            full_hands[static_cast<size_t>(p)][static_cast<size_t>(r)] =
                hand_counts_[static_cast<size_t>(p)][static_cast<size_t>(r)] +
                discard_counts_[static_cast<size_t>(p)][static_cast<size_t>(r)];
        }
    }

    HandCounts landlord_private{};
    for (int r = 0; r < NUM_RANKS; ++r) {
        landlord_private[static_cast<size_t>(r)] =
            full_hands[0][static_cast<size_t>(r)] - landlord_public_counts_[static_cast<size_t>(r)];
        if (landlord_private[static_cast<size_t>(r)] < 0) {
            throw std::runtime_error("opening_root_history_str reconstructed a negative landlord private count.");
        }
    }
    if (count_cards(landlord_private) != BASE_HAND_CARDS) {
        throw std::runtime_error("opening_root_history_str expected exactly 17 landlord private cards.");
    }
    if (count_cards(full_hands[1]) != BASE_HAND_CARDS || count_cards(full_hands[2]) != BASE_HAND_CARDS) {
        throw std::runtime_error("opening_root_history_str expected exactly 17 peasant private cards.");
    }
    if (count_cards(landlord_public_counts_) != NUM_PUBLIC_LANDLORD_CARDS) {
        throw std::runtime_error("opening_root_history_str expected exactly 3 landlord public cards.");
    }

    std::vector<std::string> segments;
    segments.reserve(NUM_PLAYERS + 1);
    segments.push_back(player_tag(0) + "+" + counts_to_cards(landlord_private));
    segments.push_back(player_tag(1) + "+" + counts_to_cards(full_hands[1]));
    segments.push_back(player_tag(2) + "+" + counts_to_cards(full_hands[2]));
    segments.push_back(player_tag(0) + "+" + counts_to_cards(landlord_public_counts_));
    return semantic_string::join_segments(segments);
}

template<typename Config>
std::string DouDiZhuGame<Config>::advance_history_str(const std::string& history_str) {
    return live_history_tracker_.advance(history_str);
}

template<typename Config>
std::vector<std::string> DouDiZhuGame<Config>::get_available_action_strs() const {
    if (player_ < 0 || player_ >= NUM_PLAYERS) {
        throw std::runtime_error("get_available_action_strs is only valid on active player decision states.");
    }
    if (phase_ != Phase::PLAYING) {
        throw std::runtime_error("get_available_action_strs is only valid during the playing phase.");
    }

    const auto actions = available_actions();
    std::vector<std::string> out;
    out.reserve(actions.size());
    for (const auto& action : actions) {
        bool found = false;
        std::string candidate;
        for (const auto& seq : current_sequences_) {
            if (stage_index_ >= static_cast<int>(seq.steps.size())) continue;
            if (seq.steps[stage_index_].tokens != action.tokens) continue;
            const std::string step_string = Endpoint::describe_live_sequence_step(seq, stage_index_);
            if (!found) {
                candidate = step_string;
                found = true;
            } else if (candidate != step_string) {
                throw std::runtime_error("Inconsistent semantic string for available action tokens.");
            }
        }
        if (!found) { throw std::runtime_error("Unable to map available action tokens to semantic string."); }
        out.push_back(candidate);
    }
    return out;
}

} // namespace nanopokerlib::doudizhu
#endif
