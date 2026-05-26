#pragma once

#include "games/doudizhu/doudizhu_endpoint.hpp"

#ifdef NANOPOKERLIB_IMPLEMENTATION
namespace nanopokerlib::doudizhu {

template<typename Config>
std::string DouDiZhuGame<Config>::Endpoint::opening_root_history_str(const Game& state) {
    return state.opening_root_history_str();
}

template<typename Config>
std::string DouDiZhuGame<Config>::Endpoint::format_history_str(
    const std::array<HandCounts, Game::NUM_PLAYERS>& initial_hands,
    const HandCounts& landlord_public_cards,
    const std::vector<std::vector<int>>& history_actions
) {
    using Game = DouDiZhuGame<Config>;
    std::array<std::string, Game::NUM_PLAYERS> private_heads{};
    std::array<bool, Game::NUM_PLAYERS> seen_heads{};
    seen_heads.fill(false);
    std::vector<std::string> suffix_segments;

    for (int view_player_id = 0; view_player_id < Game::NUM_PLAYERS; ++view_player_id) {
        const std::string infoset_str = build_infoset_str(initial_hands, landlord_public_cards, history_actions, view_player_id);
        const std::vector<std::string> segments = semantic_string::split_segments(infoset_str);
        if (segments.size() < 2) {
            throw std::runtime_error("format_history_str: expected infoset segments to include private and landlord-public heads.");
        }

        const auto head = semantic_string::try_parse_tagged_segment(segments[0]);
        if (!head.has_value() || head->op != '+') {
            throw std::runtime_error("format_history_str: infoset head must be '<player>+<cards>'.");
        }
        const int player = parse_player_tag(head->tag);
        if (player < 0 || player >= Game::NUM_PLAYERS) {
            throw std::runtime_error("format_history_str: parsed player tag out of range.");
        }
        if (seen_heads[player]) {
            throw std::runtime_error("format_history_str: duplicate private head while composing canonical history.");
        }
        seen_heads[player] = true;
        private_heads[player] = segments[0];

        if (view_player_id == 0) {
            suffix_segments.assign(segments.begin() + 1, segments.end());
        } else {
            if (segments.size() - 1 != suffix_segments.size()) {
                throw std::runtime_error("format_history_str: infoset suffix segment count mismatch across views.");
            }
            for (size_t i = 1; i < segments.size(); ++i) {
                if (segments[i] != suffix_segments[i - 1]) {
                    throw std::runtime_error("format_history_str: infoset suffix mismatch across views.");
                }
            }
        }
    }

    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        if (!seen_heads[p]) {
            throw std::runtime_error("format_history_str: missing private head for a player.");
        }
    }

    std::vector<std::string> history_segments;
    history_segments.reserve(Game::NUM_PLAYERS + suffix_segments.size());
    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        history_segments.push_back(private_heads[p]);
    }
    for (const auto& seg : suffix_segments) {
        history_segments.push_back(seg);
    }
    return semantic_string::join_segments(history_segments);
}

template<typename Config>
typename DouDiZhuGame<Config>::Endpoint::InfosetEncodeComponents
DouDiZhuGame<Config>::Endpoint::extract_infoset_encode_components(const History<Game>& infoset) {
    using Game = DouDiZhuGame<Config>;
    if (infoset.boundary() != HistoryBoundary::ENDS_WITH_ACTION) {
        throw std::runtime_error(
            "format_history_str expects ENDS_WITH_ACTION history boundary. "
            "State-ended histories are normalized by EndpointFactory before encoding."
        );
    }

    InfosetEncodeComponents components;
    const auto& observations = infoset.observations_data();
    if (observations.empty()) { throw std::runtime_error("format_history_str: infoset is empty."); }

    for (const auto& obs : observations) {
        std::array<int, Game::NUM_PLAYERS> receivers{};
        receivers.fill(-1);
        std::array<int, Game::NUM_PLAYERS> ranks{};
        ranks.fill(-1);

        bool all_hold = true;
        for (int obs_player = 0; obs_player < Game::NUM_PLAYERS; ++obs_player) {
            const auto& row = obs.tokens[obs_player];
            if (row.size() < 3 || !Game::Token::is_hold(row[2])) {
                all_hold = false;
                break;
            }
            receivers[obs_player] = Game::Token::decode_player(row[0]);
            ranks[obs_player] = Game::Token::decode_rank(row[1]);
        }
        if (!all_hold) {
            continue;
        }

        bool private_layout = true;
        for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
            if (receivers[p] != p) {
                private_layout = false;
                break;
            }
        }

        bool public_layout = true;
        const int public_rank = ranks[0];
        for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
            if (receivers[p] != 0 || ranks[p] != public_rank) {
                public_layout = false;
                break;
            }
        }

        if (private_layout) {
            for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
                components.initial_hands[p][ranks[p]] += 1;
            }
            continue;
        }

        if (public_layout) {
            components.initial_hands[0][public_rank] += 1;
            components.landlord_public_cards[public_rank] += 1;
            continue;
        }

        throw std::runtime_error("format_history_str: unexpected dealing observation layout.");
    }

    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        const int have = count_cards(components.initial_hands[p]);
        const int expected = expected_cards_for_player(p);
        if (have != expected) {
            throw std::runtime_error(
                "format_history_str: failed to reconstruct initial hand counts for player " + std::to_string(p) + "."
            );
        }
    }
    if (count_cards(components.landlord_public_cards) != Game::NUM_PUBLIC_LANDLORD_CARDS) {
        throw std::runtime_error("format_history_str: failed to reconstruct landlord public cards.");
    }

    std::vector<std::vector<int>> history_actions;
    const auto& action_rows = infoset.action_taken_tokens_data();
    history_actions.reserve(action_rows.size());
    for (const auto& row : action_rows) {
        if (!row.empty()) {
            history_actions.push_back(row);
        }
    }

    components.history_actions = std::move(history_actions);
    return components;
}

template<typename Config>
std::string DouDiZhuGame<Config>::Endpoint::format_history_str(const History<Game>& infoset) {
    const auto components = extract_infoset_encode_components(infoset);
    return format_history_str(components.initial_hands, components.landlord_public_cards, components.history_actions);
}

template<typename Config>
std::string DouDiZhuGame<Config>::Endpoint::format_infoset_str(const History<Game>& infoset) {
    using Game = DouDiZhuGame<Config>;
    if (infoset.boundary() != HistoryBoundary::ENDS_WITH_ACTION) {
        throw std::runtime_error(
            "format_infoset_str expects ENDS_WITH_ACTION history boundary. "
            "State-ended histories are normalized by EndpointFactory before encoding."
        );
    }

    const auto components = extract_infoset_encode_components(infoset);

    Endpoint state;
    state.phase_ = Game::Phase::PLAYING;
    state.player_ = 0;
    state.dealing_step_index_ = 0;
    for (auto& seq : state.deal_private_cards_)
        seq.clear();
    state.deal_landlord_public_cards_.clear();
    state.hand_counts_ = components.initial_hands;
    state.landlord_public_counts_ = components.landlord_public_cards;
    for (auto& dc : state.discard_counts_)
        dc.fill(0);
    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        state.cards_remaining_[p] = count_cards(components.initial_hands[p]);
    }
    state.action_prefix_.assign(1, Game::Token::encode_player(state.player_));
    state.stage_index_ = 0;
    state.current_sequences_ = state.build_sequences_for_player(state.player_);
    state.active_combo_.reset();
    state.current_leader_ = state.player_;
    state.consecutive_passes_ = 0;
    state.bomb_count_ = 0;

    for (const auto& tokens : components.history_actions) {
        auto matches = match_action_sequences(state, tokens);
        if (matches.empty()) {
            throw std::runtime_error("format_infoset_str: history action tokens are not legal at current state.");
        }
        (void)state.step(Action<Game>{tokens});
    }
    if (state.stage_index_ != 0) {
        throw std::runtime_error("format_infoset_str: action-ended history ends mid-sequence.");
    }

    const int view_player_id = state.player_;
    if (view_player_id < 0 || view_player_id >= Game::NUM_PLAYERS) {
        throw std::runtime_error(
            "format_infoset_str expects action-ended history to end at an active player decision state."
        );
    }

    const std::string history_str =
        format_history_str(components.initial_hands, components.landlord_public_cards, components.history_actions);
    const std::vector<std::string> segments = semantic_string::split_segments(history_str);
    if (segments.size() < static_cast<size_t>(Game::NUM_PLAYERS + 1)) {
        throw std::runtime_error("format_infoset_str: canonical history is missing required private/public heads.");
    }

    const std::string prefix = player_tag(view_player_id) + "+";
    std::string private_head;
    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        const auto& seg = segments[static_cast<size_t>(p)];
        if (seg.rfind(prefix, 0) == 0) {
            private_head = seg;
            break;
        }
    }
    if (private_head.empty()) {
        throw std::runtime_error("format_infoset_str: failed to locate active-player private head.");
    }

    std::vector<std::string> infoset_segments;
    infoset_segments.reserve(1 + segments.size() - static_cast<size_t>(Game::NUM_PLAYERS));
    infoset_segments.push_back(private_head);
    for (size_t idx = static_cast<size_t>(Game::NUM_PLAYERS); idx < segments.size(); ++idx) {
        infoset_segments.push_back(segments[idx]);
    }
    return semantic_string::join_segments(infoset_segments);
}

} // namespace nanopokerlib::doudizhu
#endif
