#pragma once

#include "games/doudizhu/doudizhu_endpoint.hpp"

#ifdef NANOPOKERLIB_IMPLEMENTATION
namespace nanopokerlib::doudizhu {

template<typename Config>
typename DouDiZhuGame<Config>::Endpoint
DouDiZhuGame<Config>::Endpoint::reconstruct_state_from_infoset(const std::string& infoset_str) {
    ParsedInput parsed = parse(infoset_str);
    auto hands = build_initial_hands(parsed);

    Endpoint state;
    seed_state_with_hands(state, parsed, hands);

    const int num_deal_steps = Game::BASE_HAND_CARDS + Game::NUM_PUBLIC_LANDLORD_CARDS;
    for (int t = 0; t < num_deal_steps; ++t) {
        state.step(Action<Game>{{}});
    }
    // Transition from dealing to playing.
    state.step(Action<Game>{{}});

    for (size_t act_idx = 0; act_idx < parsed.actions.size(); ++act_idx) {
        const auto& act = parsed.actions[act_idx];
        if (state.player_ != act.player_id) {
            throw std::runtime_error("Belief decode replay failed: action order mismatch.");
        }
        const Sequence* seq_match = match_sequence(state, act);
        if (seq_match == nullptr) {
            throw std::runtime_error("Belief decode replay failed: no legal sequence matches action.");
        }
        const Sequence seq_copy = *seq_match;
        for (size_t step_idx = 0; step_idx < seq_copy.steps.size(); ++step_idx) {
            try {
                state.step(Action<Game>{seq_copy.steps[step_idx].tokens});
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Belief decode replay failed at action " << act_idx << " step " << step_idx << ": " << e.what();
                throw std::runtime_error(oss.str());
            }
        }
    }

    return state;
}

template<typename Config>
typename DouDiZhuGame<Config>::Endpoint::BeliefStepResult
DouDiZhuGame<Config>::Endpoint::expand_belief_prefix_str(const std::string& infoset_str, const std::string& belief_str) {
    using Game = DouDiZhuGame<Config>;
    // Reconstruct once to validate infoset semantics/replay legality before belief decoding.
    (void)reconstruct_state_from_infoset(infoset_str);
    const ParsedInput parsed = parse(infoset_str);
    const auto played = compute_played_counts(parsed.actions);

    HandCounts viewer_current = parsed.view_private_counts;
    if (parsed.view_player_id == 0) {
        for (int r = 0; r < Game::NUM_RANKS; ++r) {
            viewer_current[r] += parsed.landlord_public_counts[r];
        }
    }
    for (int r = 0; r < Game::NUM_RANKS; ++r) {
        viewer_current[r] -= played[parsed.view_player_id][r];
        if (viewer_current[r] < 0) {
            throw std::runtime_error("expand_belief_prefix_str: viewer current hand counts became negative.");
        }
    }

    HandCounts full_deck{};
    for (int r = 0; r < Game::NORMAL_RANKS; ++r)
        full_deck[r] = 4;
    for (int r = Game::NORMAL_RANKS; r < Game::NUM_RANKS; ++r)
        full_deck[r] = 1;

    HandCounts played_total{};
    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        for (int r = 0; r < Game::NUM_RANKS; ++r) {
            played_total[r] += played[p][r];
        }
    }

    HandCounts unseen_counts{};
    int unseen_total = 0;
    for (int r = 0; r < Game::NUM_RANKS; ++r) {
        unseen_counts[r] = full_deck[r] - viewer_current[r] - played_total[r];
        if (unseen_counts[r] < 0) {
            throw std::runtime_error("expand_belief_prefix_str: unseen opponent card counts became negative.");
        }
        unseen_total += unseen_counts[r];
    }

    std::vector<int> canonical_rank_rows;
    canonical_rank_rows.reserve(static_cast<size_t>(unseen_total));
    for (int rank = Game::NUM_RANKS - 1; rank >= 0; --rank) {
        const int copies = unseen_counts[rank];
        for (int i = 0; i < copies; ++i) {
            canonical_rank_rows.push_back(rank);
        }
    }

    std::array<int, Game::NUM_PLAYERS> expected_current_sizes{};
    expected_current_sizes.fill(0);
    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        const int expected = expected_cards_for_player(p) - count_cards(played[p]);
        if (expected < 0) {
            throw std::runtime_error("expand_belief_prefix_str: expected current hand size became negative.");
        }
        expected_current_sizes[p] = expected;
    }

    HandCounts landlord_min_current{};
    if (parsed.view_player_id != 0) {
        for (int r = 0; r < Game::NUM_RANKS; ++r) {
            landlord_min_current[r] = std::max(0, parsed.landlord_public_counts[r] - played[0][r]);
            if (landlord_min_current[r] > unseen_counts[r]) {
                throw std::runtime_error(
                    "expand_belief_prefix_str: infoset/history/public cards imply impossible landlord ownership for a rank."
                );
            }
        }
    }

    std::vector<std::pair<int, int>> prefix_rows;
    prefix_rows.reserve(Game::MAX_BELIEF_ROWS);
    HandCounts assigned_ranks{};
    HandCounts assigned_landlord_ranks{};
    std::array<int, Game::NUM_PLAYERS> assigned_by_owner{};
    assigned_by_owner.fill(0);
    const auto landlord_remaining_feasible_after = [&](int extra_owner, int extra_rank) -> bool {
        if (parsed.view_player_id == 0) {
            return true;
        }

        const int extra_for_landlord = (extra_owner == 0) ? 1 : 0;
        const int remaining_landlord_capacity =
            expected_current_sizes[0]
            - (assigned_by_owner[0] + extra_for_landlord);
        if (remaining_landlord_capacity < 0) {
            return false;
        }

        int min_landlord_needed_total = 0;
        for (int r = 0; r < Game::NUM_RANKS; ++r) {
            const int assigned_rank_after = assigned_ranks[r] + ((r == extra_rank) ? 1 : 0);
            const int assigned_landlord_after =
                assigned_landlord_ranks[r] + ((r == extra_rank) ? extra_for_landlord : 0);
            const int remaining_rank = unseen_counts[r] - assigned_rank_after;
            if (remaining_rank < 0) {
                return false;
            }
            const int min_landlord_needed = std::max(0, landlord_min_current[r] - assigned_landlord_after);
            if (min_landlord_needed > remaining_rank) {
                return false;
            }
            min_landlord_needed_total += min_landlord_needed;
        }
        return min_landlord_needed_total <= remaining_landlord_capacity;
    };
    for (const std::string& seg : semantic_string::split_segments(belief_str)) {
        const auto tagged = semantic_string::try_parse_tagged_segment(seg);
        if (!tagged.has_value() || tagged->op != '+') {
            throw std::runtime_error("belief_str segments must use '<player>+<single_rank>' format.");
        }
        const int owner = parse_player_tag(tagged->tag);
        if (owner == parsed.view_player_id) {
            throw std::runtime_error("expand_belief_prefix_str: belief_str cannot assign hidden cards to the viewer.");
        }
        auto [counts, order] = parse_cards_with_order(tagged->payload);
        if (count_cards(counts) != 1 || order.size() != 1) {
            throw std::runtime_error("belief_str payload must contain exactly one rank symbol per segment.");
        }
        const int rank = order[0];
        prefix_rows.push_back({owner, rank});
        assigned_by_owner[owner] += 1;
        if (assigned_by_owner[owner] > expected_current_sizes[owner]) {
            throw std::runtime_error("expand_belief_prefix_str: belief_str exceeds hidden-card capacity for an owner.");
        }
        assigned_ranks[rank] += 1;
        if (assigned_ranks[rank] > unseen_counts[rank]) {
            throw std::runtime_error("expand_belief_prefix_str: belief_str rank multiplicity exceeds unseen-card counts.");
        }
        if (owner == 0) {
            assigned_landlord_ranks[rank] += 1;
        }
        if (parsed.view_player_id != 0) {
            const int remaining_of_rank = unseen_counts[rank] - assigned_ranks[rank];
            if (assigned_landlord_ranks[rank] + remaining_of_rank < landlord_min_current[rank]) {
                throw std::runtime_error(
                    "expand_belief_prefix_str: belief_str violates landlord-public ownership constraints."
                );
            }
            if (!landlord_remaining_feasible_after(-1, -1)) {
                throw std::runtime_error(
                    "expand_belief_prefix_str: belief_str violates landlord-public ownership constraints."
                );
            }
        }
        if (prefix_rows.size() > canonical_rank_rows.size()) {
            throw std::runtime_error("expand_belief_prefix_str: belief_str is longer than the unseen-card trajectory.");
        }
        const size_t row_idx = prefix_rows.size() - 1;
        if (rank != canonical_rank_rows[row_idx]) {
            throw std::runtime_error("expand_belief_prefix_str: belief_str violates canonical rank ordering.");
        }
    }

    const BeliefTriple pad = {Game::Token::PAD_TOKEN, Game::Token::PAD_TOKEN, Game::Token::PAD_TOKEN};
    auto pair_to_segment = [](int owner, int rank) -> std::string {
        return player_tag(owner) + "+" + std::string(1, static_cast<char>(Game::RANKS[rank]));
    };

    std::string canonical_prefix;
    for (size_t i = 0; i < prefix_rows.size(); ++i) {
        if (i > 0) canonical_prefix.push_back('|');
        canonical_prefix += pair_to_segment(prefix_rows[i].first, prefix_rows[i].second);
    }

    BeliefStepResult out{};
    out.tokens.fill(pad);
    out.valid.fill(0);
    out.belief_strs.fill("");

    if (prefix_rows.size() >= canonical_rank_rows.size()) {
        return out;
    }

    const int next_rank = canonical_rank_rows[prefix_rows.size()];
    int owner_count = 0;
    for (int owner = 0; owner < Game::NUM_PLAYERS; ++owner) {
        if (owner == parsed.view_player_id) continue;
        const int remaining_owner_capacity = expected_current_sizes[owner] - assigned_by_owner[owner];
        if (remaining_owner_capacity <= 0) continue;
        if (parsed.view_player_id != 0) {
            const int assigned_next_rank = assigned_ranks[next_rank] + 1;
            const int assigned_landlord_next_rank = assigned_landlord_ranks[next_rank] + (owner == 0 ? 1 : 0);
            const int remaining_next_rank = unseen_counts[next_rank] - assigned_next_rank;
            if (assigned_landlord_next_rank + remaining_next_rank < landlord_min_current[next_rank]) {
                continue;
            }
            if (!landlord_remaining_feasible_after(owner, next_rank)) {
                continue;
            }
        }
        if (owner_count >= Game::MAX_BELIEF_CANDIDATES) {
            throw std::runtime_error("expand_belief_prefix_str: candidate owner count exceeded MAX_BELIEF_CANDIDATES.");
        }
        const BeliefTriple triple = {
            Game::Token::encode_player(owner),
            Game::Token::encode_rank(next_rank),
            Game::Token::encode_hold(),
        };
        out.tokens[static_cast<size_t>(owner_count)] = triple;
        out.valid[static_cast<size_t>(owner_count)] = 1;
        const std::string seg = pair_to_segment(owner, next_rank);
        out.belief_strs[static_cast<size_t>(owner_count)] = canonical_prefix.empty() ? seg : (canonical_prefix + "|" + seg);
        owner_count += 1;
    }

    if (owner_count == 0) {
        throw std::runtime_error("Belief decode failed to enumerate next candidates.");
    }

    return out;
}

template<typename Config>
typename DouDiZhuGame<Config>::Endpoint::BeliefStepBatchResult
DouDiZhuGame<Config>::Endpoint::expand_belief_prefix_strs(
    const std::vector<std::string>& infoset_strs,
    const std::vector<std::string>& belief_strs
) {
    if (!belief_strs.empty() && belief_strs.size() != infoset_strs.size()) {
        throw std::runtime_error("belief_strs must match the number of infoset_strs.");
    }

    BeliefStepBatchResult out;
    out.items.reserve(infoset_strs.size());
    const std::string empty;
    for (size_t i = 0; i < infoset_strs.size(); ++i) {
        const std::string& belief = belief_strs.empty() ? empty : belief_strs[i];
        out.items.push_back(expand_belief_prefix_str(infoset_strs[i], belief));
    }
    return out;
}

template<typename Config>
std::string DouDiZhuGame<Config>::Endpoint::determinize_history_str(const std::string& infoset_str, const std::string& belief_str) {
    ParsedInput parsed = parse(infoset_str);
    const auto played = compute_played_counts(parsed.actions);

    HandCounts viewer_current = parsed.view_private_counts;
    if (parsed.view_player_id == 0) {
        for (int r = 0; r < Game::NUM_RANKS; ++r) {
            viewer_current[r] += parsed.landlord_public_counts[r];
        }
    }
    for (int r = 0; r < Game::NUM_RANKS; ++r) {
        viewer_current[r] -= played[parsed.view_player_id][r];
        if (viewer_current[r] < 0) {
            throw std::runtime_error("determinize_history_str: viewer current hand counts became negative.");
        }
    }

    HandCounts full_deck{};
    for (int r = 0; r < Game::NORMAL_RANKS; ++r)
        full_deck[r] = 4;
    for (int r = Game::NORMAL_RANKS; r < Game::NUM_RANKS; ++r)
        full_deck[r] = 1;

    HandCounts played_total{};
    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        for (int r = 0; r < Game::NUM_RANKS; ++r) {
            played_total[r] += played[p][r];
        }
    }

    HandCounts unseen_counts{};
    for (int r = 0; r < Game::NUM_RANKS; ++r) {
        unseen_counts[r] = full_deck[r] - viewer_current[r] - played_total[r];
        if (unseen_counts[r] < 0) {
            throw std::runtime_error("determinize_history_str: unseen opponent card counts became negative.");
        }
    }
    const int unseen_total = count_cards(unseen_counts);

    std::array<HandCounts, Game::NUM_PLAYERS> current_counts{};
    current_counts[parsed.view_player_id] = viewer_current;
    HandCounts assigned_unseen{};
    int seen_rows = 0;
    for (const std::string& seg : semantic_string::split_segments(belief_str)) {
        const auto tagged = semantic_string::try_parse_tagged_segment(seg);
        if (!tagged.has_value() || tagged->op != '+') {
            throw std::runtime_error("determinize_history_str: belief_str segments must use '<player>+<single_rank>' format.");
        }
        const int owner = parse_player_tag(tagged->tag);
        if (owner == parsed.view_player_id) {
            throw std::runtime_error("determinize_history_str: belief_str cannot assign hidden cards to the viewer.");
        }
        auto [counts, order] = parse_cards_with_order(tagged->payload);
        if (count_cards(counts) != 1 || order.size() != 1) {
            throw std::runtime_error("determinize_history_str: belief_str payload must contain exactly one rank symbol.");
        }
        const int rank = order[0];
        current_counts[owner][rank] += 1;
        assigned_unseen[rank] += 1;
        seen_rows += 1;
    }

    if (seen_rows != unseen_total) {
        throw std::runtime_error(
            "determinize_history_str: belief_str must specify exactly "
            + std::to_string(unseen_total)
            + " hidden-card rows."
        );
    }
    if (assigned_unseen != unseen_counts) {
        throw std::runtime_error("determinize_history_str: belief_str does not match the unseen card multiset.");
    }

    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        const int expected_current_size = expected_cards_for_player(p) - count_cards(played[p]);
        const int current_size = count_cards(current_counts[p]);
        if (current_size != expected_current_size) {
            throw std::runtime_error("determinize_history_str: belief_str violates per-player current hand-size constraints.");
        }
    }

    std::array<HandCounts, Game::NUM_PLAYERS> private_initial{};
    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        HandCounts private_counts = current_counts[p];
        for (int r = 0; r < Game::NUM_RANKS; ++r) {
            private_counts[r] += played[p][r];
        }
        if (p == 0) {
            for (int r = 0; r < Game::NUM_RANKS; ++r) {
                private_counts[r] -= parsed.landlord_public_counts[r];
                if (private_counts[r] < 0) {
                    throw std::runtime_error("determinize_history_str: landlord private hand counts became negative.");
                }
            }
        }
        if (count_cards(private_counts) != Game::BASE_HAND_CARDS) {
            throw std::runtime_error("determinize_history_str: reconstructed private hand count mismatch.");
        }
        private_initial[p] = private_counts;
    }

    std::vector<std::string> segments;
    segments.reserve(static_cast<size_t>(Game::NUM_PLAYERS + 1 + parsed.actions.size()));
    segments.push_back("L+" + counts_to_cards(private_initial[0]));
    segments.push_back("P1+" + counts_to_cards(private_initial[1]));
    segments.push_back("P2+" + counts_to_cards(private_initial[2]));
    segments.push_back("L+" + counts_to_cards(parsed.landlord_public_counts));
    for (const ParsedAction& action : parsed.actions) {
        const std::string payload = action.is_pass ? "PASS" : counts_to_cards(action.counts);
        segments.push_back(player_tag(action.player_id) + "-" + payload);
    }
    return semantic_string::join_segments(segments);
}

template<typename Config>
std::vector<std::string> DouDiZhuGame<Config>::Endpoint::determinize_history_strs(
    const std::vector<std::string>& infoset_strs,
    const std::vector<std::string>& belief_strs
) {
    if (infoset_strs.size() != belief_strs.size()) {
        throw std::runtime_error("determinize_history_strs: belief_strs must match infoset_strs.");
    }
    std::vector<std::string> out;
    out.reserve(infoset_strs.size());
    for (size_t i = 0; i < infoset_strs.size(); ++i) {
        out.push_back(determinize_history_str(infoset_strs[i], belief_strs[i]));
    }
    return out;
}

template<typename Config>
std::string DouDiZhuGame<Config>::Endpoint::infoset_str_from_history_str(const std::string& history_str) {
    const std::vector<std::string> segments = semantic_string::split_segments(history_str);
    if (segments.size() < static_cast<size_t>(Game::NUM_PLAYERS + 1)) {
        throw std::runtime_error("infoset_str_from_history_str: history_str is missing required private/public heads.");
    }

    std::array<HandCounts, Game::NUM_PLAYERS> private_hands{};
    std::array<bool, Game::NUM_PLAYERS> seen_private{};
    seen_private.fill(false);
    for (int idx = 0; idx < Game::NUM_PLAYERS; ++idx) {
        const auto tagged = semantic_string::try_parse_tagged_segment(segments[static_cast<size_t>(idx)]);
        if (!tagged.has_value() || tagged->op != '+') {
            throw std::runtime_error("infoset_str_from_history_str: private head must use '<player>+<cards>' format.");
        }
        const int player = parse_player_tag(tagged->tag);
        if (seen_private[player]) {
            throw std::runtime_error("infoset_str_from_history_str: duplicate private head tag.");
        }
        auto [counts, _order] = parse_cards_with_order(tagged->payload);
        if (count_cards(counts) != Game::BASE_HAND_CARDS) {
            throw std::runtime_error("infoset_str_from_history_str: private head card-count mismatch.");
        }
        private_hands[player] = counts;
        seen_private[player] = true;
    }
    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        if (!seen_private[p]) {
            throw std::runtime_error("infoset_str_from_history_str: missing private head for a player.");
        }
    }

    const auto public_tagged = semantic_string::try_parse_tagged_segment(segments[static_cast<size_t>(Game::NUM_PLAYERS)]);
    if (!public_tagged.has_value() || public_tagged->op != '+') {
        throw std::runtime_error("infoset_str_from_history_str: missing landlord public segment.");
    }
    if (parse_player_tag(public_tagged->tag) != 0) {
        throw std::runtime_error("infoset_str_from_history_str: public segment must use landlord tag 'L+'.");
    }
    auto [public_counts, _public_order] = parse_cards_with_order(public_tagged->payload);
    if (count_cards(public_counts) != Game::NUM_PUBLIC_LANDLORD_CARDS) {
        throw std::runtime_error("infoset_str_from_history_str: landlord public card-count mismatch.");
    }

    auto [_env, step] = reconstruct_env(history_str, "", 0);
    const int acting_player = step.player_id;
    if (step.is_terminal || acting_player < 0 || acting_player >= Game::NUM_PLAYERS) {
        throw std::runtime_error(
            "infoset_str_from_history_str requires a non-terminal active-player decision state."
        );
    }

    HandCounts view_private = private_hands[acting_player];

    std::vector<std::string> infoset_segments;
    infoset_segments.reserve(segments.size() - static_cast<size_t>(Game::NUM_PLAYERS) + 1);
    infoset_segments.push_back(player_tag(acting_player) + "+" + counts_to_cards(view_private));
    infoset_segments.push_back("L+" + counts_to_cards(public_counts));
    for (size_t idx = static_cast<size_t>(Game::NUM_PLAYERS + 1); idx < segments.size(); ++idx) {
        const auto tagged = semantic_string::try_parse_tagged_segment(segments[idx]);
        if (!tagged.has_value() || tagged->op != '-') {
            throw std::runtime_error("infoset_str_from_history_str: action suffix must use '<player>-<move>' format.");
        }
        const int player = parse_player_tag(tagged->tag);
        const std::string payload = normalize_action_string(tagged->payload);
        infoset_segments.push_back(player_tag(player) + "-" + (payload.empty() ? "PASS" : payload));
    }
    return semantic_string::join_segments(infoset_segments);
}

template<typename Config>
std::vector<std::string>
DouDiZhuGame<Config>::Endpoint::infoset_str_from_history_strs(const std::vector<std::string>& history_strs) {
    std::vector<std::string> out;
    out.reserve(history_strs.size());
    for (const std::string& history_str : history_strs) {
        out.push_back(infoset_str_from_history_str(history_str));
    }
    return out;
}

} // namespace nanopokerlib::doudizhu
#endif
