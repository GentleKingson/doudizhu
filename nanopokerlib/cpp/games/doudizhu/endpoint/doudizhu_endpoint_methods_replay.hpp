#pragma once

#include "games/doudizhu/doudizhu_endpoint.hpp"

#ifdef NANOPOKERLIB_IMPLEMENTATION
namespace nanopokerlib::doudizhu {

template<typename Config>
History<DouDiZhuGame<Config>>
DouDiZhuGame<Config>::Endpoint::parse_infoset_str(const std::string& infoset_str, const std::string& pending_action) {
    using Game = DouDiZhuGame<Config>;
    ParsedInput parsed = parse(infoset_str);
    auto hands = build_initial_hands(parsed);

    Endpoint state;
    seed_state_with_hands(state, parsed, hands);

    std::vector<StepResult<Game>> infoset_steps;
    std::vector<std::vector<int>> action_taken_tokens;
    Observation<Game> init_obs;
    for (auto& tok_vec : init_obs.tokens) {
        tok_vec = {Game::Token::INIT_TOKEN};
    }
    init_obs.channels = state.build_channels();
    std::array<float, Game::NUM_PLAYERS> zero_rewards{};
    infoset_steps.push_back(
        StepResult<Game>{zero_rewards, false, init_obs, state.player_, {ActionProto<Game>{{}, 1.f, false}}}
    );
    auto apply_and_record = [&](std::vector<int> tokens) {
        infoset_steps.push_back(state.step(Action<Game>{tokens}));
        action_taken_tokens.push_back(std::move(tokens));
    };

    const int num_deal_steps = Game::BASE_HAND_CARDS + Game::NUM_PUBLIC_LANDLORD_CARDS;
    for (int t = 0; t < num_deal_steps; ++t) {
        apply_and_record({});
    }
    // One extra chance step transitions the game into the normal playing phase.
    apply_and_record({});

    for (size_t act_idx = 0; act_idx < parsed.actions.size(); ++act_idx) {
        const auto& act = parsed.actions[act_idx];
        if (state.player_ != act.player_id) { throw std::runtime_error("Action order mismatch while replaying history."); }
        const Sequence* seq_match = match_sequence(state, act);
        if (seq_match == nullptr) { throw std::runtime_error("No legal sequence matches action for player " + std::to_string(act.player_id)); }
        const Sequence seq_copy = *seq_match; // copy to avoid invalidation after step()
        for (size_t step_idx = 0; step_idx < seq_copy.steps.size(); ++step_idx) {
                const auto& step_desc = seq_copy.steps[step_idx];
                try {
                    apply_and_record(step_desc.tokens);
                } catch (const std::exception& e) {
                    std::ostringstream oss;
                    oss << "Replay failed for player " << act.player_id << " at action " << act_idx << " step " << step_idx << ": " << e.what();
                throw std::runtime_error(oss.str());
            }
        }
    }

    std::vector<std::vector<int>> pending_tokens = resolve_pending_actions(state, pending_action);
    for (const auto& tok : pending_tokens) {
        apply_and_record(tok);
    }

    const std::vector<std::string> last_available_action_strs =
        available_action_strings(state, infoset_steps.back().available_actions);
    action_taken_tokens.push_back({});
    return History<Game>::from_inference_steps(
        std::move(infoset_steps),
        std::move(action_taken_tokens),
        std::move(last_available_action_strs)
    );
}

template<typename Game>
History<Game> parse_infoset_str(const std::string& infoset_str, const std::string& pending_action) {
    return Game::Endpoint::parse_infoset_str(infoset_str, pending_action);
}

template<typename Config>
HistoryBatch<DouDiZhuGame<Config>> DouDiZhuGame<Config>::Endpoint::parse_infoset_strs(
    const std::vector<std::string>& infoset_strs, const std::vector<std::string>& pending_actions
) {
    using Game = DouDiZhuGame<Config>;
    if (!pending_actions.empty() && pending_actions.size() != infoset_strs.size()) {
        throw std::runtime_error("pending_actions must match the number of infoset_strs.");
    }
    std::vector<History<Game>> infosets;
    infosets.reserve(infoset_strs.size());
    const std::string empty;
    for (size_t i = 0; i < infoset_strs.size(); ++i) {
        const std::string& pending = pending_actions.empty() ? empty : pending_actions[i];
        infosets.push_back(parse_infoset_str(infoset_strs[i], pending));
    }
    return HistoryBatch<Game>(std::move(infosets));
}

template<typename Game>
HistoryBatch<Game> parse_infoset_strs(const std::vector<std::string>& infoset_strs, const std::vector<std::string>& pending_actions) {
    return Game::Endpoint::parse_infoset_strs(infoset_strs, pending_actions);
}

template<typename Config>
std::pair<DouDiZhuGame<Config>, StepResult<DouDiZhuGame<Config>>> DouDiZhuGame<Config>::Endpoint::reconstruct_env(
    const std::string& history_str,
    const std::string& pending_action,
    int seed
) {
    using Game = DouDiZhuGame<Config>;

    // Empty history_str: fresh game with random deal from seed.
    if (history_str.empty()) {
        if (!pending_action.empty()) { throw std::runtime_error("Doudizhu reconstruct_env: pending_action must be empty when history_str is empty."); }
        Endpoint state;
        state.rng_.seed(static_cast<std::mt19937::result_type>(seed));
        state.live_history_tracker_.reset();
        state.init({0.f, 0.f, 0.f});
        state.live_history_tracker_.record("");

        StepResult<Game> last_step{};
        const int num_deal_steps = Game::BASE_HAND_CARDS + Game::NUM_PUBLIC_LANDLORD_CARDS;
        for (int t = 0; t < num_deal_steps; ++t) {
            last_step = state.step(Action<Game>{{}});
        }
        // Transition from dealing to normal playing phase.
        last_step = state.step(Action<Game>{{}});

        Game out_state = state;
        return {out_state, last_step};
    }

    const std::vector<std::string> segments = semantic_string::split_segments(history_str);
    if (segments.size() < static_cast<size_t>(Game::NUM_PLAYERS + 1)) {
        throw std::runtime_error(
            "History string must start with "
            + std::to_string(Game::NUM_PLAYERS)
            + " '<player>+<private_cards>' segments followed by 'L+<landlord_public>'."
        );
    }

    std::array<HandCounts, Game::NUM_PLAYERS> private_hands{};
    std::array<std::vector<int>, Game::NUM_PLAYERS> private_orders{};
    std::array<bool, Game::NUM_PLAYERS> seen_private{};
    seen_private.fill(false);

    for (int idx = 0; idx < Game::NUM_PLAYERS; ++idx) {
        const auto tagged = semantic_string::try_parse_tagged_segment(segments[static_cast<size_t>(idx)]);
        if (!tagged.has_value() || tagged->op != '+') {
            throw std::runtime_error(
                "History string must start with "
                + std::to_string(Game::NUM_PLAYERS)
                + " '<player>+<private_cards>' segments."
            );
        }
        const int player = parse_player_tag(tagged->tag);
        if (player < 0 || player >= Game::NUM_PLAYERS) {
            throw std::runtime_error("History string private-hand segment has invalid player tag.");
        }
        if (seen_private[player]) {
            throw std::runtime_error("History string includes duplicate private-hand segment for a player.");
        }

        auto [counts, order] = parse_cards_with_order(tagged->payload);
        if (count_cards(counts) != Game::BASE_HAND_CARDS) {
            throw std::runtime_error(
                "History string private-hand segment must contain exactly "
                + std::to_string(Game::BASE_HAND_CARDS)
                + " cards."
            );
        }

        private_hands[player] = counts;
        private_orders[player] = std::move(order);
        seen_private[player] = true;
    }

    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        if (!seen_private[p]) {
            throw std::runtime_error("History string must include exactly one private-hand segment per player.");
        }
    }

    const auto public_tagged = semantic_string::try_parse_tagged_segment(segments[static_cast<size_t>(Game::NUM_PLAYERS)]);
    if (!public_tagged.has_value() || public_tagged->op != '+') {
        throw std::runtime_error("History string must include landlord public-card segment 'L+<cards>' after private heads.");
    }
    const int public_player = parse_player_tag(public_tagged->tag);
    if (public_player != 0) {
        throw std::runtime_error("Landlord public-card segment must use landlord tag 'L+'.");
    }
    auto [landlord_public_counts, landlord_public_order] = parse_cards_with_order(public_tagged->payload);
    if (count_cards(landlord_public_counts) != Game::NUM_PUBLIC_LANDLORD_CARDS) {
        throw std::runtime_error(
            "Landlord public-card segment must contain exactly "
            + std::to_string(Game::NUM_PUBLIC_LANDLORD_CARDS)
            + " cards."
        );
    }

    ParsedInput replay_input;
    replay_input.view_player_id = 0;
    replay_input.view_private_counts = private_hands[0];
    replay_input.view_private_order = private_orders[0];
    replay_input.has_view_private = true;
    replay_input.landlord_public_counts = landlord_public_counts;
    replay_input.landlord_public_order = std::move(landlord_public_order);
    replay_input.has_landlord_public = true;
    replay_input.actions.clear();

    for (size_t idx = static_cast<size_t>(Game::NUM_PLAYERS + 1); idx < segments.size(); ++idx) {
        const auto tagged = semantic_string::try_parse_tagged_segment(segments[idx]);
        if (!tagged.has_value() || tagged->op != '-') {
            throw std::runtime_error("History string action suffix must use '<player>-<move>' segments.");
        }
        ParsedAction action;
        action.player_id = parse_player_tag(tagged->tag);
        const std::string payload = semantic_string::normalize_compact_upper(tagged->payload);
        if (payload.empty() || payload == "PASS") {
            action.is_pass = true;
        } else {
            action.counts = parse_cards(tagged->payload);
        }
        replay_input.actions.push_back(std::move(action));
    }

    std::array<HandCounts, Game::NUM_PLAYERS> initial_hands{};
    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        initial_hands[p] = private_hands[p];
    }
    for (int r = 0; r < Game::NUM_RANKS; ++r) {
        initial_hands[0][r] += landlord_public_counts[r];
    }

    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        const int expected = expected_cards_for_player(p);
        const int have = count_cards(initial_hands[p]);
        if (have != expected) {
            throw std::runtime_error(
                "History string hand-card count mismatch for player "
                + std::to_string(p)
                + " (expected "
                + std::to_string(expected)
                + ", got "
                + std::to_string(have)
                + ")."
            );
        }
    }

    HandCounts deck{};
    for (int r = 0; r < Game::NORMAL_RANKS; ++r)
        deck[r] = 4;
    for (int r = Game::NORMAL_RANKS; r < Game::NUM_RANKS; ++r)
        deck[r] = 1;
    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        for (int r = 0; r < Game::NUM_RANKS; ++r) {
            deck[r] -= initial_hands[p][r];
            if (deck[r] < 0) {
                throw std::runtime_error("History string hand assignments exceed deck card counts.");
            }
        }
    }
    if (count_cards(deck) != 0) {
        throw std::runtime_error("History string hand assignments do not consume the full deck.");
    }

    Endpoint state;
    state.rng_.seed(static_cast<std::mt19937::result_type>(seed));
    seed_state_with_hands(state, replay_input, initial_hands);

    StepResult<Game> last_step{};
    const int num_deal_steps = Game::BASE_HAND_CARDS + Game::NUM_PUBLIC_LANDLORD_CARDS;
    for (int t = 0; t < num_deal_steps; ++t) {
        last_step = state.step(Action<Game>{{}});
    }
    // Transition from dealing to normal playing phase.
    last_step = state.step(Action<Game>{{}});

    for (size_t act_idx = 0; act_idx < replay_input.actions.size(); ++act_idx) {
        const auto& act = replay_input.actions[act_idx];
        if (state.player_ != act.player_id) {
            throw std::runtime_error("reconstruct_env replay failed: action order mismatch.");
        }
        const Sequence* seq_match = match_sequence(state, act);
        if (seq_match == nullptr) {
            throw std::runtime_error("reconstruct_env replay failed: no legal sequence matches action segment.");
        }
        const Sequence seq_copy = *seq_match;
        for (size_t step_idx = 0; step_idx < seq_copy.steps.size(); ++step_idx) {
            try {
                last_step = state.step(Action<Game>{seq_copy.steps[step_idx].tokens});
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "reconstruct_env replay failed at action " << act_idx << " step " << step_idx << ": " << e.what();
                throw std::runtime_error(oss.str());
            }
        }
    }

    std::vector<std::vector<int>> pending_tokens = resolve_pending_actions(state, pending_action);
    for (const auto& tokens : pending_tokens) {
        last_step = state.step(Action<Game>{tokens});
    }

    Game out_state = state;
    return {out_state, last_step};
}

template<typename Config>
std::string DouDiZhuGame<Config>::Endpoint::build_infoset_str(
    const std::array<HandCounts, Game::NUM_PLAYERS>& initial_hands, const HandCounts& landlord_public_cards,
    const std::vector<std::vector<int>>& history_actions, int view_player_id
) {
    using Game = DouDiZhuGame<Config>;
    if (view_player_id < 0 || view_player_id >= Game::NUM_PLAYERS) { throw std::runtime_error("Invalid view_player_id index"); }

    Endpoint state;
    state.phase_ = Game::Phase::PLAYING;
    state.player_ = 0;
    state.dealing_step_index_ = 0;
    for (auto& seq : state.deal_private_cards_)
        seq.clear();
    state.deal_landlord_public_cards_.clear();
    state.hand_counts_ = initial_hands;
    state.landlord_public_counts_ = landlord_public_cards;
    for (auto& dc : state.discard_counts_)
        dc.fill(0);
    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
        state.cards_remaining_[p] = count_cards(initial_hands[p]);
    }
    state.action_prefix_.assign(1, Game::Token::encode_player(state.player_));
    state.stage_index_ = 0;
    state.current_sequences_ = state.build_sequences_for_player(state.player_);
    state.active_combo_.reset();
    state.current_leader_ = state.player_;
    state.consecutive_passes_ = 0;
    state.bomb_count_ = 0;

    std::ostringstream oss;
    HandCounts view_private = initial_hands[view_player_id];
    if (view_player_id == 0) {
        for (int r = 0; r < Game::NUM_RANKS; ++r) {
            view_private[r] -= landlord_public_cards[r];
            if (view_private[r] < 0) { throw std::runtime_error("build_infoset_str: landlord_public_cards exceed landlord hand counts."); }
        }
    }
    oss << player_tag(view_player_id) << "+" << counts_to_cards(view_private);
    oss << "|" << player_tag(0) << "+" << counts_to_cards(landlord_public_cards);

    for (const auto& tokens : history_actions) {
        int acting = state.player_;
        auto matches = match_action_sequences(state, tokens);
        if (matches.empty()) { throw std::runtime_error("build_infoset_str: tokens not legal at current stage"); }
        const auto& step_desc = matches.front()->steps[state.stage_index_];
        const bool completes = step_desc.completes;
        const Combo combo = matches.front()->combo;
        const auto combo_card_counts = combo_counts(combo);
        state.step(Action<Game>{tokens});
        if (completes) {
            oss << "|" << player_tag(acting) << "-";
            if (combo.type == Combo::Type::PASS) {
                oss << "PASS";
            } else {
                oss << counts_to_cards(combo_card_counts);
            }
        }
    }

    if (state.stage_index_ != 0) { throw std::runtime_error("build_infoset_str: history ends mid-sequence"); }

    return oss.str();
}

} // namespace nanopokerlib::doudizhu
#endif
