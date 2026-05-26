#pragma once

#include "games/doudizhu/doudizhu_core.hpp"

#ifdef NANOPOKERLIB_IMPLEMENTATION
namespace nanopokerlib::doudizhu {

template<typename Config>
std::vector<typename DouDiZhuGame<Config>::Sequence> DouDiZhuGame<Config>::build_sequences_for_player(int player) const {
    std::vector<Sequence> sequences;

    const bool must_beat = static_cast<bool>(active_combo_);

    if (must_beat) {
        Combo pass;
        pass.type = Combo::Type::PASS;
        auto seq = build_sequence(pass, player);
        if (!seq.steps.empty()) { sequences.push_back(std::move(seq)); }
    }

    std::vector<Combo> combos = enumerate_combos(hand_counts_[player]);

    for (const auto& combo : combos) {
        if (must_beat && !beats(combo, *active_combo_)) continue;
        auto seq = build_sequence(combo, player);
        if (seq.steps.empty()) continue;
        sequences.push_back(std::move(seq));
    }

    return sequences;
}

template<typename Config>
StepResult<DouDiZhuGame<Config>> DouDiZhuGame<Config>::step(const Action<DouDiZhuGame<Config>>& action) {
    using Game = DouDiZhuGame<Config>;
    if (phase_ == Phase::DEALING) {
        std::vector<ActionProto<Game>> actions = {ActionProto<Game>{{}, 1.f, false}};

        // Deal private cards (one per player per step), then reveal the landlord's public cards (broadcast).
        if (dealing_step_index_ < Game::BASE_HAND_CARDS) {
            Observation<Game> obs;
            for (int p = 0; p < NUM_PLAYERS; ++p) {
                const int rank = deal_private_cards_[p][static_cast<size_t>(dealing_step_index_)];
                hand_counts_[p][rank] += 1;
                cards_remaining_[p] += 1;
                obs.tokens[p] = {Token::encode_player(p), Token::encode_rank(rank), Token::encode_hold()};
            }
            dealing_step_index_ += 1;
            obs.channels = build_channels();
            live_history_tracker_.record(live_history_tracker_.live_history_str);
            return StepResult<Game>{{0.f, 0.f, 0.f}, false, obs, player_, actions};
        }

        if (dealing_step_index_ < Game::BASE_HAND_CARDS + Game::NUM_PUBLIC_LANDLORD_CARDS) {
            Observation<Game> obs;
            const int public_index = dealing_step_index_ - Game::BASE_HAND_CARDS;
            const int rank = deal_landlord_public_cards_[static_cast<size_t>(public_index)];
            hand_counts_[0][rank] += 1;
            landlord_public_counts_[rank] += 1;
            cards_remaining_[0] += 1;
            for (int p = 0; p < NUM_PLAYERS; ++p) {
                obs.tokens[p] = {Token::encode_player(0), Token::encode_rank(rank), Token::encode_hold()};
            }
            dealing_step_index_ += 1;
            obs.channels = build_channels();
            live_history_tracker_.record(live_history_tracker_.live_history_str);
            return StepResult<Game>{{0.f, 0.f, 0.f}, false, obs, player_, actions};
        }

        // Transition step: after all chance draws, move into the normal playing phase.
        phase_ = Phase::PLAYING;
        player_ = 0; // Landlord starts
        current_leader_ = player_;
        action_prefix_.assign(1, Token::encode_player(player_));
        stage_index_ = 0;
        current_sequences_ = build_sequences_for_player(player_);

        Observation<Game> obs = action_observation();
        obs.channels = build_channels();
        live_history_tracker_.record(opening_root_history_str());
        return StepResult<Game>{{0.f, 0.f, 0.f}, false, obs, player_, available_actions()};
    }

    if (phase_ != Phase::PLAYING) { throw std::runtime_error("step called in unknown phase"); }
    if (action.tokens.empty()) { throw std::runtime_error("Empty action tokens"); }

    auto avail = available_actions();
    bool allowed = false;
    for (const auto& opt : avail) {
        if (opt.tokens == action.tokens) {
            allowed = true;
            break;
        }
    }
    if (!allowed) throw std::runtime_error("Action not allowed in current stage");

    // Apply step
    std::vector<const Sequence*> matches;
    for (const auto& seq : current_sequences_) {
        if (stage_index_ >= static_cast<int>(seq.steps.size())) continue;
        if (seq.steps[stage_index_].tokens == action.tokens) { matches.push_back(&seq); }
    }
    if (matches.empty()) throw std::runtime_error("No matching sequence");

    const SequenceStep& step_desc = matches.front()->steps[stage_index_];
    std::string committed_action_str;
    if (step_desc.completes) {
        bool found = false;
        for (const auto* seq : matches) {
            const std::string candidate = Game::Endpoint::describe_live_sequence_step(*seq, stage_index_);
            if (!found) {
                committed_action_str = candidate;
                found = true;
            } else if (candidate != committed_action_str) {
                throw std::runtime_error("Inconsistent semantic string for committed live DouDiZhu action.");
            }
        }
        if (!found || committed_action_str.empty()) {
            throw std::runtime_error("Unable to map committed DouDiZhu action tokens to a semantic string.");
        }
    }
    apply_step(step_desc, player_);
    action_prefix_ = action.tokens;

    Observation<Game> obs = action_observation();
    obs.channels = build_channels();

    if (!step_desc.completes) {
        std::vector<Sequence> next_sequences;
        next_sequences.reserve(matches.size());
        for (const auto* seq : matches) {
            next_sequences.push_back(*seq);
        }
        current_sequences_ = std::move(next_sequences);
        stage_index_ += 1;
        auto actions = available_actions();
        live_history_tracker_.record(live_history_tracker_.live_history_str);
        return StepResult<Game>{{0.f, 0.f, 0.f}, false, obs, player_, actions};
    }

    // Sequence finished
    const Combo& chosen = matches.front()->combo;
    const int acting_player = player_;
    const auto player_tag = [](int player) {
        if (player == 0) return std::string("L");
        if (player == 1) return std::string("P1");
        if (player == 2) return std::string("P2");
        throw std::runtime_error("DouDiZhu committed action player index out of range.");
    };
    const std::string next_history_str =
        append_history_segment(live_history_tracker_.live_history_str, player_tag(acting_player) + "-" + committed_action_str);
    bool reset_lead = false;
    if (chosen.type == Combo::Type::PASS) {
        ++consecutive_passes_;
        if (!active_combo_) throw std::runtime_error("Pass without active combo");
        if (consecutive_passes_ >= NUM_PLAYERS - 1) {
            active_combo_.reset();
            consecutive_passes_ = 0;
            reset_lead = true;
        }
    } else {
        active_combo_ = chosen;
        current_leader_ = player_;
        consecutive_passes_ = 0;
        if (chosen.type == Combo::Type::BOMB || chosen.type == Combo::Type::ROCKET) { bomb_count_++; }

        if (cards_remaining_[player_] == 0) {
            float mult = std::pow(2.f, static_cast<float>(bomb_count_));
            std::array<float, NUM_PLAYERS> rewards{};
            if (player_ == 0) {
                rewards[0] = 2.f * mult;
                rewards[1] = rewards[2] = -1.f * mult;
            } else {
                rewards[0] = -2.f * mult;
                rewards[1] = rewards[2] = 1.f * mult;
            }
            // Immediately reset and return the initial frame so terminal steps still expose actions.
            StepResult<Game> terminal_step = init(rewards);
            live_history_tracker_.record(next_history_str, true);
            return terminal_step;
        }
    }

    player_ = reset_lead ? current_leader_ : next_player<DouDiZhuGame<Config>>(player_);
    action_prefix_.assign(1, Token::encode_player(player_));
    stage_index_ = 0;
    current_sequences_ = build_sequences_for_player(player_);

    auto actions = available_actions();
    live_history_tracker_.record(next_history_str);
    return StepResult<Game>{{0.f, 0.f, 0.f}, false, obs, player_, actions};
}

} // namespace nanopokerlib::doudizhu
#endif
