#pragma once

#include "games/doudizhu/doudizhu_core.hpp"

#ifdef NANOPOKERLIB_IMPLEMENTATION
namespace nanopokerlib::doudizhu {

template<typename Config>
DouDiZhuGame<Config>::DouDiZhuGame() {
    init({0.f, 0.f, 0.f});
}

template<typename Config>
StepResult<DouDiZhuGame<Config>> DouDiZhuGame<Config>::init(const std::array<float, NUM_PLAYERS>& rewards) {
    phase_ = Phase::DEALING;
    player_ = PLAYER_CHANCE_ENVIRONMENT;
    dealing_step_index_ = 0;
    for (auto& seq : deal_private_cards_)
        seq.clear();
    deal_landlord_public_cards_.clear();
    hand_counts_.fill({});
    landlord_public_counts_.fill(0);
    discard_counts_.fill({});
    cards_remaining_.fill(0);
    action_prefix_.clear();
    stage_index_ = 0;
    current_sequences_.clear();
    active_combo_.reset();
    current_leader_ = -1;
    consecutive_passes_ = 0;
    bomb_count_ = 0;

    deal_shuffled_deck();

    Observation<DouDiZhuGame<Config>> obs;
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        obs.tokens[p] = {Token::INIT_TOKEN};
    }
    obs.channels = build_channels();

    std::vector<ActionProto<DouDiZhuGame<Config>>> actions = {ActionProto<DouDiZhuGame<Config>>{{}, 1.f, false}};

    return StepResult<DouDiZhuGame<Config>>{
        rewards, true, obs, player_, actions,
    };
}

template<typename Config>
StepResult<DouDiZhuGame<Config>> DouDiZhuGame<Config>::reset(int seed) {
    rng_.seed(seed);
    live_history_tracker_.reset();
    StepResult<DouDiZhuGame<Config>> step = init({0.f, 0.f, 0.f});
    live_history_tracker_.record("");
    return step;
}

template<typename Config>
void DouDiZhuGame<Config>::deal_shuffled_deck() {
    std::vector<int> deck;
    deck.reserve(NUM_CARDS);
    for (int r = 0; r < NORMAL_RANKS; ++r) {
        for (int i = 0; i < 4; ++i)
            deck.push_back(r);
    }
    deck.push_back(BLACK_JOKER_RANK); // B
    deck.push_back(RED_JOKER_RANK); // R
    std::shuffle(deck.begin(), deck.end(), rng_);


    for (auto& seq : deal_private_cards_) {
        seq.clear();
        seq.reserve(BASE_HAND_CARDS);
    }
    deal_landlord_public_cards_.clear();
    deal_landlord_public_cards_.reserve(NUM_PUBLIC_LANDLORD_CARDS);

    for (int i = 0; i < BASE_HAND_CARDS; ++i) {
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            deal_private_cards_[p].push_back(deck[static_cast<size_t>(i * NUM_PLAYERS + p)]);
        }
    }
    for (auto& hand : deal_private_cards_) {
        std::sort(hand.begin(), hand.end());
    }
    const size_t public_offset = static_cast<size_t>(BASE_HAND_CARDS * NUM_PLAYERS);
    for (int i = 0; i < NUM_PUBLIC_LANDLORD_CARDS; ++i) {
        deal_landlord_public_cards_.push_back(deck[public_offset + static_cast<size_t>(i)]);
    }
    std::sort(deal_landlord_public_cards_.begin(), deal_landlord_public_cards_.end());
}

template<typename Config>
std::array<std::vector<ChannelValue>, DouDiZhuGame<Config>::NUM_PLAYERS> DouDiZhuGame<Config>::build_channels() const {
    std::array<std::vector<ChannelValue>, NUM_PLAYERS> channels;
    for (auto& vec : channels)
        vec.assign(NUM_CHANNELS, ChannelValue{0});

    for (int viewer = 0; viewer < NUM_PLAYERS; ++viewer) {
        auto& vec = channels[viewer];
        Channels::encode_cards(vec, Channels::SELF_OFFSET, hand_counts_[viewer]);
        Channels::encode_cards(vec, Channels::LANDLORD_PUBLIC_OFFSET, landlord_public_counts_);
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            Channels::encode_cards(vec, Channels::discard_base(p), discard_counts_[p]);
        }
        const int bomb_state = std::clamp(bomb_count_, 0, Channels::NUM_BOMB_COUNT_CHANNELS - 1);
        Channels::encode_bomb_count_channels(vec, bomb_state);
        if (player_ >= 0 && player_ < NUM_PLAYERS) { vec[Channels::encode_turn(player_)] = ChannelValue{1}; }
    }
    return channels;
}

template<typename Config>
typename DouDiZhuGame<Config>::BeliefRows DouDiZhuGame<Config>::get_belief_rows() const {
    BeliefRows out{};
    const BeliefTriple pad = {Token::PAD_TOKEN, Token::PAD_TOKEN, Token::PAD_TOKEN};
    out.fill(pad);

    if (phase_ != Phase::PLAYING) return out;
    if (player_ < 0 || player_ >= NUM_PLAYERS) return out;
    // Only expose labels on the first action token stage for the active player.
    if (action_prefix_.size() > 1) return out;

    int row = 0;
    // Canonical order: rank high -> low, then owner L -> P1 -> P2 (excluding self).
    for (int rank = NUM_RANKS - 1; rank >= 0; --rank) {
        for (int owner = 0; owner < NUM_PLAYERS; ++owner) {
            if (owner == player_) continue;
            int copies = hand_counts_[owner][rank];
            for (int i = 0; i < copies; ++i) {
                if (row >= MAX_BELIEF_ROWS) {
                    throw std::runtime_error("get_belief_rows exceeded configured row bound.");
                }
                out[static_cast<size_t>(row)] = {
                    Token::encode_player(owner),
                    Token::encode_rank(rank),
                    Token::encode_hold(),
                };
                row += 1;
            }
        }
    }
    return out;
}

template<typename Config>
Observation<DouDiZhuGame<Config>> DouDiZhuGame<Config>::action_observation() const {
    Observation<DouDiZhuGame<Config>> obs;
    const std::vector<int> payload = action_prefix_.empty() ? std::vector<int>{Token::encode_player(player_)} : action_prefix_;
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        obs.tokens[p] = payload;
    }
    return obs;
}

template<typename Config>
std::vector<ActionProto<DouDiZhuGame<Config>>> DouDiZhuGame<Config>::available_actions() const {
    std::map<std::vector<int>, bool> commit_by_tokens;
    for (const auto& seq : current_sequences_) {
        if (stage_index_ >= static_cast<int>(seq.steps.size())) continue;
        const auto& step = seq.steps[stage_index_];
        const auto iter = commit_by_tokens.find(step.tokens);
        if (iter == commit_by_tokens.end()) {
            commit_by_tokens.emplace(step.tokens, step.completes);
        } else if (iter->second != step.completes) {
            throw std::runtime_error("Inconsistent commit flag for token prefix at current stage.");
        }
    }

    std::vector<ActionProto<DouDiZhuGame<Config>>> actions;
    actions.reserve(commit_by_tokens.size());
    for (const auto& [tok, is_commit] : commit_by_tokens) {
        actions.push_back({tok, 1.f, is_commit});
    }
    return actions;
}

template<typename Config>
void DouDiZhuGame<Config>::apply_groups(const std::vector<std::pair<int, int>>& groups, int player) {
    for (const auto& [qty, rank] : groups) {
        hand_counts_[player][rank] -= qty;
        discard_counts_[player][rank] += qty;
        cards_remaining_[player] -= qty;
        if (hand_counts_[player][rank] < 0) throw std::runtime_error("Negative hand count");
    }
}

template<typename Config>
void DouDiZhuGame<Config>::apply_step(const SequenceStep& step, int player) {
    apply_groups(step.apply, player);
}

template<typename Config>
bool DouDiZhuGame<Config>::beats(const Combo& candidate, const Combo& prev) const {
    if (prev.type == Combo::Type::PASS) return candidate.type != Combo::Type::PASS;
    if (candidate.type == Combo::Type::ROCKET) return true;
    if (prev.type == Combo::Type::ROCKET) return false;
    if (candidate.type == Combo::Type::BOMB && prev.type != Combo::Type::BOMB) return true;
    if (prev.type == Combo::Type::BOMB) {
        if (candidate.type != Combo::Type::BOMB) return false;
        return candidate.main_rank > prev.main_rank;
    }
    if (candidate.type != prev.type) return false;
    if (candidate.length != prev.length) return false;
    return candidate.main_rank > prev.main_rank;
}

} // namespace nanopokerlib::doudizhu
#endif
