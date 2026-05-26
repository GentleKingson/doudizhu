#pragma once

#include <array>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.hpp"
#include "core/live_history_tracker.hpp"

namespace nanopokerlib::doudizhu {
struct DouDiZhuConfig {
    inline static const std::string GAME_IDENTIFIER = "DouDiZhu";
    static constexpr int NUM_PLAYERS = 3;
    static constexpr int NORMAL_RANKS = 13;
    static constexpr std::string_view RANKS = "3456789TJQKA2BR";
};

struct DouDiZhuSmallConfig {
    inline static const std::string GAME_IDENTIFIER = "DouDiZhuSmall";
    static constexpr int NUM_PLAYERS = 3;
    static constexpr int NORMAL_RANKS = 7; // 9, T, J, Q, K, A, 2
    static constexpr std::string_view RANKS = "9TJQKA2BR";
};

struct DouDiZhuTinyConfig {
    inline static const std::string GAME_IDENTIFIER = "DouDiZhuTiny";
    static constexpr int NUM_PLAYERS = 3;
    static constexpr int NORMAL_RANKS = 4; // Q, K, A, 2
    static constexpr std::string_view RANKS = "QKA2BR";
};

template<typename Config = DouDiZhuConfig>
struct DouDiZhuGame {
public:
    DouDiZhuGame();
    struct Channels;
    struct Endpoint;

    StepResult<DouDiZhuGame<Config>> reset(int seed = 0);
    StepResult<DouDiZhuGame<Config>> step(const Action<DouDiZhuGame<Config>>& action);
    std::string advance_history_str(const std::string& history_str);
    std::vector<std::string> get_available_action_strs() const;
    template<typename C>
    friend std::ostream& operator<<(std::ostream& os, const DouDiZhuGame<C>& state);

    enum class Phase { DEALING = 0, PLAYING };

    struct Combo {
        enum class Type {
            PASS = 0,
            ROCKET,
            SINGLE,
            PAIR,
            TRIPLE,
            TRIPLE_SINGLE,
            TRIPLE_PAIR,
            STRAIGHT,
            STRAIGHT_PAIR,
            STRAIGHT_TRIPLE,
            PLANE_SINGLE,
            PLANE_PAIR,
            FOUR_TWO_SINGLE,
            FOUR_TWO_PAIR,
            BOMB,
        };

        Type type{Type::PASS};
        std::vector<std::pair<int, int>> main; // {count, rank}
        std::vector<std::pair<int, int>> kickers; // {count, rank}
        int main_rank{-1};
        int length{0}; // straight/chain length (in ranks)
    };

    inline static const std::string GAME_IDENTIFIER = Config::GAME_IDENTIFIER;
    static constexpr int NUM_PLAYERS = Config::NUM_PLAYERS; // 0=Landlord, 1=Peasant1, 2=Peasant2 (clockwise)
    static constexpr int NORMAL_RANKS = Config::NORMAL_RANKS;
    static constexpr std::string_view RANKS = Config::RANKS;
    static constexpr int NUM_JOKERS = 2;
    static constexpr int NUM_RANKS = static_cast<int>(RANKS.size());
    static_assert(NUM_RANKS == NORMAL_RANKS + NUM_JOKERS, "Config ranks must include two jokers");
    static_assert(NORMAL_RANKS > 0, "Config must include at least one normal rank");
    static constexpr int BLACK_JOKER_RANK = NORMAL_RANKS;
    static constexpr int RED_JOKER_RANK = NORMAL_RANKS + 1;
    static constexpr int NUM_CARDS = NORMAL_RANKS * 4 + NUM_JOKERS;
    static constexpr int NUM_PUBLIC_LANDLORD_CARDS = 3;
    static_assert((NUM_CARDS - NUM_PUBLIC_LANDLORD_CARDS) % NUM_PLAYERS == 0, "Deck must split evenly after landlord bonus");
    static constexpr int BASE_HAND_CARDS = (NUM_CARDS - NUM_PUBLIC_LANDLORD_CARDS) / NUM_PLAYERS;
    static constexpr int LANDLORD_HAND_CARDS = BASE_HAND_CARDS + NUM_PUBLIC_LANDLORD_CARDS;
    static constexpr int MAX_SOLO_KICKERS = 5; // Plane length upper bound
    static constexpr int MAX_PAIR_KICKERS = 4;
    static constexpr int SEQ_MAX_RANK = NORMAL_RANKS - 2; // Highest rank allowed in sequences (excludes 2 and jokers)
    static_assert(SEQ_MAX_RANK >= 0, "Config must allow at least one rank in straights");
    static constexpr int NUM_SEQ_FROM = SEQ_MAX_RANK; // start ranks for sequences (3..K in full game)
    static constexpr int NUM_SEQ_TO = SEQ_MAX_RANK + 1; // end ranks for sequences (4..A in full game)
    static constexpr int NUM_PAIR_KICKER_RANKS = NORMAL_RANKS; // Kicker ranks cannot be jokers
    static constexpr int MAX_ACTION_TOKENS = 5;
    static constexpr int MAX_OBSERVATION_TOKENS = MAX_ACTION_TOKENS + 1;
    static constexpr int MAX_AVAILABLE_ACTIONS = 35;
    static constexpr int MAX_BELIEF_CANDIDATES = NUM_PLAYERS - 1;
    static constexpr int MAX_BELIEF_TOKENS = 3;
    static constexpr int MAX_BELIEF_ROWS = 2 * BASE_HAND_CARDS + NUM_PUBLIC_LANDLORD_CARDS;
    static constexpr int NUM_CARD_CHANNELS = 54;
    static constexpr int NUM_BOMB_COUNT_CHANNELS = 15;
    static constexpr int NUM_CHANNELS = 5 * NUM_CARD_CHANNELS + NUM_BOMB_COUNT_CHANNELS + NUM_PLAYERS; // =288
    static constexpr int MAX_GAME_HORIZON = NUM_CARDS * NUM_PLAYERS + NUM_RANKS;

    struct Token;
    using HandCounts = std::array<int, NUM_RANKS>;
    using BeliefTriple = std::array<int, MAX_BELIEF_TOKENS>;
    using BeliefRows = std::array<BeliefTriple, MAX_BELIEF_ROWS>;

    inline Phase phase() const { return phase_; }
    inline int player() const { return player_; }
    [[nodiscard]] BeliefRows get_belief_rows() const;

protected:
    struct SequenceStep {
        std::vector<int> tokens;
        std::vector<std::pair<int, int>> apply; // groups consumed this step
        bool completes{false};
    };

    struct Sequence {
        Combo combo;
        std::vector<SequenceStep> steps;
    };

    Phase phase_{Phase::DEALING};
    int player_{PLAYER_CHANCE_ENVIRONMENT};
    int dealing_step_index_{0};
    std::array<std::vector<int>, NUM_PLAYERS> deal_private_cards_{};
    std::vector<int> deal_landlord_public_cards_{};
    std::array<HandCounts, NUM_PLAYERS> hand_counts_{};
    HandCounts landlord_public_counts_{};
    std::array<HandCounts, NUM_PLAYERS> discard_counts_{};
    std::array<int, NUM_PLAYERS> cards_remaining_{};
    std::mt19937 rng_{};

    std::vector<int> action_prefix_;
    int stage_index_{0};
    std::vector<Sequence> current_sequences_;
    std::optional<Combo> active_combo_;
    int current_leader_{-1};
    int consecutive_passes_{0};
    int bomb_count_{0};
    LiveHistoryTracker live_history_tracker_{};

    StepResult<DouDiZhuGame<Config>> init(const std::array<float, NUM_PLAYERS>& rewards);
    void deal_shuffled_deck();
    Observation<DouDiZhuGame<Config>> action_observation() const;
    std::array<std::vector<ChannelValue>, NUM_PLAYERS> build_channels() const;
    std::string opening_root_history_str() const;
    std::string append_history_segment(const std::string& history_str, const std::string& segment) const;

    std::vector<Combo> enumerate_combos(const HandCounts& hand) const;
    bool beats(const Combo& candidate, const Combo& prev) const;
    Sequence build_sequence(const Combo& combo, int player) const;
    std::vector<Sequence> build_sequences_for_player(int player) const;
    std::vector<ActionProto<DouDiZhuGame<Config>>> available_actions() const;
    void apply_groups(const std::vector<std::pair<int, int>>& groups, int player);
    void apply_step(const SequenceStep& step, int player);
};

using DouDiZhu = DouDiZhuGame<DouDiZhuConfig>;
using DouDiZhuSmall = DouDiZhuGame<DouDiZhuSmallConfig>;
using DouDiZhuTiny = DouDiZhuGame<DouDiZhuTinyConfig>;
} // namespace nanopokerlib::doudizhu
