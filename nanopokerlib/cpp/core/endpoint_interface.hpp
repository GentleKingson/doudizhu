#pragma once

#include <concepts>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "types.hpp"

namespace nanopokerlib {
// Endpoint contract:
// - parse_infoset_str reconstructs an infoset trajectory (observations/channels/actions)
//   from a semantic state description that may include partial private information.
// - parse_infoset_strs is always provided by EndpointFactory as a scalar-loop wrapper.
// - format_infoset_str serializes an History object back into semantic string form.
template<typename Game>
concept HasParseInfosetStrEndpoint = requires(const std::string& infoset_str, const std::string& pending_action) {
    { Game::Endpoint::parse_infoset_str(infoset_str, pending_action) } -> std::same_as<History<Game>>;
};

template<typename Game>
concept HasFormatInfosetStrEndpoint = requires(const History<Game>& history) {
    { Game::Endpoint::format_infoset_str(history) } -> std::same_as<std::string>;
};

template<typename Game>
concept HasFormatHistoryStrEndpoint = requires(const History<Game>& history) {
    { Game::Endpoint::format_history_str(history) } -> std::same_as<std::string>;
};

template<typename Game>
concept HasReconstructEnvEndpoint = requires(const std::string& history_str, const std::string& pending_action, int seed) {
    { Game::Endpoint::reconstruct_env(history_str, pending_action, seed) } -> std::same_as<std::pair<Game, StepResult<Game>>>;
};

template<typename Game>
concept HasParseMatchStrEndpoint = requires(const std::string& match_str) {
    { Game::Endpoint::parse_match_str(match_str) } -> std::same_as<MatchStateView>;
};

template<typename Game>
concept HasFormatMatchStrEndpoint = requires(const MatchStateView& match_state) {
    { Game::Endpoint::format_match_str(match_state) } -> std::same_as<std::string>;
};

template<typename Game>
concept HasActionDecisionChainEndpoint = requires(const std::string& committed_action) {
    { Game::Endpoint::action_decision_chain(committed_action) } -> std::same_as<std::vector<std::string>>;
};

template<typename Game>
concept HasExpandBeliefPrefixStrEndpoint = requires(const std::string& infoset_str, const std::string& belief_str) {
    Game::Endpoint::expand_belief_prefix_str(infoset_str, belief_str);
};

template<typename Game>
concept HasDeterminizeHistoryStrEndpoint = requires(const std::string& infoset_str, const std::string& belief_str) {
    { Game::Endpoint::determinize_history_str(infoset_str, belief_str) } -> std::same_as<std::string>;
};

template<typename Game>
concept HasInfosetStrFromHistoryStrEndpoint = requires(const std::string& history_str) {
    { Game::Endpoint::infoset_str_from_history_str(history_str) } -> std::same_as<std::string>;
};

template<typename Game>
struct EndpointCapabilities {
    static constexpr bool has_parse_infoset_str = HasParseInfosetStrEndpoint<Game>;
    static constexpr bool has_format_infoset_str = HasFormatInfosetStrEndpoint<Game>;
    static constexpr bool has_parse_match_str = HasParseMatchStrEndpoint<Game>;
    static constexpr bool has_format_match_str = HasFormatMatchStrEndpoint<Game>;
    static constexpr bool has_expand_belief_prefix_str = HasExpandBeliefPrefixStrEndpoint<Game>;
    static constexpr bool has_determinize_history_str = HasDeterminizeHistoryStrEndpoint<Game>;
    static constexpr bool has_infoset_str_from_history_str = HasInfosetStrFromHistoryStrEndpoint<Game>;
};

template<typename Game>
struct EndpointFactory {
    static History<Game> parse_infoset_str(const std::string& infoset_str, const std::string& pending_action = "") {
        static_assert(HasParseInfosetStrEndpoint<Game>, "Game::Endpoint must define parse_infoset_str(infoset_str, pending_action).");
        return Game::Endpoint::parse_infoset_str(infoset_str, pending_action);
    }

    static HistoryBatch<Game>
    parse_infoset_strs(const std::vector<std::string>& infoset_strs, const std::vector<std::string>& pending_actions = {}) {
        if (!pending_actions.empty() && pending_actions.size() != infoset_strs.size()) {
            throw std::runtime_error("pending_actions must match the number of infoset_strs.");
        }
        std::vector<History<Game>> sequences;
        sequences.reserve(infoset_strs.size());
        for (size_t i = 0; i < infoset_strs.size(); ++i) {
            const std::string pending = pending_actions.empty() ? std::string() : pending_actions[i];
            sequences.push_back(parse_infoset_str(infoset_strs[i], pending));
        }
        return HistoryBatch<Game>(std::move(sequences));
    }

    static std::string format_infoset_str(const History<Game>& history) {
        if constexpr (HasFormatInfosetStrEndpoint<Game>) {
            const History<Game> normalized = history.normalized_for_encode();
            return Game::Endpoint::format_infoset_str(normalized);
        }
        throw std::runtime_error("Endpoint does not support format_infoset_str for this game.");
    }

    static std::vector<std::string> format_infoset_strs(const std::vector<History<Game>>& histories) {
        std::vector<std::string> out;
        out.reserve(histories.size());
        for (const History<Game>& history : histories) {
            out.push_back(format_infoset_str(history));
        }
        return out;
    }

    static std::string format_history_str(const History<Game>& history) {
        if constexpr (HasFormatHistoryStrEndpoint<Game>) { return Game::Endpoint::format_history_str(history.normalized_for_encode()); }
        throw std::runtime_error("Endpoint does not support format_history_str for this game.");
    }

    static std::vector<std::string> format_history_strs(const std::vector<History<Game>>& histories) {
        std::vector<std::string> out;
        out.reserve(histories.size());
        for (const History<Game>& history : histories) {
            out.push_back(format_history_str(history));
        }
        return out;
    }

    static std::vector<std::string> action_decision_chain(const std::string& committed_action) {
        if constexpr (HasActionDecisionChainEndpoint<Game>) { return Game::Endpoint::action_decision_chain(committed_action); }
        throw std::runtime_error("Endpoint does not support action_decision_chain for this game.");
    }

    static std::vector<std::vector<std::string>> action_decision_chain_batch(const std::vector<std::string>& committed_actions) {
        std::vector<std::vector<std::string>> out;
        out.reserve(committed_actions.size());
        for (const std::string& committed_action : committed_actions) {
            out.push_back(action_decision_chain(committed_action));
        }
        return out;
    }

    static std::pair<Game, StepResult<Game>>
    reconstruct_env(const std::string& history_str, const std::string& pending_action = "", int seed = 0) {
        if constexpr (HasReconstructEnvEndpoint<Game>) { return Game::Endpoint::reconstruct_env(history_str, pending_action, seed); }
        throw std::runtime_error("Endpoint does not support reconstruct_env for this game.");
    }

    static std::pair<EnvBatch<Game>, StepBatch<Game>>
    reconstruct_envs(const std::vector<std::string>& history_strs, const std::vector<std::string>& pending_actions = {}, int seed = 0) {
        if (!pending_actions.empty() && pending_actions.size() != history_strs.size()) {
            throw std::runtime_error("reconstruct_envs: pending_actions must match history_strs.");
        }

        const int n = static_cast<int>(history_strs.size());
        EnvBatch<Game> env(n);

        std::mt19937 seeder(static_cast<std::mt19937::result_type>(seed));
        std::vector<Game> states;
        std::vector<StepResult<Game>> steps;
        states.reserve(static_cast<size_t>(n));
        steps.reserve(static_cast<size_t>(n));

        const std::string empty;
        for (int i = 0; i < n; ++i) {
            const std::string& pending = pending_actions.empty() ? empty : pending_actions[static_cast<size_t>(i)];
            const int env_seed = static_cast<int>(seeder());
            auto out = reconstruct_env(history_strs[static_cast<size_t>(i)], pending, env_seed);
            states.push_back(std::move(out.first));
            steps.push_back(std::move(out.second));
        }

        auto step = env.reset_to_states(std::move(states), std::move(steps));
        return std::make_pair(std::move(env), std::move(step));
    }

    static auto expand_belief_prefix_str(const std::string& infoset_str, const std::string& belief_str = "") {
        if constexpr (HasExpandBeliefPrefixStrEndpoint<Game>) { return Game::Endpoint::expand_belief_prefix_str(infoset_str, belief_str); }
        throw std::runtime_error("Endpoint does not support expand_belief_prefix_str for this game.");
    }

    static std::string determinize_history_str(const std::string& infoset_str, const std::string& belief_str) {
        if constexpr (HasDeterminizeHistoryStrEndpoint<Game>) { return Game::Endpoint::determinize_history_str(infoset_str, belief_str); }
        throw std::runtime_error("Endpoint does not support determinize_history_str for this game.");
    }

    static std::string infoset_str_from_history_str(const std::string& history_str) {
        if constexpr (HasInfosetStrFromHistoryStrEndpoint<Game>) { return Game::Endpoint::infoset_str_from_history_str(history_str); }
        throw std::runtime_error("Endpoint does not support infoset_str_from_history_str for this game.");
    }

    static MatchStateView parse_match_str(const std::string& match_str) {
        if constexpr (HasParseMatchStrEndpoint<Game>) { return Game::Endpoint::parse_match_str(match_str); }
        throw std::runtime_error("Endpoint does not support parse_match_str for this game.");
    }

    static std::vector<MatchStateView> parse_match_strs(const std::vector<std::string>& match_strs) {
        std::vector<MatchStateView> out;
        out.reserve(match_strs.size());
        for (const std::string& match_str : match_strs) {
            out.push_back(parse_match_str(match_str));
        }
        return out;
    }

    static std::string format_match_str(const MatchStateView& match_state) {
        if constexpr (HasFormatMatchStrEndpoint<Game>) { return Game::Endpoint::format_match_str(match_state); }
        throw std::runtime_error("Endpoint does not support format_match_str for this game.");
    }

    static std::vector<std::string> format_match_strs(const std::vector<MatchStateView>& match_states) {
        std::vector<std::string> out;
        out.reserve(match_states.size());
        for (const MatchStateView& match_state : match_states) {
            out.push_back(format_match_str(match_state));
        }
        return out;
    }
};
} // namespace nanopokerlib
