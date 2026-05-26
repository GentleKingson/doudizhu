#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>

#include "action_proto.hpp"
#include "available_actions.hpp"
#include "constants.hpp"
#include "np_converters.hpp"
#include "observation.hpp"
#include "observation_batch.hpp"
#include "step_result.hpp"

namespace py = pybind11;

namespace nanopokerlib {
// A boundary-aware infoset container that stores observations + taken actions.
// ENDS_WITH_STATE: canonical decode output `s,a,...,s` with final decision metadata.
// ENDS_WITH_ACTION: canonical replay/encode trajectory `s,a,...,s,a`.
template<typename Game>
class History {
private:
    HistoryBoundary boundary_{HistoryBoundary::ENDS_WITH_STATE};
    std::vector<Observation<Game>> observations_history_;
    std::vector<std::vector<int>> action_taken_tokens_;
    std::vector<int> player_ids_history_;
    std::vector<ActionProto<Game>> last_available_actions_;
    std::vector<std::string> last_available_action_strs_;

    static void validate_observation_rows(const std::vector<Observation<Game>>& observations_history) {
        for (const auto& obs : observations_history) {
            for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
                if (obs.tokens[p].size() > static_cast<size_t>(Game::MAX_OBSERVATION_TOKENS)) {
                    throw std::runtime_error(
                        "History observations contain too many tokens for player " + std::to_string(p) + "."
                    );
                }
                if (obs.channels[p].size() > static_cast<size_t>(Game::NUM_CHANNELS)) {
                    throw std::runtime_error(
                        "History observations contain too many channels for player " + std::to_string(p) + "."
                    );
                }
            }
        }
    }

    static void validate_action_rows(const std::vector<std::vector<int>>& action_taken_tokens) {
        for (const auto& row : action_taken_tokens) {
            if (row.size() > static_cast<size_t>(Game::MAX_ACTION_TOKENS)) {
                throw std::runtime_error(
                    "History action_taken_tokens row exceeds MAX_ACTION_TOKENS=" + std::to_string(Game::MAX_ACTION_TOKENS) + "."
                );
            }
        }
    }

    static History build(
        HistoryBoundary boundary,
        std::vector<Observation<Game>> observations_history,
        std::vector<std::vector<int>> action_taken_tokens,
        std::vector<int> player_ids_history,
        std::vector<ActionProto<Game>> last_available_actions,
        std::vector<std::string> last_available_action_strs
    ) {
        if (observations_history.empty()) {
            throw std::runtime_error("History requires at least one observation step.");
        }
        if (action_taken_tokens.size() != observations_history.size()) {
            throw std::runtime_error(
                "History action_taken_tokens must have one row per observation step (shape [T, MAX_ACTION_TOKENS])."
            );
        }

        validate_observation_rows(observations_history);
        validate_action_rows(action_taken_tokens);
        if (player_ids_history.size() != observations_history.size()) {
            throw std::runtime_error(
                "History player_ids must have one row per observation step (shape [T])."
            );
        }
        if (boundary == HistoryBoundary::ENDS_WITH_STATE) {
            if (!action_taken_tokens.back().empty()) {
                throw std::runtime_error(
                    "ENDS_WITH_STATE history invariant violated: final action_taken_tokens row must be PAD-only (empty token list)."
                );
            }
            if (last_available_actions.size() > static_cast<size_t>(Game::MAX_AVAILABLE_ACTIONS)) {
                throw std::runtime_error(
                    "ENDS_WITH_STATE history invariant violated: last_available_actions exceeds MAX_AVAILABLE_ACTIONS=" +
                    std::to_string(Game::MAX_AVAILABLE_ACTIONS) + "."
                );
            }
            if (!last_available_action_strs.empty() && last_available_action_strs.size() != last_available_actions.size()) {
                throw std::runtime_error(
                    "ENDS_WITH_STATE history invariant violated: last_available_action_strs must align 1:1 with last_available_actions."
                );
            }
        } else if (boundary == HistoryBoundary::ENDS_WITH_ACTION) {
            if (!last_available_actions.empty()) {
                throw std::runtime_error("ENDS_WITH_ACTION history must not include last available actions.");
            }
            if (!last_available_action_strs.empty()) {
                throw std::runtime_error("ENDS_WITH_ACTION history must not include last available action strings.");
            }
        } else {
            throw std::runtime_error("History boundary is invalid.");
        }

        History out;
        out.boundary_ = boundary;
        out.observations_history_ = std::move(observations_history);
        out.action_taken_tokens_ = std::move(action_taken_tokens);
        out.player_ids_history_ = std::move(player_ids_history);
        out.last_available_actions_ = std::move(last_available_actions);
        out.last_available_action_strs_ = std::move(last_available_action_strs);
        return out;
    }

    History() = default;

public:
    static History from_state_boundary(
        std::vector<Observation<Game>> observations_history,
        std::vector<std::vector<int>> action_taken_tokens,
        std::vector<int> player_ids_history,
        std::vector<ActionProto<Game>> last_available_actions,
        std::vector<std::string> last_available_action_strs = {}
    ) {
        return build(
            HistoryBoundary::ENDS_WITH_STATE,
            std::move(observations_history),
            std::move(action_taken_tokens),
            std::move(player_ids_history),
            std::move(last_available_actions),
            std::move(last_available_action_strs)
        );
    }

    static History from_action_boundary(
        std::vector<Observation<Game>> observations_history,
        std::vector<std::vector<int>> action_taken_tokens,
        std::vector<int> player_ids_history
    ) {
        return build(
            HistoryBoundary::ENDS_WITH_ACTION,
            std::move(observations_history),
            std::move(action_taken_tokens),
            std::move(player_ids_history),
            {},
            {}
        );
    }

    // Convenience adapter: derive inference History from StepResult replay output,
    // while keeping History storage canonical (observations + action rows + last metadata).
    static History from_inference_steps(
        std::vector<StepResult<Game>> steps,
        std::vector<std::vector<int>> action_taken_tokens,
        std::vector<std::string> last_available_action_strs = {}
    ) {
        if (steps.empty()) {
            throw std::runtime_error("from_inference_steps requires a non-empty step list.");
        }

        std::vector<Observation<Game>> observations_history;
        std::vector<int> player_ids_history;
        observations_history.reserve(steps.size());
        player_ids_history.reserve(steps.size());
        for (const auto& step : steps) {
            observations_history.push_back(step.observations);
            player_ids_history.push_back(step.player_id);
        }

        const auto& last = steps.back();
        return from_state_boundary(
            std::move(observations_history),
            std::move(action_taken_tokens),
            std::move(player_ids_history),
            last.available_actions,
            std::move(last_available_action_strs)
        );
    }

    // Indexed access reconstructs StepResult views from canonical history tensors.
    StepResult<Game> operator[](int index) const {
        if (index < 0 || index >= static_cast<int>(observations_history_.size())) {
            throw std::out_of_range("History index out of range: " + std::to_string(index));
        }

        const auto idx = static_cast<size_t>(index);
        StepResult<Game> step{};
        step.rewards.fill(0.f);
        step.is_terminal = false;
        step.observations = observations_history_[idx];
        step.player_id = player_ids_history_[idx];

        // Expose final-step decision metadata in state-ended mode only.
        if (boundary_ == HistoryBoundary::ENDS_WITH_STATE && idx + 1 == observations_history_.size()) {
            step.available_actions = last_available_actions_;
            return step;
        }

        step.available_actions.clear();
        return step;
    }

    [[nodiscard]] int size() const { return static_cast<int>(observations_history_.size()); }

    [[nodiscard]] int num_steps() const { return static_cast<int>(observations_history_.size()); }

    [[nodiscard]] int get_boundary() const { return static_cast<int>(boundary_); }

    [[nodiscard]] HistoryBoundary boundary() const { return boundary_; }

    [[nodiscard]] const std::vector<Observation<Game>>& observations_data() const { return observations_history_; }
    [[nodiscard]] const std::vector<std::vector<int>>& action_taken_tokens_data() const { return action_taken_tokens_; }
    [[nodiscard]] const std::vector<int>& player_ids_data() const { return player_ids_history_; }

    [[nodiscard]] const std::vector<ActionProto<Game>>& last_available_actions_data() const {
        if (boundary_ != HistoryBoundary::ENDS_WITH_STATE) {
            throw std::runtime_error("last_available_actions is only available for ENDS_WITH_STATE history.");
        }
        return last_available_actions_;
    }

    [[nodiscard]] const std::vector<std::string>& last_available_action_strs_data() const {
        if (boundary_ != HistoryBoundary::ENDS_WITH_STATE) {
            throw std::runtime_error("last_available_action_strs is only available for ENDS_WITH_STATE history.");
        }
        if (!last_available_actions_.empty() && last_available_action_strs_.empty()) {
            throw std::runtime_error(
                "last_available_action_strs are not populated for this history; use parse_infoset_str* outputs."
            );
        }
        if (!last_available_action_strs_.empty() && last_available_action_strs_.size() != last_available_actions_.size()) {
            throw std::runtime_error(
                "last_available_action_strs must align 1:1 with last_available_actions."
            );
        }
        return last_available_action_strs_;
    }

    // Collate observation tokens for each step into a ObservationBatch
    // struct with shape [num_steps, Game::NUM_PLAYERS,
    //                    Game::MAX_OBSERVATION_TOKENS].
    [[nodiscard]] ObservationBatch get_observations() const;

    // Collate taken-action tokens for each step into shape [num_steps, MAX_ACTION_TOKENS].
    [[nodiscard]] py::array_t<int> get_action_taken_tokens() const;

    // Collate player ids for each step into shape [num_steps].
    [[nodiscard]] py::array_t<int> get_player_ids() const;

    // History boundary enum encoded as int.
    [[nodiscard]] int get_boundary_property() const { return get_boundary(); }

    [[nodiscard]] AvailableActions get_last_available_actions() const;
    [[nodiscard]] std::vector<std::string> get_last_available_action_strs() const { return last_available_action_strs_data(); }

    // Normalize state-ended trajectories into action-ended replay trajectories for encode paths.
    [[nodiscard]] History normalized_for_encode() const {
        if (boundary_ == HistoryBoundary::ENDS_WITH_ACTION) {
            return *this;
        }

        // Prefer dropping the trailing state frame when possible.
        if (observations_history_.size() > 1) {
            std::vector<Observation<Game>> obs_trimmed(observations_history_.begin(), observations_history_.end() - 1);
            std::vector<std::vector<int>> actions_trimmed(action_taken_tokens_.begin(), action_taken_tokens_.end() - 1);
            std::vector<int> player_ids_trimmed(player_ids_history_.begin(), player_ids_history_.end() - 1);
            return from_action_boundary(
                std::move(obs_trimmed),
                std::move(actions_trimmed),
                std::move(player_ids_trimmed)
            );
        }

        // Degenerate single-state prefixes keep one row to satisfy History shape invariants.
        return from_action_boundary(observations_history_, action_taken_tokens_, player_ids_history_);
    }
};

// --- History<Game> method definitions ---

template<typename Game>
ObservationBatch History<Game>::get_observations() const {
    auto n = static_cast<py::ssize_t>(observations_history_.size());
    py::array_t<int> obs_tokens(std::vector<py::ssize_t>{n, Game::NUM_PLAYERS, Game::MAX_OBSERVATION_TOKENS});
    py::array_t<ChannelValue> obs_channels(std::vector<py::ssize_t>{n, Game::NUM_PLAYERS, Game::NUM_CHANNELS});
    int* ptr = static_cast<int*>(obs_tokens.request().ptr);
    auto* ptr_channels = static_cast<ChannelValue*>(obs_channels.request().ptr);

    std::fill(ptr, ptr + n * Game::NUM_PLAYERS * Game::MAX_OBSERVATION_TOKENS, PAD_TOKEN);
    std::fill(ptr_channels, ptr_channels + n * Game::NUM_PLAYERS * Game::NUM_CHANNELS, ChannelValue{0});

    for (py::ssize_t i = 0; i < n; ++i) {
        const auto& obs = observations_history_[static_cast<size_t>(i)];
        for (int j = 0; j < Game::NUM_PLAYERS; ++j) {
            const auto& tokens = obs.tokens[j];
            for (int k = 0; k < static_cast<int>(tokens.size()) && k < Game::MAX_OBSERVATION_TOKENS; ++k) {
                ptr[(i * Game::NUM_PLAYERS + j) * Game::MAX_OBSERVATION_TOKENS + k] = tokens[static_cast<size_t>(k)];
            }

            const auto& channels = obs.channels[j];
            for (int k = 0; k < static_cast<int>(channels.size()) && k < Game::NUM_CHANNELS; ++k) {
                ptr_channels[(i * Game::NUM_PLAYERS + j) * Game::NUM_CHANNELS + k] = channels[static_cast<size_t>(k)];
            }
        }
    }

    return {obs_tokens, obs_channels};
}

template<typename Game>
py::array_t<int> History<Game>::get_action_taken_tokens() const {
    const auto n = static_cast<py::ssize_t>(action_taken_tokens_.size());
    py::array_t<int> out(std::vector<py::ssize_t>{n, Game::MAX_ACTION_TOKENS});
    auto* ptr = static_cast<int*>(out.request().ptr);
    std::fill(ptr, ptr + n * Game::MAX_ACTION_TOKENS, PAD_TOKEN);

    for (py::ssize_t i = 0; i < n; ++i) {
        const auto& row = action_taken_tokens_[static_cast<size_t>(i)];
        for (int k = 0; k < static_cast<int>(row.size()) && k < Game::MAX_ACTION_TOKENS; ++k) {
            ptr[i * Game::MAX_ACTION_TOKENS + k] = row[static_cast<size_t>(k)];
        }
    }

    return out;
}

template<typename Game>
py::array_t<int> History<Game>::get_player_ids() const {
    const auto n = static_cast<py::ssize_t>(player_ids_history_.size());
    py::array_t<int> out(std::vector<py::ssize_t>{n});
    auto* ptr = static_cast<int*>(out.request().ptr);
    for (py::ssize_t i = 0; i < n; ++i) {
        ptr[i] = player_ids_history_[static_cast<size_t>(i)];
    }
    return out;
}

template<typename Game>
AvailableActions History<Game>::get_last_available_actions() const {
    const auto& actions = last_available_actions_data();
    const auto n_actions = static_cast<py::ssize_t>(actions.size());

    py::array_t<int> tokens(std::vector<py::ssize_t>{n_actions, static_cast<py::ssize_t>(Game::MAX_ACTION_TOKENS)});
    auto* ptr_tokens = static_cast<int*>(tokens.request().ptr);
    std::fill(ptr_tokens, ptr_tokens + n_actions * static_cast<py::ssize_t>(Game::MAX_ACTION_TOKENS), PAD_TOKEN);

    py::array_t<float> weights(std::vector<py::ssize_t>{n_actions});
    auto* ptr_weights = static_cast<float*>(weights.request().ptr);

    py::array_t<std::uint8_t> is_commit(std::vector<py::ssize_t>{n_actions});
    auto* ptr_commit = static_cast<std::uint8_t*>(is_commit.request().ptr);

    for (py::ssize_t i = 0; i < n_actions; ++i) {
        const auto& proto = actions[static_cast<size_t>(i)];
        if (proto.tokens.size() > static_cast<size_t>(Game::MAX_ACTION_TOKENS)) {
            throw std::runtime_error(
                "last_available_actions: action tokens size mismatch: " + std::to_string(proto.tokens.size()) + " > " +
                std::to_string(Game::MAX_ACTION_TOKENS)
            );
        }

        for (int k = 0; k < static_cast<int>(proto.tokens.size()); ++k) {
            ptr_tokens[i * static_cast<py::ssize_t>(Game::MAX_ACTION_TOKENS) + k] = proto.tokens[static_cast<size_t>(k)];
        }
        ptr_weights[i] = proto.weight;
        ptr_commit[i] = static_cast<std::uint8_t>(proto.is_commit);
    }

    return {tokens, weights, is_commit, player_ids_history_.back()};
}

template<typename Game>
History<Game> history_from_np(
    int boundary,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& obs_tokens,
    const py::array_t<ChannelValue, py::array::c_style | py::array::forcecast>& obs_channels,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& action_taken_tokens,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& player_ids,
    const py::object& last_avail_tokens,
    const py::object& last_avail_weights,
    const py::object& last_avail_is_commit
) {
    const auto boundary_enum = static_cast<HistoryBoundary>(boundary);
    if (boundary_enum != HistoryBoundary::ENDS_WITH_STATE && boundary_enum != HistoryBoundary::ENDS_WITH_ACTION) {
        throw std::runtime_error(
            "history_from_np: boundary must be HistoryBoundary.ENDS_WITH_STATE or HistoryBoundary.ENDS_WITH_ACTION."
        );
    }

    auto obs_tokens_info = obs_tokens.request();
    auto obs_channels_info = obs_channels.request();
    auto act_taken_info = action_taken_tokens.request();
    auto player_ids_info = player_ids.request();

    if (obs_tokens_info.ndim != 3 || obs_tokens_info.shape[1] != Game::NUM_PLAYERS || obs_tokens_info.shape[2] != Game::MAX_OBSERVATION_TOKENS) {
        throw std::runtime_error(
            "history_from_np: obs_tokens must have shape [T, NUM_PLAYERS, MAX_OBSERVATION_TOKENS]."
        );
    }

    const py::ssize_t T = obs_tokens_info.shape[0];

    if (obs_channels_info.ndim != 3 || obs_channels_info.shape[0] != T || obs_channels_info.shape[1] != Game::NUM_PLAYERS ||
        obs_channels_info.shape[2] != Game::NUM_CHANNELS) {
        throw std::runtime_error("history_from_np: obs_channels must have shape [T, NUM_PLAYERS, NUM_CHANNELS].");
    }

    if (act_taken_info.ndim != 2 || act_taken_info.shape[0] != T || act_taken_info.shape[1] != Game::MAX_ACTION_TOKENS) {
        throw std::runtime_error("history_from_np: action_taken_tokens must have shape [T, MAX_ACTION_TOKENS].");
    }
    if (player_ids_info.ndim != 1 || player_ids_info.shape[0] != T) {
        throw std::runtime_error("history_from_np: player_ids must have shape [T].");
    }

    if (T <= 0) {
        throw std::runtime_error("history_from_np: T must be > 0.");
    }

    const auto* ptr_obs_tokens = static_cast<const int*>(obs_tokens_info.ptr);
    const auto* ptr_obs_channels = static_cast<const ChannelValue*>(obs_channels_info.ptr);
    const auto* ptr_action_taken = static_cast<const int*>(act_taken_info.ptr);
    const auto* ptr_player_ids = static_cast<const int*>(player_ids_info.ptr);

    std::vector<Observation<Game>> observations;
    observations.reserve(static_cast<size_t>(T));

    std::vector<std::vector<int>> action_rows;
    action_rows.reserve(static_cast<size_t>(T));
    std::vector<int> player_ids_history;
    player_ids_history.reserve(static_cast<size_t>(T));

    for (py::ssize_t t = 0; t < T; ++t) {
        Observation<Game> obs;
        for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
            auto& tok_row = obs.tokens[p];
            tok_row.reserve(Game::MAX_OBSERVATION_TOKENS);
            for (int k = 0; k < Game::MAX_OBSERVATION_TOKENS; ++k) {
                const py::ssize_t offset = (t * Game::NUM_PLAYERS + p) * Game::MAX_OBSERVATION_TOKENS + k;
                const int tok = ptr_obs_tokens[offset];
                if (tok != PAD_TOKEN) {
                    tok_row.push_back(tok);
                }
            }

            auto& ch_row = obs.channels[p];
            ch_row.resize(Game::NUM_CHANNELS);
            for (int k = 0; k < Game::NUM_CHANNELS; ++k) {
                const py::ssize_t offset = (t * Game::NUM_PLAYERS + p) * Game::NUM_CHANNELS + k;
                ch_row[static_cast<size_t>(k)] = ptr_obs_channels[offset];
            }
        }
        observations.push_back(std::move(obs));

        std::vector<int> action_tokens_row;
        action_tokens_row.reserve(Game::MAX_ACTION_TOKENS);
        for (int k = 0; k < Game::MAX_ACTION_TOKENS; ++k) {
            const int tok = ptr_action_taken[t * Game::MAX_ACTION_TOKENS + k];
            if (tok != PAD_TOKEN) {
                action_tokens_row.push_back(tok);
            }
        }
        action_rows.push_back(std::move(action_tokens_row));
        player_ids_history.push_back(ptr_player_ids[t]);
    }

    if (boundary_enum == HistoryBoundary::ENDS_WITH_ACTION) {
        if (!last_avail_tokens.is_none() || !last_avail_weights.is_none() || !last_avail_is_commit.is_none()) {
            throw std::runtime_error(
                "history_from_np: ENDS_WITH_ACTION boundary must not provide last_available_actions tensors."
            );
        }
        return History<Game>::from_action_boundary(
            std::move(observations),
            std::move(action_rows),
            std::move(player_ids_history)
        );
    }

    if (last_avail_tokens.is_none() || last_avail_weights.is_none() || last_avail_is_commit.is_none()) {
        throw std::runtime_error(
            "history_from_np: ENDS_WITH_STATE boundary requires last_available_actions tensors."
        );
    }

    py::array_t<int, py::array::c_style | py::array::forcecast> last_tokens_arr(last_avail_tokens);
    py::array_t<float, py::array::c_style | py::array::forcecast> last_weights_arr(last_avail_weights);
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> last_commit_arr(last_avail_is_commit);

    auto last_tokens_info = last_tokens_arr.request();
    auto last_weights_info = last_weights_arr.request();
    auto last_commit_info = last_commit_arr.request();

    if (last_tokens_info.ndim != 2 || last_tokens_info.shape[1] != Game::MAX_ACTION_TOKENS) {
        throw std::runtime_error("history_from_np: last_avail_tokens must have shape [A, MAX_ACTION_TOKENS].");
    }
    const py::ssize_t A = last_tokens_info.shape[0];
    if (A > static_cast<py::ssize_t>(Game::MAX_AVAILABLE_ACTIONS)) {
        throw std::runtime_error(
            "history_from_np: last_avail_tokens has too many rows A=" + std::to_string(static_cast<int>(A)) +
            " > MAX_AVAILABLE_ACTIONS=" + std::to_string(Game::MAX_AVAILABLE_ACTIONS) + "."
        );
    }

    if (last_weights_info.ndim != 1 || last_weights_info.shape[0] != A) {
        throw std::runtime_error("history_from_np: last_avail_weights must have shape [A].");
    }
    if (last_commit_info.ndim != 1 || last_commit_info.shape[0] != A) {
        throw std::runtime_error("history_from_np: last_avail_is_commit must have shape [A].");
    }

    const auto* ptr_last_tokens = static_cast<const int*>(last_tokens_info.ptr);
    const auto* ptr_last_weights = static_cast<const float*>(last_weights_info.ptr);
    const auto* ptr_last_commit = static_cast<const std::uint8_t*>(last_commit_info.ptr);

    std::vector<ActionProto<Game>> last_actions;
    last_actions.reserve(static_cast<size_t>(A));
    for (py::ssize_t a = 0; a < A; ++a) {
        ActionProto<Game> proto;
        proto.weight = ptr_last_weights[a];
        proto.is_commit = ptr_last_commit[a] != 0;

        bool has_non_pad = false;
        proto.tokens.reserve(Game::MAX_ACTION_TOKENS);
        for (int k = 0; k < Game::MAX_ACTION_TOKENS; ++k) {
            const int tok = ptr_last_tokens[a * Game::MAX_ACTION_TOKENS + k];
            if (tok != PAD_TOKEN) {
                has_non_pad = true;
                proto.tokens.push_back(tok);
            }
        }

        if (has_non_pad || proto.weight != 0.f || proto.is_commit) {
            last_actions.push_back(std::move(proto));
        }
    }

    return History<Game>::from_state_boundary(
        std::move(observations),
        std::move(action_rows),
        std::move(player_ids_history),
        std::move(last_actions)
    );
}

} // namespace nanopokerlib
