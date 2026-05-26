#pragma once

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>

#include "available_actions_batch.hpp"
#include "history.hpp"
#include "observation_batch.hpp"

namespace py = pybind11;

namespace nanopokerlib {
template<typename Game>
class HistoryBatch {
private:
    std::vector<History<Game>> sequences;
    HistoryBoundary boundary_{HistoryBoundary::ENDS_WITH_STATE};

public:
    explicit HistoryBatch(std::vector<History<Game>> seqs) : sequences(std::move(seqs)) {
        if (sequences.empty()) {
            boundary_ = HistoryBoundary::ENDS_WITH_STATE;
            return;
        }
        boundary_ = sequences.front().boundary();
        for (const auto& seq : sequences) {
            if (seq.boundary() != boundary_) {
                throw std::runtime_error("HistoryBatch requires all History entries to share the same boundary.");
            }
        }
    }

    [[nodiscard]] inline int size() const { return static_cast<int>(sequences.size()); }

    [[nodiscard]] int get_boundary() const { return static_cast<int>(boundary_); }

    [[nodiscard]] inline int max_steps() const {
        int max_len = 0;
        for (const auto& seq : sequences)
            max_len = std::max(max_len, seq.num_steps());
        return max_len;
    }

    [[nodiscard]] py::array_t<int> get_num_steps() const {
        auto B = static_cast<py::ssize_t>(sequences.size());
        py::array_t<int> out(std::vector<py::ssize_t>{B});
        auto buf = out.request();
        auto* ptr = static_cast<int*>(buf.ptr);
        for (py::ssize_t i = 0; i < B; ++i) {
            ptr[i] = sequences[i].num_steps();
        }
        return out;
    }

    [[nodiscard]] ObservationBatch get_observations() const {
        auto B = static_cast<py::ssize_t>(sequences.size());
        auto T = static_cast<py::ssize_t>(max_steps());
        py::array_t<int> out_tokens(std::vector<py::ssize_t>{B, T, Game::NUM_PLAYERS, Game::MAX_OBSERVATION_TOKENS});
        py::array_t<ChannelValue> out_channels(std::vector<py::ssize_t>{B, T, Game::NUM_PLAYERS, Game::NUM_CHANNELS});
        auto buf_tokens = out_tokens.request();
        auto buf_channels = out_channels.request();
        auto* ptr = static_cast<int*>(buf_tokens.ptr);
        auto* ptr_channels = static_cast<ChannelValue*>(buf_channels.ptr);
        std::fill(ptr, ptr + B * T * Game::NUM_PLAYERS * Game::MAX_OBSERVATION_TOKENS, PAD_TOKEN);
        std::fill(ptr_channels, ptr_channels + B * T * Game::NUM_PLAYERS * Game::NUM_CHANNELS, ChannelValue{0});

        for (py::ssize_t b = 0; b < B; ++b) {
            const auto& observations = sequences[b].observations_data();
            for (py::ssize_t t = 0; t < static_cast<py::ssize_t>(observations.size()); ++t) {
                for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
                    const auto& tokens = observations[static_cast<size_t>(t)].tokens[p];
                    for (py::ssize_t k = 0; k < static_cast<py::ssize_t>(tokens.size()) && k < Game::MAX_OBSERVATION_TOKENS; ++k) {
                        const py::ssize_t offset = (((b * T) + t) * Game::NUM_PLAYERS + p) * Game::MAX_OBSERVATION_TOKENS + k;
                        ptr[offset] = tokens[k];
                    }
                    const auto& channels = observations[static_cast<size_t>(t)].channels[p];
                    for (py::ssize_t k = 0; k < static_cast<py::ssize_t>(channels.size()) && k < Game::NUM_CHANNELS; ++k) {
                        const py::ssize_t offset = (((b * T) + t) * Game::NUM_PLAYERS + p) * Game::NUM_CHANNELS + k;
                        ptr_channels[offset] = channels[k];
                    }
                }
            }
        }
        return {out_tokens, out_channels};
    }

    [[nodiscard]] py::array_t<int> get_action_taken_tokens() const {
        auto B = static_cast<py::ssize_t>(sequences.size());
        auto T = static_cast<py::ssize_t>(max_steps());
        py::array_t<int> out(std::vector<py::ssize_t>{B, T, Game::MAX_ACTION_TOKENS});
        auto buf = out.request();
        auto* ptr = static_cast<int*>(buf.ptr);
        std::fill(ptr, ptr + B * T * Game::MAX_ACTION_TOKENS, PAD_TOKEN);

        for (py::ssize_t b = 0; b < B; ++b) {
            const auto& rows = sequences[b].action_taken_tokens_data();
            for (py::ssize_t t = 0; t < static_cast<py::ssize_t>(rows.size()); ++t) {
                const auto& tokens = rows[static_cast<size_t>(t)];
                for (py::ssize_t k = 0; k < static_cast<py::ssize_t>(tokens.size()) && k < Game::MAX_ACTION_TOKENS; ++k) {
                    const py::ssize_t offset = ((b * T) + t) * Game::MAX_ACTION_TOKENS + k;
                    ptr[offset] = tokens[static_cast<size_t>(k)];
                }
            }
        }
        return out;
    }

    [[nodiscard]] py::array_t<int> get_player_ids() const {
        auto B = static_cast<py::ssize_t>(sequences.size());
        auto T = static_cast<py::ssize_t>(max_steps());
        py::array_t<int> out(std::vector<py::ssize_t>{B, T});
        auto* ptr = static_cast<int*>(out.request().ptr);
        std::fill(ptr, ptr + B * T, PLAYER_CHANCE_TRAINER);

        for (py::ssize_t b = 0; b < B; ++b) {
            const auto& rows = sequences[b].player_ids_data();
            for (py::ssize_t t = 0; t < static_cast<py::ssize_t>(rows.size()); ++t) {
                ptr[b * T + t] = rows[static_cast<size_t>(t)];
            }
        }
        return out;
    }

    [[nodiscard]] AvailableActionsBatch get_last_available_actions() const {
        if (boundary_ != HistoryBoundary::ENDS_WITH_STATE) {
            throw std::runtime_error("HistoryBatch.last_available_actions is only available for ENDS_WITH_STATE histories.");
        }

        auto B = static_cast<py::ssize_t>(sequences.size());
        py::array_t<int> out_tokens(std::vector<py::ssize_t>{B, Game::MAX_AVAILABLE_ACTIONS, Game::MAX_ACTION_TOKENS});
        py::array_t<float> out_weights(std::vector<py::ssize_t>{B, Game::MAX_AVAILABLE_ACTIONS});
        py::array_t<std::uint8_t> out_is_commit(std::vector<py::ssize_t>{B, Game::MAX_AVAILABLE_ACTIONS});
        py::array_t<int> out_player_ids(std::vector<py::ssize_t>{B});

        auto* ptr_tokens = static_cast<int*>(out_tokens.request().ptr);
        auto* ptr_weights = static_cast<float*>(out_weights.request().ptr);
        auto* ptr_is_commit = static_cast<std::uint8_t*>(out_is_commit.request().ptr);
        auto* ptr_player_ids = static_cast<int*>(out_player_ids.request().ptr);

        std::fill(ptr_tokens, ptr_tokens + B * Game::MAX_AVAILABLE_ACTIONS * Game::MAX_ACTION_TOKENS, PAD_TOKEN);
        std::fill(ptr_weights, ptr_weights + B * Game::MAX_AVAILABLE_ACTIONS, 0.f);
        std::fill(ptr_is_commit, ptr_is_commit + B * Game::MAX_AVAILABLE_ACTIONS, static_cast<std::uint8_t>(0));

        for (py::ssize_t b = 0; b < B; ++b) {
            const auto& seq = sequences[static_cast<size_t>(b)];
            ptr_player_ids[b] = seq.player_ids_data().back();
            const auto& actions = seq.last_available_actions_data();
            for (py::ssize_t a = 0; a < static_cast<py::ssize_t>(actions.size()) && a < Game::MAX_AVAILABLE_ACTIONS; ++a) {
                const auto& proto = actions[static_cast<size_t>(a)];
                ptr_weights[b * Game::MAX_AVAILABLE_ACTIONS + a] = proto.weight;
                ptr_is_commit[b * Game::MAX_AVAILABLE_ACTIONS + a] = static_cast<std::uint8_t>(proto.is_commit);
                for (py::ssize_t k = 0; k < static_cast<py::ssize_t>(proto.tokens.size()) && k < Game::MAX_ACTION_TOKENS; ++k) {
                    const py::ssize_t offset = (b * Game::MAX_AVAILABLE_ACTIONS + a) * Game::MAX_ACTION_TOKENS + k;
                    ptr_tokens[offset] = proto.tokens[static_cast<size_t>(k)];
                }
            }
        }

        return {out_tokens, out_weights, out_is_commit, out_player_ids};
    }

    [[nodiscard]] std::vector<std::vector<std::string>> get_last_available_action_strs() const {
        if (boundary_ != HistoryBoundary::ENDS_WITH_STATE) {
            throw std::runtime_error("HistoryBatch.last_available_action_strs is only available for ENDS_WITH_STATE histories.");
        }

        std::vector<std::vector<std::string>> out;
        out.reserve(sequences.size());
        for (size_t b = 0; b < sequences.size(); ++b) {
            try {
                out.push_back(sequences[b].last_available_action_strs_data());
            } catch (const std::runtime_error& e) {
                throw std::runtime_error(
                    "HistoryBatch.last_available_action_strs failed at batch index " + std::to_string(static_cast<int>(b)) + ": " + e.what()
                );
            }
        }
        return out;
    }

    [[nodiscard]] const History<Game>& operator[](py::ssize_t index) const {
        if (index < 0 || index >= static_cast<py::ssize_t>(sequences.size())) throw std::out_of_range("HistoryBatch index out of range");
        return sequences[index];
    }
};

template<typename Game>
inline std::ostream& operator<<(std::ostream& os, const Action<Game>& action) {
    os << "Action(tokens=[";
    for (size_t i = 0; i < action.tokens.size(); ++i) {
        os << (i > 0 ? ", " : "") << Game::Token::decode_token(action.tokens[i]);
    }
    os << "])";
    return os;
}

template<typename Game>
inline std::ostream& operator<<(std::ostream& os, const ActionProto<Game>& action_proto) {
    os << "ActionProto(tokens=[";
    for (size_t i = 0; i < action_proto.tokens.size(); ++i) {
        os << (i > 0 ? ", " : "") << Game::Token::decode_token(action_proto.tokens[i]);
    }
    os << "]";
    if (action_proto.weight != 1.0f) { os << ", weight=" << action_proto.weight; }
    if (!action_proto.is_commit) { os << ", is_commit=false"; }
    os << ")";
    return os;
}

template<typename Game>
inline std::ostream& operator<<(std::ostream& os, const Observation<Game>& observations) {
    os << "Observation(tokens=[";
    for (int i = 0; i < Game::NUM_PLAYERS; ++i) {
        os << "[";
        for (int j = 0; j < observations.tokens[i].size(); ++j)
            os << (j > 0 ? ", " : "") << Game::Token::decode_token(observations.tokens[i][j]);
        os << "]" << (i < Game::NUM_PLAYERS - 1 ? ", " : "");
    }
    os << "], channels=[";
    for (int i = 0; i < Game::NUM_PLAYERS; ++i) {
        os << "[";
        bool first = true;
        for (size_t j = 0; j < observations.channels[i].size(); ++j) {
            if (observations.channels[i][j] == 0) continue;
            os << (first ? "" : ", ");
            first = false;
            os << Game::Channels::decode_channel(static_cast<int>(j));
        }
        os << "]" << (i < Game::NUM_PLAYERS - 1 ? ", " : "");
    }
    os << "]";
    os << ")";
    return os;
}

template<typename Game>
inline std::ostream& operator<<(std::ostream& os, const StepResult<Game>& env_step) {
    std::string indent = "  ";
    os << "StepResult(\n";
    os << indent << "rewards=[";
    for (int i = 0; i < Game::NUM_PLAYERS; ++i) {
        os << (i > 0 ? ", " : "") << env_step.rewards[i];
    }
    os << "],\n"; // End rewards line with comma and newline
    os << indent << "is_terminal=" << (env_step.is_terminal ? "true" : "false") << ",\n";
    os << indent << "player=" << env_step.player_id << ",\n";
    os << indent << "observations=" << env_step.observations << ",\n";
    os << indent << "available_actions=[\n";
    if (!env_step.available_actions.empty()) {
        for (size_t i = 0; i < env_step.available_actions.size(); ++i) {
            os << indent << indent << env_step.available_actions[i];
            if (i < env_step.available_actions.size() - 1) { os << ","; }
            os << "\n";
        }
    } else {
        os << indent << indent << "(empty)\n";
    }
    os << indent << "]\n";
    os << ")";
    return os;
}

} // namespace nanopokerlib
