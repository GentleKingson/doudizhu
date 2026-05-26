#pragma once

#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>

#include "action.hpp"
#include "available_actions_batch.hpp"
#include "constants.hpp"
#include "observation_batch.hpp"
#include "step_result.hpp"

namespace py = pybind11;

namespace nanopokerlib {
template<typename Game>
class StepBatch {
private:
    int n;
    std::vector<StepResult<Game>> env_step_batch;

public:
    explicit StepBatch(std::vector<StepResult<Game>> env_step_batch);

    [[nodiscard]] py::array_t<float> get_rewards() const;
    [[nodiscard]] py::array_t<int> get_is_terminal() const;
    [[nodiscard]] py::array_t<int> get_player_ids() const;
    [[nodiscard]] ObservationBatch get_observations() const;
    [[nodiscard]] AvailableActionsBatch get_available_actions() const;

    StepResult<Game> operator[](int index) const {
        if (index < 0 || index >= env_step_batch.size()) {
            throw std::out_of_range("StepBatch index out of range: " + std::to_string(index) + " (Size: " + std::to_string(n) + ")");
        }
        return env_step_batch[index];
    }

    [[nodiscard]] inline int size() const { return n; }
};

template<typename Game>
class EnvBatch {
private:
    int n;
    std::vector<Game> states;

public:
    explicit EnvBatch(int n);
    ~EnvBatch() = default;
    StepBatch<Game> reset(int seed);
    StepBatch<Game> reset_to_states(std::vector<Game> new_states, std::vector<StepResult<Game>> current_steps);
    StepBatch<Game> step(const std::vector<Action<Game>>& action_batch);

    inline Game& operator[](int index) {
        if (index < 0 || index >= n) {
            throw std::out_of_range("EnvBatch index out of range: " + std::to_string(index) + " (Size: " + std::to_string(n) + ")");
        }
        return states[index];
    }

    inline const Game& operator[](int index) const {
        if (index < 0 || index >= n) {
            throw std::out_of_range("EnvBatch index out of range: " + std::to_string(index) + " (Size: " + std::to_string(n) + ")");
        }
        return states[index];
    }

    [[nodiscard]] inline int size() const { return n; }
};

template<typename Game>
StepBatch<Game>::StepBatch(std::vector<StepResult<Game>> env_step_batch) {
    n = static_cast<int>(env_step_batch.size());
    this->env_step_batch = env_step_batch;
}

template<typename Game>
py::array_t<float> StepBatch<Game>::get_rewards() const {
    auto rewards = py::array_t<float>(std::vector<py::ssize_t>{n, Game::NUM_PLAYERS});
    {
        float* ptr = static_cast<float*>(rewards.request().ptr);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < Game::NUM_PLAYERS; ++j)
                ptr[i * Game::NUM_PLAYERS + j] = env_step_batch[i].rewards[j];
    }
    return rewards;
}

template<typename Game>
py::array_t<int> StepBatch<Game>::get_is_terminal() const {
    auto is_terminal = py::array_t<int>(std::vector<py::ssize_t>{n});
    {
        int* ptr = static_cast<int*>(is_terminal.request().ptr);
        for (int i = 0; i < n; ++i)
            ptr[i] = env_step_batch[i].is_terminal;
    }
    return is_terminal;
}

template<typename Game>
py::array_t<int> StepBatch<Game>::get_player_ids() const {
    auto player = py::array_t<int>(std::vector<py::ssize_t>{n});
    {
        int* ptr = static_cast<int*>(player.request().ptr);
        for (int i = 0; i < n; ++i)
            ptr[i] = env_step_batch[i].player_id;
    }
    return player;
}

template<typename Game>
ObservationBatch StepBatch<Game>::get_observations() const {
    py::array_t<int> obs_tokens(std::vector<py::ssize_t>{n, Game::NUM_PLAYERS, Game::MAX_OBSERVATION_TOKENS});
    py::array_t<ChannelValue> obs_channels(std::vector<py::ssize_t>{n, Game::NUM_PLAYERS, Game::NUM_CHANNELS});
    {
        auto* ptr = static_cast<int*>(obs_tokens.request().ptr);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < Game::NUM_PLAYERS; ++j) {
                if (env_step_batch[i].observations.tokens[j].size() > Game::MAX_OBSERVATION_TOKENS)
                    throw std::runtime_error(
                        "Observation tokens size mismatch: " + std::to_string(env_step_batch[i].observations.tokens[j].size()) + " > " +
                        std::to_string(Game::MAX_OBSERVATION_TOKENS)
                    );
                for (int k = 0; k < Game::MAX_OBSERVATION_TOKENS; ++k)
                    if (k < env_step_batch[i].observations.tokens[j].size())
                        ptr[(i * Game::NUM_PLAYERS + j) * Game::MAX_OBSERVATION_TOKENS + k] = env_step_batch[i].observations.tokens[j][k];
                    else
                        ptr[(i * Game::NUM_PLAYERS + j) * Game::MAX_OBSERVATION_TOKENS + k] = PAD_TOKEN; // Padding
            }
    }
    {
        auto* ptr = static_cast<ChannelValue*>(obs_channels.request().ptr);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < Game::NUM_PLAYERS; ++j) {
                const auto& channels = env_step_batch[i].observations.channels[j];
                if (channels.size() > Game::NUM_CHANNELS)
                    throw std::runtime_error(
                        "Observation channels size mismatch: " + std::to_string(channels.size()) + " > " +
                        std::to_string(Game::NUM_CHANNELS)
                    );
                for (int k = 0; k < Game::NUM_CHANNELS; ++k)
                    if (k < static_cast<int>(channels.size()))
                        ptr[(i * Game::NUM_PLAYERS + j) * Game::NUM_CHANNELS + k] = channels[k];
                    else
                        ptr[(i * Game::NUM_PLAYERS + j) * Game::NUM_CHANNELS + k] = ChannelValue{0};
            }
    }
    return {obs_tokens, obs_channels};
}

template<typename Game>
AvailableActionsBatch StepBatch<Game>::get_available_actions() const {
    py::array_t<int> avail_tokens(std::vector<py::ssize_t>{n, Game::MAX_AVAILABLE_ACTIONS, Game::MAX_ACTION_TOKENS});
    {
        auto* ptr = static_cast<int*>(avail_tokens.request().ptr);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < Game::MAX_AVAILABLE_ACTIONS; ++j) {
                if (j < env_step_batch[i].available_actions.size() &&
                    env_step_batch[i].available_actions[j].tokens.size() > Game::MAX_ACTION_TOKENS)
                    throw std::runtime_error(
                        "Available action tokens size mismatch: " + std::to_string(env_step_batch[i].available_actions[j].tokens.size()) +
                        " > " + std::to_string(Game::MAX_AVAILABLE_ACTIONS)
                    );
                for (int k = 0; k < Game::MAX_ACTION_TOKENS; ++k)
                    if (j < env_step_batch[i].available_actions.size() && k < env_step_batch[i].available_actions[j].tokens.size())
                        ptr[(i * Game::MAX_AVAILABLE_ACTIONS + j) * Game::MAX_ACTION_TOKENS + k] =
                            env_step_batch[i].available_actions[j].tokens[k];
                    else
                        ptr[(i * Game::MAX_AVAILABLE_ACTIONS + j) * Game::MAX_ACTION_TOKENS + k] = PAD_TOKEN; // Padding
            }
    }
    py::array_t<float> avail_weights(std::vector<py::ssize_t>{n, Game::MAX_AVAILABLE_ACTIONS});
    {
        auto* ptr = static_cast<float*>(avail_weights.request().ptr);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < Game::MAX_AVAILABLE_ACTIONS; ++j)
                if (j < env_step_batch[i].available_actions.size())
                    ptr[i * Game::MAX_AVAILABLE_ACTIONS + j] = env_step_batch[i].available_actions[j].weight;
                else
                    ptr[i * Game::MAX_AVAILABLE_ACTIONS + j] = 0.f; // Padding
    }
    py::array_t<std::uint8_t> avail_is_commit(std::vector<py::ssize_t>{n, Game::MAX_AVAILABLE_ACTIONS});
    {
        auto* ptr = static_cast<std::uint8_t*>(avail_is_commit.request().ptr);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < Game::MAX_AVAILABLE_ACTIONS; ++j)
                if (env_step_batch[i].player_id >= 0 && j < env_step_batch[i].available_actions.size())
                    ptr[i * Game::MAX_AVAILABLE_ACTIONS + j] = static_cast<std::uint8_t>(env_step_batch[i].available_actions[j].is_commit);
                else
                    ptr[i * Game::MAX_AVAILABLE_ACTIONS + j] = static_cast<std::uint8_t>(0);
    }
    py::array_t<int> player_ids(std::vector<py::ssize_t>{n});
    {
        auto* ptr = static_cast<int*>(player_ids.request().ptr);
        for (int i = 0; i < n; ++i) {
            ptr[i] = env_step_batch[i].player_id;
        }
    }
    return {avail_tokens, avail_weights, avail_is_commit, player_ids};
}

template<typename Game>
EnvBatch<Game>::EnvBatch(int n) : states(n), n(n) {
    for (int i = 0; i < n; ++i) {
        states[i] = Game();
    }
}

template<typename Game>
StepBatch<Game> EnvBatch<Game>::reset(int seed) {
    std::vector<StepResult<Game>> batch;
    batch.resize(n);

    // Use a deterministic RNG seeded with the provided seed to
    // generate unique seeds for each environment instance.
    std::mt19937 seeder(static_cast<std::mt19937::result_type>(seed));
    for (int i = 0; i < n; ++i) {
        // Generate a unique seed for each environment using the RNG
        int env_seed = static_cast<int>(seeder());
        batch[i] = states[i].reset(env_seed);
    }
    return StepBatch<Game>(batch);
}

template<typename Game>
StepBatch<Game> EnvBatch<Game>::step(const std::vector<Action<Game>>& action_batch) {
    std::vector<StepResult<Game>> batch;
    batch.resize(n);
    for (int i = 0; i < n; ++i) {
        batch[i] = states[i].step(action_batch[i]);
    }
    return StepBatch<Game>(batch);
}

template<typename Game>
StepBatch<Game> EnvBatch<Game>::reset_to_states(std::vector<Game> new_states, std::vector<StepResult<Game>> current_steps) {
    if (static_cast<int>(new_states.size()) != n) {
        throw std::runtime_error(
            "reset_to_states: new_states size mismatch: " + std::to_string(new_states.size()) + " != " + std::to_string(n)
        );
    }
    if (static_cast<int>(current_steps.size()) != n) {
        throw std::runtime_error(
            "reset_to_states: current_steps size mismatch: " + std::to_string(current_steps.size()) + " != " + std::to_string(n)
        );
    }
    states = std::move(new_states);
    return StepBatch<Game>(std::move(current_steps));
}

} // namespace nanopokerlib
