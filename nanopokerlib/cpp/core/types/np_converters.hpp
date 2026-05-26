#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "action.hpp"
#include "constants.hpp"

namespace py = pybind11;

namespace nanopokerlib {

template<typename Game>
class History;

template<typename Game>
std::vector<Action<Game>> actions_from_np(const py::array_t<int, py::array::c_style | py::array::forcecast>& tokens);

template<typename Game>
Action<Game> action_from_np(const py::array_t<int, py::array::c_style | py::array::forcecast>& tokens);

template<typename Game>
History<Game> history_from_np(
    int boundary,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& obs_tokens,
    const py::array_t<ChannelValue, py::array::c_style | py::array::forcecast>& obs_channels,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& action_taken_tokens,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& player_ids,
    const py::object& last_avail_tokens = py::none(),
    const py::object& last_avail_weights = py::none(),
    const py::object& last_avail_is_commit = py::none()
);

template<typename Game>
std::vector<Action<Game>> actions_from_np(const py::array_t<int, py::array::c_style | py::array::forcecast>& tokens) {
    auto info_tokens = tokens.request();
    if (info_tokens.ndim != 2) {
        throw std::runtime_error(
            "Input 'tokens' must be a 2D NumPy array (N x MAX_ACTION_TOKENS). Got " + std::to_string(info_tokens.ndim) +
            " dimensions."
        );
    }

    const py::ssize_t num_actions_tokens = info_tokens.shape[0];
    const py::ssize_t max_tokens_dim = info_tokens.shape[1];
    if (max_tokens_dim != Game::MAX_ACTION_TOKENS) {
        throw std::runtime_error(
            "Dimension 1 of 'tokens' array must equal Game::MAX_ACTION_TOKENS (" + std::to_string(Game::MAX_ACTION_TOKENS) +
            "). Got " + std::to_string(max_tokens_dim) + "."
        );
    }

    std::vector<Action<Game>> action_batch;
    action_batch.reserve(static_cast<size_t>(num_actions_tokens));
    auto* tokens_ptr = static_cast<int*>(info_tokens.ptr);
    const auto num_actions = static_cast<size_t>(num_actions_tokens);

    for (size_t i = 0; i < num_actions; ++i) {
        Action<Game> action;
        int* current_token_start = tokens_ptr + i * Game::MAX_ACTION_TOKENS;
        for (int j = 0; j < Game::MAX_ACTION_TOKENS; ++j) {
            const int token_val = *(current_token_start + j);
            if (token_val != PAD_TOKEN) {
                action.tokens.push_back(token_val);
            }
        }
        action_batch.push_back(std::move(action));
    }
    return action_batch;
}

template<typename Game>
Action<Game> action_from_np(const py::array_t<int, py::array::c_style | py::array::forcecast>& tokens) {
    auto info_tokens = tokens.request();
    if (info_tokens.ndim != 1) {
        throw std::runtime_error(
            "Input 'tokens' must be a 1D NumPy array (MAX_ACTION_TOKENS). Got " + std::to_string(info_tokens.ndim) +
            " dimensions."
        );
    }

    const py::ssize_t width = info_tokens.shape[0];
    if (width > Game::MAX_ACTION_TOKENS) {
        throw std::runtime_error(
            "Dimension 0 of 'tokens' array must be <= Game::MAX_ACTION_TOKENS (" + std::to_string(Game::MAX_ACTION_TOKENS) +
            "). Got " + std::to_string(width) + "."
        );
    }

    Action<Game> action;
    const auto* tokens_ptr = static_cast<const int*>(info_tokens.ptr);
    action.tokens.reserve(static_cast<size_t>(width));
    for (py::ssize_t j = 0; j < width; ++j) {
        const int token_val = tokens_ptr[j];
        if (token_val != PAD_TOKEN) {
            action.tokens.push_back(token_val);
        }
    }
    return action;
}

} // namespace nanopokerlib
