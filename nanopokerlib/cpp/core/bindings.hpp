#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <sstream>
#include <string>
#include <cctype>
#include <utility>
#include <type_traits>

#include "types.hpp" // Your existing core types
#include "endpoint_interface.hpp"

namespace py = pybind11;

namespace nanopokerlib {
namespace detail {
inline std::string lowercase(const std::string& name) {
    std::string lower = name;
    for (char& ch : lower) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return lower;
}

template<typename Value>
void bind_constant(py::handle module, const char* name, Value&& value) {
    py::object obj;
    using ValueType = std::remove_reference_t<Value>;
    if constexpr (std::is_base_of_v<py::detail::pyobject_tag, std::remove_cv_t<ValueType>>) {
        obj = py::reinterpret_borrow<py::object>(std::forward<Value>(value));
    } else {
        obj = py::cast(std::forward<Value>(value));
    }
    module.attr(name) = obj;
    module.attr(lowercase(name).c_str()) = obj;
}

} // namespace detail

template<typename Game>
void bind_common_game_structs(py::module_& m_submodule) {
    static_assert(HasParseInfosetStrEndpoint<Game>, "Game::Endpoint must define parse_infoset_str(infoset_str, pending_action).");
    detail::bind_constant(m_submodule, "HAS_PARSE_INFOSET_STR", EndpointCapabilities<Game>::has_parse_infoset_str);
    detail::bind_constant(m_submodule, "HAS_FORMAT_INFOSET_STR", EndpointCapabilities<Game>::has_format_infoset_str);
    detail::bind_constant(m_submodule, "HAS_PARSE_MATCH_STR", EndpointCapabilities<Game>::has_parse_match_str);
    detail::bind_constant(m_submodule, "HAS_FORMAT_MATCH_STR", EndpointCapabilities<Game>::has_format_match_str);
    detail::bind_constant(m_submodule, "HAS_EXPAND_BELIEF_PREFIX_STR", EndpointCapabilities<Game>::has_expand_belief_prefix_str);
    detail::bind_constant(m_submodule, "HAS_DETERMINIZE_HISTORY_STR", EndpointCapabilities<Game>::has_determinize_history_str);
    detail::bind_constant(m_submodule, "HAS_INFOSET_STR_FROM_HISTORY_STR", EndpointCapabilities<Game>::has_infoset_str_from_history_str);

    // Action<Game>
    py::class_<Action<Game>>(m_submodule, "Action", "Represents a game action.")
        .def(py::init<>())
        .def(
            py::init([](const std::vector<int>& tokens) {
                Action<Game> a;
                a.tokens = tokens;
                return a;
            }),
            py::arg("tokens")
        )
        .def_readonly("tokens", &Action<Game>::tokens)
        .def("__str__", [](const Action<Game>& a) {
            std::ostringstream oss;
            oss << a; // Leverages the templated operator<< from core/types.hpp
            return oss.str();
        });

    // ActionProto<Game>
    py::class_<ActionProto<Game>>(m_submodule, "ActionProto", "Describes an available action type.")
        .def(py::init<>())
        .def_readonly("tokens", &ActionProto<Game>::tokens)
        .def_readonly("weight", &ActionProto<Game>::weight) // 'weight' as per core/types.hpp
        .def_readonly("is_commit", &ActionProto<Game>::is_commit, "Whether choosing this action commits a player decision.")
        .def("__str__", [](const ActionProto<Game>& action_proto) {
            std::ostringstream oss;
            oss << action_proto; // Leverages the templated operator<< from core/types.hpp
            return oss.str();
        });

    // Observation<Game>
    py::class_<Observation<Game>>(m_submodule, "Observation", "Observation for each player.")
        .def(py::init<>())
        .def_readonly("tokens", &Observation<Game>::tokens)
        .def_readonly("channels", &Observation<Game>::channels)
        .def("__str__", [](const Observation<Game>& observations) {
            std::ostringstream oss;
            oss << observations; // Leverages the templated operator<< from core/types.hpp
            return oss.str();
        });

    // StepResult<Game>
    py::class_<StepResult<Game>>(m_submodule, "StepResult", "Result of a single environment step.")
        .def(py::init<>())
        .def_readonly("rewards", &StepResult<Game>::rewards)
        .def_readonly("observations", &StepResult<Game>::observations)
        .def_readonly("is_terminal", &StepResult<Game>::is_terminal)
        .def_readonly("player_id", &StepResult<Game>::player_id)
        .def_readonly("available_actions", &StepResult<Game>::available_actions)
        .def("__str__", [](const StepResult<Game>& env_step) {
            std::ostringstream oss;
            oss << env_step; // Leverages the templated operator<< from core/types.hpp
            return oss.str();
        });

    // History of StepResult<Game>
    py::class_<History<Game>>(m_submodule, "History", "Sequential StepResult history.")
        .def("__len__", &History<Game>::size)
        .def("size", &History<Game>::size)
        .def("__getitem__", &History<Game>::operator[], py::arg("index"))
        .def_property_readonly("num_steps", &History<Game>::num_steps)
        .def_property_readonly("boundary", &History<Game>::get_boundary_property, "History boundary encoded as HistoryBoundary enum value.")
        .def_property_readonly("observations", &History<Game>::get_observations, "Observation tokens for the entire history.")
        .def_property_readonly("action_taken_tokens", &History<Game>::get_action_taken_tokens, "Taken-action token rows for each step.")
        .def_property_readonly("player_ids", &History<Game>::get_player_ids, "Per-step player ids aligned with the observation rows.")
        .def_property_readonly(
            "last_available_actions", &History<Game>::get_last_available_actions,
            "Final-step available actions as padded arrays (state-ended boundary only)."
        )
        .def_property_readonly(
            "last_available_action_strs", &History<Game>::get_last_available_action_strs,
            "Final-step semantic action strings aligned with last_available_actions (state-ended boundary only)."
        );

    py::class_<HistoryBatch<Game>>(m_submodule, "HistoryBatch", "Batch of infoset histories.")
        .def(py::init<std::vector<History<Game>>>())
        .def("size", &HistoryBatch<Game>::size)
        .def("max_steps", &HistoryBatch<Game>::max_steps)
        .def_property_readonly("boundary", &HistoryBatch<Game>::get_boundary)
        .def_property_readonly("num_steps", &HistoryBatch<Game>::get_num_steps)
        .def_property_readonly("observations", &HistoryBatch<Game>::get_observations)
        .def_property_readonly("action_taken_tokens", &HistoryBatch<Game>::get_action_taken_tokens)
        .def_property_readonly("player_ids", &HistoryBatch<Game>::get_player_ids)
        .def_property_readonly("last_available_actions", &HistoryBatch<Game>::get_last_available_actions)
        .def_property_readonly("last_available_action_strs", &HistoryBatch<Game>::get_last_available_action_strs)
        .def("__getitem__", &HistoryBatch<Game>::operator[], py::arg("index"), py::return_value_policy::reference_internal);

    // NumPy related helpers that are templated on Game
    m_submodule.def(
        "actions_from_np", &actions_from_np<Game>, py::arg("tokens"),
        "Converts NumPy arrays of action tokens into a std::vector<Action<Game>>.\n"
        "Requires shape [N, MAX_ACTION_TOKENS]. Inputs are canonicalized to C-contiguous int32."
    );
    m_submodule.def(
        "action_from_np", &action_from_np<Game>, py::arg("tokens"),
        "Converts one action-token NumPy row into Action<Game>.\n"
        "Requires shape [L] with L <= MAX_ACTION_TOKENS. Inputs are canonicalized to C-contiguous int32."
    );
    m_submodule.def(
        "history_from_np", &history_from_np<Game>, py::arg("boundary"), py::arg("obs_tokens"), py::arg("obs_channels"),
        py::arg("action_taken_tokens"), py::arg("player_ids"), py::arg("last_avail_tokens") = py::none(),
        py::arg("last_avail_weights") = py::none(), py::arg("last_avail_is_commit") = py::none(),
        "Build an History from per-step rollout tensors.\n"
        "Expected shapes:\n"
        "  boundary: HistoryBoundary enum value (ENDS_WITH_STATE or ENDS_WITH_ACTION)\n"
        "  obs_tokens: [T, NUM_PLAYERS, MAX_OBSERVATION_TOKENS]\n"
        "  obs_channels: [T, NUM_PLAYERS, NUM_CHANNELS]\n"
        "  action_taken_tokens: [T, MAX_ACTION_TOKENS]\n"
        "  player_ids: [T]\n"
        "  last_avail_tokens: [A, MAX_ACTION_TOKENS] (state-ended boundary only)\n"
        "  last_avail_weights: [A] (state-ended boundary only)\n"
        "  last_avail_is_commit: [A] (state-ended boundary only)\n"
        "Expected dtypes/layout:\n"
        "  canonicalized to int32: obs_tokens, action_taken_tokens, player_ids, last_avail_tokens\n"
        "  canonicalized to uint8: obs_channels, last_avail_is_commit\n"
        "  canonicalized to float32: last_avail_weights\n"
        "  canonicalized to C-contiguous layout\n"
        "Boundary invariants:\n"
        "  ENDS_WITH_STATE: action_taken_tokens[-1] must be PAD-only and last_* metadata is required.\n"
        "  ENDS_WITH_ACTION: last_ava_* metadata is forbidden."
    );

    // Environment<Game>
    py::class_<Game>(m_submodule, "Environment", "Represents a single game environment.")
        .def(py::init<>(), "Setup a single game state.")
        .def("reset", &Game::reset, py::arg("seed") = 0, "Resets the game state and returns the initial StepResult.")
        .def("step", &Game::step, py::arg("action"), "Applies an action and returns the resulting StepResult.")
        .def("__str__", [](const Game& state) {
            std::ostringstream oss;
            oss << state;
            return oss.str();
        });

    // EnvBatch<Game>
    py::class_<EnvBatch<Game>>(m_submodule, "EnvBatch", "Manages a batch of environments.")
        .def(py::init<int>(), py::arg("num_envs"))
        .def(
            "reset", &EnvBatch<Game>::reset, py::arg("seed") = 0,
            "Resets all environments and returns a batch of initial StepResult objects."
        )
        .def(
            "step", &EnvBatch<Game>::step, py::arg("action_batch"), "Applies a batch of actions and returns a batch of StepResult objects."
        )
        .def(
            "__getitem__", [](EnvBatch<Game>& envs, int index) -> Game& { return envs[index]; }, py::arg("index"),
            py::return_value_policy::reference_internal
        )
        .def("size", &EnvBatch<Game>::size)
        .def("__len__", &EnvBatch<Game>::size)
        .def("__str__", [](const EnvBatch<Game>& env) {
            return "<nanopokerlib.EnvBatch<" + std::string(typeid(Game).name()) + "> size=" + std::to_string(env.size()) + ">";
        });

    py::class_<StepBatch<Game>>(m_submodule, "StepBatch", "Batch of step results collated into NumPy arrays.")
        .def_property_readonly("rewards", &StepBatch<Game>::get_rewards, "Rewards for each environment step.")
        .def_property_readonly("is_terminal", &StepBatch<Game>::get_is_terminal, "Terminal state flags for each environment step.")
        .def_property_readonly("player_ids", &StepBatch<Game>::get_player_ids, "Current player for each environment step.")
        .def_property_readonly("observations", &StepBatch<Game>::get_observations, "Observations for each environment step.")
        .def_property_readonly("available_actions", &StepBatch<Game>::get_available_actions, "Available actions for each environment step.")
        .def("__getitem__", &StepBatch<Game>::operator[], py::arg("index"), py::return_value_policy::reference_internal)
        .def("size", &StepBatch<Game>::size, "Returns the number of environment steps.")
        .def("__len__", &StepBatch<Game>::size)
        .def("__str__", [](const StepBatch<Game>& vec_env_step) {
            return "<nanopokerlib.StepBatch<" + std::string(typeid(Game).name()) + "> size=" + std::to_string(vec_env_step.size()) + ">";
        });
}

// Function to bind truly common (non-templated or one-time) NumPy wrapper structs
// This should be called only once in your main PYBIND11_MODULE.
inline void bind_numpy_wrappers(py::module_& m) {
    if (!hasattr(m, "HistoryBoundary")) {
        py::enum_<HistoryBoundary>(m, "HistoryBoundary", "History boundary semantics.")
            .value("ENDS_WITH_STATE", HistoryBoundary::ENDS_WITH_STATE)
            .value("ENDS_WITH_ACTION", HistoryBoundary::ENDS_WITH_ACTION)
            .export_values();
    }

    // ObservationBatch
    // Check if it's already bound to prevent errors if called multiple times from different game binding initializers.
    // A more robust way is to ensure this function is called only once.
    if (!hasattr(m, "ObservationBatch")) {
        py::class_<ObservationBatch>(m, "ObservationBatch", "Observation data padded into NumPy arrays.")
            .def(py::init<>())
            .def_readwrite("tokens", &ObservationBatch::tokens)
            .def_readwrite("channels", &ObservationBatch::channels);
    }

    // AvailableActionsBatch
    if (!hasattr(m, "AvailableActionsBatch")) {
        py::class_<AvailableActionsBatch>(m, "AvailableActionsBatch", "Available actions data padded into NumPy arrays.")
            .def(py::init<>())
            .def_readwrite("tokens", &AvailableActionsBatch::tokens)
            .def_readwrite("weights", &AvailableActionsBatch::weights)
            .def_readwrite("is_commit", &AvailableActionsBatch::is_commit)
            .def_readwrite("player_ids", &AvailableActionsBatch::player_ids);
    }

    if (!hasattr(m, "AvailableActions")) {
        py::class_<AvailableActions>(m, "AvailableActions", "Final-step available actions padded into NumPy arrays.")
            .def(py::init<>())
            .def_readwrite("tokens", &AvailableActions::tokens)
            .def_readwrite("weights", &AvailableActions::weights)
            .def_readwrite("is_commit", &AvailableActions::is_commit)
            .def_readwrite("player_id", &AvailableActions::player_id);
    }

    if (!hasattr(m, "MatchStateView")) {
        py::class_<MatchStateView>(m, "MatchStateView", "Game-agnostic match-state side stream for a single environment.")
            .def(py::init<>())
            .def_readwrite("rewards", &MatchStateView::rewards)
            .def_readwrite("is_terminal", &MatchStateView::is_terminal)
            .def_readwrite("channels", &MatchStateView::channels);
    }

    if (!hasattr(m, "MatchStateBatch")) {
        py::class_<MatchStateBatch>(m, "MatchStateBatch", "Game-agnostic match-state side stream padded into NumPy arrays.")
            .def(py::init<>())
            .def_readwrite("rewards", &MatchStateBatch::rewards)
            .def_readwrite("is_terminal", &MatchStateBatch::is_terminal)
            .def_readwrite("channels", &MatchStateBatch::channels);
    }
}
} // namespace nanopokerlib
