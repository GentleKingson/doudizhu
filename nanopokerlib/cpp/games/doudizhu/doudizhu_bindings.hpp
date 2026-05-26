#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>

#include "core/bindings.hpp"
#include "doudizhu_core.hpp"
#include "doudizhu_endpoint.hpp"

namespace py = pybind11;

namespace nanopokerlib {
template<typename Game>
inline void init_doudizhu_bindings(py::module_& m_doudizhu, const char* docstring = nullptr) {
    using Token = typename Game::Token;
    using Combo = typename Game::Combo;
    using Channels = typename Game::Channels;
    using namespace detail;

    if (docstring != nullptr) { m_doudizhu.doc() = docstring; }

    py::enum_<typename Game::Phase>(m_doudizhu, "Phase")
        .value("DEALING", Game::Phase::DEALING)
        .value("PLAYING", Game::Phase::PLAYING)
        .export_values();

    py::enum_<typename Combo::Type>(m_doudizhu, "ComboType")
        .value("PASS", Combo::Type::PASS)
        .value("ROCKET", Combo::Type::ROCKET)
        .value("SINGLE", Combo::Type::SINGLE)
        .value("PAIR", Combo::Type::PAIR)
        .value("TRIPLE", Combo::Type::TRIPLE)
        .value("TRIPLE_SINGLE", Combo::Type::TRIPLE_SINGLE)
        .value("TRIPLE_PAIR", Combo::Type::TRIPLE_PAIR)
        .value("STRAIGHT", Combo::Type::STRAIGHT)
        .value("STRAIGHT_PAIR", Combo::Type::STRAIGHT_PAIR)
        .value("STRAIGHT_TRIPLE", Combo::Type::STRAIGHT_TRIPLE)
        .value("PLANE_SINGLE", Combo::Type::PLANE_SINGLE)
        .value("PLANE_PAIR", Combo::Type::PLANE_PAIR)
        .value("FOUR_TWO_SINGLE", Combo::Type::FOUR_TWO_SINGLE)
        .value("FOUR_TWO_PAIR", Combo::Type::FOUR_TWO_PAIR)
        .value("BOMB", Combo::Type::BOMB)
        .export_values();

    bind_constant(m_doudizhu, "NUM_PLAYERS", Game::NUM_PLAYERS);
    bind_constant(m_doudizhu, "NUM_RANKS", Game::NUM_RANKS);
    bind_constant(m_doudizhu, "MAX_SOLO_KICKERS", Game::MAX_SOLO_KICKERS);
    bind_constant(m_doudizhu, "MAX_PAIR_KICKERS", Game::MAX_PAIR_KICKERS);
    bind_constant(m_doudizhu, "NUM_SEQ_FROM", Game::NUM_SEQ_FROM);
    bind_constant(m_doudizhu, "NUM_SEQ_TO", Game::NUM_SEQ_TO);
    bind_constant(m_doudizhu, "NUM_PAIR_KICKER_RANKS", Game::NUM_PAIR_KICKER_RANKS);
    bind_constant(m_doudizhu, "MAX_OBSERVATION_TOKENS", Game::MAX_OBSERVATION_TOKENS);
    bind_constant(m_doudizhu, "MAX_ACTION_TOKENS", Game::MAX_ACTION_TOKENS);
    bind_constant(m_doudizhu, "MAX_AVAILABLE_ACTIONS", Game::MAX_AVAILABLE_ACTIONS);
    bind_constant(m_doudizhu, "MAX_BELIEF_CANDIDATES", Game::MAX_BELIEF_CANDIDATES);
    bind_constant(m_doudizhu, "MAX_BELIEF_TOKENS", Game::MAX_BELIEF_TOKENS);
    bind_constant(m_doudizhu, "MAX_BELIEF_ROWS", Game::MAX_BELIEF_ROWS);
    bind_constant(m_doudizhu, "NUM_PUBLIC_LANDLORD_CARDS", Game::NUM_PUBLIC_LANDLORD_CARDS);
    bind_constant(m_doudizhu, "BASE_HAND_CARDS", Game::BASE_HAND_CARDS);
    bind_constant(m_doudizhu, "NUM_CHANNELS", Game::NUM_CHANNELS);
    bind_constant(m_doudizhu, "MAX_GAME_HORIZON", Game::MAX_GAME_HORIZON);
    bind_constant(m_doudizhu, "GAME_IDENTIFIER", Game::GAME_IDENTIFIER);
    bind_constant(m_doudizhu, "MATCH_CHANNELS", 0);
    bind_constant(m_doudizhu, "PAD_TOKEN", Token::PAD_TOKEN);
    bind_constant(m_doudizhu, "INIT_TOKEN", Token::INIT_TOKEN);
    bind_constant(m_doudizhu, "NUM_TOKENS", Token::NUM_TOKENS);

    auto m_token = m_doudizhu.def_submodule("token", "Encoding helpers for DouDiZhu");
    bind_constant(m_token, "PAD_TOKEN", Token::PAD_TOKEN);
    bind_constant(m_token, "INIT_TOKEN", Token::INIT_TOKEN);
    bind_constant(m_token, "PLAYER_OFFSET", Token::PLAYER_OFFSET);
    bind_constant(m_token, "ACTION_CLASS_OFFSET", Token::ACTION_CLASS_OFFSET);
    bind_constant(m_token, "RANK_OFFSET", Token::RANK_OFFSET);
    bind_constant(m_token, "SEQ_FROM_OFFSET", Token::SEQ_FROM_OFFSET);
    bind_constant(m_token, "SEQ_TO_OFFSET", Token::SEQ_TO_OFFSET);
    bind_constant(m_token, "DEPRECATED_SEQ_TO_TOKEN", Token::DEPRECATED_SEQ_TO_TOKEN);
    bind_constant(m_token, "KICKER_TYPE_OFFSET", Token::KICKER_TYPE_OFFSET);
    bind_constant(m_token, "SOLO_KICKER_OFFSET", Token::SOLO_KICKER_OFFSET);
    bind_constant(m_token, "PAIR_KICKER_OFFSET", Token::PAIR_KICKER_OFFSET);
    bind_constant(m_token, "PASS_TOKEN", Token::PASS_TOKEN);
    bind_constant(m_token, "HOLD_TOKEN", Token::HOLD_TOKEN);
    bind_constant(m_token, "NUM_TOKENS", Token::NUM_TOKENS);

    py::enum_<typename Token::ActionClass>(m_token, "ActionClass")
        .value("DISCARD_SOLO", Token::ActionClass::DISCARD_SOLO)
        .value("DISCARD_PAIR", Token::ActionClass::DISCARD_PAIR)
        .value("DISCARD_TRIO", Token::ActionClass::DISCARD_TRIO)
        .value("DISCARD_QUAD", Token::ActionClass::DISCARD_QUAD)
        .value("DISCARD_STRAIGHT", Token::ActionClass::DISCARD_STRAIGHT)
        .value("DISCARD_PAIR_SISTERS", Token::ActionClass::DISCARD_PAIR_SISTERS)
        .value("DISCARD_TRIO_CHAIN", Token::ActionClass::DISCARD_TRIO_CHAIN)
        .value("DISCARD_ROCKET", Token::ActionClass::DISCARD_ROCKET)
        .export_values();

    m_token.def("encode_player", &Token::encode_player, py::arg("player"));
    m_token.def("encode_action_class", &Token::encode_action_class, py::arg("action_class"));
    m_token.def("encode_rank", &Token::encode_rank, py::arg("rank"));
    m_token.def("encode_seq_from", &Token::encode_seq_from, py::arg("rank"));
    m_token.def("encode_seq_to", &Token::encode_seq_to, py::arg("rank"));
    m_token.def("encode_no_kicker", &Token::encode_no_kicker);
    m_token.def("encode_solo_kicker_count", &Token::encode_solo_kicker_count, py::arg("count"));
    m_token.def("encode_pair_kicker_count", &Token::encode_pair_kicker_count, py::arg("count"));
    m_token.def("encode_pass", &Token::encode_pass);
    m_token.def("encode_hold", &Token::encode_hold);
    m_token.def("decode_player", &Token::decode_player, py::arg("token"));
    m_token.def("decode_action_class", &Token::decode_action_class, py::arg("token"));
    m_token.def("decode_rank", &Token::decode_rank, py::arg("token"));
    m_token.def("decode_seq_from", &Token::decode_seq_from, py::arg("token"));
    m_token.def("decode_seq_to", &Token::decode_seq_to, py::arg("token"));
    m_token.def("decode_solo_kicker_count", &Token::decode_solo_kicker_count, py::arg("token"));
    m_token.def("decode_pair_kicker_count", &Token::decode_pair_kicker_count, py::arg("token"));
    m_token.def("is_hold", &Token::is_hold, py::arg("token"));
    m_token.def("decode_token", &Token::decode_token, py::arg("token"));
    m_token.def("rank_to_string", &Token::rank_to_string, py::arg("rank"));
    m_token.def("action_class_to_string", &Token::action_class_to_string, py::arg("cls"));

    m_doudizhu.def(
        "parse_infoset_str",
        [](const std::string& infoset_str, const std::string& pending_action) {
            return doudizhu::parse_infoset_str<Game>(infoset_str, pending_action);
        },
        py::arg("infoset_str"), py::arg("pending_action") = std::string(),
        "Reconstruct an History from a compact string such as\n"
        "  'L+34457899TJQQQKKA22BR|L-4|P1-9|P2-PASS'.\n"
        "pending_action is a human-readable action-prefix string (e.g. '333+' or 'PASS')\n"
        "that advances the stage-level action before returning.\n"
        "Use last_available_action_strs (aligned with last_available_actions) for final-step semantic action strings."
    );
    m_doudizhu.def(
        "parse_infoset_strs",
        [](const std::vector<std::string>& infoset_strs, const std::vector<std::string>& pending_actions) {
            return doudizhu::parse_infoset_strs<Game>(infoset_strs, pending_actions);
        },
        py::arg("infoset_strs"), py::arg("pending_actions") = std::vector<std::string>(),
        "Batch version of parse_infoset_str with per-infoset pending_action prefixes."
    );
    m_doudizhu.def(
        "expand_belief_prefix_str",
        [](const std::string& infoset_str, const std::string& belief_str) {
            const auto step = Game::Endpoint::expand_belief_prefix_str(infoset_str, belief_str);
            py::array_t<int> candidate_tokens(
                std::vector<py::ssize_t>{Game::MAX_BELIEF_CANDIDATES, Game::MAX_BELIEF_TOKENS}
            );
            py::array_t<std::uint8_t> candidate_valid(std::vector<py::ssize_t>{Game::MAX_BELIEF_CANDIDATES});

            auto* tok_ptr = static_cast<int*>(candidate_tokens.request().ptr);
            auto* val_ptr = static_cast<std::uint8_t*>(candidate_valid.request().ptr);
            for (int a = 0; a < Game::MAX_BELIEF_CANDIDATES; ++a) {
                val_ptr[a] = step.valid[static_cast<size_t>(a)];
                for (int e = 0; e < Game::MAX_BELIEF_TOKENS; ++e) {
                    tok_ptr[a * Game::MAX_BELIEF_TOKENS + e] = step.tokens[static_cast<size_t>(a)][static_cast<size_t>(e)];
                }
            }

            py::list candidate_belief_strs;
            for (int a = 0; a < Game::MAX_BELIEF_CANDIDATES; ++a) {
                candidate_belief_strs.append(step.belief_strs[static_cast<size_t>(a)]);
            }
            py::dict out;
            out["candidate_tokens"] = candidate_tokens;
            out["candidate_valid"] = candidate_valid;
            out["candidate_belief_strs"] = candidate_belief_strs;
            return out;
        },
        py::arg("infoset_str"), py::arg("belief_str") = std::string(),
        "Autoregressive belief decode step.\n"
        "Returns dict with keys: candidate_tokens [A, E] int32, candidate_valid [A] uint8,\n"
        "candidate_belief_strs list[str] length A.\n"
        "E is MAX_BELIEF_TOKENS.\n"
        "A is MAX_BELIEF_CANDIDATES."
    );
    m_doudizhu.def(
        "determinize_history_str",
        &EndpointFactory<Game>::determinize_history_str,
        py::arg("infoset_str"),
        py::arg("belief_str"),
        "Construct one canonical history string from infoset semantics plus a complete belief_str trajectory."
    );
    m_doudizhu.def(
        "determinize_history_strs",
        &Game::Endpoint::determinize_history_strs,
        py::arg("infoset_strs"),
        py::arg("belief_strs"),
        "Batch version of determinize_history_str."
    );
    m_doudizhu.def(
        "infoset_str_from_history_str",
        &EndpointFactory<Game>::infoset_str_from_history_str,
        py::arg("history_str"),
        "Derive active-player infoset string from canonical history string at a non-terminal decision state."
    );
    m_doudizhu.def(
        "infoset_str_from_history_strs",
        &Game::Endpoint::infoset_str_from_history_strs,
        py::arg("history_strs"),
        "Batch version of infoset_str_from_history_str."
    );
    m_doudizhu.def(
        "expand_belief_prefix_strs",
        [](const std::vector<std::string>& infoset_strs, const std::vector<std::string>& belief_strs) {
            const auto batch = Game::Endpoint::expand_belief_prefix_strs(infoset_strs, belief_strs);
            const int bsz = static_cast<int>(batch.items.size());
            py::array_t<int> candidate_tokens(
                std::vector<py::ssize_t>{bsz, Game::MAX_BELIEF_CANDIDATES, Game::MAX_BELIEF_TOKENS}
            );
            py::array_t<std::uint8_t> candidate_valid(std::vector<py::ssize_t>{bsz, Game::MAX_BELIEF_CANDIDATES});

            auto* tok_ptr = static_cast<int*>(candidate_tokens.request().ptr);
            auto* val_ptr = static_cast<std::uint8_t*>(candidate_valid.request().ptr);
            py::list candidate_belief_str_rows;
            for (int b = 0; b < bsz; ++b) {
                const auto& item = batch.items[static_cast<size_t>(b)];
                py::list row;
                for (int a = 0; a < Game::MAX_BELIEF_CANDIDATES; ++a) {
                    val_ptr[b * Game::MAX_BELIEF_CANDIDATES + a] = item.valid[static_cast<size_t>(a)];
                    for (int e = 0; e < Game::MAX_BELIEF_TOKENS; ++e) {
                        tok_ptr[(b * Game::MAX_BELIEF_CANDIDATES + a) * Game::MAX_BELIEF_TOKENS + e] =
                            item.tokens[static_cast<size_t>(a)][static_cast<size_t>(e)];
                    }
                    row.append(item.belief_strs[static_cast<size_t>(a)]);
                }
                candidate_belief_str_rows.append(row);
            }

            py::dict out;
            out["candidate_tokens"] = candidate_tokens;
            out["candidate_valid"] = candidate_valid;
            out["candidate_belief_strs"] = candidate_belief_str_rows;
            return out;
        },
        py::arg("infoset_strs"), py::arg("belief_strs") = std::vector<std::string>(),
        "Batch version of expand_belief_prefix_str.\n"
        "Returns dict with keys: candidate_tokens [B, A, E] int32, candidate_valid [B, A] uint8,\n"
        "candidate_belief_strs list[list[str]].\n"
        "E is MAX_BELIEF_TOKENS.\n"
        "A is MAX_BELIEF_CANDIDATES."
    );
    m_doudizhu.def(
        "format_history_str",
        &EndpointFactory<Game>::format_history_str,
        py::arg("history"),
        "Serialize an History into canonical omniscient history format:\n"
        "  L+<landlord_private>|P1+<peasant1_private>|P2+<peasant2_private>|L+<landlord_public>|...\n"
        "with shared committed action suffix segments."
    );
    m_doudizhu.def(
        "format_history_strs",
        &EndpointFactory<Game>::format_history_strs,
        py::arg("histories"),
        "Batch version of format_history_str."
    );
    m_doudizhu.def(
        "format_infoset_str",
        &EndpointFactory<Game>::format_infoset_str,
        py::arg("history"),
        "Serialize an History into single-view semantic infoset format (active player's private head plus shared public suffix)."
    );
    m_doudizhu.def(
        "format_infoset_strs",
        &EndpointFactory<Game>::format_infoset_strs,
        py::arg("histories"),
        "Batch version of format_infoset_str."
    );
    m_doudizhu.def(
        "reconstruct_env",
        &EndpointFactory<Game>::reconstruct_env,
        py::arg("history_str"),
        py::arg("pending_action") = "",
        py::arg("seed") = 0,
        "Reconstruct a single DouDiZhu environment state from canonical history string:\n"
        "  L+<landlord_private>|P1+<peasant1_private>|P2+<peasant2_private>|L+<landlord_public>|...\n"
        "Returns tuple[Environment, StepResult]."
    );
    m_doudizhu.def(
        "reconstruct_envs",
        &EndpointFactory<Game>::reconstruct_envs,
        py::arg("history_strs"),
        py::arg("pending_actions") = std::vector<std::string>(),
        py::arg("seed") = 0,
        "Batch version of reconstruct_env with optional per-item pending_action strings."
    );

    const std::string channel_doc = "Channel encoding/decoding helpers for DouDiZhu";
    auto m_channel = m_doudizhu.def_submodule("channel", channel_doc.c_str());
    bind_constant(m_channel, "NUM_CHANNELS", Channels::NUM_CHANNELS);
    bind_constant(m_channel, "NUM_CARD_CHANNELS", Channels::NUM_CARD_CHANNELS);
    bind_constant(m_channel, "NUM_BOMB_COUNT_CHANNELS", Channels::NUM_BOMB_COUNT_CHANNELS);
    bind_constant(m_channel, "SELF_OFFSET", Channels::SELF_OFFSET);
    bind_constant(m_channel, "LANDLORD_PUBLIC_OFFSET", Channels::LANDLORD_PUBLIC_OFFSET);
    bind_constant(m_channel, "DISCARD_OFFSET", Channels::DISCARD_OFFSET);
    bind_constant(m_channel, "DISCARD_STRIDE", Channels::DISCARD_STRIDE);
    bind_constant(m_channel, "BOMB_COUNT_OFFSET", Channels::BOMB_COUNT_OFFSET);
    bind_constant(m_channel, "TURN_OFFSET", Channels::TURN_OFFSET);
    m_channel.def("encode_self_card", &Channels::encode_self_card, py::arg("card_index"));
    m_channel.def("encode_landlord_public_card", &Channels::encode_landlord_public_card, py::arg("card_index"));
    m_channel.def("discard_base", &Channels::discard_base, py::arg("player"));
    m_channel.def("encode_discard_card", &Channels::encode_discard_card, py::arg("player"), py::arg("card_index"));
    m_channel.def("encode_bomb_count", &Channels::encode_bomb_count, py::arg("count"));
    m_channel.def("encode_turn", &Channels::encode_turn, py::arg("player"));
    m_channel.def("decode_channel", &Channels::decode_channel, py::arg("channel"));

    bind_common_game_structs<Game>(m_doudizhu);
    const py::object env_type = m_doudizhu.attr("Environment");
    env_type.attr("get_available_action_strs") = py::cpp_function(
        [](const Game& env) { return env.get_available_action_strs(); },
        py::is_method(env_type),
        "Return canonical semantic action strings aligned 1:1 with the current live available_actions rows.\n"
        "This helper is valid on active player decision states."
    );
    env_type.attr("advance_history_str") = py::cpp_function(
        [](Game& env, const std::string& history_str) { return env.advance_history_str(history_str); },
        py::is_method(env_type),
        py::arg("history_str"),
        "Advance one canonical DouDiZhu history string using the latest live transition.\n"
        "This consumes exactly one reset()/step() update and may return the same string unchanged\n"
        "for non-semantic transitions such as dealing steps or non-commit action-prefix steps."
    );
    env_type.attr("belief_supervision") = py::cpp_function(
        [](const Game& env) {
            py::array_t<int> candidate_tokens(
                std::vector<py::ssize_t>{Game::MAX_BELIEF_ROWS, Game::MAX_BELIEF_CANDIDATES, Game::MAX_BELIEF_TOKENS}
            );
            py::array_t<std::uint8_t> candidate_valid(
                std::vector<py::ssize_t>{Game::MAX_BELIEF_ROWS, Game::MAX_BELIEF_CANDIDATES}
            );
            py::array_t<int> target_index(std::vector<py::ssize_t>{Game::MAX_BELIEF_ROWS});
            auto* tok_ptr = static_cast<int*>(candidate_tokens.request().ptr);
            auto* val_ptr = static_cast<std::uint8_t*>(candidate_valid.request().ptr);
            auto* idx_ptr = static_cast<int*>(target_index.request().ptr);
            int belief_length = 0;

            for (int r = 0; r < Game::MAX_BELIEF_ROWS; ++r) {
                idx_ptr[r] = -1;
                for (int a = 0; a < Game::MAX_BELIEF_CANDIDATES; ++a) {
                    val_ptr[r * Game::MAX_BELIEF_CANDIDATES + a] = 0;
                    for (int e = 0; e < Game::MAX_BELIEF_TOKENS; ++e) {
                        tok_ptr[(r * Game::MAX_BELIEF_CANDIDATES + a) * Game::MAX_BELIEF_TOKENS + e] = Game::Token::PAD_TOKEN;
                    }
                }
            }

            const auto rows = env.get_belief_rows();
            const int viewer = env.player();
            std::array<int, Game::MAX_BELIEF_CANDIDATES> opponent_ids{};
            int opponent_count = 0;
            if (viewer >= 0 && viewer < Game::NUM_PLAYERS) {
                for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
                    if (p == viewer) continue;
                    if (opponent_count >= Game::MAX_BELIEF_CANDIDATES) break;
                    opponent_ids[static_cast<size_t>(opponent_count)] = p;
                    opponent_count += 1;
                }
            }

            for (int r = 0; r < Game::MAX_BELIEF_ROWS; ++r) {
                if (rows[static_cast<size_t>(r)][0] == Game::Token::PAD_TOKEN) continue;
                if (opponent_count != Game::MAX_BELIEF_CANDIDATES) {
                    throw std::runtime_error("belief_supervision expected a valid active viewer with two opponents.");
                }
                const int true_owner = Game::Token::decode_player(rows[static_cast<size_t>(r)][0]);
                int matched_target = -1;
                for (int a = 0; a < Game::MAX_BELIEF_CANDIDATES; ++a) {
                    const int owner = opponent_ids[static_cast<size_t>(a)];
                    val_ptr[r * Game::MAX_BELIEF_CANDIDATES + a] = 1;
                    tok_ptr[(r * Game::MAX_BELIEF_CANDIDATES + a) * Game::MAX_BELIEF_TOKENS + 0] =
                        Game::Token::encode_player(owner);
                    tok_ptr[(r * Game::MAX_BELIEF_CANDIDATES + a) * Game::MAX_BELIEF_TOKENS + 1] =
                        rows[static_cast<size_t>(r)][1];
                    tok_ptr[(r * Game::MAX_BELIEF_CANDIDATES + a) * Game::MAX_BELIEF_TOKENS + 2] =
                        rows[static_cast<size_t>(r)][2];
                    if (owner == true_owner) matched_target = a;
                }
                if (matched_target < 0) {
                    throw std::runtime_error("belief_supervision failed to map true owner to candidate index.");
                }
                idx_ptr[r] = matched_target;
                belief_length += 1;
            }
            py::dict out;
            out["candidate_tokens"] = candidate_tokens;
            out["candidate_valid"] = candidate_valid;
            out["target_index"] = target_index;
            out["belief_length"] = py::int_(belief_length);
            return out;
        },
        py::is_method(env_type),
        "Return belief-table supervision for one environment.\n"
        "Fields:\n"
        "  candidate_tokens [S, A, E] int32\n"
        "  candidate_valid [S, A] uint8\n"
        "  target_index [S] int32\n"
        "  belief_length int\n"
        "where S=MAX_BELIEF_ROWS, A=MAX_BELIEF_CANDIDATES, E=MAX_BELIEF_TOKENS."
    );
    const py::object env_batch_type = m_doudizhu.attr("EnvBatch");
    env_batch_type.attr("get_available_action_strs") = py::cpp_function(
        [](const EnvBatch<Game>& envs) {
            py::list rows;
            for (int idx = 0; idx < envs.size(); ++idx) {
                rows.append(py::cast(envs[idx].get_available_action_strs()));
            }
            return rows;
        },
        py::is_method(env_batch_type),
        "Batch version of Environment.get_available_action_strs().\n"
        "Returns a list of per-env semantic action-string rows."
    );
    env_batch_type.attr("advance_history_strs") = py::cpp_function(
        [](EnvBatch<Game>& envs, const std::vector<std::string>& history_strs) {
            if (static_cast<int>(history_strs.size()) != envs.size()) {
                throw std::runtime_error("advance_history_strs: history_strs must match EnvBatch size.");
            }
            py::list out;
            for (int idx = 0; idx < envs.size(); ++idx) {
                out.append(envs[idx].advance_history_str(history_strs[static_cast<size_t>(idx)]));
            }
            return out;
        },
        py::is_method(env_batch_type),
        py::arg("history_strs"),
        "Batch version of Environment.advance_history_str().\n"
        "Consumes one semantic update per env and returns aligned canonical history strings."
    );
    env_batch_type.attr("belief_supervision") = py::cpp_function(
        [](const EnvBatch<Game>& envs) {
            const int bsz = envs.size();
            py::array_t<int> candidate_tokens(
                std::vector<py::ssize_t>{bsz, Game::MAX_BELIEF_ROWS, Game::MAX_BELIEF_CANDIDATES, Game::MAX_BELIEF_TOKENS}
            );
            py::array_t<std::uint8_t> candidate_valid(
                std::vector<py::ssize_t>{bsz, Game::MAX_BELIEF_ROWS, Game::MAX_BELIEF_CANDIDATES}
            );
            py::array_t<int> target_index(std::vector<py::ssize_t>{bsz, Game::MAX_BELIEF_ROWS});
            py::array_t<int> belief_length(std::vector<py::ssize_t>{bsz});
            auto* tok_ptr = static_cast<int*>(candidate_tokens.request().ptr);
            auto* val_ptr = static_cast<std::uint8_t*>(candidate_valid.request().ptr);
            auto* idx_ptr = static_cast<int*>(target_index.request().ptr);
            auto* len_ptr = static_cast<int*>(belief_length.request().ptr);

            for (int b = 0; b < bsz; ++b) {
                len_ptr[b] = 0;
                for (int r = 0; r < Game::MAX_BELIEF_ROWS; ++r) {
                    idx_ptr[b * Game::MAX_BELIEF_ROWS + r] = -1;
                    for (int a = 0; a < Game::MAX_BELIEF_CANDIDATES; ++a) {
                        val_ptr[(b * Game::MAX_BELIEF_ROWS + r) * Game::MAX_BELIEF_CANDIDATES + a] = 0;
                        for (int e = 0; e < Game::MAX_BELIEF_TOKENS; ++e) {
                            tok_ptr[
                                ((b * Game::MAX_BELIEF_ROWS + r) * Game::MAX_BELIEF_CANDIDATES + a) *
                                    Game::MAX_BELIEF_TOKENS +
                                e
                            ] = Game::Token::PAD_TOKEN;
                        }
                    }
                }
            }

            for (int b = 0; b < bsz; ++b) {
                const auto rows = envs[b].get_belief_rows();
                const int viewer = envs[b].player();
                std::array<int, Game::MAX_BELIEF_CANDIDATES> opponent_ids{};
                int opponent_count = 0;
                if (viewer >= 0 && viewer < Game::NUM_PLAYERS) {
                    for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
                        if (p == viewer) continue;
                        if (opponent_count >= Game::MAX_BELIEF_CANDIDATES) break;
                        opponent_ids[static_cast<size_t>(opponent_count)] = p;
                        opponent_count += 1;
                    }
                }
                for (int r = 0; r < Game::MAX_BELIEF_ROWS; ++r) {
                    if (rows[static_cast<size_t>(r)][0] == Game::Token::PAD_TOKEN) continue;
                    if (opponent_count != Game::MAX_BELIEF_CANDIDATES) {
                        throw std::runtime_error("belief_supervision expected a valid active viewer with two opponents.");
                    }
                    const int true_owner = Game::Token::decode_player(rows[static_cast<size_t>(r)][0]);
                    int matched_target = -1;
                    for (int a = 0; a < Game::MAX_BELIEF_CANDIDATES; ++a) {
                        const int owner = opponent_ids[static_cast<size_t>(a)];
                        val_ptr[(b * Game::MAX_BELIEF_ROWS + r) * Game::MAX_BELIEF_CANDIDATES + a] = 1;
                        tok_ptr[
                            ((b * Game::MAX_BELIEF_ROWS + r) * Game::MAX_BELIEF_CANDIDATES + a) * Game::MAX_BELIEF_TOKENS +
                            0
                        ] = Game::Token::encode_player(owner);
                        tok_ptr[
                            ((b * Game::MAX_BELIEF_ROWS + r) * Game::MAX_BELIEF_CANDIDATES + a) * Game::MAX_BELIEF_TOKENS +
                            1
                        ] = rows[static_cast<size_t>(r)][1];
                        tok_ptr[
                            ((b * Game::MAX_BELIEF_ROWS + r) * Game::MAX_BELIEF_CANDIDATES + a) * Game::MAX_BELIEF_TOKENS +
                            2
                        ] = rows[static_cast<size_t>(r)][2];
                        if (owner == true_owner) matched_target = a;
                    }
                    if (matched_target < 0) {
                        throw std::runtime_error("belief_supervision failed to map true owner to candidate index.");
                    }
                    idx_ptr[b * Game::MAX_BELIEF_ROWS + r] = matched_target;
                    len_ptr[b] += 1;
                }
            }
            py::dict out;
            out["candidate_tokens"] = candidate_tokens;
            out["candidate_valid"] = candidate_valid;
            out["target_index"] = target_index;
            out["belief_length"] = belief_length;
            return out;
        },
        py::is_method(env_batch_type),
        "Batch belief-table supervision.\n"
        "Fields:\n"
        "  candidate_tokens [B, S, A, E] int32\n"
        "  candidate_valid [B, S, A] uint8\n"
        "  target_index [B, S] int32\n"
        "  belief_length [B] int32\n"
        "where S=MAX_BELIEF_ROWS, A=MAX_BELIEF_CANDIDATES, E=MAX_BELIEF_TOKENS."
    );
    // Convenience alias
    m_doudizhu.attr("Game") = m_doudizhu.attr("Environment");
}
} // namespace nanopokerlib
