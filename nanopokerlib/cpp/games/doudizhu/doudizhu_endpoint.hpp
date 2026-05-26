#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/semantic_string.hpp"
#include "doudizhu_core.hpp"

namespace nanopokerlib::doudizhu {
template<typename Config>
struct DouDiZhuGame<Config>::Endpoint : DouDiZhuGame<Config> {
    using Game = DouDiZhuGame<Config>;
    using HandCounts = typename Game::HandCounts;
    using Combo = typename Game::Combo;
    using Sequence = typename Game::Sequence;

    struct ParsedAction {
        int player_id{0};
        bool is_pass{false};
        HandCounts counts{};
    };

    struct ParsedInput {
        int view_player_id{-1};
        HandCounts view_private_counts{};
        std::vector<int> view_private_order;
        bool has_view_private{false};

        HandCounts landlord_public_counts{};
        std::vector<int> landlord_public_order;
        bool has_landlord_public{false};
        std::vector<ParsedAction> actions;
    };

    struct BeliefStepResult {
        std::array<BeliefTriple, Game::MAX_BELIEF_CANDIDATES> tokens{};
        std::array<std::uint8_t, Game::MAX_BELIEF_CANDIDATES> valid{};
        std::array<std::string, Game::MAX_BELIEF_CANDIDATES> belief_strs{};
    };

    struct BeliefStepBatchResult {
        std::vector<BeliefStepResult> items;
    };

    static History<Game> parse_infoset_str(const std::string& infoset_str, const std::string& pending_action = "");

    static HistoryBatch<Game>
    parse_infoset_strs(const std::vector<std::string>& infoset_strs, const std::vector<std::string>& pending_actions);

    static BeliefStepResult expand_belief_prefix_str(const std::string& infoset_str, const std::string& belief_str = "");
    static BeliefStepBatchResult
    expand_belief_prefix_strs(const std::vector<std::string>& infoset_strs, const std::vector<std::string>& belief_strs = {});
    static std::string determinize_history_str(const std::string& infoset_str, const std::string& belief_str);
    static std::vector<std::string>
    determinize_history_strs(const std::vector<std::string>& infoset_strs, const std::vector<std::string>& belief_strs);
    static std::string infoset_str_from_history_str(const std::string& history_str);
    static std::vector<std::string> infoset_str_from_history_strs(const std::vector<std::string>& history_strs);

    static std::string format_history_str(const History<Game>& infoset);

    static std::string format_infoset_str(const History<Game>& infoset);

    static std::pair<Game, StepResult<Game>> reconstruct_env(
        const std::string& history_str,
        const std::string& pending_action = "",
        int seed = 0
    );

    static std::string describe_live_sequence_step(const Sequence& seq, int step_idx) { return sequence_step_string(seq, step_idx); }
    static std::string opening_root_history_str(const Game& state);

private:
    struct InfosetEncodeComponents {
        std::array<HandCounts, Game::NUM_PLAYERS> initial_hands{};
        HandCounts landlord_public_cards{};
        std::vector<std::vector<int>> history_actions;
    };

    static std::string build_infoset_str(
        const std::array<HandCounts, Game::NUM_PLAYERS>& initial_hands, const HandCounts& landlord_public_cards,
        const std::vector<std::vector<int>>& history_actions, int view_player_id
    );

    static std::string trim(std::string_view in) {
        return semantic_string::trim_ascii_whitespace(in);
    }

    static int parse_player_tag(std::string_view tag) {
        const std::string upper = semantic_string::normalize_compact_upper(tag);
        if (upper == "L" || upper == "LANDLORD") return 0;
        if (upper == "P1" || upper == "PEASANT1") return 1;
        if (upper == "P2" || upper == "PEASANT2") return 2;
        throw std::runtime_error("Unknown player tag: " + std::string(tag));
    }

    static HandCounts parse_cards(std::string_view cards) {
        HandCounts counts{};
        for (char ch : cards) {
            if (std::isspace(static_cast<unsigned char>(ch))) continue;
            char up = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            size_t idx = Game::RANKS.find(up);
            if (idx == std::string_view::npos) { throw std::runtime_error(std::string("Unknown rank symbol: ") + up); }
            counts[static_cast<int>(idx)] += 1;
        }
        return counts;
    }

    static std::pair<HandCounts, std::vector<int>> parse_cards_with_order(std::string_view cards) {
        HandCounts counts{};
        std::vector<int> order;
        order.reserve(cards.size());
        for (char ch : cards) {
            if (std::isspace(static_cast<unsigned char>(ch))) continue;
            char up = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            size_t idx = Game::RANKS.find(up);
            if (idx == std::string_view::npos) { throw std::runtime_error(std::string("Unknown rank symbol: ") + up); }
            const int rank = static_cast<int>(idx);
            counts[rank] += 1;
            order.push_back(rank);
        }
        std::sort(order.begin(), order.end());
        return {counts, std::move(order)};
    }

    static std::string normalize_action_string(std::string_view in) {
        return semantic_string::normalize_compact_upper(in);
    }

    static ParsedInput parse(const std::string& infoset_str) {
        ParsedInput parsed;
        for (const std::string& seg : semantic_string::split_segments(infoset_str)) {
            const auto tagged = semantic_string::try_parse_tagged_segment(seg);
            if (!tagged.has_value()) { throw std::runtime_error("Segment missing + or -: " + seg); }
            const int player = parse_player_tag(tagged->tag);
            if (tagged->op == '+') {
                auto [counts, order] = parse_cards_with_order(tagged->payload);
                if (!parsed.has_view_private) {
                    parsed.view_player_id = player;
                    parsed.view_private_counts = counts;
                    parsed.view_private_order = std::move(order);
                    parsed.has_view_private = true;
                } else {
                    if (player != 0) {
                        throw std::runtime_error("Only landlord public cards may be provided in a second + segment.");
                    }
                    if (parsed.has_landlord_public) {
                        throw std::runtime_error("Only one landlord public + segment is allowed.");
                    }
                    parsed.landlord_public_counts = counts;
                    parsed.landlord_public_order = std::move(order);
                    parsed.has_landlord_public = true;
                }
            } else if (tagged->op == '-') {
                ParsedAction action;
                action.player_id = player;
                const std::string upper_payload = semantic_string::normalize_compact_upper(tagged->payload);
                if (upper_payload == "PASS" || upper_payload.empty()) {
                    action.is_pass = true;
                } else {
                    action.counts = parse_cards(tagged->payload);
                }
                parsed.actions.push_back(std::move(action));
            } else {
                throw std::runtime_error("Unknown segment operator: " + std::string(1, tagged->op));
            }
        }

        if (parsed.view_player_id < 0 || !parsed.has_view_private) { throw std::runtime_error("Missing + segment with the acting player's hand."); }
        if (!parsed.has_landlord_public) { throw std::runtime_error("Missing landlord public cards segment (expected '|L+...')."); }
        return parsed;
    }

    static int expected_cards_for_player(int player) {
        return (player == 0) ? Game::LANDLORD_HAND_CARDS : Game::BASE_HAND_CARDS;
    }

    static int count_cards(const HandCounts& counts) {
        int total = 0;
        for (int c : counts)
            total += c;
        return total;
    }

    static std::array<HandCounts, Game::NUM_PLAYERS> compute_played_counts(const std::vector<ParsedAction>& actions) {
        std::array<HandCounts, Game::NUM_PLAYERS> played{};
        for (const auto& act : actions) {
            if (act.is_pass) continue;
            for (int r = 0; r < Game::NUM_RANKS; ++r) {
                played[act.player_id][r] += act.counts[r];
            }
        }
        return played;
    }

    static std::array<HandCounts, Game::NUM_PLAYERS> build_initial_hands(const ParsedInput& parsed) {
        if (!parsed.has_view_private) { throw std::runtime_error("Missing view player's private hand segment."); }
        if (!parsed.has_landlord_public) { throw std::runtime_error("Missing landlord public cards segment."); }

        const int view_private_count = count_cards(parsed.view_private_counts);
        if (view_private_count > Game::BASE_HAND_CARDS) {
            throw std::runtime_error("Viewing player's private hand segment has too many cards.");
        }
        if (count_cards(parsed.landlord_public_counts) != Game::NUM_PUBLIC_LANDLORD_CARDS) {
            throw std::runtime_error("Landlord public cards segment must contain exactly 3 cards.");
        }

        auto played = compute_played_counts(parsed.actions);

        HandCounts deck{};
        for (int r = 0; r < Game::NORMAL_RANKS; ++r)
            deck[r] = 4;
        for (int r = Game::NORMAL_RANKS; r < Game::NUM_RANKS; ++r)
            deck[r] = 1;

        auto subtract_or_throw = [](HandCounts& dst, const HandCounts& src, const std::string& context) {
            for (int r = 0; r < Game::NUM_RANKS; ++r) {
                dst[r] -= src[r];
                if (dst[r] < 0) { throw std::runtime_error("Not enough cards left in deck while assigning " + context); }
            }
        };

        subtract_or_throw(deck, parsed.view_private_counts, "view private hand");
        subtract_or_throw(deck, parsed.landlord_public_counts, "landlord public cards");

        HandCounts view_full = parsed.view_private_counts;
        if (parsed.view_player_id == 0) {
            for (int r = 0; r < Game::NUM_RANKS; ++r) {
                view_full[r] += parsed.landlord_public_counts[r];
            }
        }
        for (int r = 0; r < Game::NUM_RANKS; ++r) {
            if (view_full[r] < played[parsed.view_player_id][r]) {
                throw std::runtime_error("Known hand does not cover the viewing player's action history.");
            }
        }

        std::array<HandCounts, Game::NUM_PLAYERS> hands{};

        // Start with known view hand and known landlord public cards.
        hands[parsed.view_player_id] = parsed.view_private_counts;
        for (int r = 0; r < Game::NUM_RANKS; ++r) {
            hands[0][r] += parsed.landlord_public_counts[r];
        }

        // Seed unknown hands with the cards they played in the history.
        for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
            if (p == parsed.view_player_id) continue;
            for (int r = 0; r < Game::NUM_RANKS; ++r) {
                const int missing = std::max(played[p][r] - hands[p][r], 0);
                if (missing > deck[r]) {
                    throw std::runtime_error("History spends more of rank " + std::to_string(r) + " than the deck contains.");
                }
                hands[p][r] += missing;
                deck[r] -= missing;
            }
        }

        // Fill remaining unknown cards deterministically (lowest ranks first).
        for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
            const int expected = expected_cards_for_player(p);
            const int have = count_cards(hands[p]);
            if (have > expected) { throw std::runtime_error("Player " + std::to_string(p) + " has more cards than expected."); }
            int need = expected - have;
            for (int r = 0; r < Game::NUM_RANKS && need > 0; ++r) {
                int take = std::min(deck[r], need);
                hands[p][r] += take;
                deck[r] -= take;
                need -= take;
            }
            if (need > 0) { throw std::runtime_error("Not enough cards to fill player " + std::to_string(p) + "'s hand."); }
        }

        if (count_cards(deck) != 0) { throw std::runtime_error("Card allocation mismatch; leftover cards remain after reconstruction."); }
        return hands;
    }

    static Endpoint reconstruct_state_from_infoset(const std::string& infoset_str);
    static HandCounts combo_counts(const Combo& combo) {
        HandCounts counts{};
        auto add = [&](const std::vector<std::pair<int, int>>& groups) {
            for (const auto& [qty, rank] : groups) {
                counts[rank] += qty;
            }
        };
        add(combo.main);
        add(combo.kickers);
        return counts;
    }

    static const Sequence* match_sequence(const Endpoint& state, const ParsedAction& action) {
        for (const auto& seq : state.current_sequences_) {
            if (action.is_pass) {
                if (seq.combo.type == Combo::Type::PASS) return &seq;
                continue;
            }
            if (combo_counts(seq.combo) == action.counts) { return &seq; }
        }
        return nullptr;
    }

    static std::string player_tag(int player) {
        if (player == 0) return "L";
        if (player == 1) return "P1";
        if (player == 2) return "P2";
        return "P" + std::to_string(player);
    }

    static std::string counts_to_cards(const HandCounts& counts) {
        std::string out;
        for (int r = 0; r < Game::NUM_RANKS; ++r) {
            for (int c = 0; c < counts[r]; ++c) {
                out.push_back(static_cast<char>(Game::RANKS[r]));
            }
        }
        return out;
    }

    static std::string groups_to_cards(const std::vector<std::pair<int, int>>& groups) {
        std::string out;
        for (const auto& [qty, rank] : groups) {
            for (int c = 0; c < qty; ++c) {
                out.push_back(static_cast<char>(Game::RANKS[rank]));
            }
        }
        return out;
    }

    static std::string chain_string(int start_rank, int length, int repeat) {
        std::string out;
        for (int r = start_rank; r < start_rank + length; ++r) {
            for (int c = 0; c < repeat; ++c) {
                out.push_back(static_cast<char>(Game::RANKS[r]));
            }
        }
        return out;
    }

    static std::string kicker_suffix_with_placeholders(const Combo& combo, int chosen_groups) {
        std::string out;
        int chosen_cards = 0;
        for (int i = 0; i < chosen_groups && i < static_cast<int>(combo.kickers.size()); ++i) {
            const auto& [qty, rank] = combo.kickers[i];
            for (int c = 0; c < qty; ++c) {
                out.push_back(static_cast<char>(Game::RANKS[rank]));
            }
            chosen_cards += qty;
        }
        int total_cards = 0;
        for (const auto& [qty, _] : combo.kickers) {
            total_cards += qty;
        }
        int remaining = total_cards - chosen_cards;
        if (remaining > 0) { out.append(static_cast<size_t>(remaining), '.'); }
        return out;
    }

    static std::string sequence_step_string(const Sequence& seq, int step_idx) {
        if (step_idx < 0 || step_idx >= static_cast<int>(seq.steps.size())) {
            throw std::runtime_error("sequence_step_string: step index out of range");
        }
        const Combo& combo = seq.combo;
        const std::string main_cards = groups_to_cards(combo.main);
        const int start_rank = combo.main_rank - combo.length + 1;
        switch (combo.type) {
        case Combo::Type::PASS:
            return "PASS";
        case Combo::Type::ROCKET:
        case Combo::Type::SINGLE:
        case Combo::Type::PAIR:
            return main_cards;
        case Combo::Type::TRIPLE:
            return (step_idx == 0) ? (main_cards + "?") : main_cards;
        case Combo::Type::TRIPLE_SINGLE:
        case Combo::Type::TRIPLE_PAIR: {
            if (step_idx == 0) return main_cards + "?";
            if (step_idx == 1) return main_cards + kicker_suffix_with_placeholders(combo, 0);
            return main_cards + kicker_suffix_with_placeholders(combo, 1);
        }
        case Combo::Type::BOMB:
            return (step_idx == 0) ? (main_cards + "?") : main_cards;
        case Combo::Type::FOUR_TWO_SINGLE:
        case Combo::Type::FOUR_TWO_PAIR: {
            if (step_idx == 0) return main_cards + "?";
            const int chosen_groups = step_idx - 1;
            return main_cards + kicker_suffix_with_placeholders(combo, chosen_groups);
        }
        case Combo::Type::STRAIGHT:
            return (step_idx == 0) ? (chain_string(start_rank, 1, 1) + "+") : chain_string(start_rank, combo.length, 1);
        case Combo::Type::STRAIGHT_PAIR:
            return (step_idx == 0) ? (chain_string(start_rank, 1, 2) + "+") : chain_string(start_rank, combo.length, 2);
        case Combo::Type::STRAIGHT_TRIPLE:
            return (step_idx == 0) ? (chain_string(start_rank, 1, 3) + "+") : chain_string(start_rank, combo.length, 3);
        case Combo::Type::PLANE_SINGLE:
        case Combo::Type::PLANE_PAIR: {
            if (step_idx == 0) { return chain_string(start_rank, 1, 3) + "+"; }
            const std::string main_chain = chain_string(start_rank, combo.length, 3);
            if (step_idx == 1) { return main_chain + kicker_suffix_with_placeholders(combo, 0); }
            const int chosen_groups = step_idx - 1;
            return main_chain + kicker_suffix_with_placeholders(combo, chosen_groups);
        }
        }
        throw std::runtime_error("sequence_step_string: unknown combo type");
    }

    static std::vector<std::vector<int>> resolve_pending_actions(const Endpoint& state, const std::string& pending_action) {
        std::string target = normalize_action_string(pending_action);
        if (target.empty()) return {};

        const Sequence* match = nullptr;
        int match_step = -1;
        for (const auto& seq : state.current_sequences_) {
            for (int idx = state.stage_index_; idx < static_cast<int>(seq.steps.size()); ++idx) {
                if (sequence_step_string(seq, idx) == target) {
                    match = &seq;
                    match_step = idx;
                    break;
                }
            }
            if (match != nullptr) break;
        }
        if (match == nullptr) { throw std::runtime_error("Pending action string not reachable: " + target); }

        std::vector<std::vector<int>> tokens;
        tokens.reserve(match_step - state.stage_index_ + 1);
        for (int idx = state.stage_index_; idx <= match_step; ++idx) {
            tokens.push_back(match->steps[idx].tokens);
        }
        return tokens;
    }

    static std::vector<std::string>
    available_action_strings(const Endpoint& state, const std::vector<ActionProto<Game>>& available_actions) {
        std::vector<std::string> out;
        out.reserve(available_actions.size());
        for (const auto& action : available_actions) {
            bool found = false;
            std::string candidate;
            for (const auto& seq : state.current_sequences_) {
                if (state.stage_index_ >= static_cast<int>(seq.steps.size())) continue;
                if (seq.steps[state.stage_index_].tokens != action.tokens) continue;
                const std::string step_string = sequence_step_string(seq, state.stage_index_);
                if (!found) {
                    candidate = step_string;
                    found = true;
                } else if (candidate != step_string) {
                    throw std::runtime_error("Inconsistent semantic string for available action tokens");
                }
            }
            if (!found) { throw std::runtime_error("Unable to map available action tokens to semantic string"); }
            out.push_back(candidate);
        }
        return out;
    }

    static std::vector<const Sequence*> match_action_sequences(const Endpoint& state, const std::vector<int>& tokens) {
        std::vector<const Sequence*> matches;
        for (const auto& seq : state.current_sequences_) {
            if (state.stage_index_ >= static_cast<int>(seq.steps.size())) continue;
            if (seq.steps[state.stage_index_].tokens == tokens) { matches.push_back(&seq); }
        }
        return matches;
    }

    static void seed_state_with_hands(Endpoint& state, const ParsedInput& parsed, const std::array<HandCounts, Game::NUM_PLAYERS>& hands) {
        state.phase_ = Game::Phase::DEALING;
        state.player_ = PLAYER_CHANCE_ENVIRONMENT;
        state.dealing_step_index_ = 0;
        for (auto& seq : state.deal_private_cards_)
            seq.clear();
        state.deal_landlord_public_cards_.clear();
        state.hand_counts_.fill({});
        state.landlord_public_counts_.fill(0);
        for (auto& dc : state.discard_counts_)
            dc.fill(0);
        state.cards_remaining_.fill(0);
        state.action_prefix_.clear();
        state.stage_index_ = 0;
        state.current_sequences_.clear();
        state.active_combo_.reset();
        state.current_leader_ = -1;
        state.consecutive_passes_ = 0;
        state.bomb_count_ = 0;

        HandCounts landlord_private{};
        for (int r = 0; r < Game::NUM_RANKS; ++r) {
            landlord_private[r] = hands[0][r] - parsed.landlord_public_counts[r];
            if (landlord_private[r] < 0) { throw std::runtime_error("Negative landlord private card count during reconstruction."); }
        }
        if (count_cards(landlord_private) != Game::BASE_HAND_CARDS) {
            throw std::runtime_error("Landlord private card count mismatch during reconstruction.");
        }

        for (int p = 0; p < Game::NUM_PLAYERS; ++p) {
            const HandCounts& private_counts = (p == 0) ? landlord_private : hands[p];
            std::vector<int> order;
            order.reserve(static_cast<size_t>(Game::BASE_HAND_CARDS));

            if (p == parsed.view_player_id) {
                order = parsed.view_private_order;
                HandCounts missing = private_counts;
                for (int r = 0; r < Game::NUM_RANKS; ++r) {
                    missing[r] -= parsed.view_private_counts[r];
                    if (missing[r] < 0) { throw std::runtime_error("View private hand segment exceeds expected cards."); }
                }
                for (int r = 0; r < Game::NUM_RANKS; ++r) {
                    for (int i = 0; i < missing[r]; ++i)
                        order.push_back(r);
                }
            } else {
                for (int r = 0; r < Game::NUM_RANKS; ++r) {
                    for (int i = 0; i < private_counts[r]; ++i)
                        order.push_back(r);
                }
            }

            if (order.size() != static_cast<size_t>(Game::BASE_HAND_CARDS)) {
                throw std::runtime_error("Private deal order length mismatch during reconstruction.");
            }
            std::sort(order.begin(), order.end());
            state.deal_private_cards_[p] = std::move(order);
        }

        state.deal_landlord_public_cards_ = parsed.landlord_public_order;
        if (state.deal_landlord_public_cards_.size() != static_cast<size_t>(Game::NUM_PUBLIC_LANDLORD_CARDS)) {
            throw std::runtime_error("Landlord public deal order length mismatch during reconstruction.");
        }
        std::sort(state.deal_landlord_public_cards_.begin(), state.deal_landlord_public_cards_.end());
    }

    static std::string format_history_str(
        const std::array<HandCounts, Game::NUM_PLAYERS>& initial_hands,
        const HandCounts& landlord_public_cards,
        const std::vector<std::vector<int>>& history_actions
    );

    static InfosetEncodeComponents extract_infoset_encode_components(const History<Game>& infoset);
};

} // namespace nanopokerlib::doudizhu

#ifdef NANOPOKERLIB_IMPLEMENTATION
#include "endpoint/doudizhu_endpoint_methods_replay.hpp"
#include "endpoint/doudizhu_endpoint_methods_belief.hpp"
#include "endpoint/doudizhu_endpoint_methods_encode.hpp"
#endif
