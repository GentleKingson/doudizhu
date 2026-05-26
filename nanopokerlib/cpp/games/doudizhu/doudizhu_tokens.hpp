#pragma once

#include <stdexcept>
#include <string>

#include "doudizhu_struct.hpp"

namespace nanopokerlib::doudizhu {
template<typename Config>
struct DouDiZhuGame<Config>::Token {
    using Game = DouDiZhuGame<Config>;

    static constexpr int PAD_TOKEN = nanopokerlib::PAD_TOKEN;
    static constexpr int INIT_TOKEN = nanopokerlib::INIT_TOKEN;

    // Player identity
    static constexpr int PLAYER_OFFSET = INIT_TOKEN + 1;

    // Action class
    static constexpr int ACTION_CLASS_OFFSET = PLAYER_OFFSET + NUM_PLAYERS;
    static constexpr int NUM_ACTION_CLASSES = 8; // Solo, Pair, Trio, Quad, Straight, Sisters, TrioChain, Rocket

    // Rank and sequence tokens
    static constexpr int RANK_OFFSET = ACTION_CLASS_OFFSET + NUM_ACTION_CLASSES;
    static constexpr int SEQ_FROM_OFFSET = RANK_OFFSET + Game::NUM_RANKS; // 3..K
    static constexpr int SEQ_TO_OFFSET = SEQ_FROM_OFFSET + Game::NUM_SEQ_FROM; // 3..A (SeqTo(0) is deprecated/unreachable)
    static constexpr int DEPRECATED_SEQ_TO_TOKEN = SEQ_TO_OFFSET; // == encode_seq_to(0); reserved/deprecated

    // Kicker count/type tokens: none, solo count (1..5), pair count (1..4)
    static constexpr int KICKER_TYPE_OFFSET = SEQ_TO_OFFSET + Game::NUM_SEQ_TO;
    static constexpr int NUM_KICKER_TYPE_TOKENS = 1 + Game::MAX_SOLO_KICKERS + Game::MAX_PAIR_KICKERS;

    // Reserved deprecated kicker-rank tokens (legacy; never emitted by the environment).
    static constexpr int SOLO_KICKER_OFFSET = KICKER_TYPE_OFFSET + NUM_KICKER_TYPE_TOKENS;
    static constexpr int PAIR_KICKER_OFFSET = SOLO_KICKER_OFFSET + Game::NUM_RANKS; // 3..2 (jokers disallowed)

    // Pass terminator
    static constexpr int PASS_TOKEN = PAIR_KICKER_OFFSET + Game::NUM_PAIR_KICKER_RANKS;

    // Observation-only marker used during dealing: [PLAYER, RANK, HOLD]
    static constexpr int HOLD_TOKEN = PASS_TOKEN + 1;

    static constexpr int NUM_TOKENS = HOLD_TOKEN + 1;

    enum class ActionClass {
        DISCARD_SOLO = 0,
        DISCARD_PAIR,
        DISCARD_TRIO,
        DISCARD_QUAD,
        DISCARD_STRAIGHT,
        DISCARD_PAIR_SISTERS,
        DISCARD_TRIO_CHAIN,
        DISCARD_ROCKET,
    };

    // Encoding helpers
    static constexpr int encode_player(int player) {
        if (player < 0 || player >= Game::NUM_PLAYERS) { throw std::out_of_range("encode_player: invalid player index"); }
        return PLAYER_OFFSET + player;
    }

    static constexpr int encode_action_class(ActionClass cls) {
        const int idx = static_cast<int>(cls);
        if (idx < 0 || idx >= NUM_ACTION_CLASSES) { throw std::out_of_range("encode_action_class: invalid class index"); }
        return ACTION_CLASS_OFFSET + idx;
    }

    static constexpr int encode_rank(int rank) {
        if (rank < 0 || rank >= Game::NUM_RANKS) { throw std::out_of_range("encode_rank: invalid rank index"); }
        return RANK_OFFSET + rank;
    }

    static constexpr int encode_seq_from(int rank) {
        if (rank < 0 || rank >= Game::NUM_SEQ_FROM) { throw std::out_of_range("encode_seq_from: invalid start rank"); }
        return SEQ_FROM_OFFSET + rank;
    }

    static constexpr int encode_seq_to(int rank) {
        if (rank < 0 || rank >= Game::NUM_SEQ_TO) { throw std::out_of_range("encode_seq_to: invalid end rank"); }
        return SEQ_TO_OFFSET + rank;
    }

    static constexpr int encode_no_kicker() { return KICKER_TYPE_OFFSET; }

    static constexpr int encode_solo_kicker_count(int count) {
        if (count < 1 || count > Game::MAX_SOLO_KICKERS) { throw std::out_of_range("encode_solo_kicker_count: invalid count"); }
        return KICKER_TYPE_OFFSET + count;
    }

    static constexpr int encode_pair_kicker_count(int count) {
        if (count < 1 || count > Game::MAX_PAIR_KICKERS) { throw std::out_of_range("encode_pair_kicker_count: invalid count"); }
        return KICKER_TYPE_OFFSET + 1 + Game::MAX_SOLO_KICKERS + (count - 1);
    }

    static constexpr int encode_pass() { return PASS_TOKEN; }

    static constexpr int encode_hold() { return HOLD_TOKEN; }

    // Decoding helpers
    static constexpr int decode_player(int token) {
        if (token < PLAYER_OFFSET || token >= PLAYER_OFFSET + Game::NUM_PLAYERS) {
            throw std::out_of_range("decode_player: token out of range");
        }
        return token - PLAYER_OFFSET;
    }

    static constexpr ActionClass decode_action_class(int token) {
        if (!is_action_class(token)) { throw std::out_of_range("decode_action_class: token out of range"); }
        return static_cast<ActionClass>(token - ACTION_CLASS_OFFSET);
    }

    static constexpr int decode_rank(int token) {
        if (!is_rank(token)) { throw std::out_of_range("decode_rank: token out of range"); }
        return token - RANK_OFFSET;
    }

    static constexpr int decode_seq_from(int token) {
        if (!is_seq_from(token)) { throw std::out_of_range("decode_seq_from: token out of range"); }
        return token - SEQ_FROM_OFFSET;
    }

    static constexpr int decode_seq_to(int token) {
        if (!is_seq_to(token)) { throw std::out_of_range("decode_seq_to: token out of range"); }
        return token - SEQ_TO_OFFSET;
    }

    static constexpr int decode_solo_kicker_count(int token) {
        const int offset = token - KICKER_TYPE_OFFSET;
        if (offset <= 0 || offset > Game::MAX_SOLO_KICKERS) { throw std::out_of_range("decode_solo_kicker_count: token out of range"); }
        return offset;
    }

    static constexpr int decode_pair_kicker_count(int token) {
        const int offset = token - (KICKER_TYPE_OFFSET + 1 + Game::MAX_SOLO_KICKERS);
        if (offset < 0 || offset >= Game::MAX_PAIR_KICKERS) { throw std::out_of_range("decode_pair_kicker_count: token out of range"); }
        return offset + 1;
    }

    static constexpr bool is_pass(int token) { return token == PASS_TOKEN; }

    static constexpr bool is_hold(int token) { return token == HOLD_TOKEN; }

    static constexpr bool is_action_class(int token) {
        return token >= ACTION_CLASS_OFFSET && token < ACTION_CLASS_OFFSET + NUM_ACTION_CLASSES;
    }

    static constexpr bool is_rank(int token) { return token >= RANK_OFFSET && token < RANK_OFFSET + Game::NUM_RANKS; }

    static constexpr bool is_seq_from(int token) { return token >= SEQ_FROM_OFFSET && token < SEQ_FROM_OFFSET + Game::NUM_SEQ_FROM; }

    static constexpr bool is_seq_to(int token) { return token >= SEQ_TO_OFFSET && token < SEQ_TO_OFFSET + Game::NUM_SEQ_TO; }

    static constexpr bool is_kicker_type(int token) {
        return token >= KICKER_TYPE_OFFSET && token < KICKER_TYPE_OFFSET + NUM_KICKER_TYPE_TOKENS;
    }

    // Pretty-printers
    static std::string rank_to_string(int rank) {
        if (rank < 0 || rank >= static_cast<int>(Game::RANKS.size())) { return "?"; }
        return std::string(1, static_cast<char>(Game::RANKS[rank]));
    }

    static std::string action_class_to_string(ActionClass cls) {
        switch (cls) {
        case ActionClass::DISCARD_SOLO:
            return "DiscardSolo";
        case ActionClass::DISCARD_PAIR:
            return "DiscardPair";
        case ActionClass::DISCARD_TRIO:
            return "DiscardTrio";
        case ActionClass::DISCARD_QUAD:
            return "DiscardQuad";
        case ActionClass::DISCARD_STRAIGHT:
            return "DiscardStraight";
        case ActionClass::DISCARD_PAIR_SISTERS:
            return "DiscardPairSisters";
        case ActionClass::DISCARD_TRIO_CHAIN:
            return "DiscardTrioChain";
        case ActionClass::DISCARD_ROCKET:
            return "DiscardRocket";
        }
        return "Discard?";
    }

    static std::string kicker_count_to_string(int token) {
        if (token == encode_no_kicker()) return "NoKicker";
        if (token >= KICKER_TYPE_OFFSET + 1 && token < KICKER_TYPE_OFFSET + 1 + Game::MAX_SOLO_KICKERS) {
            return "SoloKicker(x" + std::to_string(decode_solo_kicker_count(token)) + ")";
        }
        if (token >= KICKER_TYPE_OFFSET + 1 + Game::MAX_SOLO_KICKERS &&
            token < KICKER_TYPE_OFFSET + 1 + Game::MAX_SOLO_KICKERS + Game::MAX_PAIR_KICKERS) {
            return "PairKicker(x" + std::to_string(decode_pair_kicker_count(token)) + ")";
        }
        return "Kicker?";
    }

    static std::string decode_token(int token) {
        if (token == PAD_TOKEN) return "PAD";
        if (token == INIT_TOKEN) return "INIT";
        if (token >= PLAYER_OFFSET && token < PLAYER_OFFSET + NUM_PLAYERS) {
            return "PLAYER(" + std::to_string(decode_player(token)) + ")";
        }
        if (is_action_class(token)) return action_class_to_string(decode_action_class(token));
        if (is_rank(token)) return "RANK(" + rank_to_string(decode_rank(token)) + ")";
        if (is_seq_from(token)) return "SeqFrom(" + rank_to_string(decode_seq_from(token)) + ")";
        if (token == DEPRECATED_SEQ_TO_TOKEN) return "SeqTo_DEPRECATED(" + rank_to_string(0) + ")";
        if (is_seq_to(token)) return "SeqTo(" + rank_to_string(decode_seq_to(token)) + ")";
        if (is_kicker_type(token)) return kicker_count_to_string(token);
        if (token >= SOLO_KICKER_OFFSET && token < SOLO_KICKER_OFFSET + Game::NUM_RANKS) {
            return "SoloKicker_DEPRECATED(" + rank_to_string(token - SOLO_KICKER_OFFSET) + ")";
        }
        if (token >= PAIR_KICKER_OFFSET && token < PAIR_KICKER_OFFSET + Game::NUM_PAIR_KICKER_RANKS) {
            return "PairKicker_DEPRECATED(" + rank_to_string(token - PAIR_KICKER_OFFSET) + ")";
        }
        if (token == PASS_TOKEN) return "PASS";
        if (token == HOLD_TOKEN) return "HOLD";
        return "UNK(" + std::to_string(token) + ")";
    }
};
} // namespace nanopokerlib::doudizhu
