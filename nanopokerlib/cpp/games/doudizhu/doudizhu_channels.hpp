#pragma once

#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "doudizhu_struct.hpp"

namespace nanopokerlib::doudizhu {
template<typename Config>
struct DouDiZhuGame<Config>::Channels {
    using Game = DouDiZhuGame<Config>;

    static constexpr int NUM_PLAYERS = Game::NUM_PLAYERS;
    static constexpr int NUM_CARD_CHANNELS = Game::NUM_CARD_CHANNELS; // always 54 (full deck embedding)
    static constexpr int NUM_BOMB_COUNT_CHANNELS = Game::NUM_BOMB_COUNT_CHANNELS; // 0..14 cumulative thresholds

    static constexpr int SELF_OFFSET = 0;
    static constexpr int LANDLORD_PUBLIC_OFFSET = SELF_OFFSET + NUM_CARD_CHANNELS;
    static constexpr int DISCARD_OFFSET = LANDLORD_PUBLIC_OFFSET + NUM_CARD_CHANNELS;
    static constexpr int DISCARD_STRIDE = NUM_CARD_CHANNELS;
    static constexpr int BOMB_COUNT_OFFSET = DISCARD_OFFSET + NUM_PLAYERS * DISCARD_STRIDE;
    static constexpr int TURN_OFFSET = BOMB_COUNT_OFFSET + NUM_BOMB_COUNT_CHANNELS;
    static constexpr int NUM_CHANNELS = TURN_OFFSET + NUM_PLAYERS;
    static_assert(NUM_CHANNELS == Game::NUM_CHANNELS, "Channel count mismatch with game constants");

    using HandCounts = typename Game::HandCounts;

    inline static constexpr std::string_view FULL_RANKS = "3456789TJQKA2BR";
    static constexpr int FULL_NORMAL_RANKS = 13;
    static constexpr int FULL_DECK_CARDS = FULL_NORMAL_RANKS * 4 + 2;
    static_assert(NUM_CARD_CHANNELS == FULL_DECK_CARDS, "DouDiZhu channel encoding expects 54 card slots");

    static constexpr int encode_card(int base_offset, int card_index) {
        if (base_offset < 0 || base_offset + NUM_CARD_CHANNELS > NUM_CHANNELS) { throw std::out_of_range("encode_card: invalid base"); }
        if (card_index < 0 || card_index >= NUM_CARD_CHANNELS) { throw std::out_of_range("encode_card: invalid card_index"); }
        return base_offset + card_index;
    }

    static constexpr int encode_self_card(int card_index) { return encode_card(SELF_OFFSET, card_index); }

    static constexpr int encode_landlord_public_card(int card_index) { return encode_card(LANDLORD_PUBLIC_OFFSET, card_index); }

    static constexpr int discard_base(int player) {
        if (player < 0 || player >= NUM_PLAYERS) { throw std::out_of_range("discard_base: invalid player index"); }
        return DISCARD_OFFSET + player * DISCARD_STRIDE;
    }

    static constexpr int encode_discard_card(int player, int card_index) { return encode_card(discard_base(player), card_index); }

    static constexpr int encode_bomb_count(int count) {
        if (count < 0 || count >= NUM_BOMB_COUNT_CHANNELS) { throw std::out_of_range("encode_bomb_count: invalid count"); }
        return BOMB_COUNT_OFFSET + count;
    }

    static void encode_bomb_count_channels(std::vector<ChannelValue>& out, int count) {
        if (out.size() < static_cast<size_t>(NUM_CHANNELS)) { out.assign(NUM_CHANNELS, ChannelValue{0}); }
        if (count < 0 || count >= NUM_BOMB_COUNT_CHANNELS) {
            throw std::out_of_range("encode_bomb_count_channels: invalid count");
        }
        // Zero-anchored cumulative prefix:
        // - 0 bombs -> threshold 0 set
        // - 1 bomb  -> thresholds 0..1 set
        // Restore the historical one-hot contract by replacing this loop with a single
        // `out[encode_bomb_count(count)] = ChannelValue{1};` write in local forks.
        for (int threshold = 0; threshold <= count; ++threshold) {
            out[encode_bomb_count(threshold)] = ChannelValue{1};
        }
    }

    static constexpr int encode_turn(int player) {
        if (player < 0 || player >= NUM_PLAYERS) { throw std::out_of_range("encode_turn: invalid player index"); }
        return TURN_OFFSET + player;
    }

    static constexpr int full_rank_index(char symbol) {
        for (size_t i = 0; i < FULL_RANKS.size(); ++i) {
            if (FULL_RANKS[i] == symbol) return static_cast<int>(i);
        }
        return -1;
    }

    static constexpr int joker_card_index(int full_rank) {
        if (full_rank == FULL_NORMAL_RANKS) return FULL_NORMAL_RANKS * 4; // B
        if (full_rank == FULL_NORMAL_RANKS + 1) return FULL_NORMAL_RANKS * 4 + 1; // R
        throw std::out_of_range("joker_card_index: invalid joker rank");
    }

    static void encode_cards(std::vector<ChannelValue>& out, int base_offset, const HandCounts& counts) {
        if (out.size() < static_cast<size_t>(NUM_CHANNELS)) { out.assign(NUM_CHANNELS, ChannelValue{0}); }
        for (int rank = 0; rank < Game::NUM_RANKS; ++rank) {
            const int count = counts[rank];
            if (count < 0) { throw std::out_of_range("encode_cards: negative count"); }
            const char symbol = static_cast<char>(std::toupper(static_cast<unsigned char>(Game::RANKS[rank])));
            const int full = full_rank_index(symbol);
            if (full < 0) { throw std::runtime_error("encode_cards: rank symbol missing from FULL_RANKS"); }
            if (full < FULL_NORMAL_RANKS) {
                if (count > 4) { throw std::out_of_range("encode_cards: normal rank count must be within [0, 4]"); }
                const int base = full * 4;
                for (int i = 0; i < count; ++i) {
                    out[encode_card(base_offset, base + i)] = ChannelValue{1};
                }
            } else {
                if (count > 1) { throw std::out_of_range("encode_cards: joker count must be within [0, 1]"); }
                if (count == 1) {
                    out[encode_card(base_offset, joker_card_index(full))] = ChannelValue{1};
                }
            }
        }
    }

    static std::string card_to_string(int card_index) {
        if (card_index < 0 || card_index >= NUM_CARD_CHANNELS) return "?";
        if (card_index < FULL_NORMAL_RANKS * 4) {
            const int full_rank = card_index / 4;
            const int copy = card_index % 4;
            std::string out;
            out.push_back(static_cast<char>(FULL_RANKS[static_cast<size_t>(full_rank)]));
            out.push_back('#');
            out.push_back(static_cast<char>('0' + copy));
            return out;
        }
        if (card_index == FULL_NORMAL_RANKS * 4) return "B";
        if (card_index == FULL_NORMAL_RANKS * 4 + 1) return "R";
        return "?";
    }

    static std::string decode_channel(int channel) {
        if (channel < 0 || channel >= NUM_CHANNELS) { return "UNK_CH(" + std::to_string(channel) + ")"; }
        if (channel < LANDLORD_PUBLIC_OFFSET) { return "SELF:" + card_to_string(channel - SELF_OFFSET); }
        if (channel < DISCARD_OFFSET) { return "L_PUB:" + card_to_string(channel - LANDLORD_PUBLIC_OFFSET); }
        if (channel < BOMB_COUNT_OFFSET) {
            const int rel = channel - DISCARD_OFFSET;
            const int player = rel / DISCARD_STRIDE;
            const int card = rel % DISCARD_STRIDE;
            return "D" + std::to_string(player) + ":" + card_to_string(card);
        }
        if (channel < TURN_OFFSET) { return "BOMBS>=" + std::to_string(channel - BOMB_COUNT_OFFSET); }
        return "TURN_P" + std::to_string(channel - TURN_OFFSET);
    }
};
} // namespace nanopokerlib::doudizhu
