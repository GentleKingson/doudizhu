#pragma once

#include "games/doudizhu/doudizhu_core.hpp"

#ifdef NANOPOKERLIB_IMPLEMENTATION
namespace nanopokerlib::doudizhu {

template<typename Config>
std::vector<typename DouDiZhuGame<Config>::Combo> DouDiZhuGame<Config>::enumerate_combos(const HandCounts& hand) const {
    using Game = DouDiZhuGame<Config>;
    std::vector<Combo> combos;
    combos.reserve(256);
    const size_t MAX_COMBOS = 200000;
    bool capped = false;

    auto push_combo = [&](Combo&& c) {
        if (capped) return;
        sort_groups<Game>(c.main);
        sort_groups<Game>(c.kickers);
        combos.push_back(std::move(c));
        if (combos.size() >= MAX_COMBOS) { capped = true; }
    };

    // Rocket
    if (hand[Game::BLACK_JOKER_RANK] > 0 && hand[Game::RED_JOKER_RANK] > 0) {
        Combo c;
        c.type = Combo::Type::ROCKET;
        c.main = {{1, Game::BLACK_JOKER_RANK}, {1, Game::RED_JOKER_RANK}};
        c.main_rank = Game::RED_JOKER_RANK;
        c.length = 1;
        push_combo(std::move(c));
    }

    // Bombs
    for (int r = 0; r < NORMAL_RANKS; ++r) {
        if (hand[r] >= 4) {
            Combo c;
            c.type = Combo::Type::BOMB;
            c.main = {{4, r}};
            c.main_rank = r;
            c.length = 1;
            push_combo(std::move(c));
        }
    }

    // Singles / pairs / triples
    for (int r = 0; r < NUM_RANKS; ++r) {
        if (hand[r] >= 1) {
            Combo c;
            c.type = Combo::Type::SINGLE;
            c.main = {{1, r}};
            c.main_rank = r;
            c.length = 1;
            push_combo(std::move(c));
        }
        if (hand[r] >= 2) {
            Combo c;
            c.type = Combo::Type::PAIR;
            c.main = {{2, r}};
            c.main_rank = r;
            c.length = 1;
            push_combo(std::move(c));
        }
        if (hand[r] >= 3) {
            Combo c;
            c.type = Combo::Type::TRIPLE;
            c.main = {{3, r}};
            c.main_rank = r;
            c.length = 1;
            push_combo(std::move(c));
        }
    }

    // Triple with attachments
    for (int tr = 0; tr < NUM_RANKS; ++tr) {
        if (hand[tr] < 3) continue;
        for (int s = 0; s < NUM_RANKS; ++s) {
            if (s == tr) continue;
            if (hand[s] >= 1) {
                Combo c;
                c.type = Combo::Type::TRIPLE_SINGLE;
                c.main = {{3, tr}};
                c.kickers = {{1, s}};
                c.main_rank = tr;
                c.length = 1;
                push_combo(std::move(c));
            }
            if (hand[s] >= 2) {
                Combo c;
                c.type = Combo::Type::TRIPLE_PAIR;
                c.main = {{3, tr}};
                c.kickers = {{2, s}};
                c.main_rank = tr;
                c.length = 1;
                push_combo(std::move(c));
            }
        }
    }

    // Straights
    collect_runs<Game>(hand, 1, 5, SEQ_MAX_RANK, Combo::Type::STRAIGHT, 0, combos);
    collect_runs<Game>(hand, 2, 3, SEQ_MAX_RANK, Combo::Type::STRAIGHT_PAIR, 0, combos);
    collect_runs<Game>(hand, 3, 2, SEQ_MAX_RANK, Combo::Type::STRAIGHT_TRIPLE, MAX_SOLO_KICKERS, combos);

    // Quad with attachments
    for (int r = 0; r < NUM_RANKS; ++r) {
        if (hand[r] < 4) continue;
        // Two singles (may repeat if counts allow)
        std::vector<int> single_ranks, single_counts;
        for (int s = 0; s < NUM_RANKS; ++s) {
            if (s == r) continue;
            if (hand[s] > 0) {
                single_ranks.push_back(s);
                single_counts.push_back(hand[s]);
            }
        }
        std::vector<std::vector<std::pair<int, int>>> single_choices;
        std::vector<std::pair<int, int>> cur;
        choose_units_with_counts(single_ranks, single_counts, 2, 1, 0, cur, single_choices);
        for (auto wings : single_choices) {
            Combo c;
            c.type = Combo::Type::FOUR_TWO_SINGLE;
            c.main = {{4, r}};
            sort_groups<Game>(wings);
            c.kickers = std::move(wings);
            c.main_rank = r;
            c.length = 1;
            push_combo(std::move(c));
        }

        // Two pairs (distinct ranks)
        std::vector<int> pair_ranks;
        for (int s = 0; s < NUM_RANKS; ++s) {
            if (s == r) continue;
            if (hand[s] >= 2) pair_ranks.push_back(s);
        }
        std::vector<std::vector<int>> pair_choices;
        std::vector<int> cur_pair;
        choose_distinct_ranks(pair_ranks, 2, 0, cur_pair, pair_choices);
        for (auto ranks_choice : pair_choices) {
            Combo c;
            c.type = Combo::Type::FOUR_TWO_PAIR;
            c.main = {{4, r}};
            for (int pr : ranks_choice)
                c.kickers.emplace_back(2, pr);
            c.main_rank = r;
            c.length = 1;
            push_combo(std::move(c));
        }
    }

    // Planes (triple chains) with/without wings
    HandCounts triple_mask{};
    for (int i = 0; i < NUM_RANKS; ++i)
        triple_mask[i] = hand[i] >= 3 ? 1 : 0;
    int start = 0;
    const int max_seq_rank = SEQ_MAX_RANK;
    while (start <= max_seq_rank) {
        while (start <= max_seq_rank && triple_mask[start] == 0)
            ++start;
        if (start > max_seq_rank) break;
        int end = start;
        while (end <= max_seq_rank && triple_mask[end] == 1)
            ++end;
        int run_length = end - start;
        if (run_length >= 2) {
            for (int len = 2; len <= run_length && len <= MAX_SOLO_KICKERS; ++len) {
                for (int s = start; s + len <= end; ++s) {
                    std::vector<int> triple_ranks;
                    for (int r = s; r < s + len; ++r)
                        triple_ranks.push_back(r);

                    Combo plain;
                    plain.type = Combo::Type::STRAIGHT_TRIPLE;
                    plain.length = len;
                    plain.main_rank = s + len - 1;
                    for (int r : triple_ranks)
                        plain.main.emplace_back(3, r);
                    push_combo(std::move(plain));

                    // Wings: singles
                    HandCounts available = hand;
                    for (int r : triple_ranks)
                        available[r] = 0; // exclude all cards from triple ranks
                    std::vector<int> wing_ranks, wing_counts;
                    for (int r = 0; r < NUM_RANKS; ++r) {
                        if (available[r] > 0) {
                            wing_ranks.push_back(r);
                            wing_counts.push_back(available[r]);
                        }
                    }
                    std::vector<std::vector<std::pair<int, int>>> single_wings;
                    std::vector<std::pair<int, int>> wing_cur;
                    choose_units_with_counts(wing_ranks, wing_counts, len, 1, 0, wing_cur, single_wings);
                    for (auto wings : single_wings) {
                        Combo c;
                        c.type = Combo::Type::PLANE_SINGLE;
                        c.length = len;
                        c.main_rank = s + len - 1;
                        for (int r : triple_ranks)
                            c.main.emplace_back(3, r);
                        sort_groups<Game>(wings);
                        c.kickers = std::move(wings);
                        push_combo(std::move(c));
                    }

                    // Wings: pairs
                    std::vector<int> pair_r;
                    for (int r = 0; r < NUM_RANKS; ++r) {
                        if (std::find(triple_ranks.begin(), triple_ranks.end(), r) != triple_ranks.end()) continue;
                        if (hand[r] >= 2) pair_r.push_back(r);
                    }
                    std::vector<std::vector<int>> pair_wings;
                    std::vector<int> tmp;
                    choose_distinct_ranks(pair_r, len, 0, tmp, pair_wings);
                    for (auto wings : pair_wings) {
                        Combo c;
                        c.type = Combo::Type::PLANE_PAIR;
                        c.length = len;
                        c.main_rank = s + len - 1;
                        for (int r : triple_ranks)
                            c.main.emplace_back(3, r);
                        for (int pr : wings)
                            c.kickers.emplace_back(2, pr);
                        push_combo(std::move(c));
                    }
                }
            }
        }
        start = end + 1;
    }

    return combos;
}

template<typename Config>
typename DouDiZhuGame<Config>::Sequence DouDiZhuGame<Config>::build_sequence(const Combo& combo, int player) const {
    using Token = typename DouDiZhuGame<Config>::Token;
    Sequence seq;
    seq.combo = combo;
    const int player_tok = Token::encode_player(player);

    auto add_step = [&](std::vector<int> tokens, std::vector<std::pair<int, int>> apply, bool completes) {
        seq.steps.push_back({std::move(tokens), std::move(apply), completes});
    };

    using ActionClass = typename Token::ActionClass;
    auto class_tok = [&](ActionClass cls) { return Token::encode_action_class(cls); };
    auto solo_tok = [&](int rank) {
        return std::vector<int>{player_tok, class_tok(Token::ActionClass::DISCARD_SOLO), Token::encode_rank(rank)};
    };
    auto pair_tok = [&](int rank) {
        return std::vector<int>{player_tok, class_tok(Token::ActionClass::DISCARD_PAIR), Token::encode_rank(rank)};
    };

    switch (combo.type) {
    case Combo::Type::PASS: {
        add_step({player_tok, Token::encode_pass()}, {}, true);
        break;
    }
    case Combo::Type::ROCKET: {
        add_step({player_tok, class_tok(Token::ActionClass::DISCARD_ROCKET)}, combo.main, true);
        break;
    }
    case Combo::Type::SINGLE: {
        add_step({player_tok, class_tok(Token::ActionClass::DISCARD_SOLO), Token::encode_rank(combo.main_rank)}, combo.main, true);
        break;
    }
    case Combo::Type::PAIR: {
        add_step({player_tok, class_tok(Token::ActionClass::DISCARD_PAIR), Token::encode_rank(combo.main_rank)}, combo.main, true);
        break;
    }
    case Combo::Type::TRIPLE:
    case Combo::Type::TRIPLE_SINGLE:
    case Combo::Type::TRIPLE_PAIR: {
        std::vector<int> base{player_tok, class_tok(Token::ActionClass::DISCARD_TRIO), Token::encode_rank(combo.main_rank)};
        add_step(base, {}, false);
        if (combo.type == Combo::Type::TRIPLE) {
            std::vector<int> tok = base;
            tok.push_back(Token::encode_no_kicker());
            add_step(tok, combo.main, true);
        } else if (combo.type == Combo::Type::TRIPLE_SINGLE) {
            std::vector<int> tok = base;
            tok.push_back(Token::encode_solo_kicker_count(1));
            add_step(tok, combo.main, false);
            int kick_rank = combo.kickers.front().second;
            add_step(solo_tok(kick_rank), {{1, kick_rank}}, true);
        } else {
            std::vector<int> tok = base;
            tok.push_back(Token::encode_pair_kicker_count(1));
            add_step(tok, combo.main, false);
            int kick_rank = combo.kickers.front().second;
            add_step(pair_tok(kick_rank), {{2, kick_rank}}, true);
        }
        break;
    }
    case Combo::Type::BOMB:
    case Combo::Type::FOUR_TWO_SINGLE:
    case Combo::Type::FOUR_TWO_PAIR: {
        std::vector<int> base{player_tok, class_tok(Token::ActionClass::DISCARD_QUAD), Token::encode_rank(combo.main_rank)};
        add_step(base, {}, false);
        if (combo.type == Combo::Type::BOMB) {
            std::vector<int> tok = base;
            tok.push_back(Token::encode_no_kicker());
            add_step(tok, combo.main, true);
        } else if (combo.type == Combo::Type::FOUR_TWO_SINGLE) {
            std::vector<int> tok = base;
            tok.push_back(Token::encode_solo_kicker_count(2));
            add_step(tok, combo.main, false);
            for (size_t i = 0; i < combo.kickers.size(); ++i) {
                int r = combo.kickers[i].second;
                add_step(solo_tok(r), {{1, r}}, i + 1 == combo.kickers.size());
            }
        } else {
            std::vector<int> tok = base;
            tok.push_back(Token::encode_pair_kicker_count(2));
            add_step(tok, combo.main, false);
            for (size_t i = 0; i < combo.kickers.size(); ++i) {
                int r = combo.kickers[i].second;
                add_step(pair_tok(r), {{2, r}}, i + 1 == combo.kickers.size());
            }
        }
        break;
    }
    case Combo::Type::STRAIGHT: {
        int start = combo.main_rank - combo.length + 1;
        std::vector<int> first{player_tok, class_tok(Token::ActionClass::DISCARD_STRAIGHT), Token::encode_seq_from(start)};
        std::vector<int> second = first;
        second.push_back(Token::encode_seq_to(combo.main_rank));
        add_step(first, {}, false);
        add_step(second, combo.main, true);
        break;
    }
    case Combo::Type::STRAIGHT_PAIR: {
        int start = combo.main_rank - combo.length + 1;
        std::vector<int> first{player_tok, class_tok(Token::ActionClass::DISCARD_PAIR_SISTERS), Token::encode_seq_from(start)};
        std::vector<int> second = first;
        second.push_back(Token::encode_seq_to(combo.main_rank));
        add_step(first, {}, false);
        add_step(second, combo.main, true);
        break;
    }
    case Combo::Type::STRAIGHT_TRIPLE:
    case Combo::Type::PLANE_SINGLE:
    case Combo::Type::PLANE_PAIR: {
        int start = combo.main_rank - combo.length + 1;
        std::vector<int> first{player_tok, class_tok(Token::ActionClass::DISCARD_TRIO_CHAIN), Token::encode_seq_from(start)};
        std::vector<int> mid = first;
        mid.push_back(Token::encode_seq_to(combo.main_rank));
        if (combo.type == Combo::Type::STRAIGHT_TRIPLE) {
            add_step(first, {}, false);
            add_step(mid, combo.main, true);
        } else if (combo.type == Combo::Type::PLANE_SINGLE) {
            mid.push_back(Token::encode_solo_kicker_count(static_cast<int>(combo.kickers.size())));
            add_step(first, {}, false);
            add_step(mid, combo.main, false);
            for (size_t i = 0; i < combo.kickers.size(); ++i) {
                int r = combo.kickers[i].second;
                add_step(solo_tok(r), {{1, r}}, i + 1 == combo.kickers.size());
            }
        } else {
            mid.push_back(Token::encode_pair_kicker_count(static_cast<int>(combo.kickers.size())));
            add_step(first, {}, false);
            add_step(mid, combo.main, false);
            for (size_t i = 0; i < combo.kickers.size(); ++i) {
                int r = combo.kickers[i].second;
                add_step(pair_tok(r), {{2, r}}, i + 1 == combo.kickers.size());
            }
        }
        break;
    }
    }
    return seq;
}

} // namespace nanopokerlib::doudizhu
#endif
