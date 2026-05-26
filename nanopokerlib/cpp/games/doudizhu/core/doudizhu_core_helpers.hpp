#pragma once

#include "games/doudizhu/doudizhu_core.hpp"

#ifdef NANOPOKERLIB_IMPLEMENTATION
namespace nanopokerlib::doudizhu {

template<typename Game>
inline int next_player(int p) {
    return (p + 1) % Game::NUM_PLAYERS;
}

// Utility helpers ----------------------------------------------------------------
template<typename Game>
static void sort_groups(std::vector<std::pair<int, int>>& groups) {
    std::sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    });
}

template<typename Game>
static void collect_runs(
    const typename Game::HandCounts& hand, int min_count, int min_length, int max_rank_inclusive, typename Game::Combo::Type type,
    int max_length_inclusive, std::vector<typename Game::Combo>& out
) {
    int start = 0;
    while (start <= max_rank_inclusive) {
        while (start <= max_rank_inclusive && hand[start] < min_count) {
            ++start;
        }
        if (start > max_rank_inclusive) break;
        int end = start;
        while (end <= max_rank_inclusive && hand[end] >= min_count) {
            ++end;
        }
        int run_length = end - start;
        const int max_len = (max_length_inclusive > 0) ? std::min(run_length, max_length_inclusive) : run_length;
        for (int len = min_length; len <= max_len; ++len) {
            for (int s = start; s + len <= end; ++s) {
                typename Game::Combo combo;
                combo.type = type;
                combo.length = len;
                combo.main_rank = s + len - 1;
                for (int r = s; r < s + len; ++r) {
                    combo.main.emplace_back(min_count, r);
                }
                out.push_back(std::move(combo));
            }
        }
        start = end + 1;
    }
}

static void choose_units_with_counts(
    const std::vector<int>& ranks, const std::vector<int>& counts, int need, int unit_size, size_t index,
    std::vector<std::pair<int, int>>& current, std::vector<std::vector<std::pair<int, int>>>& out
) {
    if (need == 0) {
        out.push_back(current);
        return;
    }
    if (index >= ranks.size()) return;
    int max_units = counts[index] / unit_size;
    for (int take = 0; take <= std::min(max_units, need); ++take) {
        for (int i = 0; i < take; ++i)
            current.emplace_back(unit_size, ranks[index]);
        choose_units_with_counts(ranks, counts, need - take, unit_size, index + 1, current, out);
        for (int i = 0; i < take; ++i)
            current.pop_back();
    }
}

static void choose_distinct_ranks(
    const std::vector<int>& ranks, int need, size_t index, std::vector<int>& current, std::vector<std::vector<int>>& out
) {
    if (need == 0) {
        out.push_back(current);
        return;
    }
    if (index >= ranks.size()) return;
    choose_distinct_ranks(ranks, need, index + 1, current, out);
    current.push_back(ranks[index]);
    choose_distinct_ranks(ranks, need - 1, index + 1, current, out);
    current.pop_back();
}

} // namespace nanopokerlib::doudizhu
#endif
