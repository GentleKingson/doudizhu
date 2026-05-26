#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace nanopokerlib {

struct LiveHistoryTracker {
    std::string live_history_str{};
    std::string pending_history_str{};
    std::uint64_t pending_history_prev_fingerprint{fingerprint("")};
    std::size_t pending_history_prev_size{0};
    std::uint64_t history_transition_serial{0};
    std::uint64_t history_consumed_serial{0};

    static std::uint64_t fingerprint(std::string_view history) {
        std::uint64_t hash = 1469598103934665603ull;
        for (unsigned char ch : history) {
            hash ^= static_cast<std::uint64_t>(ch);
            hash *= 1099511628211ull;
        }
        return hash;
    }

    void reset() {
        live_history_str.clear();
        pending_history_str.clear();
        pending_history_prev_fingerprint = fingerprint("");
        pending_history_prev_size = 0;
        history_transition_serial = 0;
        history_consumed_serial = 0;
    }

    void record(const std::string& next_history_str, bool reset_live_history_after_record = false) {
        pending_history_prev_fingerprint = fingerprint(live_history_str);
        pending_history_prev_size = live_history_str.size();
        pending_history_str = next_history_str;
        history_transition_serial += 1;
        if (reset_live_history_after_record) {
            live_history_str.clear();
        } else {
            live_history_str = next_history_str;
        }
    }

    std::string advance(const std::string& history_str) {
        if (history_transition_serial == history_consumed_serial) {
            throw std::runtime_error("advance_history_str requires an unconsumed reset()/step() transition.");
        }
        if (
            history_str.size() != pending_history_prev_size ||
            fingerprint(history_str) != pending_history_prev_fingerprint
        ) {
            throw std::runtime_error("advance_history_str input does not match the expected prior canonical history.");
        }
        history_consumed_serial = history_transition_serial;
        return pending_history_str;
    }
};

} // namespace nanopokerlib
