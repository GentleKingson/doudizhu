#pragma once

#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace nanopokerlib::semantic_string {

struct TaggedSegment {
    std::string tag;
    char op{'\0'};
    std::string payload;
};

inline std::string trim_ascii_whitespace(std::string_view in) {
    size_t start = 0;
    size_t end = in.size();
    while (start < end && std::isspace(static_cast<unsigned char>(in[start])))
        ++start;
    while (end > start && std::isspace(static_cast<unsigned char>(in[end - 1])))
        --end;
    return std::string(in.substr(start, end - start));
}

inline std::string normalize_compact_upper(std::string_view in) {
    std::string out;
    out.reserve(in.size());
    for (char ch : in) {
        if (std::isspace(static_cast<unsigned char>(ch))) continue;
        if (std::isalpha(static_cast<unsigned char>(ch))) {
            out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

inline std::vector<std::string> split_segments(std::string_view text) {
    std::vector<std::string> out;
    std::string current;
    current.reserve(text.size());
    for (size_t i = 0; i <= text.size(); ++i) {
        if (i == text.size() || text[i] == '|') {
            std::string seg = trim_ascii_whitespace(current);
            current.clear();
            if (!seg.empty()) { out.push_back(std::move(seg)); }
            continue;
        }
        current.push_back(text[i]);
    }
    return out;
}

inline std::string join_segments(const std::vector<std::string>& segments) {
    if (segments.empty()) return "";
    size_t total_chars = segments.size() - 1;
    for (const auto& seg : segments) {
        total_chars += seg.size();
    }

    std::string out;
    out.reserve(total_chars);
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i > 0) { out.push_back('|'); }
        out += segments[i];
    }
    return out;
}

inline std::optional<TaggedSegment> try_parse_tagged_segment(std::string_view segment) {
    const std::string trimmed = trim_ascii_whitespace(segment);
    if (trimmed.empty()) return std::nullopt;

    const size_t plus_pos = trimmed.find('+');
    const size_t minus_pos = trimmed.find('-');
    const size_t colon_pos = trimmed.find(':');
    const size_t split_pos = std::min(
        std::min(plus_pos == std::string::npos ? trimmed.size() : plus_pos, minus_pos == std::string::npos ? trimmed.size() : minus_pos),
        colon_pos == std::string::npos ? trimmed.size() : colon_pos
    );
    if (split_pos == trimmed.size()) { return std::nullopt; }

    TaggedSegment out;
    out.op = trimmed[split_pos];
    out.tag = trim_ascii_whitespace(std::string_view(trimmed).substr(0, split_pos));
    out.payload = trim_ascii_whitespace(std::string_view(trimmed).substr(split_pos + 1));
    return out;
}

inline TaggedSegment parse_tagged_segment(std::string_view segment, const std::string& context) {
    auto parsed = try_parse_tagged_segment(segment);
    if (!parsed.has_value()) {
        throw std::runtime_error(context + ": expected '<tag>+<payload>', '<tag>-<payload>', or '<tag>:<payload>' segment.");
    }
    return *parsed;
}

} // namespace nanopokerlib::semantic_string
