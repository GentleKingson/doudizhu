#pragma once

namespace nanopokerlib::math {
constexpr int max_pow(int base, int exp) {
    int value = 1;
    for (int i = 0; i < exp; ++i) {
        value *= base;
    }
    return value;
}
} // namespace nanopokerlib::math

