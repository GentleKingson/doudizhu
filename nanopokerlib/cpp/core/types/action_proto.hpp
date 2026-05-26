#pragma once

#include <vector>

namespace nanopokerlib {

template<typename Game>
struct ActionProto {
    std::vector<int> tokens;
    float weight{};
    bool is_commit{true};
};

} // namespace nanopokerlib
