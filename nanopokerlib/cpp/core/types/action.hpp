#pragma once

#include <vector>

namespace nanopokerlib {

template<typename Game>
struct Action {
    std::vector<int> tokens;
};

} // namespace nanopokerlib
