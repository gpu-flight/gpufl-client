// Ticks a counter through the INSTALLED package. The check script greps the
// line below, so a silent link against a stub or a handle that failed to
// register fails the test rather than passing quietly.
#include <cstdio>

#include "gpufl.hpp"

int main() {
    auto tokens = gpufl::counter("tokens");
    for (int i = 0; i < 1000; ++i) tokens.add(8);
    std::printf("consumer ok valid=%d\n", tokens.valid() ? 1 : 0);
    return tokens.valid() ? 0 : 1;
}
