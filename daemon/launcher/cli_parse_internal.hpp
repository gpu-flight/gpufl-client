#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gpufl::launcher::detail {

// The spelling of one option token, e.g. `--flag` or `--flag=value`.
struct FlagBreak {
    std::string key;
    std::optional<std::string> inline_value;
};

FlagBreak splitFlag(const std::string& token);

// Consume the value of flag from either its inline spelling or the next argv
// token. Returns an empty string on success, otherwise a user-facing error.
std::string takeFlagValue(const FlagBreak& flag,
                          const std::vector<std::string>& argv,
                          std::size_t& index,
                          std::string& value);

std::string trim(const std::string& value);

bool parseDurationMs(const std::string& value, std::int64_t& out_ms);
bool parseUint64(const std::string& value, std::uint64_t& out);
bool parseNonNegativeInt(const std::string& value, int& out);
bool parsePositiveInt(const std::string& value, int& out);
bool parseByteSize(const std::string& value, std::uint64_t& out);

}  // namespace gpufl::launcher::detail
