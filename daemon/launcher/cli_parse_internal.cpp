#include "cli_parse_internal.hpp"

#include <charconv>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <system_error>

namespace gpufl::launcher::detail {

FlagBreak splitFlag(const std::string& token) {
    const auto equal = token.find('=');
    if (equal == std::string::npos) return {token, std::nullopt};
    return {token.substr(0, equal), token.substr(equal + 1)};
}

std::string takeFlagValue(const FlagBreak& flag,
                          const std::vector<std::string>& argv,
                          std::size_t& index,
                          std::string& value) {
    if (flag.inline_value) {
        value = *flag.inline_value;
        return {};
    }
    if (index + 1 >= argv.size()) return "missing value for " + flag.key;
    value = argv[++index];
    return {};
}

std::string trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t");
    if (begin == std::string::npos) return {};
    const auto end = value.find_last_not_of(" \t");
    return value.substr(begin, end - begin + 1);
}

bool parseDurationMs(const std::string& value, std::int64_t& out_ms) {
    if (value.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const double number = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || errno == ERANGE || !std::isfinite(number) ||
        number < 0) {
        return false;
    }

    const std::string unit = trim(end);
    double multiplier_ms;
    if (unit.empty() || unit == "s") multiplier_ms = 1000.0;
    else if (unit == "ms") multiplier_ms = 1.0;
    else if (unit == "m") multiplier_ms = 60.0 * 1000.0;
    else if (unit == "h") multiplier_ms = 60.0 * 60.0 * 1000.0;
    else return false;

    const double milliseconds = number * multiplier_ms;
    if (!std::isfinite(milliseconds) ||
        milliseconds >= static_cast<double>(
            (std::numeric_limits<std::int64_t>::max)()) ||
        (number > 0 && milliseconds < 1.0)) {
        return false;
    }
    out_ms = static_cast<std::int64_t>(milliseconds);
    return true;
}

bool parseUint64(const std::string& value, std::uint64_t& out) {
    const char* begin = value.data();
    const char* end = begin + value.size();
    while (begin != end &&
           std::isspace(static_cast<unsigned char>(*begin))) {
        ++begin;
    }
    if (begin != end && *begin == '+') ++begin;
    if (begin == end) return false;

    std::uint64_t parsed = 0;
    const auto result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end) {
        return false;
    }
    out = parsed;
    return true;
}

bool parseNonNegativeInt(const std::string& value, int& out) {
    std::uint64_t parsed = 0;
    if (!parseUint64(value, parsed) ||
        parsed > static_cast<std::uint64_t>((std::numeric_limits<int>::max)())) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

bool parsePositiveInt(const std::string& value, int& out) {
    return parseNonNegativeInt(value, out) && out > 0;
}

}  // namespace gpufl::launcher::detail
