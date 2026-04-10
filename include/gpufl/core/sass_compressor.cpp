#include "gpufl/core/sass_compressor.hpp"

#include <cctype>
#include <sstream>

namespace gpufl {

std::string SassCompressor::normalizeForCompare(const std::string& sass) {
    // Replace sequences of digits (optionally with sign, decimal point, 'e'
    // exponent, and 'x' for hex prefix) with a single '#' placeholder.
    // This collapses:
    //   integer literals:  42, 0x1F, -3
    //   float literals:    1.0099999904632568359, 9.999e-05
    //   hex addresses:     0x380
    //
    // Register names like R0, UR4, SR_TID contain digits but are preceded
    // by a letter, so we only replace digit sequences that are NOT preceded
    // by a letter.
    std::string out;
    out.reserve(sass.size());

    size_t i = 0;
    const size_t len = sass.size();
    while (i < len) {
        char c = sass[i];

        // Check if this is the start of a numeric literal.
        // A number starts with a digit, or a sign followed by a digit,
        // but NOT if the previous char is a letter (e.g. "R0", "UR4").
        bool isNumStart = false;
        if (std::isdigit(static_cast<unsigned char>(c))) {
            isNumStart = out.empty() ||
                         !std::isalpha(static_cast<unsigned char>(out.back()));
        } else if ((c == '-' || c == '+') && i + 1 < len &&
                   std::isdigit(static_cast<unsigned char>(sass[i + 1]))) {
            // Signed literal like -3 or +1.5
            // Only treat as number if preceded by a non-alphanumeric
            isNumStart = out.empty() ||
                         (!std::isalnum(static_cast<unsigned char>(out.back())) &&
                          out.back() != '_');
        }

        if (isNumStart) {
            // Skip over the entire numeric literal
            if (c == '-' || c == '+') ++i;  // skip sign
            // Skip hex prefix
            if (i + 1 < len && sass[i] == '0' &&
                (sass[i + 1] == 'x' || sass[i + 1] == 'X')) {
                i += 2;
                while (i < len && std::isxdigit(static_cast<unsigned char>(sass[i])))
                    ++i;
            } else {
                // Decimal integer or float
                while (i < len && std::isdigit(static_cast<unsigned char>(sass[i])))
                    ++i;
                // Decimal point
                if (i < len && sass[i] == '.') {
                    ++i;
                    while (i < len &&
                           std::isdigit(static_cast<unsigned char>(sass[i])))
                        ++i;
                }
                // Exponent
                if (i < len && (sass[i] == 'e' || sass[i] == 'E')) {
                    ++i;
                    if (i < len && (sass[i] == '+' || sass[i] == '-')) ++i;
                    while (i < len &&
                           std::isdigit(static_cast<unsigned char>(sass[i])))
                        ++i;
                }
            }
            out += '#';
        } else {
            out += c;
            ++i;
        }
    }
    return out;
}

bool SassCompressor::isSameShape(const std::string& a, const std::string& b) {
    return normalizeForCompare(a) == normalizeForCompare(b);
}

std::vector<SassCompressor::CompressedEntry> SassCompressor::compress(
    const std::vector<std::pair<uint32_t, std::string>>& entries) {
    std::vector<CompressedEntry> result;
    if (entries.empty()) return result;
    result.reserve(entries.size());  // worst case: no compression

    size_t i = 0;
    while (i < entries.size()) {
        const auto& [pc, sass] = entries[i];
        const std::string normalized = normalizeForCompare(sass);

        // Count consecutive entries with the same normalized form
        size_t runEnd = i + 1;
        while (runEnd < entries.size() &&
               normalizeForCompare(entries[runEnd].second) == normalized) {
            ++runEnd;
        }

        result.push_back({pc, sass, static_cast<uint32_t>(runEnd - i)});
        i = runEnd;
    }
    return result;
}

}  // namespace gpufl
