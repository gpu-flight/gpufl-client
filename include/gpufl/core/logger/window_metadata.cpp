#include "gpufl/core/logger/window_metadata.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

#include <zlib.h>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/json/json.hpp"
#include "gpufl/core/logger/log_salvage.hpp"

namespace gpufl {
namespace fs = std::filesystem;
namespace {

std::string generateUuidV4() {
    std::array<unsigned char, 16> bytes{};
    std::random_device random;
    for (auto& byte : bytes) {
        byte = static_cast<unsigned char>(random());
    }
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0fU) | 0x40U);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3fU) | 0x80U);

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out << '-';
        out << std::setw(2) << static_cast<unsigned>(bytes[i]);
    }
    return out.str();
}

std::string jsonEscape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char c : value) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += c; break;
        }
    }
    return escaped;
}

bool payloadFingerprint(const fs::path& payload, std::uint64_t& bytes,
                        std::uint32_t& checksum) {
    std::ifstream input(payload, std::ios::binary);
    if (!input) return false;
    uLong crc = crc32(0L, Z_NULL, 0);
    bytes = 0;
    std::array<unsigned char, 64 * 1024> buffer{};
    while (input) {
        input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        const auto count = input.gcount();
        if (count <= 0) continue;
        crc = crc32(crc, buffer.data(), static_cast<uInt>(count));
        bytes += static_cast<std::uint64_t>(count);
    }
    if (input.bad()) return false;
    checksum = static_cast<std::uint32_t>(crc);
    return true;
}

bool existingMetadataMatches(
    const fs::path& metadata_path,
    const std::string& session_id,
    const std::string& channel,
    const std::size_t sequence,
    const std::string& published_name,
    const std::uint64_t payload_bytes,
    const std::uint32_t payload_crc32) {
    const auto metadata = json::loadFile(metadata_path.string());
    constexpr std::uint64_t kMissing =
        std::numeric_limits<std::uint64_t>::max();
    if (!metadata.is_object() ||
        metadata.value<std::string>("type", "") != "transport_window" ||
        metadata.value<std::string>("window_id", "").empty() ||
        metadata.value<std::string>("session_id", "") != session_id ||
        metadata.value<std::string>("channel", "") != channel ||
        metadata.value<std::uint64_t>("window_sequence", kMissing) !=
            sequence ||
        metadata.value<std::string>("payload_file", "") != published_name ||
        metadata.value<std::uint64_t>("payload_bytes", kMissing) !=
            payload_bytes ||
        metadata.value<std::uint64_t>("payload_crc32", kMissing) !=
            payload_crc32) {
        GFL_LOG_ERROR(
            "[Logger] immutable metadata '", metadata_path.string(),
            "' does not describe the payload waiting to claim its window "
            "sequence; refusing to publish either identity.");
        return false;
    }
    return true;
}

}  // namespace

fs::path windowMetadataPath(const fs::path& session_dir,
                            const std::string& channel,
                            const std::size_t sequence) {
    return session_dir /
           (".gpufl-window." + channel + "." + std::to_string(sequence) +
            ".json");
}

bool ensureWindowMetadata(const fs::path& session_dir,
                          const std::string& session_id,
                          const std::string& channel,
                          const std::size_t sequence,
                          const fs::path& payload,
                          const WindowTiming& timing) {
    return ensureWindowMetadata(session_dir, session_id, channel, sequence,
                                payload, payload.filename().string(), timing);
}

bool ensureWindowMetadata(const fs::path& session_dir,
                          const std::string& session_id,
                          const std::string& channel,
                          const std::size_t sequence,
                          const fs::path& fingerprint_source,
                          const std::string& published_name,
                          const WindowTiming& timing) {
    const fs::path target =
        windowMetadataPath(session_dir, channel, sequence);
    std::uint64_t payload_bytes = 0;
    std::uint32_t payload_crc32 = 0;
    // Read the file that exists NOW; record the name it will have when a
    // consumer sees it. A publish is a rename of these exact bytes, so the
    // fingerprint stays valid across it.
    if (!payloadFingerprint(fingerprint_source, payload_bytes,
                            payload_crc32)) {
        GFL_LOG_ERROR("[Logger] cannot fingerprint window payload '",
                      fingerprint_source.string(),
                      "'; metadata not published.");
        return false;
    }

    std::error_code state_ec;
    if (fs::is_regular_file(target, state_ec)) {
        return existingMetadataMatches(
            target, session_id, channel, sequence, published_name,
            payload_bytes, payload_crc32);
    }

    WindowMetadata metadata;
    metadata.window_id = generateUuidV4();
    metadata.session_id = session_id;
    metadata.channel = channel;
    metadata.window_sequence = sequence;
    metadata.timing = timing;
    metadata.created_wall_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    metadata.payload_file = published_name;
    metadata.payload_bytes = payload_bytes;
    metadata.payload_crc32 = payload_crc32;

    const fs::path partial =
        target.string() + ".part." + metadata.window_id;
    {
        std::ofstream out(partial, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out << "{\"schema_version\":1,\"type\":\"transport_window\","
            << "\"window_id\":\"" << metadata.window_id << "\","
            << "\"session_id\":\"" << jsonEscape(metadata.session_id)
            << "\",\"channel\":\"" << jsonEscape(metadata.channel)
            << "\",\"window_sequence\":" << metadata.window_sequence
            << ",\"opened_mono_ms\":" << metadata.timing.opened_mono_ms
            << ",\"closed_mono_ms\":" << metadata.timing.closed_mono_ms
            << ",\"created_wall_ms\":" << metadata.created_wall_ms
            << ",\"payload_file\":\""
            << jsonEscape(metadata.payload_file)
            << "\",\"payload_bytes\":" << metadata.payload_bytes
            << ",\"payload_crc32\":" << metadata.payload_crc32 << "}\n";
        out.flush();
        if (!out.good()) {
            out.close();
            fs::remove(partial, state_ec);
            return false;
        }
    }

    std::error_code move_ec;
    const auto moved = moveFileNoReplace(partial, target, move_ec);
    if (moved == MoveFileNoReplaceResult::Moved) {
        fs::remove(partial, state_ec);
        return true;
    }
    if (moved == MoveFileNoReplaceResult::DestinationExists) {
        fs::remove(partial, state_ec);
        std::error_code target_ec;
        if (fs::is_regular_file(target, target_ec)) {
            return existingMetadataMatches(
                target, session_id, channel, sequence, published_name,
                payload_bytes, payload_crc32);
        }
        GFL_LOG_ERROR("[Logger] immutable window metadata path '",
                      target.string(),
                      "' exists but is not a regular file.");
        return false;
    }
    fs::remove(partial, state_ec);
    GFL_LOG_ERROR("[Logger] cannot publish immutable window metadata '",
                  target.string(), "': ", move_ec.message());
    return false;
}

void scanWindowMetadataMaxSequence(const fs::path& session_dir,
                                   const std::string& channel,
                                   std::size_t& max_sequence) {
    const std::string prefix = ".gpufl-window." + channel + ".";
    constexpr const char* suffix = ".json";
    std::error_code ec;
    if (!fs::is_directory(session_dir, ec)) return;
    for (const auto& entry : fs::directory_iterator(session_dir, ec)) {
        std::error_code file_ec;
        if (!entry.is_regular_file(file_ec)) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) != 0 ||
            name.size() <= prefix.size() + std::char_traits<char>::length(suffix) ||
            name.compare(name.size() - std::char_traits<char>::length(suffix),
                         std::char_traits<char>::length(suffix), suffix) != 0) {
            continue;
        }
        const std::size_t count =
            name.size() - prefix.size() -
            std::char_traits<char>::length(suffix);
        try {
            const auto sequence =
                static_cast<std::size_t>(
                    std::stoull(name.substr(prefix.size(), count)));
            max_sequence = std::max(max_sequence, sequence);
        } catch (...) {
            // Ignore unrelated dot-files.
        }
    }
}

}  // namespace gpufl
