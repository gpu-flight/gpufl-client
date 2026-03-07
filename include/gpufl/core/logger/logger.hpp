#pragma once

#include <cstddef>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "gpufl/core/model/serializable.hpp"

namespace gpufl {

class IFileCompressor;
class LogFileRotator;

class Logger {
   public:
    struct Options {
        std::string base_path;
        std::size_t rotate_bytes = 64 * 1024 * 1024;  // 64 MiB default
        std::size_t max_files = 100;
        bool compress_rotated = true;
        bool flush_always = false;
        int system_sample_rate_ms = 0;
    };

    Logger();
    ~Logger();

    bool open(const Options& opt);
    void close();

    void write(const IJsonSerializable& model);

   private:
    class LogChannel {
       public:
        LogChannel(std::string name, Options opt);
        ~LogChannel();

        void write(const std::string& line);
        void close();
        bool isOpen() const;

       private:
        void ensureOpenLocked();
        void rotateLocked();
        void closeLocked();

        std::string name_;
        Options opt_;
        std::unique_ptr<IFileCompressor> compressor_;
        std::unique_ptr<LogFileRotator> rotator_;

        std::ofstream stream_;
        size_t current_bytes_ = 0;

        mutable std::mutex mu_;
        bool opened_ = false;
    };

    LogChannel* resolveChannel(Channel ch)const;

    Options opt_;
    std::unique_ptr<LogChannel> chanDevice_;
    std::unique_ptr<LogChannel> chanScope_;
    std::unique_ptr<LogChannel> chanSystem_;
};

}  // namespace gpufl
