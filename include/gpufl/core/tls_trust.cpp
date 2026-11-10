#include "gpufl/core/tls_trust.hpp"

#include <cstdlib>
#include <mutex>

#if defined(__linux__)
#include <sys/stat.h>
#include <unistd.h>
#endif

#if GPUFL_HTTPLIB_TLS
#include <openssl/crypto.h>
#include <openssl/x509.h>
#endif

#include "gpufl/core/debug_logger.hpp"

namespace gpufl::tls {

namespace {

// Bundle files, most common layout first.
constexpr const char* kCaFiles[] = {
    "/etc/ssl/certs/ca-certificates.crt",                 // Debian, Ubuntu, Alpine, Arch, Gentoo
    "/etc/pki/tls/certs/ca-bundle.crt",                   // RHEL, Rocky, Alma, Fedora, Amazon Linux
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",  // RHEL, Fedora
    "/etc/ssl/ca-bundle.pem",                             // openSUSE, SLES
    "/etc/pki/tls/cacert.pem",
    "/etc/ssl/cert.pem",
};

constexpr const char* kCaDirs[] = {
    "/etc/ssl/certs",
    "/etc/pki/tls/certs",
};

#if defined(__linux__)
bool hostIsFile(const std::string& path) {
    struct stat st {};
    return !path.empty() && ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) &&
           st.st_size > 0 && ::access(path.c_str(), R_OK) == 0;
}

bool hostIsDir(const std::string& path) {
    struct stat st {};
    return !path.empty() && ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
#endif

}  // namespace

CaPaths chooseCaPaths(const CaProbe& probe) {
    if (probe.getenv("SSL_CERT_FILE") != nullptr || probe.getenv("SSL_CERT_DIR") != nullptr) {
        return {};
    }
    for (const char* file : kCaFiles) {
        if (probe.is_file(file)) return {file, ""};
    }
    if (const char* conda = probe.getenv("CONDA_PREFIX"); conda != nullptr && *conda != '\0') {
        const std::string file = std::string(conda) + "/ssl/cacert.pem";
        if (probe.is_file(file)) return {file, ""};
    }
    for (const char* dir : kCaDirs) {
        if (probe.is_dir(dir)) return {"", dir};
    }
    return {};
}

bool privateOpenSslBuild() {
#if defined(__linux__) && defined(GPUFL_PRIVATE_OPENSSL) && GPUFL_PRIVATE_OPENSSL
    return true;
#else
    return false;
#endif
}

CaPaths caPathsForThisBuild() {
#if defined(__linux__)
    if (!privateOpenSslBuild()) return {};
    CaProbe probe;
    probe.getenv = [](const char* name) -> const char* { return std::getenv(name); };
    probe.is_file = hostIsFile;
    probe.is_dir = hostIsDir;
    CaPaths paths = chooseCaPaths(probe);
    if (!paths.empty()) {
        static std::once_flag logged;
        std::call_once(logged, [&paths] {
            GFL_LOG_DEBUG("[tls] CA bundle: ", paths.file.empty() ? paths.dir : paths.file);
        });
    }
    return paths;
#else
    return {};
#endif
}

std::string describeTls() {
#if GPUFL_HTTPLIB_TLS
    std::string out = OpenSSL_version(OPENSSL_VERSION);
    out += privateOpenSslBuild() ? " (private, static); CA: " : "; CA: ";
    if (const char* file = std::getenv("SSL_CERT_FILE")) {
        out += std::string("SSL_CERT_FILE=") + file;
    } else if (const char* dir = std::getenv("SSL_CERT_DIR")) {
        out += std::string("SSL_CERT_DIR=") + dir;
    } else if (const CaPaths paths = caPathsForThisBuild(); !paths.empty()) {
        out += paths.file.empty() ? paths.dir : paths.file;
    } else {
        out += std::string("OpenSSL default (") + X509_get_default_cert_file() + ")";
    }
    return out;
#else
    return "no TLS (HTTP only)";
#endif
}

std::string verificationFailureHint() {
    const bool user_set = std::getenv("SSL_CERT_FILE") != nullptr ||
                          std::getenv("SSL_CERT_DIR") != nullptr;
    return "TLS certificate verification failed; " + describeTls() +
           (user_set ? "; check that SSL_CERT_FILE or SSL_CERT_DIR holds the server's CA"
                     : "; set SSL_CERT_FILE to your CA bundle");
}

}  // namespace gpufl::tls
