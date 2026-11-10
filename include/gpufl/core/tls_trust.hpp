#pragma once

#include <functional>
#include <string>

// Where gpufl's own HTTPS clients (uploads, the session-complete signal, the
// backend version probe) find the host's CA certificates.
//
// Source builds link the system OpenSSL, which already knows its distro's CA
// store, so these helpers return nothing there and httplib keeps calling
// SSL_CTX_set_default_verify_paths. Linux wheels link a private static OpenSSL
// (GPUFL_PRIVATE_OPENSSL; see scripts/build-static-openssl.sh), whose built-in
// default directory cannot match every distro, so the wheel looks for the
// host's bundle itself.

namespace gpufl::tls {

struct CaPaths {
    std::string file;  // a PEM bundle, or empty
    std::string dir;   // a hashed certificate directory, or empty
    bool empty() const { return file.empty() && dir.empty(); }
};

// The host lookups chooseCaPaths makes, injectable for tests.
struct CaProbe {
    std::function<const char*(const char*)> getenv;
    // A regular, readable, non-empty file.
    std::function<bool(const std::string&)> is_file;
    std::function<bool(const std::string&)> is_dir;
};

// The CA locations to hand httplib, in this order:
//  1. nothing when SSL_CERT_FILE or SSL_CERT_DIR is set, even to "": OpenSSL
//     then reads both variables itself, so the user's choice always wins;
//  2. the first bundle file a known distro layout provides;
//  3. $CONDA_PREFIX/ssl/cacert.pem;
//  4. the first hashed certificate directory;
//  5. nothing, which leaves OpenSSL's own default.
CaPaths chooseCaPaths(const CaProbe& probe);

// True in a Linux build linked with the private static OpenSSL.
bool privateOpenSslBuild();

// chooseCaPaths over this host in a private-OpenSSL build; empty otherwise.
CaPaths caPathsForThisBuild();

// One line for diagnostics, for example
// "OpenSSL 3.5.9 ... (private, static); CA: /etc/ssl/certs/ca-certificates.crt".
std::string describeTls();

// What to tell the user when the server's certificate failed verification:
// "TLS certificate verification failed; <describeTls()>; <what to try>".
std::string verificationFailureHint();

}  // namespace gpufl::tls
