#include <gtest/gtest.h>

#include <map>
#include <set>
#include <string>

#include "gpufl/core/tls_trust.hpp"

using gpufl::tls::CaPaths;
using gpufl::tls::CaProbe;
using gpufl::tls::chooseCaPaths;

namespace {

// A host made of an environment and the files and directories that exist.
struct FakeHost {
    std::map<std::string, std::string> env;
    std::set<std::string> files;
    std::set<std::string> dirs;

    CaProbe probe() const {
        CaProbe p;
        p.getenv = [this](const char* name) -> const char* {
            const auto it = env.find(name);
            return it == env.end() ? nullptr : it->second.c_str();
        };
        p.is_file = [this](const std::string& path) { return files.count(path) > 0; };
        p.is_dir = [this](const std::string& path) { return dirs.count(path) > 0; };
        return p;
    }
};

CaPaths choose(const FakeHost& host) { return chooseCaPaths(host.probe()); }

}  // namespace

TEST(TlsTrust, UbuntuUsesTheDebianBundle) {
    FakeHost host;
    host.files = {"/etc/ssl/certs/ca-certificates.crt"};
    host.dirs = {"/etc/ssl/certs"};
    const CaPaths ca = choose(host);
    EXPECT_EQ(ca.file, "/etc/ssl/certs/ca-certificates.crt");
    EXPECT_EQ(ca.dir, "");
}

TEST(TlsTrust, RhelPrefersCaBundleCrt) {
    FakeHost host;
    host.files = {"/etc/pki/tls/certs/ca-bundle.crt",
                  "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
                  "/etc/pki/tls/cert.pem"};
    host.dirs = {"/etc/pki/tls/certs"};
    EXPECT_EQ(choose(host).file, "/etc/pki/tls/certs/ca-bundle.crt");
}

TEST(TlsTrust, OpenSuseBundle) {
    FakeHost host;
    host.files = {"/etc/ssl/ca-bundle.pem"};
    host.dirs = {"/etc/ssl/certs"};
    EXPECT_EQ(choose(host).file, "/etc/ssl/ca-bundle.pem");
}

TEST(TlsTrust, AlpineBundle) {
    FakeHost host;
    host.files = {"/etc/ssl/certs/ca-certificates.crt", "/etc/ssl/cert.pem"};
    EXPECT_EQ(choose(host).file, "/etc/ssl/certs/ca-certificates.crt");
}

TEST(TlsTrust, EmptySslCertFileStillDefersToOpenSsl) {
    FakeHost host;
    host.env = {{"SSL_CERT_FILE", ""}};
    host.files = {"/etc/ssl/certs/ca-certificates.crt"};
    EXPECT_TRUE(choose(host).empty());
}

TEST(TlsTrust, SslCertDirAloneDefersToOpenSsl) {
    FakeHost host;
    host.env = {{"SSL_CERT_DIR", "/my/certs"}};
    host.files = {"/etc/ssl/certs/ca-certificates.crt"};
    EXPECT_TRUE(choose(host).empty());
}

TEST(TlsTrust, CondaBundleWhenTheHostHasNone) {
    FakeHost host;
    host.env = {{"CONDA_PREFIX", "/c"}};
    host.files = {"/c/ssl/cacert.pem"};
    EXPECT_EQ(choose(host).file, "/c/ssl/cacert.pem");
}

TEST(TlsTrust, HostBundleBeatsConda) {
    FakeHost host;
    host.env = {{"CONDA_PREFIX", "/c"}};
    host.files = {"/c/ssl/cacert.pem", "/etc/pki/tls/certs/ca-bundle.crt"};
    EXPECT_EQ(choose(host).file, "/etc/pki/tls/certs/ca-bundle.crt");
}

TEST(TlsTrust, HashedDirectoryOnly) {
    FakeHost host;
    host.dirs = {"/etc/ssl/certs"};
    const CaPaths ca = choose(host);
    EXPECT_EQ(ca.file, "");
    EXPECT_EQ(ca.dir, "/etc/ssl/certs");
}

TEST(TlsTrust, NothingLeavesOpenSslDefault) {
    FakeHost host;
    EXPECT_TRUE(choose(host).empty());
}

TEST(TlsTrust, SourceBuildsKeepOpenSslDefaults) {
    if (gpufl::tls::privateOpenSslBuild()) {
        GTEST_SKIP() << "built with GPUFL_PRIVATE_OPENSSL";
    }
    EXPECT_TRUE(gpufl::tls::caPathsForThisBuild().empty());
}
