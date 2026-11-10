#!/usr/bin/env bash
# Build the private, static OpenSSL that Linux wheels link into every gpufl
# binary (CMake option GPUFL_PRIVATE_OPENSSL; see CMakeLists.txt).
#
#   scripts/build-static-openssl.sh <install prefix>
#
# The wheel workflows run this in CIBW_BEFORE_ALL_LINUX and point
# OPENSSL_ROOT_DIR at the prefix. It installs only headers, libssl.a,
# libcrypto.a and the pkg-config files. Set GPUFL_OPENSSL_TARBALL to a local
# copy of the release tarball to build without downloading; its SHA-256 is
# checked either way.
#
# Why a private copy: `gpufl trace` preloads libgpufl_inject.so into the traced
# program. A shared OpenSSL that the inject NEEDs joins that program's global
# symbol scope, and the program's own OpenSSL users (Python's _ssl, libcurl)
# then bind to it, with its foreign OPENSSLDIR and CA paths. Linked statically
# and hidden with --exclude-libs, gpufl's OpenSSL is invisible to them.
#
# Configure flags:
#   no-shared            static archives only.
#   no-module            no providers loaded from disk (legacy, fips).
#   no-autoload-config   never read the host's openssl.cnf or OPENSSL_CONF
#                        inside a traced program.
#   no-atexit            no OPENSSL_cleanup at exit underneath gpufl's
#                        detached version-probe thread.
#   --openssldir=/etc/ssl  only the last-resort default; gpufl finds the host
#                        CA bundle itself (include/gpufl/core/tls_trust.cpp).
#
# Security updates: OpenSSL fixes in this copy reach users only through a gpufl
# release. For an OpenSSL advisory, change OPENSSL_VERSION and OPENSSL_SHA256
# below together (the SHA-256 is published beside the tarball on the OpenSSL
# release page), let the wheel CI build it, then cut a release. 3.5 is an LTS
# line, supported until 2030-04.
set -euo pipefail

OPENSSL_VERSION=3.5.9
OPENSSL_SHA256=603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a

if [ $# -ne 1 ] || [ -z "$1" ]; then
    echo "usage: $0 <install prefix>" >&2
    exit 2
fi
prefix=$1
case $prefix in
    /*) ;;
    *) prefix=$(pwd)/$prefix ;;
esac

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"

tarball=openssl-$OPENSSL_VERSION.tar.gz
if [ -n "${GPUFL_OPENSSL_TARBALL:-}" ]; then
    cp "$GPUFL_OPENSSL_TARBALL" "$tarball"
else
    # EL8's curl 7.61 has no --retry-all-errors.
    curl -fsSL --retry 5 --retry-delay 5 -o "$tarball" \
        "https://github.com/openssl/openssl/releases/download/openssl-$OPENSSL_VERSION/$tarball"
fi
echo "$OPENSSL_SHA256  $tarball" | sha256sum -c -
tar xzf "$tarball"
cd "openssl-$OPENSSL_VERSION"

./Configure "linux-$(uname -m)" \
    --prefix="$prefix" --libdir=lib --openssldir=/etc/ssl \
    no-shared no-module no-autoload-config no-atexit no-tests no-docs \
    -fPIC
make -j"$(nproc)" build_libs
make install_dev

for f in lib/libssl.a lib/libcrypto.a lib/pkgconfig/openssl.pc include/openssl/opensslv.h; do
    if [ ! -f "$prefix/$f" ]; then
        echo "build-static-openssl: $prefix/$f is missing" >&2
        exit 1
    fi
done
grep -E '^# *define OPENSSL_VERSION_TEXT' "$prefix/include/openssl/opensslv.h" || true
echo "build-static-openssl: OpenSSL $OPENSSL_VERSION installed in $prefix"
