# Run as a POST_BUILD step by gpufl_check_private_openssl() in CMakeLists.txt
# when GPUFL_PRIVATE_OPENSSL is ON:
#
#   cmake -DFILE=<ELF> -DREADELF=<readelf> -P check_private_openssl.cmake
#
# Fails the build unless FILE keeps OpenSSL entirely to itself: no DT_NEEDED
# on libssl or libcrypto, no OpenSSL name in .dynsym (defined or undefined),
# no OPENSSL_* symbol version, and the private copy's version string present,
# so TLS was compiled in rather than dropped. tests/package/check_wheel_launcher.py
# checks the same invariants on the installed wheel.

if(NOT FILE OR NOT EXISTS "${FILE}")
    message(FATAL_ERROR "check_private_openssl: FILE '${FILE}' does not exist")
endif()
if(NOT READELF)
    find_program(READELF readelf)
endif()
if(NOT READELF)
    message(FATAL_ERROR "check_private_openssl: readelf not found")
endif()

execute_process(
    COMMAND "${READELF}" -W -d --dyn-syms -V "${FILE}"
    OUTPUT_VARIABLE elf
    ERROR_VARIABLE elf_err
    RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "check_private_openssl: ${READELF} failed on ${FILE}: ${elf_err}")
endif()

set(problems "")
string(REPLACE "\n" ";" lines "${elf}")
foreach(line IN LISTS lines)
    if(line MATCHES "\\(NEEDED\\).*\\[lib(ssl|crypto)[-.]")
        list(APPEND problems "NEEDED: ${line}")
    # readelf --dyn-syms: "Num: Value Size Type Bind Vis Ndx Name".
    elseif(line MATCHES "^ +[0-9]+: [0-9a-f]+ .* (UND|ABS|COM|[0-9]+) +(SSL|SSL_CTX|TLS|DTLS|OPENSSL|OSSL|ossl|X509|EVP|BIO|CRYPTO|ERR|PEM|ASN1|BN|RSA|EC|HMAC|RAND|OBJ|d2i|i2d)_")
        list(APPEND problems "dynamic symbol: ${line}")
    elseif(line MATCHES "OPENSSL_[0-9]")
        list(APPEND problems "symbol version: ${line}")
    endif()
endforeach()

file(STRINGS "${FILE}" version_text REGEX "OpenSSL 3\\.[0-9]+\\.[0-9]+" LIMIT_COUNT 1)
if(NOT version_text)
    list(APPEND problems "no 'OpenSSL 3.x.y' string: the private OpenSSL is not linked in")
endif()

if(problems)
    list(LENGTH problems n)
    if(n GREATER 20)
        list(SUBLIST problems 0 20 problems)
        list(APPEND problems "... and more")
    endif()
    list(JOIN problems "\n  " shown)
    message(FATAL_ERROR
        "check_private_openssl: ${FILE} does not keep OpenSSL private (${n} problems):\n  ${shown}")
endif()
message(STATUS "check_private_openssl: ${FILE} keeps OpenSSL private")
