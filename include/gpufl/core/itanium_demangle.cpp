// Itanium ABI demangler — focused implementation for CUDA kernel names.
//
// See itanium_demangle.hpp for the rationale and scope. This file is
// the implementation. The parser is a single-pass recursive-descent
// over the mangled byte stream with one substitution + one template-
// parameter table. Any parse failure throws ParseError, which is
// caught by the public entry point so callers always get a string
// back (the original mangled name on failure).
//
// Reference: https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling

#include "gpufl/core/itanium_demangle.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpufl::core {
namespace {

class ParseError : public std::exception {};

class ItaniumParser {
public:
    explicit ItaniumParser(const char* s)
        : begin_(s), p_(s), end_(s + std::strlen(s)) {}

    std::string parse() {
        if (end_ - p_ < 2 || p_[0] != '_' || p_[1] != 'Z') return begin_;
        p_ += 2;
        return parseEncoding();
    }

private:
    const char* begin_;
    const char* p_;
    const char* end_;

    // Substitution table — every "useful" type the parser consumes
    // gets appended here. Later refs `S_`, `S0_`, `S1_`, … look up
    // by 0-based index. Per the spec, S_ == subs_[0], S0_ == subs_[1],
    // S1_ == subs_[2], etc. (Yes, off-by-one. The naked `S_` is index 0;
    // subsequent S<n>_ are index n+1.)
    std::vector<std::string> subs_;

    // Template parameter table — populated by parseTemplateArgs() so
    // later references like `T_`, `T0_` resolve back into the captured
    // arg's rendered string. Same off-by-one convention as subs_:
    // T_ == templates_[0], T0_ == templates_[1], …
    std::vector<std::string> templates_;

    // Was the last name we parsed a template instantiation? If yes,
    // the next type in the bare-function-type is the RETURN type
    // (Itanium spec quirk — ordinary names omit the return type from
    // mangling, templated names include it because overload resolution
    // depends on it).
    bool lastNameWasTemplate_ = false;

    // ── Byte-level helpers ────────────────────────────────────────
    bool eof() const { return p_ >= end_; }
    char peek() const { return eof() ? '\0' : *p_; }
    char get() { if (eof()) throw ParseError(); return *p_++; }
    bool consume(char c) {
        if (peek() == c) { ++p_; return true; }
        return false;
    }

    int parseNumber() {
        int n = 0;
        bool any = false;
        while (!eof() && peek() >= '0' && peek() <= '9') {
            n = n * 10 + (get() - '0');
            any = true;
        }
        if (!any) throw ParseError();
        return n;
    }

    // ── Grammar productions ───────────────────────────────────────

    // <encoding> ::= <name> [<bare_function_type>]
    std::string parseEncoding() {
        lastNameWasTemplate_ = false;
        std::string name = parseName();

        // Bare function type follows when there's still input. The
        // first type is the return type IFF the name was templated;
        // remaining types are parameters. For non-templated names,
        // every type is a parameter.
        if (eof()) return name;

        std::string returnType;
        if (lastNameWasTemplate_) {
            try {
                returnType = parseType();
            } catch (const ParseError&) {
                // Couldn't parse the return type — emit the name alone.
                return name;
            }
        }

        std::vector<std::string> params;
        while (!eof()) {
            std::string t;
            try {
                t = parseType();
            } catch (const ParseError&) {
                break;
            }
            // A lone "v" with no other args means "(void)" → empty arg list.
            if (params.empty() && t == "void" && eof()) break;
            params.push_back(std::move(t));
        }

        std::string out;
        if (!returnType.empty()) {
            out = returnType + " ";
        }
        out += name;
        out += "(";
        for (size_t i = 0; i < params.size(); ++i) {
            if (i) out += ", ";
            out += params[i];
        }
        out += ")";
        return out;
    }

    // <name> ::= <nested_name>
    //          | <unscoped_name>
    //          | <unscoped_template_name> <template_args>
    //          | <substitution>
    std::string parseName() {
        char c = peek();
        if (c == 'N') return parseNestedName();
        if (c == 'S') {
            // Substitution at the start of a name — rare for CUDA
            // kernels; substitution refers back to a previously seen
            // type/name. Most kernel symbols start with an explicit
            // nested name `N…E` or a bare source name.
            std::string sub = parseSubstitution();
            // May be followed by template_args.
            if (peek() == 'I') {
                std::string targs = parseTemplateArgs();
                lastNameWasTemplate_ = true;
                std::string out = sub + targs;
                subs_.push_back(out);
                return out;
            }
            lastNameWasTemplate_ = false;
            return sub;
        }
        // Unscoped name: source_name [template_args]
        std::string n = parseUnqualifiedName();
        if (peek() == 'I') {
            std::string targs = parseTemplateArgs();
            lastNameWasTemplate_ = true;
            std::string out = n + targs;
            subs_.push_back(out);
            return out;
        }
        lastNameWasTemplate_ = false;
        return n;
    }

    // <nested_name> ::= "N" [<cv-qualifier>]* <prefix> <unqualified_name> "E"
    //                 | "N" [<cv-qualifier>]* <template_prefix> <template_args> "E"
    //
    // We collapse this to: read N, eat any CV qualifiers, then parse
    // a sequence of components separated by "::". Components are
    // either unqualified_names or template_args (which apply to the
    // most-recent component). "E" terminates.
    std::string parseNestedName() {
        if (!consume('N')) throw ParseError();
        // Skip CV qualifiers; we don't render them on nested-name
        // wrappers because for kernel names they're decorative.
        while (peek() == 'K' || peek() == 'V' || peek() == 'r') ++p_;
        // Reference qualifiers (R / O) on member functions — also skip.
        while (peek() == 'R' || peek() == 'O') ++p_;

        std::vector<std::string> parts;
        while (!eof() && peek() != 'E') {
            const char c = peek();
            if (c == 'I') {
                // Template args attach to the previous component.
                if (parts.empty()) throw ParseError();
                const std::string targs = parseTemplateArgs();
                parts.back() += targs;
                // The substitution table stores the cumulative prefix
                // (per the ABI). For our purposes, recording the most
                // recent qualified+templated form is enough for T_/S_
                // refs that show up later.
                subs_.push_back(joinScope(parts));
            } else if (c == 'S') {
                // A substitution can appear as a name component (e.g.
                // when a long namespace was seen earlier and is now
                // reused). The sub string is taken whole; it may
                // already contain `::`.
                std::string sub = parseSubstitution();
                parts.push_back(sub);
            } else if (c == 'T') {
                // Template parameter as a name component — rare but
                // valid (`N…T_…E`). Resolve and append.
                std::string tp = parseTemplateParam();
                parts.push_back(tp);
            } else {
                // Source name (length-prefixed identifier).
                std::string n = parseUnqualifiedName();
                parts.push_back(n);
                subs_.push_back(joinScope(parts));
            }
        }
        if (!consume('E')) throw ParseError();

        std::string joined = joinScope(parts);
        // Was the LAST component templated? Detect by its trailing '>'.
        lastNameWasTemplate_ = !joined.empty() && joined.back() == '>';
        return joined;
    }

    static std::string joinScope(const std::vector<std::string>& parts) {
        std::string out;
        for (size_t i = 0; i < parts.size(); ++i) {
            if (i) out += "::";
            out += parts[i];
        }
        return out;
    }

    // <unqualified_name> ::= <source_name>
    //                      | <operator_name>      (not handled — falls back)
    //                      | <ctor_dtor_name>     (not handled — falls back)
    std::string parseUnqualifiedName() {
        // For our scope: always a source_name (length-prefixed).
        // Anonymous-namespace markers like "_GLOBAL__N_…" are still
        // length-prefixed identifiers, so they fall through here too.
        if (peek() < '0' || peek() > '9') throw ParseError();
        return parseSourceName();
    }

    // <source_name> ::= <positive length number> <identifier>
    std::string parseSourceName() {
        const int len = parseNumber();
        if (len <= 0 || p_ + len > end_) throw ParseError();
        std::string s(p_, p_ + len);
        p_ += len;
        return s;
    }

    // <template_args> ::= "I" <template_arg>+ "E"
    // <template_arg>  ::= <type>
    //                   | "X" <expression> "E"     (skip expression)
    //                   | "L" <type> <value> "E"   (literal — best-effort)
    //                   | "J" <template_arg>* "E"  (parameter pack)
    std::string parseTemplateArgs() {
        if (!consume('I')) throw ParseError();
        std::string out = "<";
        bool first = true;
        while (!eof() && peek() != 'E') {
            if (!first) out += ", ";
            first = false;

            char c = peek();
            std::string arg;
            if (c == 'X' || c == 'L' || c == 'J') {
                // Skip these by consuming until the matching E. Render
                // as "?" so the reader knows something was elided.
                arg = parseUntilE();
                if (arg.empty()) arg = "?";
            } else {
                arg = parseType();
            }
            out += arg;
            templates_.push_back(arg);
        }
        if (!consume('E')) throw ParseError();
        out += ">";
        return out;
    }

    // Consume input bytes until the matching balanced E, then return a
    // best-effort textual representation. Used for X / L / J template
    // arguments where we don't fully implement the sub-grammar.
    std::string parseUntilE() {
        const char tag = get(); // X, L, or J
        std::string body;
        int depth = 1;
        while (!eof() && depth > 0) {
            char c = get();
            if (c == 'E') {
                --depth;
                if (depth == 0) break;
            } else if (c == 'I' || c == 'X' || c == 'L' || c == 'J' || c == 'N') {
                ++depth;
            }
            body += c;
        }
        // Best-effort literal: `Lif42E` -> "42" (for L<type>value).
        if (tag == 'L' && !body.empty()) {
            // Strip the leading type prefix (one char for builtin) if
            // present; otherwise return the body verbatim.
            if (body.size() > 1 && std::strchr("vwbcahstijlmxynofdeg", body[0])) {
                return body.substr(1);
            }
            return body;
        }
        return body;
    }

    // <type> ::= <builtin_type>
    //          | <function_type>     (subset — function pointers)
    //          | <class_enum_type>   (== <name>)
    //          | <array_type>
    //          | <pointer_to_member_type>
    //          | <template_param>    T_ / T<n>_
    //          | <substitution>      S_ / S<n>_
    //          | "P" <type>
    //          | "R" <type>          (lvalue ref)
    //          | "O" <type>          (rvalue ref)
    //          | "K" <type>          (const)
    //          | "V" <type>          (volatile)
    //          | "r" <type>          (restrict)
    //          | "U" <source_name> <type>  (vendor qualifier)
    //          | "Dn"                (decltype(nullptr))
    std::string parseType() {
        if (eof()) throw ParseError();
        char c = peek();

        // Builtin types ─────────────────────────────────────────
        const char* builtin = builtinName(c);
        if (builtin) {
            ++p_;
            // Builtin types are NOT added to the substitution table
            // per the ABI, so don't push.
            return builtin;
        }

        // Two-char builtins: "Dn", "Dh", "Di", etc.
        if (c == 'D') {
            if (p_ + 1 >= end_) throw ParseError();
            const char d = p_[1];
            const char* d2 = builtinNameD(d);
            if (d2) {
                p_ += 2;
                return d2;
            }
            // Fall through to treat as unknown.
        }

        if (c == 'P') { ++p_; std::string t = parseType() + "*";  subs_.push_back(t); return t; }
        if (c == 'R') { ++p_; std::string t = parseType() + "&";  subs_.push_back(t); return t; }
        if (c == 'O') { ++p_; std::string t = parseType() + "&&"; subs_.push_back(t); return t; }
        if (c == 'K') { ++p_; std::string t = parseType() + " const";    subs_.push_back(t); return t; }
        if (c == 'V') { ++p_; std::string t = parseType() + " volatile"; subs_.push_back(t); return t; }
        if (c == 'r') { ++p_; std::string t = parseType() + " restrict"; subs_.push_back(t); return t; }

        if (c == 'T') return parseTemplateParam();
        if (c == 'S') {
            // A substitution can be the start of a TYPE — and that
            // type may carry template args. E.g. `St5arrayIPcLy1EE`
            // is `std::array<char*, 1>`: parseSubstitution consumes
            // `St5array` and returns `std::array`; the following
            // `I…E` template-args block belongs to that type and
            // must be parsed + appended here. Without this, the
            // outer template-arg loop would see `I…` as a new arg
            // and choke (parseType doesn't know how to start with
            // `I`), and the whole demangle aborts.
            std::string n = parseSubstitution();
            if (!eof() && peek() == 'I') {
                n += parseTemplateArgs();
                subs_.push_back(n);
            }
            return n;
        }

        if (c == 'N') {
            // Nested name as a class type, e.g. T::Params -> NT_6ParamsE.
            std::string n = parseNestedName();
            subs_.push_back(n);
            return n;
        }

        if (c >= '0' && c <= '9') {
            // Source name as a class type, e.g. "5MyClass".
            std::string n = parseSourceName();
            subs_.push_back(n);
            return n;
        }

        // Vendor-extended type: U<source_name><type>
        if (c == 'U') {
            ++p_;
            (void)parseSourceName();   // discard vendor name
            return parseType();
        }

        // Anything else we don't recognize — bail out.
        throw ParseError();
    }

    // Map a single-character builtin code to its display name. Returns
    // nullptr for codes that aren't single-char builtins.
    static const char* builtinName(char c) {
        switch (c) {
            case 'v': return "void";
            case 'w': return "wchar_t";
            case 'b': return "bool";
            case 'c': return "char";
            case 'a': return "signed char";
            case 'h': return "unsigned char";
            case 's': return "short";
            case 't': return "unsigned short";
            case 'i': return "int";
            case 'j': return "unsigned int";
            case 'l': return "long";
            case 'm': return "unsigned long";
            case 'x': return "long long";
            case 'y': return "unsigned long long";
            case 'n': return "__int128";
            case 'o': return "unsigned __int128";
            case 'f': return "float";
            case 'd': return "double";
            case 'e': return "long double";
            case 'g': return "__float128";
            case 'z': return "...";  // ellipsis
            default:  return nullptr;
        }
    }
    static const char* builtinNameD(char c) {
        switch (c) {
            case 'd': return "decimal64";
            case 'e': return "decimal128";
            case 'f': return "decimal32";
            case 'h': return "_Float16";
            case 'i': return "char32_t";
            case 's': return "char16_t";
            case 'a': return "auto";
            case 'c': return "decltype(auto)";
            case 'n': return "decltype(nullptr)";
            default:  return nullptr;
        }
    }

    // <substitution> ::= "S_"           (subs_[0])
    //                  | "S" <seq_id> "_"   (subs_[seq_id + 1])
    //                  | "St"           (std::)
    //                  | "Sa"           (std::allocator)
    //                  | "Sb"           (std::basic_string)
    //                  | "Ss"           (std::string)
    //                  | "Si"           (std::istream)
    //                  | "So"           (std::ostream)
    //                  | "Sd"           (std::iostream)
    std::string parseSubstitution() {
        if (!consume('S')) throw ParseError();
        char c = peek();
        if (c == '_') { ++p_; if (subs_.empty()) throw ParseError(); return subs_[0]; }
        if (c == 't') {
            ++p_;
            // `St` is a NAMESPACE shortcut for `::std::`, NOT a
            // standalone type. In the wild it's almost always followed
            // by an unqualified-name (e.g. `St5array` → `std::array`,
            // `St11char_traits…` → `std::char_traits<…>`). Glue them
            // together so the caller sees one type, not "std" plus a
            // dangling source name. Without this, template-arg
            // parsing dies on the unconsumed digits and the whole
            // demangle falls back to the mangled string — which is
            // the regression we hit on the ATen
            // `vectorized_elementwise_kernel` Windows traces.
            if (!eof() && peek() >= '0' && peek() <= '9') {
                return "std::" + parseSourceName();
            }
            return "std";
        }
        if (c == 'a') { ++p_; return "std::allocator"; }
        if (c == 'b') { ++p_; return "std::basic_string"; }
        if (c == 's') { ++p_; return "std::string"; }
        if (c == 'i') { ++p_; return "std::istream"; }
        if (c == 'o') { ++p_; return "std::ostream"; }
        if (c == 'd') { ++p_; return "std::iostream"; }

        // Numeric subst-id (base-36, digits then uppercase letters).
        int seq = 0;
        bool any = false;
        while (!eof() && peek() != '_') {
            char d = get();
            int v;
            if (d >= '0' && d <= '9')      v = d - '0';
            else if (d >= 'A' && d <= 'Z') v = 10 + (d - 'A');
            else throw ParseError();
            seq = seq * 36 + v;
            any = true;
        }
        if (!any) throw ParseError();
        if (!consume('_')) throw ParseError();
        const size_t idx = static_cast<size_t>(seq) + 1;
        if (idx >= subs_.size()) throw ParseError();
        return subs_[idx];
    }

    // <template_param> ::= "T_"          (templates_[0])
    //                    | "T" <number> "_"  (templates_[number + 1])
    std::string parseTemplateParam() {
        if (!consume('T')) throw ParseError();
        if (consume('_')) {
            if (templates_.empty()) throw ParseError();
            return templates_[0];
        }
        // Numeric param-id (base-36).
        int seq = 0;
        bool any = false;
        while (!eof() && peek() != '_') {
            const char d = get();
            int v;
            if (d >= '0' && d <= '9')      v = d - '0';
            else if (d >= 'A' && d <= 'Z') v = 10 + (d - 'A');
            else throw ParseError();
            seq = seq * 36 + v;
            any = true;
        }
        if (!any) throw ParseError();
        if (!consume('_')) throw ParseError();
        const size_t idx = static_cast<size_t>(seq) + 1;
        if (idx >= templates_.size()) throw ParseError();
        return templates_[idx];
    }
};

}  // namespace

std::string DemangleItaniumName(const char* mangled) {
    if (!mangled || mangled[0] == '\0') return mangled ? mangled : "";
    try {
        ItaniumParser parser(mangled);
        return parser.parse();
    } catch (...) {
        // Any parse failure → return the original mangled string.
        // Callers always get a non-empty, displayable result.
        return mangled;
    }
}

}  // namespace gpufl::core
