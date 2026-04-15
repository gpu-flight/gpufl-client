#include "gpufl/core/json/json.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace gpufl {
namespace json {

// ── escape ─────────────────────────────────────────────────────────────────

std::string escape(const std::string& s) {
    std::ostringstream oss;
    for (unsigned char c : s) {
        switch (c) {
            case '\\': oss << "\\\\"; break;
            case '"':  oss << "\\\""; break;
            case '\n': oss << "\\n";  break;
            case '\r': oss << "\\r";  break;
            case '\t': oss << "\\t";  break;
            default:
                if (c < 0x20) oss << "\\u" << std::hex << std::setw(4)
                                  << std::setfill('0') << static_cast<int>(c)
                                  << std::dec;
                else          oss << c;
        }
    }
    return oss.str();
}

// ── Static defaults ─────────────────────────────────────────────────────────

const JsonValue JsonValue::kNull;
const std::string JsonValue::kEmptyString;
const JsonValue::Array JsonValue::kEmptyArray;
const JsonValue::Object JsonValue::kEmptyObject;

// ── Type ────────────────────────────────────────────────────────────────────

JsonValue::Type JsonValue::type() const {
    return std::visit([](auto&& v) -> Type {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, Null>)        return Type::Null;
        if constexpr (std::is_same_v<T, bool>)        return Type::Bool;
        if constexpr (std::is_same_v<T, int64_t>)     return Type::Int;
        if constexpr (std::is_same_v<T, double>)      return Type::Double;
        if constexpr (std::is_same_v<T, std::string>) return Type::String;
        if constexpr (std::is_same_v<T, Array>)       return Type::Array;
        if constexpr (std::is_same_v<T, Object>)      return Type::Object;
    }, data_);
}

// ── Value accessors ─────────────────────────────────────────────────────────

bool JsonValue::get_bool(bool def) const {
    if (auto* v = std::get_if<bool>(&data_)) return *v;
    return def;
}

int64_t JsonValue::get_int(int64_t def) const {
    if (auto* v = std::get_if<int64_t>(&data_)) return *v;
    return def;
}

double JsonValue::get_double(double def) const {
    if (auto* v = std::get_if<double>(&data_)) return *v;
    return def;
}

const std::string& JsonValue::get_string() const {
    if (auto* v = std::get_if<std::string>(&data_)) return *v;
    return kEmptyString;
}

const JsonValue::Array& JsonValue::get_array() const {
    if (auto* v = std::get_if<Array>(&data_)) return *v;
    return kEmptyArray;
}

const JsonValue::Object& JsonValue::get_object() const {
    if (auto* v = std::get_if<Object>(&data_)) return *v;
    return kEmptyObject;
}

int64_t JsonValue::as_int(int64_t def) const {
    if (auto* v = std::get_if<int64_t>(&data_)) return *v;
    if (auto* v = std::get_if<double>(&data_)) return static_cast<int64_t>(*v);
    return def;
}

double JsonValue::as_double(double def) const {
    if (auto* v = std::get_if<double>(&data_)) return *v;
    if (auto* v = std::get_if<int64_t>(&data_)) return static_cast<double>(*v);
    return def;
}

uint64_t JsonValue::as_uint64(uint64_t def) const {
    if (auto* v = std::get_if<int64_t>(&data_)) return static_cast<uint64_t>(*v);
    if (auto* v = std::get_if<double>(&data_)) return static_cast<uint64_t>(*v);
    return def;
}

// ── Object access ───────────────────────────────────────────────────────────

bool JsonValue::contains(const std::string& key) const {
    if (auto* obj = std::get_if<Object>(&data_))
        return obj->find(key) != obj->end();
    return false;
}

const JsonValue& JsonValue::operator[](const std::string& key) const {
    if (auto* obj = std::get_if<Object>(&data_)) {
        auto it = obj->find(key);
        if (it != obj->end()) return it->second;
    }
    return kNull;
}

const JsonValue& JsonValue::at(const std::string& key) const {
    return (*this)[key];
}

// ── Array access ────────────────────────────────────────────────────────────

const JsonValue& JsonValue::operator[](size_t index) const {
    if (auto* arr = std::get_if<Array>(&data_)) {
        if (index < arr->size()) return (*arr)[index];
    }
    return kNull;
}

size_t JsonValue::size() const {
    if (auto* arr = std::get_if<Array>(&data_)) return arr->size();
    if (auto* obj = std::get_if<Object>(&data_)) return obj->size();
    return 0;
}

bool JsonValue::empty() const {
    return size() == 0;
}

JsonValue::Object::const_iterator JsonValue::begin() const {
    if (auto* obj = std::get_if<Object>(&data_)) return obj->begin();
    return kEmptyObject.begin();
}

JsonValue::Object::const_iterator JsonValue::end() const {
    if (auto* obj = std::get_if<Object>(&data_)) return obj->end();
    return kEmptyObject.end();
}

// ── Parser ──────────────────────────────────────────────────────────────────

namespace {

class Parser {
   public:
    explicit Parser(const std::string& input) : src_(input), pos_(0) {}

    JsonValue parse() {
        skipWhitespace();
        if (pos_ >= src_.size()) return JsonValue::null();
        auto val = parseValue();
        return val;
    }

   private:
    const std::string& src_;
    size_t pos_;

    char peek() const { return pos_ < src_.size() ? src_[pos_] : '\0'; }
    char advance() { return pos_ < src_.size() ? src_[pos_++] : '\0'; }

    void skipWhitespace() {
        while (pos_ < src_.size() && std::isspace(static_cast<unsigned char>(src_[pos_])))
            ++pos_;
    }

    bool consume(char c) {
        skipWhitespace();
        if (peek() == c) { advance(); return true; }
        return false;
    }

    void expect(char c) {
        skipWhitespace();
        if (advance() != c)
            throw std::runtime_error("expected '" + std::string(1, c) + "'");
    }

    JsonValue parseValue() {
        skipWhitespace();
        char c = peek();
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') return parseString();
        if (c == 't' || c == 'f') return parseBool();
        if (c == 'n') return parseNull();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c)))
            return parseNumber();
        throw std::runtime_error("unexpected character");
    }

    JsonValue parseObject() {
        expect('{');
        JsonValue::Object obj;
        if (consume('}')) return JsonValue::object(std::move(obj));

        do {
            skipWhitespace();
            std::string key = parseRawString();
            expect(':');
            obj[std::move(key)] = parseValue();
        } while (consume(','));

        expect('}');
        return JsonValue::object(std::move(obj));
    }

    JsonValue parseArray() {
        expect('[');
        JsonValue::Array arr;
        if (consume(']')) return JsonValue::array(std::move(arr));

        do {
            arr.push_back(parseValue());
        } while (consume(','));

        expect(']');
        return JsonValue::array(std::move(arr));
    }

    JsonValue parseString() {
        return JsonValue::string(parseRawString());
    }

    std::string parseRawString() {
        expect('"');
        std::string result;
        result.reserve(32);
        while (pos_ < src_.size()) {
            char c = src_[pos_++];
            if (c == '"') return result;
            if (c == '\\') {
                if (pos_ >= src_.size()) break;
                char e = src_[pos_++];
                switch (e) {
                    case '"':  result += '"';  break;
                    case '\\': result += '\\'; break;
                    case '/':  result += '/';  break;
                    case 'n':  result += '\n'; break;
                    case 'r':  result += '\r'; break;
                    case 't':  result += '\t'; break;
                    case 'b':  result += '\b'; break;
                    case 'f':  result += '\f'; break;
                    case 'u': {
                        // Skip 4 hex digits (simplified: just output '?')
                        size_t end = std::min(pos_ + 4, src_.size());
                        pos_ = end;
                        result += '?';
                        break;
                    }
                    default: result += e; break;
                }
            } else {
                result += c;
            }
        }
        return result;
    }

    JsonValue parseNumber() {
        size_t start = pos_;
        bool isFloat = false;

        if (peek() == '-') advance();
        while (std::isdigit(static_cast<unsigned char>(peek()))) advance();

        if (peek() == '.') {
            isFloat = true;
            advance();
            while (std::isdigit(static_cast<unsigned char>(peek()))) advance();
        }
        if (peek() == 'e' || peek() == 'E') {
            isFloat = true;
            advance();
            if (peek() == '+' || peek() == '-') advance();
            while (std::isdigit(static_cast<unsigned char>(peek()))) advance();
        }

        std::string numStr = src_.substr(start, pos_ - start);
        if (isFloat) {
            return JsonValue::number(std::stod(numStr));
        }
        // Try int64 first; fall back to double for very large numbers
        try {
            size_t idx = 0;
            int64_t val = std::stoll(numStr, &idx);
            if (idx == numStr.size()) return JsonValue::integer(val);
        } catch (...) {}
        return JsonValue::number(std::stod(numStr));
    }

    JsonValue parseBool() {
        if (src_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            return JsonValue::boolean(true);
        }
        if (src_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            return JsonValue::boolean(false);
        }
        throw std::runtime_error("invalid boolean");
    }

    JsonValue parseNull() {
        if (src_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            return JsonValue::null();
        }
        throw std::runtime_error("invalid null");
    }
};

}  // namespace

// ── Public parse functions ──────────────────────────────────────────────────

JsonValue parseJson(const std::string& input) {
    try {
        Parser p(input);
        return p.parse();
    } catch (...) {
        return JsonValue::null();
    }
}

JsonValue loadFile(const std::string& path) {
    if (path.empty()) return JsonValue::null();

    std::ifstream file(path);
    if (!file.is_open()) return JsonValue::null();

    std::ostringstream ss;
    ss << file.rdbuf();
    return parseJson(ss.str());
}

std::vector<JsonValue> loadJsonLines(const std::string& path) {
    std::vector<JsonValue> records;
    if (path.empty()) return records;

    std::ifstream file(path);
    if (!file.is_open()) return records;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] != '{') continue;
        auto val = parseJson(line);
        if (!val.is_null()) records.push_back(std::move(val));
    }
    return records;
}

}  // namespace json
}  // namespace gpufl
