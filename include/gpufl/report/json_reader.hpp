#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace gpufl {
namespace report {

// Minimal JSON value type for reading NDJSON log files.
// Supports: null, bool, int64, double, string, array, object.
class JsonValue {
   public:
    using Object = std::map<std::string, JsonValue>;
    using Array = std::vector<JsonValue>;

    enum class Type { Null, Bool, Int, Double, String, Array, Object };

    JsonValue() : data_(Null{}) {}

    // Construction helpers
    static JsonValue null() { return {}; }
    static JsonValue boolean(bool v) { JsonValue j; j.data_ = v; return j; }
    static JsonValue integer(int64_t v) { JsonValue j; j.data_ = v; return j; }
    static JsonValue number(double v) { JsonValue j; j.data_ = v; return j; }
    static JsonValue string(std::string v) { JsonValue j; j.data_ = std::move(v); return j; }
    static JsonValue array(Array v) { JsonValue j; j.data_ = std::move(v); return j; }
    static JsonValue object(Object v) { JsonValue j; j.data_ = std::move(v); return j; }

    Type type() const;

    bool is_null() const { return type() == Type::Null; }
    bool is_bool() const { return type() == Type::Bool; }
    bool is_int() const { return type() == Type::Int; }
    bool is_double() const { return type() == Type::Double; }
    bool is_number() const { return is_int() || is_double(); }
    bool is_string() const { return type() == Type::String; }
    bool is_array() const { return type() == Type::Array; }
    bool is_object() const { return type() == Type::Object; }

    // Value accessors (return defaults if wrong type)
    bool get_bool(bool def = false) const;
    int64_t get_int(int64_t def = 0) const;
    double get_double(double def = 0.0) const;
    const std::string& get_string() const;
    const Array& get_array() const;
    const Object& get_object() const;

    // Convenience: get_int that also works on doubles
    int64_t as_int(int64_t def = 0) const;
    double as_double(double def = 0.0) const;
    uint64_t as_uint64(uint64_t def = 0) const;

    // Object key access
    bool contains(const std::string& key) const;
    const JsonValue& operator[](const std::string& key) const;
    const JsonValue& at(const std::string& key) const;

    // Array index access
    const JsonValue& operator[](size_t index) const;
    size_t size() const;
    bool empty() const;

    // Object iteration
    Object::const_iterator begin() const;
    Object::const_iterator end() const;

    // Shorthand: value<T>(key, default)
    template <typename T>
    T value(const std::string& key, const T& def) const;

   private:
    struct Null {};
    std::variant<Null, bool, int64_t, double, std::string, Array, Object> data_;
    static const JsonValue kNull;
    static const std::string kEmptyString;
    static const Array kEmptyArray;
    static const Object kEmptyObject;
};

// Parse a single JSON string into a JsonValue. Returns null on failure.
JsonValue parseJson(const std::string& input);

// Parse an NDJSON file into a vector of JsonValue objects.
std::vector<JsonValue> loadJsonLines(const std::string& path);

// ── Template implementations ────────────────────────────────────────────────

template <>
inline std::string JsonValue::value(const std::string& key, const std::string& def) const {
    if (!is_object()) return def;
    auto it = std::get<Object>(data_).find(key);
    if (it == std::get<Object>(data_).end() || !it->second.is_string()) return def;
    return it->second.get_string();
}

template <>
inline int64_t JsonValue::value(const std::string& key, const int64_t& def) const {
    if (!is_object()) return def;
    auto it = std::get<Object>(data_).find(key);
    if (it == std::get<Object>(data_).end()) return def;
    return it->second.as_int(def);
}

template <>
inline uint64_t JsonValue::value(const std::string& key, const uint64_t& def) const {
    if (!is_object()) return def;
    auto it = std::get<Object>(data_).find(key);
    if (it == std::get<Object>(data_).end()) return def;
    return it->second.as_uint64(def);
}

template <>
inline int JsonValue::value(const std::string& key, const int& def) const {
    return static_cast<int>(value<int64_t>(key, def));
}

template <>
inline unsigned JsonValue::value(const std::string& key, const unsigned& def) const {
    if (!is_object()) return def;
    auto it = std::get<Object>(data_).find(key);
    if (it == std::get<Object>(data_).end()) return def;
    return static_cast<unsigned>(it->second.as_int(def));
}

template <>
inline double JsonValue::value(const std::string& key, const double& def) const {
    if (!is_object()) return def;
    auto it = std::get<Object>(data_).find(key);
    if (it == std::get<Object>(data_).end()) return def;
    return it->second.as_double(def);
}

template <>
inline float JsonValue::value(const std::string& key, const float& def) const {
    return static_cast<float>(value<double>(key, static_cast<double>(def)));
}

template <>
inline bool JsonValue::value(const std::string& key, const bool& def) const {
    if (!is_object()) return def;
    auto it = std::get<Object>(data_).find(key);
    if (it == std::get<Object>(data_).end() || !it->second.is_bool()) return def;
    return it->second.get_bool(def);
}

}  // namespace report
}  // namespace gpufl
