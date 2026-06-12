#pragma once

// Profile-preset VALUE constants read by the injection shared library.
//
// NOTE: the environment-variable NAME constants that used to live here (kEnv*)
// moved to gpufl/core/env_vars.hpp (namespace gpufl::env) so every env-var name
// lives in exactly one place - read/set env through those. This file now holds
// only the profile-preset string VALUES below (selected via env::kInjectProfile).

namespace gpufl::inject {

// Profile-name string values. `gpufl trace --profile` is no longer exposed; the
// launcher pins comprehensive and these remain for internal/legacy env-based
// injection overrides.
constexpr const char* kProfileComprehensive  = "comprehensive";
constexpr const char* kProfileLight          = "light";
constexpr const char* kProfileMonitoringOnly = "monitoring-only";

}  // namespace gpufl::inject
