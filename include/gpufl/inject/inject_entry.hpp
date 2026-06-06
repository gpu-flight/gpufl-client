#pragma once

// Profile-preset VALUE constants shared between the launcher binary
// (`gpufl trace --profile`) and the injection shared library.
//
// NOTE: the environment-variable NAME constants that used to live here (kEnv*)
// moved to gpufl/core/env_vars.hpp (namespace gpufl::env) so every env-var name
// lives in exactly one place — read/set env through those. This file now holds
// only the profile-preset string VALUES below (selected via env::kInjectProfile).

namespace gpufl::inject {

// Profile-name string values (must stay in sync with the launcher's
// `--profile` flag parsing in cli_parse.cpp).
constexpr const char* kProfileComprehensive  = "comprehensive";
constexpr const char* kProfileLight          = "light";
constexpr const char* kProfileMonitoringOnly = "monitoring-only";

}  // namespace gpufl::inject
