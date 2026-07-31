#pragma once

#include <algorithm>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cli_parse_internal.hpp"

namespace gpufl::launcher::detail {

/**
 * Alias-to-handler dispatch for one subcommand's option table, plus the help
 * text for those options.
 *
 * Help lives next to the option on purpose: a flag and its documentation are
 * added in one place, so it is not possible to ship a flag nobody can discover.
 * An option with an empty description is deliberately undocumented (removed
 * flags kept only to print a migration hint).
 */
template <typename Args>
class CliOptionManager {
   public:
    using Handler = std::string (*)(
        const FlagBreak&, const std::vector<std::string>&, std::size_t&, Args&);

    /** Undocumented option: dispatch only, never rendered into help. */
    CliOptionManager& add(std::initializer_list<std::string_view> aliases,
                          const Handler handler) {
        return add(aliases, "", "", 0, handler);
    }

    /**
     * Documented option. `value_name` is the metavariable ("<DUR>") or "" for a
     * boolean flag; `help_section` groups the option in help output and must be
     * nonzero for the option to be rendered.
     *
     * Aliases MUST have static storage duration - string literals. Only the
     * views are stored, so a temporary std::string would dangle.
     */
    CliOptionManager& add(std::initializer_list<std::string_view> aliases,
                          const std::string_view value_name,
                          const std::string_view description,
                          const int help_section,
                          const Handler handler) {
        if (aliases.size() == 0 || handler == nullptr) {
            throw std::logic_error("CLI option requires an alias and handler");
        }
        for (const std::string_view alias : aliases) {
            if (find(alias) != nullptr) {
                throw std::logic_error(
                    "duplicate CLI option alias: " + std::string(alias));
            }
        }
        options_.push_back({std::vector<std::string_view>(aliases), value_name,
                            description, help_section, handler});
        return *this;
    }

    bool parse(const FlagBreak& flag,
               const std::vector<std::string>& argv,
               std::size_t& index,
               Args& args,
               std::string& error) const {
        const Option* option = find(flag.key);
        if (option == nullptr) return false;
        error = option->handler(flag, argv, index, args);
        return true;
    }

    /** Render every documented option of one section, in registration order. */
    std::string formatHelp(const int help_section) const {
        std::string out;
        for (const Option& option : options_) {
            if (option.help_section != help_section) continue;
            if (option.description.empty()) continue;
            appendHelpLine(out, option);
        }
        return out;
    }

    /**
     * Every alias in the table, documented or not. Lets a test assert that the
     * table and the help output agree instead of trusting review.
     */
    std::vector<std::string_view> aliases() const {
        std::vector<std::string_view> all;
        for (const Option& option : options_) {
            all.insert(all.end(), option.aliases.begin(), option.aliases.end());
        }
        return all;
    }

   private:
    struct Option {
        std::vector<std::string_view> aliases;
        std::string_view value_name;
        std::string_view description;
        int help_section = 0;
        Handler handler = nullptr;
    };

    const Option* find(const std::string_view key) const {
        const auto option = std::find_if(
            options_.begin(), options_.end(),
            [key](const Option& candidate) {
                return std::find(candidate.aliases.begin(),
                                 candidate.aliases.end(), key) !=
                       candidate.aliases.end();
            });
        return option == options_.end() ? nullptr : &*option;
    }

    // Matches the launcher's long-standing help layout: descriptions start at
    // one column, and a long-only option indents to where a "-x, " prefix would
    // have ended so both kinds line up.
    static void appendHelpLine(std::string& out, const Option& option) {
        constexpr std::size_t kShortIndent = 4;
        constexpr std::size_t kLongOnlyIndent = 8;
        constexpr std::size_t kDescriptionColumn = 28;
        constexpr std::size_t kLineWidth = 96;

        const bool has_short = std::any_of(
            option.aliases.begin(), option.aliases.end(),
            [](const std::string_view alias) {
                return alias.size() > 1 && alias[0] == '-' && alias[1] != '-';
            });
        const std::size_t indent = has_short ? kShortIndent : kLongOnlyIndent;

        std::string spelling;
        for (std::size_t i = 0; i < option.aliases.size(); ++i) {
            if (i > 0) spelling += ", ";
            spelling += option.aliases[i];
        }
        if (!option.value_name.empty()) {
            spelling += "=";
            spelling += option.value_name;
        }

        out.append(indent, ' ');
        out += spelling;
        std::size_t column = indent + spelling.size();
        // A spelling that reaches the description column takes the next line,
        // so a long flag name never pushes its description out of alignment.
        if (column >= kDescriptionColumn) {
            out += '\n';
            out.append(kDescriptionColumn, ' ');
        } else {
            out.append(kDescriptionColumn - column, ' ');
        }
        column = kDescriptionColumn;

        std::istringstream words{std::string(option.description)};
        std::string word;
        bool first = true;
        while (words >> word) {
            const std::size_t separator = first ? 0 : 1;
            if (column + separator + word.size() > kLineWidth) {
                out += '\n';
                out.append(kDescriptionColumn, ' ');
                column = kDescriptionColumn;
                first = true;
            }
            if (!first) {
                out += ' ';
                ++column;
            }
            out += word;
            column += word.size();
            first = false;
        }
        out += '\n';
    }

    std::vector<Option> options_;
};

}  // namespace gpufl::launcher::detail
