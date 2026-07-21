// -*- c++ -*- /////////////////////////////////////////////////////////////////////////
// SPARTA-GUI - A Graphical Tool to Learn and Explore the SPARTA DSMC Simulation Software
//
// Copyright (c) 2023, 2024, 2025, 2026  Axel Kohlmeyer
//
// Documentation: https://sparta.github.io/sparta-gui/
// Contact: akohlmey@gmail.com
//
// This software is distributed under the GNU General Public License version 2 or later.
////////////////////////////////////////////////////////////////////////////////////////

#ifndef INPUTCHECK_H
#define INPUTCHECK_H

#include <QHash>
#include <QList>
#include <QSet>
#include <QString>
#include <QStringList>

#include <functional>

/**
 * @brief Pure, GUI-free static validator for SPARTA input decks.
 *
 * The validator performs a static (no SPARTA execution) analysis of an input
 * script and returns a list of diagnostics that can be surfaced as inline
 * editor markers or in a diagnostics list.  It has no dependency on Qt Widgets,
 * the SPARTA library, or any running instance -- everything it needs (the set
 * of known commands, the per-command style dictionaries, and an optional
 * file-existence probe) is supplied through @ref InputCheck::Context, so the
 * core is fully unit-testable.
 *
 * The checks are intentionally conservative to avoid false positives: only
 * clearly wrong constructs (an unknown command or style, a missing argument)
 * are reported as errors; cross-reference and file checks -- which an
 * @c include file could satisfy out of view -- are reported as warnings.
 */
namespace InputCheck {

/// @brief Severity of a diagnostic.
enum class Severity { Error, Warning, Info };

/// @brief Argument specification for one command, derived from the SPARTA docs.
struct CommandSpec {
    int minArgs = 0;       ///< minimum required arguments (excluding the command word)
    bool variadic = true;  ///< true if trailing keyword/args may follow (no maximum)
    /// valid keyword names for a keyword-led command (empty = do not keyword-check)
    QSet<QString> keywords;
    /// argument index (0-based, after the command word) at which the keyword list
    /// begins, or -1 when the command is not keyword-led / has no known keywords
    int keywordStart = -1;
};

/// @brief Human-facing syntax help for one command, from the SPARTA docs.
struct CommandHelp {
    QString syntax;        ///< the verbatim Syntax: template, e.g. "create_box xlo xhi ..."
    QStringList args;      ///< required positional field names ("xlo", "xhi", ...)
    QStringList keywords;  ///< optional keyword names ("start", "stop", ...)
    int keywordStart = -1; ///< arg index where the keyword list begins (-1 = N/A)
};

/// @brief A single validation finding tied to a physical line of the deck.
struct Diagnostic {
    int line = 0;                     ///< 1-based physical line number
    int column = 0;                   ///< 1-based column, 0 = whole line
    Severity severity = Severity::Error;
    QString code;                     ///< short machine tag, e.g. "unknown-command"
    QString message;                  ///< human-readable description
};

/**
 * @brief Everything the validator needs to know about the SPARTA vocabulary.
 *
 * The GUI populates this from the command/style dictionaries (help table and,
 * when available, the live SPARTA instance's command/style lists).  For unit
 * tests it is built by hand.
 */
struct Context {
    /// known top-level command names (e.g. "run", "create_box", "fix", ...).
    /// Commands present in @ref commandSpecs are also treated as known.
    QSet<QString> commands;
    /// per-command argument specs generated from the SPARTA docs (see
    /// @ref parseSyntaxTable); drives the required/maximum argument checks
    QHash<QString, CommandSpec> commandSpecs;
    /// per-command style names, keyed by the command that introduces them:
    /// "fix", "compute", "dump", "region", "collide", "react",
    /// "surf_collide", "surf_react"
    QHash<QString, QSet<QString>> styles;
    /// optional probe returning true if a deck-referenced file exists; when
    /// null, missing-file checks are skipped
    std::function<bool(const QString &)> fileExists;
    /// enable variable / compute / fix cross-reference checks
    bool checkReferences = true;
    /// enable unknown-command / unknown-style checks (needs a complete
    /// @ref commands / @ref styles dictionary to avoid false positives)
    bool checkVocabulary = true;
};

/**
 * @brief Statically validate a deck given as a list of physical lines.
 * @param lines the deck, one physical line per entry (no trailing newlines)
 * @param ctx   the SPARTA vocabulary and options
 * @return diagnostics ordered by line then column
 */
QList<Diagnostic> checkDeck(const QStringList &lines, const Context &ctx);

/// @brief Convenience: split a deck string into lines and validate it.
QList<Diagnostic> checkDeckText(const QString &text, const Context &ctx);

/**
 * @brief Parse the generated command-syntax table into per-command specs.
 * @param tableText contents of @c resources/command_syntax.table
 *
 * The table has one command per line, "@c name @c minArgs @c variadic", with
 * @c # comment lines ignored.  See @c tools/gen_command_syntax.py.
 */
QHash<QString, CommandSpec> parseSyntaxTable(const QString &tableText);

/**
 * @brief Parse the generated command-syntax catalog (JSON) into per-command help.
 * @param json contents of @c resources/command_syntax.json
 *
 * Used by the editor's syntax-aware autocomplete and the validator's error help.
 */
QHash<QString, CommandHelp> parseSyntaxCatalog(const QByteArray &json);

} // namespace InputCheck

#endif // INPUTCHECK_H
// Local Variables:
// c-basic-offset: 4
// End:
