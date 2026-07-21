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

#include "inputcheck.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>

namespace InputCheck {

namespace {

// ---- token with its 0-based start column in the (comment-stripped) line ------
struct Token {
    QString text;
    int col; // 0-based offset in the stripped physical line
};

// A logical line: the joined text of one or more physical lines connected by a
// trailing '&', with the physical line number where it starts.  `multiline` is
// true when it spans more than one physical line (then column info is dropped).
struct LogicalLine {
    QString text;      // comment-stripped, continuation-joined
    int startLine;     // 1-based physical line number of the first piece
    bool multiline;    // spans more than one physical line
};

// Remove a trailing '#...' comment, honoring single/double quotes so a '#'
// inside a quoted string is not treated as a comment.
QString stripComment(const QString &line)
{
    QChar quote(0);
    for (int i = 0; i < line.size(); ++i) {
        const QChar c = line.at(i);
        if (quote != QChar(0)) {
            if (c == quote) quote = QChar(0);
        } else if (c == QLatin1Char('"') || c == QLatin1Char('\'')) {
            quote = c;
        } else if (c == QLatin1Char('#')) {
            return line.left(i);
        }
    }
    return line;
}

// Split a line into whitespace-separated tokens, honoring quotes, and record
// each token's start column.  Surrounding quotes are stripped from the value.
QList<Token> tokenize(const QString &line)
{
    QList<Token> out;
    int i = 0;
    const int n = line.size();
    while (i < n) {
        while (i < n && line.at(i).isSpace()) ++i;
        if (i >= n) break;
        const int start = i;
        QString cur;
        QChar quote(0);
        while (i < n) {
            const QChar c = line.at(i);
            if (quote != QChar(0)) {
                if (c == quote) quote = QChar(0);
                else cur.append(c);
            } else if (c == QLatin1Char('"') || c == QLatin1Char('\'')) {
                quote = c;
            } else if (c.isSpace()) {
                break;
            } else {
                cur.append(c);
            }
            ++i;
        }
        out.append({cur, start});
    }
    return out;
}

// Blank out lines inside triple-quoted here-docs (e.g. the embedded Python of a
// "python ... here \"\"\" ... \"\"\"" command), preserving line numbering so
// those lines are not mistaken for SPARTA commands.
QStringList stripTripleBlocks(const QStringList &lines)
{
    QStringList out;
    out.reserve(lines.size());
    bool inTriple = false;
    for (const QString &line : lines) {
        const int count = line.count(QStringLiteral("\"\"\""));
        if (inTriple) {
            out.append(QString()); // inside a here-doc: not a command
            if (count % 2 == 1) inTriple = false;
        } else {
            out.append(line); // the opening line is still a real command
            if (count % 2 == 1) inTriple = true;
        }
    }
    return out;
}

// Build the logical lines of a deck from its physical lines.
QList<LogicalLine> logicalLines(const QStringList &lines)
{
    QList<LogicalLine> out;
    QString acc;
    int startLine = 0;
    int pieces = 0;
    for (int idx = 0; idx < lines.size(); ++idx) {
        QString code = stripComment(lines.at(idx));
        // a trailing '&' (possibly followed by spaces) means continuation
        QString trimmed = code;
        while (!trimmed.isEmpty() && trimmed.back().isSpace()) trimmed.chop(1);
        const bool cont = trimmed.endsWith(QLatin1Char('&'));
        if (startLine == 0) startLine = idx + 1;
        if (cont) {
            trimmed.chop(1); // drop the '&'
            acc += trimmed + QLatin1Char(' ');
            ++pieces;
            continue;
        }
        acc += code;
        ++pieces;
        out.append({acc, startLine, pieces > 1});
        acc.clear();
        startLine = 0;
        pieces = 0;
    }
    if (!acc.isEmpty() || startLine != 0)
        out.append({acc, startLine == 0 ? int(lines.size()) : startLine, pieces > 1});
    return out;
}

// commands that introduce a style, and the token index of that style
const QHash<QString, int> &styleSlot()
{
    static const QHash<QString, int> m = {
        {QStringLiteral("fix"), 2},         {QStringLiteral("compute"), 2},
        {QStringLiteral("dump"), 2},        {QStringLiteral("region"), 2},
        {QStringLiteral("surf_collide"), 2}, {QStringLiteral("surf_react"), 2},
        {QStringLiteral("collide"), 1},     {QStringLiteral("react"), 1},
    };
    return m;
}

// commands that define an ID at token index 1
const QSet<QString> &defCommands()
{
    static const QSet<QString> s = {
        QStringLiteral("compute"), QStringLiteral("fix"),   QStringLiteral("region"),
        QStringLiteral("group"),   QStringLiteral("mixture"), QStringLiteral("surf_collide"),
        QStringLiteral("surf_react"),
    };
    return s;
}

// commands whose token[1] is a filename to check for existence
const QSet<QString> &fileCommands()
{
    static const QSet<QString> s = {
        QStringLiteral("include"),      QStringLiteral("read_surf"),
        QStringLiteral("read_grid"),    QStringLiteral("read_restart"),
        QStringLiteral("read_particles"),
    };
    return s;
}

bool looksExpanded(const QString &tok)
{
    return tok.contains(QLatin1Char('$'));
}

// strip a trailing accelerator suffix (e.g. "vss/kk" -> "vss") for style lookup.
// SPARTA style names themselves contain '/' (e.g. "emit/face", "thermal/grid"),
// so only a known accelerator suffix may be removed -- not the first '/'.
QString baseStyle(const QString &s)
{
    static const QStringList suffixes = {QStringLiteral("/kk"), QStringLiteral("/kokkos"),
                                         QStringLiteral("/omp"), QStringLiteral("/gpu"),
                                         QStringLiteral("/intel")};
    for (const auto &suf : suffixes)
        if (s.endsWith(suf)) return s.left(s.size() - suf.size());
    return s;
}

} // namespace

QList<Diagnostic> checkDeck(const QStringList &lines, const Context &ctx)
{
    QList<Diagnostic> diags;
    const QList<LogicalLine> logic = logicalLines(stripTripleBlocks(lines));

    // ---- pass 1: collect every defined identifier across the whole deck -------
    // (order-insensitive so we never false-flag a forward reference, and so an
    //  identifier defined later or in an included block is still recognized)
    QSet<QString> vars{QStringLiteral("gui_run")};
    QSet<QString> computes, fixes;
    for (const auto &ll : logic) {
        const QList<Token> t = tokenize(ll.text);
        if (t.isEmpty()) continue;
        const QString &cmd = t.first().text;
        if (cmd == QLatin1String("variable") && t.size() >= 2) vars.insert(t.at(1).text);
        else if (cmd == QLatin1String("compute") && t.size() >= 2) computes.insert(t.at(1).text);
        else if (cmd == QLatin1String("fix") && t.size() >= 2) fixes.insert(t.at(1).text);
    }

    // reference patterns
    static const QRegularExpression reVarBrace(QStringLiteral("\\$\\{([A-Za-z0-9_]+)\\}"));
    static const QRegularExpression reVarChar(QStringLiteral("\\$([A-Za-z])"));
    static const QRegularExpression reVarRef(QStringLiteral("\\bv_([A-Za-z0-9_]+)"));
    static const QRegularExpression reComputeRef(QStringLiteral("\\bc_([A-Za-z0-9_]+)"));
    static const QRegularExpression reFixRef(QStringLiteral("\\bf_([A-Za-z0-9_]+)"));

    auto addRefDiags = [&](const LogicalLine &ll, const QRegularExpression &re,
                           const QSet<QString> &defined, const QString &code,
                           const QString &what) {
        auto it = re.globalMatch(ll.text);
        while (it.hasNext()) {
            const auto m = it.next();
            const QString name = m.captured(1);
            if (defined.contains(name)) continue;
            Diagnostic d;
            d.line = ll.startLine;
            d.column = ll.multiline ? 0 : m.capturedStart(0) + 1;
            d.severity = Severity::Warning;
            d.code = code;
            d.message = QStringLiteral("%1 '%2' is not defined in this file").arg(what, name);
            diags.append(d);
        }
    };

    // ---- pass 2: per-line checks --------------------------------------------
    for (const auto &ll : logic) {
        const QList<Token> t = tokenize(ll.text);
        if (t.isEmpty()) continue;
        const Token &cmdTok = t.first();
        const QString cmd = cmdTok.text;
        const int cmdCol = ll.multiline ? 0 : cmdTok.col + 1;

        const int nargs = t.size() - 1; // arguments after the command word
        const bool known = ctx.commands.contains(cmd) || ctx.commandSpecs.contains(cmd);

        // unknown command
        if (ctx.checkVocabulary && !looksExpanded(cmd) &&
            !(ctx.commands.isEmpty() && ctx.commandSpecs.isEmpty()) && !known) {
            diags.append({ll.startLine, cmdCol, Severity::Error, QStringLiteral("unknown-command"),
                          QStringLiteral("Unknown command '%1'").arg(cmd)});
            continue; // don't cascade further checks off a bad command
        }

        // argument count from the doc-derived command spec
        const auto specIt = ctx.commandSpecs.constFind(cmd);
        bool argCountBad = false;
        if (specIt != ctx.commandSpecs.constEnd()) {
            const CommandSpec &spec = specIt.value();
            if (nargs < spec.minArgs) {
                argCountBad = true;
                diags.append({ll.startLine, cmdCol, Severity::Error,
                              QStringLiteral("too-few-args"),
                              QStringLiteral("'%1' needs at least %2 argument%3 (%4 given)")
                                  .arg(cmd)
                                  .arg(spec.minArgs)
                                  .arg(spec.minArgs == 1 ? "" : "s")
                                  .arg(nargs)});
            } else if (!spec.variadic && nargs > spec.minArgs) {
                diags.append({ll.startLine, cmdCol, Severity::Error,
                              QStringLiteral("too-many-args"),
                              QStringLiteral("'%1' takes exactly %2 argument%3 (%4 given)")
                                  .arg(cmd)
                                  .arg(spec.minArgs)
                                  .arg(spec.minArgs == 1 ? "" : "s")
                                  .arg(nargs)});
            }
        }

        // keyword-led commands (global, run, dump_modify, ...): the token that
        // begins the keyword list must be a documented keyword.  Only the first
        // keyword position is checked -- deeper positions need per-keyword value
        // arities we do not track -- which already catches e.g. "global 1 1".
        // Skipped when the keyword set is unknown (empty) to avoid false alarms.
        if (specIt != ctx.commandSpecs.constEnd() && ctx.checkVocabulary && !argCountBad) {
            const CommandSpec &spec = specIt.value();
            const int slot = spec.keywordStart + 1; // +1 to skip the command word
            if (spec.keywordStart >= 0 && !spec.keywords.isEmpty() && t.size() > slot) {
                const Token &kwTok = t.at(slot);
                if (!looksExpanded(kwTok.text) && !spec.keywords.contains(kwTok.text)) {
                    diags.append({ll.startLine, ll.multiline ? 0 : kwTok.col + 1,
                                  Severity::Error, QStringLiteral("unknown-keyword"),
                                  QStringLiteral("'%1' is not a valid %2 keyword")
                                      .arg(kwTok.text, cmd)});
                }
            }
        }

        // style commands: validity of the style-name token (arg count already
        // handled above, so only report the name when the token is present)
        auto slotIt = styleSlot().constFind(cmd);
        if (slotIt != styleSlot().constEnd() && ctx.checkVocabulary && !argCountBad) {
            const int slot = slotIt.value();
            if (t.size() > slot) {
                const Token &styTok = t.at(slot);
                const QString sty = baseStyle(styTok.text);
                const auto dictIt = ctx.styles.constFind(cmd);
                if (!looksExpanded(styTok.text) && dictIt != ctx.styles.constEnd() &&
                    !dictIt.value().isEmpty() && !dictIt.value().contains(sty)) {
                    diags.append({ll.startLine, ll.multiline ? 0 : styTok.col + 1,
                                  Severity::Error, QStringLiteral("unknown-style"),
                                  QStringLiteral("Unknown %1 style '%2'").arg(cmd, styTok.text)});
                }
            }
        }

        // file commands: existence of the referenced file (token[1])
        if (fileCommands().contains(cmd) && t.size() >= 2 && ctx.fileExists) {
            const Token &fileTok = t.at(1);
            const QString fname = fileTok.text;
            if (!looksExpanded(fname) && !fname.contains(QLatin1Char('*')) &&
                !ctx.fileExists(fname)) {
                diags.append({ll.startLine, ll.multiline ? 0 : fileTok.col + 1,
                              Severity::Warning, QStringLiteral("missing-file"),
                              QStringLiteral("File '%1' was not found").arg(fname)});
            }
        }

        // cross-reference checks (warnings: an include file could satisfy them)
        if (ctx.checkReferences) {
            addRefDiags(ll, reVarBrace, vars, QStringLiteral("undefined-variable"),
                        QStringLiteral("Variable"));
            addRefDiags(ll, reVarChar, vars, QStringLiteral("undefined-variable"),
                        QStringLiteral("Variable"));
            addRefDiags(ll, reVarRef, vars, QStringLiteral("undefined-variable"),
                        QStringLiteral("Variable"));
            addRefDiags(ll, reComputeRef, computes, QStringLiteral("undefined-compute"),
                        QStringLiteral("Compute"));
            addRefDiags(ll, reFixRef, fixes, QStringLiteral("undefined-fix"),
                        QStringLiteral("Fix"));
        }
    }

    // stable order: by line, then column
    std::stable_sort(diags.begin(), diags.end(), [](const Diagnostic &a, const Diagnostic &b) {
        if (a.line != b.line) return a.line < b.line;
        return a.column < b.column;
    });
    return diags;
}

QList<Diagnostic> checkDeckText(const QString &text, const Context &ctx)
{
    // keep empty trailing lines out but preserve interior blanks / line numbers
    const QStringList lines = text.split(QLatin1Char('\n'));
    return checkDeck(lines, ctx);
}

QHash<QString, CommandHelp> parseSyntaxCatalog(const QByteArray &json)
{
    QHash<QString, CommandHelp> out;
    const QJsonDocument doc = QJsonDocument::fromJson(json);
    if (!doc.isObject()) return out;
    const QJsonObject root = doc.object();
    for (auto it = root.constBegin(); it != root.constEnd(); ++it) {
        const QJsonObject o = it.value().toObject();
        CommandHelp h;
        h.syntax = o.value(QStringLiteral("syntax")).toString();
        for (const QJsonValue &v : o.value(QStringLiteral("args")).toArray())
            h.args.append(v.toString());
        for (const QJsonValue &v : o.value(QStringLiteral("keywords")).toArray())
            h.keywords.append(v.toString());
        h.keywordStart = o.value(QStringLiteral("keywordStart")).toInt(-1);
        out.insert(it.key(), h);
    }
    return out;
}

QHash<QString, CommandSpec> parseSyntaxTable(const QString &tableText)
{
    QHash<QString, CommandSpec> specs;
    const QStringList lines = tableText.split(QLatin1Char('\n'));
    for (const QString &raw : lines) {
        const QString line = raw.trimmed();
        if (line.isEmpty() || line.startsWith(QLatin1Char('#'))) continue;
        const QStringList f = line.split(QLatin1Char(' '), Qt::SkipEmptyParts);
        if (f.size() < 3) continue;
        CommandSpec spec;
        bool ok1 = false, ok2 = false;
        spec.minArgs = f.at(1).toInt(&ok1);
        spec.variadic = f.at(2).toInt(&ok2) != 0;
        if (ok1 && ok2) specs.insert(f.at(0), spec);
    }
    return specs;
}

} // namespace InputCheck
// Local Variables:
// c-basic-offset: 4
// End:
