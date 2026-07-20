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

#include "highlighter.h"
#include "constants.h"
#include "helpers.h"
#include <QColor>
#include <QSettings>

namespace {
/**
 * @brief Foreground colors for every token category of one color scheme.
 *
 * The highlighter recognizes eleven token categories; a scheme is just a table
 * of eleven colors (the font weight/style is fixed per category, not per
 * scheme).  Each scheme supplies a light and a dark variant so it works with
 * either application theme.
 */
struct Palette {
    QColor number, string, comment, special, variable;
    QColor lattice, setup, particle, run, read, output;
};

// ---- Classic (the legacy SPARTA-GUI palette) ------------------------------
const Palette classicLight = {Qt::blue,     Qt::darkGreen, Qt::red,     Qt::darkMagenta,
                              Qt::darkGray, Qt::darkGreen, Qt::darkCyan, Qt::darkRed,
                              Qt::darkBlue, Qt::magenta,   Qt::darkYellow};
const Palette classicDark  = {
    QColorConstants::Svg::dodgerblue, QColorConstants::Green,          QColorConstants::Red,
    QColorConstants::Magenta,         QColorConstants::Svg::lightgray, QColorConstants::Svg::lightgreen,
    QColorConstants::Cyan,            QColorConstants::Svg::indianred, QColorConstants::Svg::lightskyblue,
    QColorConstants::Svg::lightcoral, QColorConstants::Yellow};

// ---- Solarized (Ethan Schoonover, published cross-editor standard) --------
// order: number string comment special variable | lattice setup particle run read output
const Palette solarizedLight = {QColor("#6c71c4"), QColor("#2aa198"), QColor("#93a1a1"),
                                QColor("#cb4b16"), QColor("#b58900"), QColor("#268bd2"),
                                QColor("#859900"), QColor("#d33682"), QColor("#cb4b16"),
                                QColor("#6c71c4"), QColor("#b58900")};
const Palette solarizedDark  = {QColor("#6c71c4"), QColor("#2aa198"), QColor("#586e75"),
                                QColor("#cb4b16"), QColor("#b58900"), QColor("#268bd2"),
                                QColor("#859900"), QColor("#d33682"), QColor("#cb4b16"),
                                QColor("#6c71c4"), QColor("#b58900")};

// ---- VS Code (Light+ / Dark+, the default in the editor most widely used) -
const Palette vscodeLight = {QColor("#098658"), QColor("#a31515"), QColor("#008000"),
                             QColor("#af00db"), QColor("#0070c1"), QColor("#0000ff"),
                             QColor("#267f99"), QColor("#af00db"), QColor("#0000ff"),
                             QColor("#795e26"), QColor("#001080")};
const Palette vscodeDark  = {QColor("#b5cea8"), QColor("#ce9178"), QColor("#6a9955"),
                             QColor("#c586c0"), QColor("#9cdcfe"), QColor("#569cd6"),
                             QColor("#4ec9b0"), QColor("#c586c0"), QColor("#569cd6"),
                             QColor("#dcdcaa"), QColor("#4fc1ff")};

// ---- One (Atom, One Light / One Dark) -------------------------------------
const Palette oneLight = {QColor("#986801"), QColor("#50a14f"), QColor("#a0a1a7"),
                          QColor("#a626a4"), QColor("#e45649"), QColor("#4078f2"),
                          QColor("#0184bc"), QColor("#c18401"), QColor("#a626a4"),
                          QColor("#986801"), QColor("#e45649")};
const Palette oneDark  = {QColor("#d19a66"), QColor("#98c379"), QColor("#7f848e"),
                          QColor("#c678dd"), QColor("#e06c75"), QColor("#61afef"),
                          QColor("#56b6c2"), QColor("#e5c07b"), QColor("#c678dd"),
                          QColor("#d19a66"), QColor("#e06c75")};

const Palette &paletteFor(const QString &scheme, bool light)
{
    if (scheme == QLatin1String("classic")) return light ? classicLight : classicDark;
    if (scheme == QLatin1String("solarized")) return light ? solarizedLight : solarizedDark;
    if (scheme == QLatin1String("one")) return light ? oneLight : oneDark;
    // default and "vscode"
    return light ? vscodeLight : vscodeDark;
}
} // namespace

QStringList Highlighter::schemeIds()
{
    return {QStringLiteral("vscode"), QStringLiteral("solarized"), QStringLiteral("one"),
            QStringLiteral("classic")};
}

QStringList Highlighter::schemeLabels()
{
    return {QStringLiteral("VS Code"), QStringLiteral("Solarized"), QStringLiteral("One (Atom)"),
            QStringLiteral("Classic (legacy)")};
}

QString Highlighter::defaultScheme()
{
    return QStringLiteral("vscode");
}

void Highlighter::setFormats(const QString &scheme)
{
    const Palette &p = paletteFor(scheme, isLightTheme());

    // Comments are set apart with italics (and never red as an accent), so they
    // read as annotations rather than as errors regardless of the scheme.
    formatComment.setForeground(p.comment);
    formatComment.setFontItalic(true);
    formatComment.setFontWeight(QFont::Normal);

    formatNumber.setForeground(p.number);
    formatNumber.setFontWeight(QFont::Normal);
    formatString.setForeground(p.string);
    formatString.setFontWeight(QFont::Normal);

    // command families and keyword-like tokens are emphasized with bold
    formatSpecial.setForeground(p.special);
    formatSpecial.setFontWeight(QFont::Bold);
    formatVariable.setForeground(p.variable);
    formatVariable.setFontWeight(QFont::Bold);
    formatLattice.setForeground(p.lattice);
    formatLattice.setFontWeight(QFont::Bold);
    formatSetup.setForeground(p.setup);
    formatSetup.setFontWeight(QFont::Bold);
    formatParticle.setForeground(p.particle);
    formatParticle.setFontWeight(QFont::Bold);
    formatRun.setForeground(p.run);
    formatRun.setFontWeight(QFont::Bold);
    formatRead.setForeground(p.read);
    formatRead.setFontWeight(QFont::Bold);
    formatOutput.setForeground(p.output);
    formatOutput.setFontWeight(QFont::Bold);
}

void Highlighter::applyScheme(const QString &scheme)
{
    setFormats(scheme);
    rehighlight();
}

Highlighter::Highlighter(QTextDocument *parent) :
    QSyntaxHighlighter(parent),
    isLattice1(QStringLiteral("^\\s*(units|dimension|seed|timestep|global|package)\\s+(\\S+)")),
    isLattice2(QStringLiteral("^\\s*(create_box|create_grid|create_particles|create_"
                              "isurf)\\s+(\\S+)\\s+(\\S+)")),
    isLattice3(QStringLiteral("^\\s*(boundary)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)")),
    isOutput1(QStringLiteral("^\\s*(echo|log|print|restart|"
                             "stats_style|stats_modify|stats|"
                             "write_grid|write_isurf|write_restart|write_surf)\\s+(\\S+)")),
    isOutput2(QStringLiteral("^\\s*(shell|dump_modify)\\s+(\\S+)\\s+(\\S+)")),
    isRead(QStringLiteral(
        "^\\s*(include|read_restart|read_grid|read_isurf|read_particles|read_surf)\\s+(\\S+)")),
    isStyle(QStringLiteral("^\\s*(fix|compute|dump)\\s+(\\S+)\\s+(\\S+)")),
    isForce(QStringLiteral(
        "^\\s*(collide_modify|collide|react_modify|react)\\s+(\\S+)")),
    isDefine(QStringLiteral("^\\s*(group|variable|python|region|mixture|surf_collide|surf_react)"
                            "\\s+(\\S+)\\s+(\\S+)")),
    isUndo(QStringLiteral("^\\s*(unfix|uncompute|undump|label|jump|next)\\s+(\\S+)")),
    isParticle(QStringLiteral("^\\s*(species_modify|species|bound_modify|surf_modify|"
                              "move_surf|remove_surf)\\s+(\\S+)")),
    isRun(QStringLiteral("^\\s*(run|balance_grid|adapt_grid|custom|scale_particles|partition)")),
    isSetup(QStringLiteral("^\\s*(clear|quit)")),
    isSetup1(QStringLiteral("^\\s*(reset_timestep)\\s+(\\S+)")),
    isVariable(QStringLiteral("(\\$[a-z]|\\${[^} ]+}|\\$\\(\\S+\\))")),
    isReference(
        QStringLiteral("\\s+(c_\\S+|f_\\S+|v_\\S+|p_\\S+|g_\\S+|s_\\S+)")),
    isNumber1(QStringLiteral("(^|\\s+)[-+]?[0-9:*]+")), // integer and integer ranges
    isNumber2(QStringLiteral("(^|\\s+)[-+]?[0-9]+\\.[0-9]*[edED]?[-+]?[0-9]*")), // floating point 1
    isNumber3(QStringLiteral("(^|\\s+)[-+]?[0-9]*\\.[0-9]+[edED]?[-+]?[0-9]*")), // floating point 2
    isNumber4(QStringLiteral(
        "(^|\\s+)[-+]?[0-9]+([edED][-+]?[0-9]+)?")), // integer with optional exponent
    isSpecial(QStringLiteral("(\\sINF|\\sEDGE|\\sNULL|\\sSELF|if\\s|then\\s|else\\s|elif\\s)")),
    isContinue(QStringLiteral("&$")), isComment(QStringLiteral("#.*")),
    isQuotedComment(QStringLiteral("(\".*#.*\"|'.*#.*')")),
    isTriple(QStringLiteral("[^\"]*\"\"\"[^\"]*")),
    isString(QStringLiteral("(\".+?\"|'.+?'|\"\"\".*\"\"\")")), in_triple(false)
{
    // pick the syntax color palette from the stored preference (default: VS Code);
    // the light/dark variant follows the current application theme
    const QString scheme =
        QSettings().value(Keys::COLOR_SCHEME, Highlighter::defaultScheme()).toString();
    setFormats(scheme);
}

void Highlighter::highlightBlock(const QString &text)
{
    // nothing to do for empty lines
    if (text.isEmpty()) return;

    auto match = isLattice1.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatLattice);
        setFormat(match.capturedStart(2), match.capturedLength(2), formatRun);
    }

    match = isLattice2.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatLattice);
        setFormat(match.capturedStart(2), match.capturedLength(2), formatString);
        setFormat(match.capturedStart(3), match.capturedLength(3), formatRun);
    }

    match = isLattice3.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatLattice);
        setFormat(match.capturedStart(2), match.capturedLength(2), formatString);
        setFormat(match.capturedStart(3), match.capturedLength(3), formatString);
        setFormat(match.capturedStart(4), match.capturedLength(4), formatString);
    }

    match = isOutput1.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatOutput);
        setFormat(match.capturedStart(2), match.capturedLength(2), formatString);
    }

    match = isOutput2.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatOutput);
        setFormat(match.capturedStart(2), match.capturedLength(2), formatString);
        setFormat(match.capturedStart(3), match.capturedLength(3), formatRun);
    }

    match = isRead.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatRead);
        setFormat(match.capturedStart(2), match.capturedLength(2), formatString);
    }

    match = isStyle.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatParticle);
        setFormat(match.capturedStart(2), match.capturedLength(2), formatNumber);
        setFormat(match.capturedStart(3), match.capturedLength(3), formatRun);
    }

    match = isForce.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatParticle);
        setFormat(match.capturedStart(2), match.capturedLength(2), formatRun);
    }

    match = isUndo.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatSpecial);
        setFormat(match.capturedStart(2), match.capturedLength(2), formatString);
    }

    match = isDefine.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatParticle);
        setFormat(match.capturedStart(2), match.capturedLength(2), formatString);
        setFormat(match.capturedStart(3), match.capturedLength(3), formatRun);
    }

    match = isParticle.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatParticle);
        setFormat(match.capturedStart(2), match.capturedLength(2), formatString);
    }

    match = isRun.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatRun);
    }

    match = isSetup.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatSetup);
    }

    match = isSetup1.match(text);
    if (match.hasMatch()) {
        setFormat(match.capturedStart(1), match.capturedLength(1), formatSetup);
        setFormat(match.capturedStart(2), match.capturedLength(2), formatString);
    }

    // numbers
    const QRegularExpression *numbers[] = {&isNumber1, &isNumber2, &isNumber3, &isNumber4};
    for (const auto *number : numbers) {
        auto num = number->globalMatch(text);
        while (num.hasNext()) {
            auto hit = num.next();
            setFormat(hit.capturedStart(), hit.capturedLength(), formatNumber);
        }
    }

    // variables
    auto vars = isVariable.globalMatch(text);
    while (vars.hasNext()) {
        auto hit = vars.next();
        setFormat(hit.capturedStart(), hit.capturedLength(), formatVariable);
    }

    // references
    auto refs = isReference.globalMatch(text);
    while (refs.hasNext()) {
        auto hit = refs.next();
        setFormat(hit.capturedStart(), hit.capturedLength(), formatVariable);
    }

    // continuation character
    auto multiline = isContinue.match(text);
    if (multiline.hasMatch())
        setFormat(multiline.capturedStart(0), multiline.capturedLength(0), formatSpecial);

    // special keywords
    auto special = isSpecial.globalMatch(text);
    while (special.hasNext()) {
        auto hit = special.next();
        setFormat(hit.capturedStart(), hit.capturedLength(), formatSpecial);
    }

    // comments, must come before strings but after other keywords.
    auto comment = isComment.match(text);
    if (comment.hasMatch() && !isQuotedComment.match(text).hasMatch() && !in_triple) {
        setFormat(comment.capturedStart(0), comment.capturedLength(0), formatComment);
        return;
    }

    // strings, must come last so they can overwrite other formatting
    auto string = isString.globalMatch(text);
    while (string.hasNext()) {
        auto hit = string.next();
        setFormat(hit.capturedStart(), hit.capturedLength(), formatString);
    }

    auto triple = isTriple.match(text);
    if (triple.hasMatch()) {
        if (in_triple) {
            in_triple = false;
            setFormat(0, triple.capturedStart(0) + triple.capturedLength(0), formatString);
        } else {
            in_triple = true;
            setFormat(triple.capturedStart(0), -1, formatString);
        }
    } else {
        if (in_triple) setFormat(0, text.size(), formatString);
    }
}
// Local Variables:
// c-basic-offset: 4
// End:
