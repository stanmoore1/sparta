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

#ifndef HIGHLIGHTER_H
#define HIGHLIGHTER_H

#include <QRegularExpression>
#include <QSyntaxHighlighter>
#include <QTextCharFormat>

/**
 * @brief Syntax highlighter for SPARTA input scripts
 *
 * This class extends QSyntaxHighlighter to provide syntax highlighting
 * for SPARTA input files in the CodeEditor. It categorizes and styles
 * commands, keywords, variables, numbers, strings, and comments.
 */
class Highlighter : public QSyntaxHighlighter {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param parent Parent text document to highlight
     */
    explicit Highlighter(QTextDocument *parent = nullptr);

    /**
     * @brief Destructor
     */
    ~Highlighter() override = default;

    Highlighter()                               = delete;
    Highlighter(const Highlighter &)            = delete;
    Highlighter(Highlighter &&)                 = delete;
    Highlighter &operator=(const Highlighter &) = delete;
    Highlighter &operator=(Highlighter &&)      = delete;

    /**
     * @brief Switch the syntax color palette and re-highlight the document
     * @param scheme scheme id (see @ref schemeIds); unknown ids fall back to the default
     *
     * Only the token foreground colors change; the light/dark variant is
     * chosen automatically from the current application theme.
     */
    void applyScheme(const QString &scheme);

    /// @brief Machine ids of the available color schemes (stored in QSettings)
    static QStringList schemeIds();
    /// @brief Human-readable labels for the schemes, parallel to @ref schemeIds
    static QStringList schemeLabels();
    /// @brief The default scheme id used when none is stored
    static QString defaultScheme();

protected:
    /**
     * @brief Highlight a single block (line) of text
     * @param text The text to highlight
     */
    void highlightBlock(const QString &text) override;

private:
    // Regular expressions for different SPARTA command categories
    QRegularExpression isLattice1, isLattice2, isLattice3; ///< Box, grid, and setup commands
    QRegularExpression isOutput1, isOutput2, isRead;       ///< Output and input commands
    QTextCharFormat formatOutput, formatRead, formatLattice,
        formatSetup;                                         ///< Formats for setup commands
    QRegularExpression isStyle, isForce, isDefine, isUndo;   ///< Styled and define/undefine commands
    QRegularExpression isParticle, isRun, isSetup, isSetup1; ///< Particle, surface, and run commands
    QTextCharFormat formatParticle, formatRun;               ///< Formats for various command types
    QRegularExpression isVariable, isReference; ///< Variable definitions and references
    QTextCharFormat formatVariable;             ///< Format for variables
    QRegularExpression isNumber1, isNumber2, isNumber3, isNumber4; ///< Various number formats
    QTextCharFormat formatNumber;                                  ///< Format for numbers
    QRegularExpression isSpecial, isContinue; ///< Special keywords and line continuations
    QTextCharFormat formatSpecial;            ///< Format for special keywords
    QRegularExpression isComment;             ///< Comment patterns
    QRegularExpression isQuotedComment;       ///< Quoted comment patterns
    QTextCharFormat formatComment;            ///< Format for comments
    QRegularExpression isTriple;              ///< Triple-quoted strings
    QRegularExpression isString;              ///< Regular strings
    QTextCharFormat formatString;             ///< Format for strings

    bool in_triple; ///< State flag for multi-line triple-quoted strings

    /// @brief Apply a scheme's colors to the QTextCharFormat members (no rehighlight)
    void setFormats(const QString &scheme);
};
#endif
// Local Variables:
// c-basic-offset: 4
// End:
