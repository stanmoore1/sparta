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

#include "codeeditor.h"
#include "constants.h"
#include "fileviewer.h"
#include "helpers.h"
#include "spartagui.h"
#include "spartawrapper.h"
#include "linenumberarea.h"

#include <QAbstractItemView>
#include <QAction>
#include <QCompleter>
#include <QDesktopServices>
#include <QDir>
#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QDropEvent>
#include <QFileInfo>
#include <QFont>
#include <QIcon>
#include <QKeySequence>
#include <QMenu>
#include <QMimeData>
#include <QHelpEvent>
#include <QPainter>
#include <QTextEdit>
#include <QToolTip>
#include <QRect>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSettings>
#include <QShortcut>
#include <QStringListModel>
#include <QTextBlock>
#include <QTextCursor>
#include <QTextDocumentFragment>
#include <QUrl>
#include <QVariant>
#include <QWidget>

CodeEditor::CodeEditor(QWidget *parent) :
    QPlainTextEdit(parent), mainWindow(qobject_cast<SpartaGui *>(parent)), currentComp(nullptr),
    commandComp(new QCompleter(this)), fixComp(new QCompleter(this)),
    computeComp(new QCompleter(this)), dumpComp(new QCompleter(this)),
    regionComp(new QCompleter(this)), collideComp(new QCompleter(this)),
    reactComp(new QCompleter(this)), surfCollideComp(new QCompleter(this)),
    surfReactComp(new QCompleter(this)), variableComp(new QCompleter(this)),
    unitsComp(new QCompleter(this)), groupComp(new QCompleter(this)),
    varnameComp(new QCompleter(this)), fixidComp(new QCompleter(this)),
    compidComp(new QCompleter(this)), mixtureComp(new QCompleter(this)),
    fileComp(new QCompleter(this)), highlight(NO_HIGHLIGHT), highlighterror(false),
    reformatOnReturn(false), automaticCompletion(true)
{
    helpAction = new QShortcut(QKeySequence::fromString("Ctrl+?"), parent);
    connect(helpAction, &QShortcut::activated, this, &CodeEditor::getHelp);

    // set up each completer with consistent settings
    auto setupCompleter = [this](QCompleter *completer) {
        completer->setCompletionMode(QCompleter::UnfilteredPopupCompletion);
        completer->setModelSorting(QCompleter::CaseInsensitivelySortedModel);
        completer->setWidget(this);
        completer->setMaxVisibleItems(16);
        completer->setWrapAround(false);
        connect(completer, QOverload<const QString &>::of(&QCompleter::activated), this,
                &CodeEditor::insertCompletedCommand);
    };

    for (auto *c : {commandComp, fixComp, computeComp, dumpComp, regionComp, collideComp,
                    reactComp, surfCollideComp, surfReactComp, variableComp, unitsComp,
                    groupComp, varnameComp, fixidComp, compidComp, mixtureComp, fileComp})
        setupCompleter(c);

    // initialize help system
    QFile help_index(":/help_index.table");
    if (help_index.open(QIODevice::ReadOnly | QIODevice::Text)) {
        while (!help_index.atEnd()) {
            auto line  = QString(help_index.readLine());
            auto words = line.trimmed().split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
            if (words.size() > 2) {

                if (words.at(1) == "fix") {
                    fixMap[words.at(2)] = words.at(0);
                } else if (words.at(1) == "compute") {
                    computeMap[words.at(2)] = words.at(0);
                } else if (words.at(1) == "dump") {
                    dumpMap[words.at(2)] = words.at(0);
                } else if (words.at(1) == "surf_react") {
                    surfReactMap[words.at(2)] = words.at(0);
                }
            } else if (words.size() == 2) {
                cmdMap[words.at(1)] = words.at(0);
            } else {
                fprintf(stderr, "unhandled help item: %s\n", qPrintable(line.trimmed()));
            }
        }
        help_index.close();
    }

    setBackgroundRole(QPalette::Light);
    lineNumberArea = new LineNumberArea(this);
    lineNumberArea->setBackgroundRole(QPalette::Dark);
    lineNumberArea->setAutoFillBackground(true);
    connect(this, &CodeEditor::blockCountChanged, this, &CodeEditor::updateLineNumberAreaWidth);
    connect(this, &CodeEditor::updateRequest, this, &CodeEditor::updateLineNumberArea);
    updateLineNumberAreaWidth(0);
    setCursorWidth(2);

    // doc-derived command syntax help for call tips + linter error help
    QFile syntaxFile(QStringLiteral(":/command_syntax.json"));
    if (syntaxFile.open(QIODevice::ReadOnly))
        cmdHelp = InputCheck::parseSyntaxCatalog(syntaxFile.readAll());

    // banner watermark: shown only while the editor is empty; hidden once the user
    // types or a file is loaded.  Toggle it as the document's emptiness changes.
    bannerVisible = document()->isEmpty();
    refreshEditorStyle();
    connect(this, &QPlainTextEdit::textChanged, this, [this] {
        if (document()->isEmpty() != bannerVisible) refreshEditorStyle();
        // a run-error highlight is stale as soon as the deck is edited
        clearErrorHighlight();
    });
}

QString CodeEditor::editorStyleSheet(const QColor &background, const QColor &foreground,
                                     bool withBanner)
{
    // This is applied to the scroll-area viewport (the text surface), so it is a bare
    // property list, not a CodeEditor{...} rule.  The background color must live in the
    // stylesheet (not the palette): while an application-wide stylesheet is active, a
    // widget/viewport palette change to the surface is overridden.
    QString ss;
    if (withBanner)
        ss += QStringLiteral("background-position: center center; background-repeat: no-repeat; "
                             "background-image: url(:/icons/sparta-gui-banner.png);");
    if (background.isValid()) ss += QStringLiteral(" background-color: %1;").arg(background.name());
    if (foreground.isValid()) ss += QStringLiteral(" color: %1;").arg(foreground.name());
    return ss;
}

void CodeEditor::refreshEditorStyle()
{
    bannerVisible = document()->isEmpty();
    viewport()->setStyleSheet(editorStyleSheet(schemeBg, schemeFg, bannerVisible));
    viewport()->update();
}

void CodeEditor::setColorScheme(const QColor &background, const QColor &foreground)
{
    schemeBg = background;
    schemeFg = foreground;

    if (background.isValid() && foreground.isValid()) {
        // Derive a subtly contrasting gutter from the background so the line-number
        // margin stays legible without needing a separate color in every scheme.
        const bool dark     = background.lightness() < 128;
        const QColor gutter = dark ? background.lighter(140) : background.darker(112);
        lineNumberArea->setStyleSheet(QStringLiteral("background-color: %1;").arg(gutter.name()));
    } else {
        // theme default (e.g. the "Classic" scheme): drop the overrides
        lineNumberArea->setStyleSheet(QString());
    }
    refreshEditorStyle();
    lineNumberArea->update();
    update();
}

CodeEditor::~CodeEditor()
{
    // helpAction's parent is the main window (not this widget), so we must
    // delete it explicitly rather than rely on Qt's parent-child ownership.
    // It is a QPointer because, now that this widget is nested inside the
    // dock manager instead of being a direct child of the main window, Qt's
    // own child-teardown order may destroy helpAction first (it is still a
    // direct main-window child) -- QPointer turns that into a safe no-op
    // instead of a dangling-pointer delete.  All other children
    // (lineNumberArea, completers) are Qt children of this widget and are
    // automatically deleted by Qt's parent-child ownership.
    delete helpAction;
}

int CodeEditor::lineNumberAreaWidth()
{
    int digits = 1;
    int max    = qMax(1, blockCount());
    while (max >= 10) {
        max /= 10;
        ++digits;
    }

    int space = 3 + (fontMetrics().horizontalAdvance(QLatin1Char('9')) * (digits + 2));
    return space;
}

void CodeEditor::setFont(const QFont &newfont)
{
    lineNumberArea->setFont(newfont);
    document()->setDefaultFont(newfont);
}

void CodeEditor::setCursor(int block)
{
    // move cursor to given position
    auto cursor = textCursor();
    auto bl     = document()->findBlockByNumber(block);
    if (bl.isValid()) {
        cursor.setPosition(bl.position());
        setTextCursor(cursor);
    }
}

void CodeEditor::setHighlight(int block, bool error)
{
    // a separate error flag: encoding the error state in the sign of the
    // block number cannot represent an error on block 0
    highlight      = block;
    highlighterror = error;

    // also reset the cursor
    setCursor(block);

    // an error highlight also paints a full-width line background in the editor
    // body (not just the gutter marker) so the offending line stands out; the
    // progress (green) highlight stays gutter-only to avoid flashing on every
    // thermo update during a run.  refreshDiagSelections() draws both the
    // diagnostic overlays and this error line, so keep them in sync here.
    refreshDiagSelections();

    // update graphics
    repaint();
}

void CodeEditor::clearErrorHighlight()
{
    if (highlight == NO_HIGHLIGHT || !highlighterror) return;
    highlight      = NO_HIGHLIGHT;
    highlighterror = false;
    refreshDiagSelections(); // drop the red full-width band
    lineNumberArea->update(); // drop the gutter marker
}

void CodeEditor::setDiagnostics(const QList<InputCheck::Diagnostic> &diags)
{
    diagMarks.clear();
    for (const auto &d : diags) {
        const int block = d.line - 1;
        if (block < 0) continue;
        const int rank = (d.severity == InputCheck::Severity::Error)     ? 2
                         : (d.severity == InputCheck::Severity::Warning) ? 1
                                                                         : 0;
        DiagMark &m = diagMarks[block];
        m.severity = qMax(m.severity, rank);
        if (!m.tip.isEmpty()) m.tip += QLatin1Char('\n');
        m.tip += d.message;
    }
    refreshDiagSelections();
    lineNumberArea->update();
}

void CodeEditor::clearDiagnostics()
{
    if (diagMarks.isEmpty()) return;
    diagMarks.clear();
    // rebuild (rather than clear outright) so an active run-error line highlight
    // survives the diagnostics being cleared
    refreshDiagSelections();
    lineNumberArea->update();
}

void CodeEditor::refreshDiagSelections()
{
    QList<QTextEdit::ExtraSelection> sels;
    const bool light = isLightTheme();
    // translucent overlays that read on both light and dark editor backgrounds
    const QColor errBg = light ? QColor(220, 40, 40, 40) : QColor(255, 90, 90, 55);
    const QColor warnBg = light ? QColor(220, 150, 0, 40) : QColor(255, 200, 60, 45);
    for (auto it = diagMarks.constBegin(); it != diagMarks.constEnd(); ++it) {
        const QTextBlock bl = document()->findBlockByNumber(it.key());
        if (!bl.isValid()) continue;
        QTextEdit::ExtraSelection sel;
        sel.format.setBackground(it.value().severity >= 2 ? errBg : warnBg);
        sel.format.setProperty(QTextFormat::FullWidthSelection, true);
        sel.cursor = QTextCursor(bl);
        sel.cursor.clearSelection();
        sels.append(sel);
    }

    // the run-error line: a stronger full-width band so the failed (or last-run)
    // line is obvious in the editor body, drawn last so it wins over any faint
    // diagnostic overlay on the same line
    if ((highlight != NO_HIGHLIGHT) && highlighterror && (highlight >= 0)) {
        const QTextBlock bl = document()->findBlockByNumber(highlight);
        if (bl.isValid()) {
            const QColor band = light ? QColor(230, 60, 60, 80) : QColor(255, 80, 80, 90);
            QTextEdit::ExtraSelection sel;
            sel.format.setBackground(band);
            sel.format.setProperty(QTextFormat::FullWidthSelection, true);
            sel.cursor = QTextCursor(bl);
            sel.cursor.clearSelection();
            sels.append(sel);
        }
    }

    setExtraSelections(sels);
}

bool CodeEditor::event(QEvent *event)
{
    if (event->type() == QEvent::ToolTip && !diagMarks.isEmpty()) {
        auto *he = static_cast<QHelpEvent *>(event);
        const QTextCursor cur = cursorForPosition(viewport()->mapFrom(this, he->pos()));
        const auto it = diagMarks.constFind(cur.blockNumber());
        if (it != diagMarks.constEnd())
            QToolTip::showText(he->globalPos(), it.value().tip, this);
        else
            QToolTip::hideText();
        return true;
    }
    return QPlainTextEdit::event(event);
}

// reformat line

QString CodeEditor::reformatLine(const QString &line)
{
    auto words = splitLine(line);
    QString newtext;
    QSettings settings;
    settings.beginGroup(Keys::GROUP_REFORMAT);
    int cmdsize  = settings.value(Keys::COMMAND, "16").toInt();
    int typesize = settings.value(Keys::TYPE, "4").toInt();
    int idsize   = settings.value(Keys::ID, "4").toInt();
    int namesize = settings.value(Keys::NAME, "8").toInt();
    settings.endGroup();

    bool rebuildGroupComp     = false;
    bool rebuildVarNameComp   = false;
    bool rebuildComputeIDComp = false;
    bool rebuildFixIDComp     = false;
    bool rebuildMixtureIDComp = false;

    if (!words.isEmpty()) {
        // commented line. do nothing
        if (words[0][0] == '#') return line;

        // start with SPARTA command plus padding if another word follows
        newtext = words[0];
        if (words.size() > 1) {
            for (int i = words[0].size() + 1; i < cmdsize; ++i)
                newtext += ' ';
            // new/updated group command -> update completer
            if (words[0] == "group") rebuildGroupComp = true;
            // new/updated variable command -> update completer
            if (words[0] == "variable") rebuildVarNameComp = true;
            // new/updated compute command -> update completer
            if (words[0] == "compute") rebuildComputeIDComp = true;
            // new/updated fix command -> update completer
            if (words[0] == "fix") rebuildFixIDComp = true;
            // new/updated mixture command -> update completer
            if (words[0] == "mixture") rebuildMixtureIDComp = true;
        }

        // append remaining words with just a single blank added.
        for (int i = 1; i < words.size(); ++i) {
            newtext += ' ';
            newtext += words[i];

            // special cases

            if (i < 3) {
                // pad IDs and styles of fix/compute/dump commands
                if ((words[0] == "fix") || (words[0] == "compute") || (words[0] == "dump")) {
                    if (i == 1) {
                        for (int j = words[i].size(); j < idsize; ++j)
                            newtext += ' ';
                    } else if (i == 2) {
                        for (int j = words[i].size(); j < namesize; ++j)
                            newtext += ' ';
                    }
                }
            }

            if (i < 2) {
                // additional space for species in mixture assignments
                if (words[0] == "mixture")
                    for (int j = words[i].size(); j < typesize; ++j)
                        newtext += ' ';
            }
        }
    }
    if (rebuildGroupComp) setGroupList();
    if (rebuildVarNameComp) setVarNameList();
    if (rebuildComputeIDComp) setComputeIDList();
    if (rebuildFixIDComp) setFixIDList();
    if (rebuildMixtureIDComp) setMixtureIDList();
    return newtext;
}

#define COMPLETER_INIT_FUNC(keyword, Type)                                   \
    void CodeEditor::set##Type##List(const QStringList &words)               \
    {                                                                        \
        keyword##Comp->setModel(new QStringListModel(words, keyword##Comp)); \
    }

COMPLETER_INIT_FUNC(command, Command)
COMPLETER_INIT_FUNC(fix, Fix)
COMPLETER_INIT_FUNC(compute, Compute)
COMPLETER_INIT_FUNC(dump, Dump)
COMPLETER_INIT_FUNC(region, Region)
COMPLETER_INIT_FUNC(collide, Collide)
COMPLETER_INIT_FUNC(react, React)
COMPLETER_INIT_FUNC(surfCollide, SurfCollide)
COMPLETER_INIT_FUNC(surfReact, SurfReact)
COMPLETER_INIT_FUNC(variable, Variable)
COMPLETER_INIT_FUNC(units, Units)

#undef COMPLETER_INIT_FUNC

// build completer for groups by parsing through edit buffer

void CodeEditor::setGroupList()
{
    QStringList groups;
    QRegularExpression groupcmd(QStringLiteral(R"(^\s*group\s+(\S+)(\s+|$))"));

    auto saved = textCursor();
    // reposition cursor to beginning of text and search for group commands
    auto cursor = textCursor();
    cursor.movePosition(QTextCursor::Start);
    setTextCursor(cursor);
    while (find(groupcmd)) {
        auto words = splitLine(textCursor().block().text().replace('\t', ' '));
        if ((words.size() > 1) && !groups.contains(words[1])) groups << words[1];
    }
    groups.sort();
    groups.prepend(QStringLiteral("all"));

    setTextCursor(saved);
    groupComp->setModel(new QStringListModel(groups, groupComp));
}

void CodeEditor::setVarNameList()
{
    QStringList vars;

    // variable "gui_run" is always defined by SPARTA-GUI
    vars << QString("${gui_run}");
    vars << QString("v_gui_run");

    SpartaWrapper *sparta = &mainWindow->sparta;
    int nvar              = sparta->idCount("variable");
    for (int i = 0; i < nvar; ++i) {
        const QString name = sparta->variableInfo(i);
        if (!name.isEmpty()) {
            if (name.size() == 1) vars << QString("$%1").arg(name);
            vars << QString("${%1}").arg(name);
            vars << QString("v_%1").arg(name);
        }
    }

    QRegularExpression varcmd(QStringLiteral(R"(^\s*variable\s+(\S+)(\s+|$))"));
    auto saved = textCursor();
    // reposition cursor to beginning of text and search for variable commands
    auto cursor = textCursor();
    cursor.movePosition(QTextCursor::Start);
    setTextCursor(cursor);
    while (find(varcmd)) {
        auto words = splitLine(textCursor().block().text().replace('\t', ' '));
        if ((words.size() > 1)) {
            QString w = QString("$%1").arg(words[1]);
            if ((words[1].size() == 1) && !vars.contains(w)) vars << w;
            w = QString("${%1}").arg(words[1]);
            if (!vars.contains(w)) vars << w;
            w = QString("v_%1").arg(words[1]);
            if (!vars.contains(w)) vars << w;
        }
    }
    vars.sort();

    setTextCursor(saved);
    varnameComp->setModel(new QStringListModel(vars, varnameComp));
}

void CodeEditor::setComputeIDList()
{
    QStringList compid;
    QRegularExpression compcmd(QStringLiteral(R"(^\s*compute\s+(\S+)\s+)"));

    auto saved = textCursor();
    // reposition cursor to beginning of text and search for compute commands
    auto cursor = textCursor();
    cursor.movePosition(QTextCursor::Start);
    setTextCursor(cursor);
    while (find(compcmd)) {
        auto words = splitLine(textCursor().block().text().replace('\t', ' '));
        if ((words.size() > 1)) {
            QString w = QString("c_%1").arg(words[1]);
            if (!compid.contains(w)) compid << w;
            w = QString("C_%1").arg(words[1]);
            if (!compid.contains(w)) compid << w;
        }
    }
    compid.sort();

    setTextCursor(saved);
    compidComp->setModel(new QStringListModel(compid, compidComp));
}

void CodeEditor::setFixIDList()
{
    QStringList fixid;
    QRegularExpression fixcmd(QStringLiteral(R"(^\s*fix\s+(\S+)\s+)"));

    auto saved = textCursor();
    // reposition cursor to beginning of text and search for fix commands
    auto cursor = textCursor();
    cursor.movePosition(QTextCursor::Start);
    setTextCursor(cursor);
    while (find(fixcmd)) {
        auto words = splitLine(textCursor().block().text().replace('\t', ' '));
        if ((words.size() > 1)) {
            QString w = QString("f_%1").arg(words[1]);
            if (!fixid.contains(w)) fixid << w;
            w = QString("F_%1").arg(words[1]);
            if (!fixid.contains(w)) fixid << w;
        }
    }
    fixid.sort();

    setTextCursor(saved);
    fixidComp->setModel(new QStringListModel(fixid, fixidComp));
}

// build completer for mixture IDs from the SPARTA instance and the edit buffer

void CodeEditor::setMixtureIDList()
{
    QStringList mixid;

    // query mixtures known to the SPARTA instance (includes the
    // predefined mixtures "all" and "species")
    SpartaWrapper *sparta = &mainWindow->sparta;
    int nmix              = sparta->idCount("mixture");
    for (int i = 0; i < nmix; ++i) {
        const QString name = sparta->idName("mixture", i);
        if (!name.isEmpty() && !mixid.contains(name)) mixid << name;
    }
    if (!mixid.contains("all")) mixid << QStringLiteral("all");
    if (!mixid.contains("species")) mixid << QStringLiteral("species");

    QRegularExpression mixcmd(QStringLiteral(R"(^\s*mixture\s+(\S+)(\s+|$))"));
    auto saved = textCursor();
    // reposition cursor to beginning of text and search for mixture commands
    auto cursor = textCursor();
    cursor.movePosition(QTextCursor::Start);
    setTextCursor(cursor);
    while (find(mixcmd)) {
        auto words = splitLine(textCursor().block().text().replace('\t', ' '));
        if ((words.size() > 1) && !mixid.contains(words[1])) mixid << words[1];
    }
    mixid.sort();

    setTextCursor(saved);
    mixtureComp->setModel(new QStringListModel(mixid, mixtureComp));
}

void CodeEditor::setFileList()
{
    QStringList files;
    QDir dir(".");
    for (const auto &file : dir.entryInfoList(QDir::Files))
        files << file.fileName();
    files.sort();
    fileComp->setModel(new QStringListModel(files, fileComp));
}

void CodeEditor::keyPressEvent(QKeyEvent *event)
{
    const auto key = event->key();

    if (currentComp && currentComp->popup()->isVisible()) {
        // The following keys are forwarded by the completer to the widget
        switch (key) {
            case Qt::Key_Enter:
            case Qt::Key_Return:
            case Qt::Key_Escape:
            case Qt::Key_Tab:
            case Qt::Key_Backtab:
                event->ignore();
                return; // let the completer do default behavior
            default:
                break;
        }
    }

    // reformat current line and consume key event
    if (key == Qt::Key_Tab) {
        reformatCurrentLine();
        return;
    }

    // run command completion and consume key event
    if (key == Qt::Key_Backtab) {
        runCompletion();
        return;
    }

    // automatically reformat when hitting the return or enter key; the flag is
    // maintained through setReformatOnReturn() when the preferences change --
    // re-reading QSettings here would both override the setter and cost a
    // settings lookup on every keystroke
    if (reformatOnReturn && ((key == Qt::Key_Return) || (key == Qt::Key_Enter))) {
        reformatCurrentLine();
    }

    // process key event in parent class
    QPlainTextEdit::keyPressEvent(event);

    // if enabled, try pop up completion automatically after 2 characters
    if (automaticCompletion) {
        auto cursor = textCursor();
        auto line   = cursor.block().text();
        if (line.isEmpty()) return;

        // QTextCursor::WordUnderCursor is unusable here since it recognizes '/' as word boundary.
        // Work around it by manually searching for the location of the beginning of the word.
        int begin = qMin(cursor.positionInBlock(), line.length() - 1);

        while (begin >= 0) {
            if (line[begin].isSpace()) break;
            --begin;
        }
        if (((cursor.positionInBlock() - begin) > 2) ||
            ((line.length() > begin + 1) && (line[begin + 1] == '$')))
            runCompletion();
        if (currentComp && currentComp->popup()->isVisible() &&
            ((cursor.positionInBlock() - begin) < 2)) {
            currentComp->popup()->hide();
        }
    }
}

void CodeEditor::updateLineNumberAreaWidth(int /* newBlockCount */)
{
    setViewportMargins(lineNumberAreaWidth(), 0, 0, 0);
}

void CodeEditor::updateLineNumberArea(const QRect &rect, int dy)
{
    if (dy)
        lineNumberArea->scroll(0, dy);
    else
        lineNumberArea->update(0, rect.y(), lineNumberArea->width(), rect.height());

    if (rect.contains(viewport()->rect())) updateLineNumberAreaWidth(0);
}

void CodeEditor::dragEnterEvent(QDragEnterEvent *event)
{
    event->acceptProposedAction();
}

void CodeEditor::dragLeaveEvent(QDragLeaveEvent *event)
{
    event->accept();
    cut();
    QPlainTextEdit::dragLeaveEvent(event);
}

bool CodeEditor::canInsertFromMimeData(const QMimeData *source) const
{
    return source->hasUrls() || source->hasText();
}

void CodeEditor::dropEvent(QDropEvent *event)
{
    if (event->mimeData()->hasUrls()) {
        event->accept();
        auto file = event->mimeData()->urls()[0].toLocalFile();
        auto *gui = mainWindow;
        if (gui) {
            moveCursor(QTextCursor::Start, QTextCursor::MoveAnchor);
            gui->openFile(file);
        }
        // properly handle drop event in base class, but set editor
        // buffer readonly to prevent undesired changes
        setReadOnly(true);
        QPlainTextEdit::dropEvent(event);
        setReadOnly(false);
    } else if (event->mimeData()->hasText()) {
        event->accept();
        // cut selected text to clipboard before we reposition
        // the cursor and re-insert the text with drag-n-drop
        cut();
        cursorForPosition(event->position().toPoint()).insertText(event->mimeData()->text());
        // properly handle drop event in base class, but set editor
        // buffer readonly to prevent undesired changes
        setReadOnly(true);
        QPlainTextEdit::dropEvent(event);
        setReadOnly(false);
    } else
        event->ignore();
}

void CodeEditor::resizeEvent(QResizeEvent *e)
{
    QPlainTextEdit::resizeEvent(e);

    QRect cr = contentsRect();
    lineNumberArea->setGeometry(QRect(cr.left(), cr.top(), lineNumberAreaWidth(), cr.height()));
}

void CodeEditor::lineNumberAreaPaintEvent(QPaintEvent *event)
{
    QPainter painter(lineNumberArea);
    QTextBlock block = firstVisibleBlock();
    int blockNumber  = block.blockNumber();

    int top    = qRound(blockBoundingGeometry(block).translated(contentOffset()).top());
    int bottom = top + qRound(blockBoundingRect(block).height());
    while (block.isValid() && top <= event->rect().bottom()) {
        if (block.isVisible() && bottom >= event->rect().top()) {
            QString number = QString::number(blockNumber + 1) + " ";
            if ((highlight == NO_HIGHLIGHT) || (blockNumber != highlight)) {
                painter.setPen(schemeFg.isValid() ? schemeFg
                                                  : palette().color(QPalette::WindowText));
            } else {
                number = QString(">") + QString::number(blockNumber + 1) + "<";
                if (highlighterror)
                    painter.fillRect(0, top, lineNumberArea->width(), fontMetrics().height(),
                                     Qt::darkRed);
                else
                    painter.fillRect(0, top, lineNumberArea->width(), fontMetrics().height(),
                                     Qt::darkGreen);

                painter.setPen(Qt::white);
            }
            painter.drawText(0, top, lineNumberArea->width(), fontMetrics().height(),
                             Qt::AlignRight, number);

            // validation marker: a small dot at the left edge of the gutter
            const auto dm = diagMarks.constFind(blockNumber);
            if (dm != diagMarks.constEnd()) {
                const int h = fontMetrics().height();
                const int d = qMax(4, h / 3);
                painter.setRenderHint(QPainter::Antialiasing, true);
                painter.setPen(Qt::NoPen);
                painter.setBrush(dm.value().severity >= 2 ? QColor(210, 40, 40)
                                                          : QColor(220, 150, 0));
                painter.drawEllipse(2, top + (h - d) / 2, d, d);
                painter.setRenderHint(QPainter::Antialiasing, false);
            }
        }

        block  = block.next();
        top    = bottom;
        bottom = top + qRound(blockBoundingRect(block).height());
        ++blockNumber;
    }
}

void CodeEditor::contextMenuEvent(QContextMenuEvent *event)
{
    // reposition the cursor here, but only if there is no active selection
    if (!textCursor().hasSelection()) setTextCursor(cursorForPosition(event->pos()));

    QString page, help;
    findHelp(page, help);

    auto *menu = createStandardContextMenu();
    menu->addSeparator();
    auto *gui = mainWindow;
    if (textCursor().hasSelection()) {
        addMenuAction(menu, "Comment out selection", ":/icons/comment-out.svg", this,
                      &CodeEditor::commentSelection);
        addMenuAction(menu, "Uncomment selection", ":/icons/uncomment.svg", this,
                      &CodeEditor::uncommentSelection);
    } else {
        addMenuAction(menu, "Comment out line", ":/icons/comment-out.svg", this,
                      &CodeEditor::commentLine);
        addMenuAction(menu, "Uncomment line", ":/icons/uncomment.svg", this,
                      &CodeEditor::uncommentLine);
    }
    menu->addSeparator();
    SpartaWrapper *sparta = &gui->sparta;
    if (sparta->isRunning()) {
        addMenuAction(menu, "Stop SPARTA", ":/icons/process-stop.svg", gui, &SpartaGui::stopRun);
    } else {
        addMenuAction(menu, "Run SPARTA from Editor Buffer", ":/icons/system-run.svg", gui,
                      &SpartaGui::runBuffer);
        addMenuAction(menu, "Run SPARTA from File", ":/icons/run-file.svg", gui,
                      &SpartaGui::runFile);
    }
    menu->addSeparator();

    // print augmented context menu if an entry was found
    if (!help.isEmpty()) {
        addMenuAction(menu, QString("Display available completions for '%1'").arg(help),
                      ":/icons/expand-text.svg", this, &CodeEditor::runCompletion);
        menu->addSeparator();
    }

    if (!page.isEmpty()) {
        addMenuAction(menu, QString("Reformat '%1' command").arg(help),
                      ":/icons/format-indent-less-3.svg", this, &CodeEditor::reformatCurrentLine);

        menu->addSeparator();
        addMenuAction(menu, QString("View Documentation for '%1'").arg(help),
                      ":/icons/system-help.svg", this, &CodeEditor::openHelp)
            ->setData(page);
        // if we link to help with specific styles (fix, compute, pair, bond, ...)
        // also link to the docs for the primary command
        auto words = help.split(' ', Qt::SkipEmptyParts);
        if (words.size() > 1) {
            help = words.at(0);
            page = words.at(0);
            page += ".html";
            addMenuAction(menu, QString("View Documentation for '%1'").arg(help),
                          ":/icons/system-help.svg", this, &CodeEditor::openHelp)
                ->setData(page);
        }
    }

    // check if word under cursor is file
    {
        auto cursor = textCursor();
        auto line   = cursor.block().text();
        if (!line.isEmpty()) {
            // QTextCursor::WordUnderCursor is unusable here since it recognizes '/' as word
            // boundary. Work around it by manually searching for the location of the beginning of
            // the word.
            int begin = qMin(cursor.positionInBlock(), line.length() - 1);

            while (begin >= 0) {
                if (line[begin].isSpace()) break;
                --begin;
            }
            int end = begin + 1;
            while (end < line.length()) {
                if (line[end].isSpace()) break;
                ++end;
            }

            QString word = line.mid(begin, end - begin).trimmed();
            QFileInfo fi(word);
            if (fi.exists() && fi.isFile()) {
                // check if file is a SPARTA restart
                if (isRestartFile(word)) {
                    addMenuAction(menu, QString("Inspect restart file '%1'").arg(word),
                                  ":/icons/document-open.svg", this, &CodeEditor::inspectFile)
                        ->setData(word);
                } else {
                    addMenuAction(menu, QString("View file '%1'").arg(word),
                                  ":/icons/document-open.svg", this, &CodeEditor::viewFile)
                        ->setData(word);
                    // for editable (non-image/binary) files -- e.g. an include or
                    // read_surf/read_grid target -- also offer to open it for editing
                    static const QRegularExpression binext(
                        QStringLiteral("\\.(png|jpe?g|gif|bmp|ppm|tiff?|mp4|avi|mov|mpe?g|gz|bz2|"
                                       "zip|bin|so|o|a|dll|exe)$"),
                        QRegularExpression::CaseInsensitiveOption);
                    if (!binext.match(word).hasMatch())
                        addMenuAction(menu, QString("Open '%1' in editor").arg(word),
                                      ":/icons/document-open.svg", this, &CodeEditor::openInEditor)
                            ->setData(word);
                }
            }
        }
    }

    addMenuAction(menu, QString("SPARTA Commands Overview"), ":/icons/help-browser.svg", this,
                  &CodeEditor::openHelp)
        ->setData(QString("Section_commands.html"));

    addMenuAction(menu, QString("SPARTA Manual"), ":/icons/help-browser.svg", this,
                  &CodeEditor::openHelp)
        ->setData(QString("Manual.html"));

    menu->exec(event->globalPos());
    delete menu;
}

void CodeEditor::reformatCurrentLine()
{
    auto cursor  = textCursor();
    auto text    = cursor.block().text();
    auto newtext = reformatLine(text);

    // perform edit but only if text has changed
    if (QString::compare(text, newtext)) {
        cursor.beginEditBlock();
        cursor.movePosition(QTextCursor::StartOfLine);
        cursor.movePosition(QTextCursor::EndOfLine, QTextCursor::KeepAnchor, 1);
        cursor.insertText(newtext);
        cursor.endEditBlock();
    }
}

void CodeEditor::commentLine()
{
    auto cursor = textCursor();
    cursor.movePosition(QTextCursor::StartOfLine);
    cursor.insertText("#");
}

void CodeEditor::commentSelection()
{
    auto cursor = textCursor();
    auto text   = cursor.selection().toPlainText();
    auto lines  = text.split('\n');
    QString newtext;
    for (const auto &line : lines) {
        newtext.append('#');
        newtext.append(line);
        newtext.append('\n');
    }
    if (newtext.isEmpty()) newtext = "#\n";
    cursor.insertText(newtext);
    setTextCursor(cursor);
}

void CodeEditor::uncommentSelection()
{
    auto cursor = textCursor();
    auto text   = cursor.selection().toPlainText();
    auto lines  = text.split('\n');
    QString newtext;
    for (const auto &line : lines) {
        QString newline;
        bool start = true;
        for (auto letter : line) {
            if (start && (letter == '#')) {
                start = false;
                continue;
            }
            if (start && !letter.isSpace()) start = false;
            newline.append(letter);
        }
        newtext.append(newline);
        newtext.append('\n');
    }
    cursor.insertText(newtext);
    setTextCursor(cursor);
}

void CodeEditor::uncommentLine()
{
    auto cursor = textCursor();
    auto text   = cursor.block().text();
    QString newtext;
    bool start = true;
    for (auto letter : text) {
        if (start && (letter == '#')) {
            start = false;
            continue;
        }
        if (start && !letter.isSpace()) start = false;
        newtext.append(letter);
    }

    // perform edit but only if text has changed
    if (QString::compare(text, newtext)) {
        cursor.beginEditBlock();
        cursor.movePosition(QTextCursor::StartOfLine);
        cursor.movePosition(QTextCursor::EndOfLine, QTextCursor::KeepAnchor, 1);
        cursor.insertText(newtext);
        cursor.endEditBlock();
    }
}

// Pop up (or hide) the completion list of currentComp for the given prefix,
// hiding the popup of a previously active completer. Shared by all completion
// contexts past the first word in CodeEditor::runCompletion().
void CodeEditor::popupCompletion(const QString &prefix, QAbstractItemView *oldPopup)
{
    currentComp->setCompletionPrefix(prefix);
    if (oldPopup && (oldPopup != currentComp->popup())) oldPopup->hide();
    auto *popup = currentComp->popup();
    // if the word is already a complete command, remove an existing popup
    if (prefix == currentComp->currentCompletion()) {
        if (popup->isVisible()) popup->hide();
        return;
    }
    QRect cr = cursorRect();
    cr.setWidth(popup->sizeHintForColumn(0) + popup->verticalScrollBar()->sizeHint().width());
    popup->setAlternatingRowColors(true);
    currentComp->complete(cr);
}

void CodeEditor::runCompletion()
{
    QAbstractItemView *popup = nullptr;
    if (currentComp) popup = currentComp->popup();

    auto cursor = textCursor();
    auto line   = cursor.block().text().trimmed();
    // no completion possible on empty lines
    if (line.isEmpty()) return;
    auto words = splitLine(line);

    // QTextCursor::WordUnderCursor is unusable here since it recognizes '/' as word boundary.
    // Work around it by manually searching for the beginning and end position of the word
    // under the cursor and then using that substring.
    line      = cursor.block().text();
    int begin = qMin(cursor.positionInBlock(), line.length() - 1);
    while (begin >= 0) {
        if (line[begin].isSpace()) break;
        --begin;
    }
    int end = ++begin;
    while (end < line.length()) {
        if (line[end].isSpace()) break;
        ++end;
    }
    const auto selected = line.mid(begin, end - begin);

    // if on first word, try to complete command
    if ((!words.isEmpty()) && (words[0] == selected)) {
        // no completion on comment lines
        if (words[0][0] == '#') return;

        currentComp = commandComp;
        currentComp->setCompletionPrefix(words[0]);
        if (popup && (popup != currentComp->popup())) popup->hide();
        popup = currentComp->popup();
        // if the command is already a complete command, remove existing popup
        if (words[0] == currentComp->currentCompletion()) {
            if (popup->isVisible()) {
                popup->hide();
                currentComp = nullptr;
            }
            return;
        }
        QRect cr = cursorRect();
        cr.setWidth(popup->sizeHintForColumn(0) + popup->verticalScrollBar()->sizeHint().width());
        popup->setAlternatingRowColors(true);
        currentComp->complete(cr);

        // completions for second word
    } else if ((words.size() > 1) && (words[1] == selected)) {
        // no completion on comment lines
        if (words[0][0] == '#') return;

        currentComp = nullptr;
        if (words[0] == "collide")
            currentComp = collideComp;
        else if (words[0] == "react")
            currentComp = reactComp;
        else if (words[0] == "units")
            currentComp = unitsComp;
        else if (words[0] == "create_particles")
            currentComp = mixtureComp;
        else if ((words[0] == "adapt_grid") || (words[0] == "read_isurf"))
            currentComp = groupComp;
        else if ((words[0] == "include") || (words[0] == "jump") || (words[0] == "read_grid") ||
                 (words[0] == "read_restart") || (words[0] == "read_surf")) {
            if (selected.contains('/')) {
                if (popup && popup->isVisible()) popup->hide();
            } else
                currentComp = fileComp;
        } else if (selected.startsWith("v_"))
            currentComp = varnameComp;
        else if (selected.startsWith("c_") || selected.startsWith("C_"))
            currentComp = compidComp;
        else if (selected.startsWith("f_") || selected.startsWith("F_"))
            currentComp = fixidComp;

        if (currentComp) popupCompletion(words[1], popup);
        // completions for third word
    } else if ((words.size() > 2) && (words[2] == selected)) {
        // no completion on comment lines
        if (words[0][0] == '#') return;

        currentComp = nullptr;
        if (words[0] == "region")
            currentComp = regionComp;
        else if (words[0] == "variable")
            currentComp = variableComp;
        else if (words[0] == "fix")
            currentComp = fixComp;
        else if (words[0] == "compute")
            currentComp = computeComp;
        else if (words[0] == "dump")
            currentComp = dumpComp;
        else if (words[0] == "surf_collide")
            currentComp = surfCollideComp;
        else if (words[0] == "surf_react")
            currentComp = surfReactComp;
        else if (selected.startsWith("v_"))
            currentComp = varnameComp;
        else if (selected.startsWith("c_") || selected.startsWith("C_"))
            currentComp = compidComp;
        else if (selected.startsWith("f_") || selected.startsWith("F_"))
            currentComp = fixidComp;

        if (currentComp) popupCompletion(words[2], popup);
        // completions for fourth word
    } else if ((words.size() > 3) && (words[3] == selected)) {
        // no completion on comment lines
        if (words[0][0] == '#') return;

        currentComp = nullptr;
        // "dump ID style mix-ID ..." and "fix ID emit/... mix-ID ..." take a mixture ID
        if ((words[0] == "dump") || ((words[0] == "fix") && words[2].startsWith("emit/")))
            currentComp = mixtureComp;
        else if (selected.startsWith("v_"))
            currentComp = varnameComp;
        else if (selected.startsWith("c_") || selected.startsWith("C_"))
            currentComp = compidComp;
        else if (selected.startsWith("f_") || selected.startsWith("F_"))
            currentComp = fixidComp;

        if (currentComp) popupCompletion(words[3], popup);
        // reference located anywhere further right in the line
    } else if (words.size() > 4) {
        currentComp = nullptr;
        if (selected.startsWith("v_"))
            currentComp = varnameComp;
        else if (selected.startsWith("c_") || selected.startsWith("C_"))
            currentComp = compidComp;
        else if (selected.startsWith("f_") || selected.startsWith("F_"))
            currentComp = fixidComp;

        if (currentComp) popupCompletion(selected, popup);
    }
}

void CodeEditor::insertCompletedCommand(const QString &completion)
{
    auto *completer = qobject_cast<QCompleter *>(sender());
    if (completer->widget() != this) return;

    // select the entire word (non-space text) under the cursor
    // we need to do it in this complicated way, since QTextCursor does not recognize
    // special characters as part of a word.
    auto cursor = textCursor();
    auto line   = cursor.block().text();
    int begin   = qMin(cursor.positionInBlock(), line.length() - 1);

    while (begin >= 0) {
        if (line[begin].isSpace()) break;
        --begin;
    }

    int end = begin + 1;
    while (end < line.length()) {
        if (line[end].isSpace()) break;
        ++end;
    }

    cursor.setPosition(cursor.position() - cursor.positionInBlock() + begin + 1);
    cursor.movePosition(QTextCursor::NextCharacter, QTextCursor::KeepAnchor, end - begin - 1);
    cursor.insertText(completion);
    setTextCursor(cursor);

    // once a command is completed, show its required/optional fields as a call tip
    showCommandCallTip();
}

void CodeEditor::showCommandCallTip()
{
    if (cmdHelp.isEmpty()) return;
    const QString first =
        textCursor().block().text().trimmed().section(QRegularExpression("\\s+"), 0, 0);
    const auto it = cmdHelp.constFind(first);
    if (it == cmdHelp.constEnd() || it.value().syntax.isEmpty()) return;
    QString tip = QStringLiteral("<b>%1</b>").arg(it.value().syntax.toHtmlEscaped());
    if (!it.value().keywords.isEmpty())
        tip += QStringLiteral("<br><i>keywords:</i> %1")
                   .arg(it.value().keywords.join(QStringLiteral(", ")).toHtmlEscaped());
    QToolTip::showText(viewport()->mapToGlobal(cursorRect().bottomLeft()), tip, this);
}

void CodeEditor::getHelp()
{
    QString page, help;
    findHelp(page, help);
    // the SPARTA online documentation is not versioned
    if (!page.isEmpty())
        QDesktopServices::openUrl(QUrl(QString("%1/doc/%2").arg(Cfg::DOCS_URL, page)));
}

void CodeEditor::findHelp(QString &page, QString &help)
{
    // process line of text where the cursor is
    auto text = textCursor().block().text().replace('\t', ' ').trimmed();
    help.clear();
    page.clear();

    // fix/compute/dump/surf_collide/surf_react have their style as the third word
    auto style = QRegularExpression(R"(^(fix|compute|dump|surf_collide|surf_react)\s+\w+\s+(\S+))")
                     .match(text);
    if (style.hasMatch()) {
        help = QString("%1 %2").arg(style.captured(1), style.captured(2));
        if (style.captured(1) == "fix") {
            page = fixMap.value(style.captured(2), QString());
        } else if (style.captured(1) == "compute") {
            page = computeMap.value(style.captured(2), QString());
        } else if (style.captured(1) == "dump") {
            page = dumpMap.value(style.captured(2), QString());
        } else if (style.captured(1) == "surf_react") {
            page = surfReactMap.value(style.captured(2), QString());
        }
    }

    // could not find a matching "style", now try the plain command
    if (page.isEmpty() && !text.isEmpty()) {
        auto cmd = text.section(' ', 0, 0);
        help     = cmd;
        page     = cmdMap.value(cmd, QString());
    }
}

void CodeEditor::openHelp()
{
    auto *act = qobject_cast<QAction *>(sender());
    // the SPARTA online documentation is not versioned
    QDesktopServices::openUrl(
        QUrl(QString("%1/doc/%2").arg(Cfg::DOCS_URL, act->data().toString())));
}

// forward requests to view or inspect files to the corresponding SpartaGui methods

void CodeEditor::viewFile()
{
    auto *act     = qobject_cast<QAction *>(sender());
    auto *guimain = mainWindow;
    guimain->viewFile(act->data().toString());
}

void CodeEditor::openInEditor()
{
    auto *act = qobject_cast<QAction *>(sender());
    if (act && mainWindow) mainWindow->openFile(act->data().toString());
}

void CodeEditor::inspectFile()
{
    auto *act     = qobject_cast<QAction *>(sender());
    auto *guimain = mainWindow;
    guimain->inspectFile(act->data().toString());
}

// Local Variables:
// c-basic-offset: 4
// End:
