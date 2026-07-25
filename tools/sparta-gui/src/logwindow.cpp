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

#include "logwindow.h"

#include "constants.h"
#include "flagwarnings.h"
#include "helpers.h"
#include "spartagui.h"

#include <QAction>
#include <QDesktopServices>
#include <QFile>
#include <QFileDialog>
#include <QFont>
#include <QFontInfo>
#include <QHBoxLayout>
#include <QIcon>
#include <QKeySequence>
#include <QLabel>
#include <QMenu>
#include <QPushButton>
#include <QRegularExpression>
#include <QSettings>
#include <QShortcut>
#include <QString>
#include <QTextStream>

namespace {
constexpr auto YAML_REGEX = R"(^(keywords:.*$|data:$|---$|\.\.\.$|  - \[.*\]$))";
constexpr auto URL_REGEX  = "^.*(https://sparta.github.io/err[0-9]+).*$";
QRegularExpression is_yaml(YAML_REGEX, QRegularExpression::MultilineOption);
constexpr int BADGE_MARGIN = 3;
} // namespace

LogWindow::LogWindow(const QString &_filename, SpartaGui *_spartagui, QWidget *parent) :
    QPlainTextEdit(parent), filename(_filename), spartagui(_spartagui), warnings(nullptr)
{
    QSettings settings;
    resize(settings.value(Keys::LOGX, 500).toInt(), settings.value(Keys::LOGY, 320).toInt());

    document()->setDefaultFont(monoFontFromSettings());

    summary = new QLabel("0 Warnings / Errors - 0 Lines");
    summary->setMargin(1);

    auto *frame = new QFrame;
    frame->setAutoFillBackground(true);
    frame->setFrameStyle(QFrame::Box | QFrame::Plain);
    frame->setLineWidth(2);

    auto *button = new QPushButton(QIcon(":/icons/warning.svg"), "");
    button->setToolTip("Jump to next warning");
    connect(button, &QPushButton::released, this, &LogWindow::nextWarning);

    auto *panel = new QHBoxLayout(frame);
    panel->addWidget(summary);
    panel->addWidget(button);
    panel->setStretchFactor(summary, 10);
    panel->setStretchFactor(button, 1);

    // The badge sits in a strip reserved below the text, not on top of it. It
    // used to be positioned by a layout installed on this widget, which put it
    // over the last couple of lines of output -- exactly the lines someone
    // watching a run is reading. setViewportMargins() takes the strip out of
    // the scrolling area instead, so text can never flow underneath.
    warningBadge = frame;
    // the layout used to adopt it; without one it needs a parent of its own
    warningBadge->setParent(this);
    warningBadge->show();
    warningBadge->adjustSize();
    setViewportMargins(0, 0, 0, warningBadge->sizeHint().height() + 2 * BADGE_MARGIN);

    warnings = new FlagWarnings(summary, document());

    // the summary is rewritten as output arrives, and a longer line count needs
    // a wider badge
    connect(document(), &QTextDocument::contentsChanged, this, &LogWindow::placeWarningBadge);

    // WidgetWithChildrenShortcut (rather than the default WindowShortcut) so
    // these only compete with the identically-bound main-window menu
    // shortcuts while focus is actually inside this docked panel, instead of
    // unconditionally while its window (now shared with the main window) is
    // active; the eventFilter() override below resolves that in-focus case
    auto *action = new QShortcut(QKeySequence("Ctrl+S"), this);
    action->setContext(Qt::WidgetWithChildrenShortcut);
    connect(action, &QShortcut::activated, this, &LogWindow::saveAs);
    action = new QShortcut(QKeySequence("Ctrl+Y"), this);
    action->setContext(Qt::WidgetWithChildrenShortcut);
    connect(action, &QShortcut::activated, this, &LogWindow::extractYaml);
    action = new QShortcut(QKeySequence("Ctrl+Q"), this);
    action->setContext(Qt::WidgetWithChildrenShortcut);
    connect(action, &QShortcut::activated, this, &LogWindow::quit);
    action = new QShortcut(QKeySequence("Ctrl+N"), this);
    action->setContext(Qt::WidgetWithChildrenShortcut);
    connect(action, &QShortcut::activated, this, &LogWindow::nextWarning);
    action = new QShortcut(QKeySequence("Ctrl+/"), this);
    action->setContext(Qt::WidgetWithChildrenShortcut);
    connect(action, &QShortcut::activated, this, &LogWindow::stopRun);
    action = new QShortcut(QKeySequence("Ctrl+Return"), this);
    action->setContext(Qt::WidgetWithChildrenShortcut);
    connect(action, &QShortcut::activated, this, &LogWindow::runBuffer);

    installEventFilter(this);
    applyWindowFlags(this);
}

// warnings and summary are Qt-parented and cleaned up by their parents
LogWindow::~LogWindow() = default;

void LogWindow::closeEvent(QCloseEvent *event)
{
    QSettings settings;
    if (!isMaximized()) {
        settings.setValue(Keys::LOGX, width());
        settings.setValue(Keys::LOGY, height());
    }
    QPlainTextEdit::closeEvent(event);
}

void LogWindow::quit()
{
    if (spartagui) spartagui->quit();
}

void LogWindow::stopRun()
{
    if (spartagui) spartagui->stopRun();
}

void LogWindow::runBuffer()
{
    if (spartagui) spartagui->runBuffer();
}

// Centre the badge in the strip reserved for it under the text.
void LogWindow::placeWarningBadge()
{
    if (!warningBadge) return;
    // Re-fit every time: the summary starts at "0 Warnings / Errors - 0 Lines"
    // and grows as the run produces output, so a width measured once at
    // construction ends up cutting the last character off.
    warningBadge->adjustSize();
    const QSize hint = warningBadge->size();
    const QRect box  = contentsRect();
    warningBadge->setGeometry(box.center().x() - hint.width() / 2,
                              box.bottom() - hint.height() - BADGE_MARGIN, hint.width(),
                              hint.height());
}

void LogWindow::resizeEvent(QResizeEvent *event)
{
    QPlainTextEdit::resizeEvent(event);
    placeWarningBadge();
}

void LogWindow::showEvent(QShowEvent *event)
{
    QPlainTextEdit::showEvent(event);
    // the widget has its real size only once shown
    placeWarningBadge();
}

void LogWindow::nextWarning()
{
    auto regex = QRegularExpression(QStringLiteral("^(ERROR|WARNING).*$"));

    if (warnings->getNWarnings() > 0) {
        // wrap around search
        if (!find(regex)) {
            moveCursor(QTextCursor::Start, QTextCursor::MoveAnchor);
            find(regex);
        }
        // move cursor to unselect
        moveCursor(QTextCursor::NextBlock, QTextCursor::MoveAnchor);
    }
}

void LogWindow::saveAs()
{
    QString defaultname = filename + ".log";
    if (filename.isEmpty()) defaultname = "sparta.log";
    QString logFileName = QFileDialog::getSaveFileName(this, "Save Log to File", defaultname,
                                                       "Log files (*.log *.out *.txt)");
    if (logFileName.isEmpty()) return;

    QFileInfo path(logFileName);
    QFile file(path.absoluteFilePath());

    if (!file.open(QIODevice::WriteOnly | QFile::Text)) {
        warning(this, "LogWindow Warning", "Cannot save to file " + logFileName + ":",
                file.errorString());
        return;
    }

    QTextStream out(&file);
    QString text = toPlainText();
    out << text;
    if (!text.endsWith('\n')) out << "\n"; // add final newline if missing
    file.close();
}

bool LogWindow::checkYaml()
{
    return document()->find(is_yaml).isNull() == false;
}

void LogWindow::extractYaml()
{
    // ignore if no YAML format lines in buffer
    if (!checkYaml()) return;

    QString defaultname = filename + ".yaml";
    if (filename.isEmpty()) defaultname = "sparta.yaml";
    QString yamlFileName = QFileDialog::getSaveFileName(this, "Save YAML data to File", defaultname,
                                                        "YAML files (*.yaml *.yml)");
    // cannot save without filename
    if (yamlFileName.isEmpty()) return;

    QFileInfo path(yamlFileName);
    QFile file(path.absoluteFilePath());
    if (!file.open(QIODevice::WriteOnly | QFile::Text)) {
        warning(this, "LogWindow Warning", "Cannot save to file " + yamlFileName + ":",
                file.errorString());
        return;
    }

    QTextStream out(&file);
    for (auto block = document()->begin(); block != document()->end(); block = block.next()) {
        auto line = block.text();
        if (is_yaml.match(line).hasMatch()) out << line << '\n';
    }
    file.close();
}

void LogWindow::openErrorUrl()
{
    if (!errorurl.isEmpty()) QDesktopServices::openUrl(QUrl(errorurl));
}

void LogWindow::mouseDoubleClickEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
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

        auto text = cursor.selectedText();
        auto url  = QRegularExpression(URL_REGEX).match(text);
        if (url.hasMatch()) {
            errorurl = url.captured(1);
            if (!errorurl.isEmpty()) {
                QDesktopServices::openUrl(QUrl(errorurl));
                return;
            }
        }
    }
    // forward event to parent class for all unhandled cases
    QPlainTextEdit::mouseDoubleClickEvent(event);
}

void LogWindow::contextMenuEvent(QContextMenuEvent *event)
{
    // reposition the cursor here, but only if there is no active selection
    if (!textCursor().hasSelection()) setTextCursor(cursorForPosition(event->pos()));

    // show augmented context menu
    auto *menu = createStandardContextMenu();
    menu->addSeparator();
    addMenuAction(menu, QString("Save Log to File ..."), ":/icons/document-save-as.svg", this,
                  &LogWindow::saveAs)
        ->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_S));
    // only show export-to-yaml entry if there is YAML format content.
    if (checkYaml()) {
        addMenuAction(menu, QString("&Export YAML Data to File ..."), ":/icons/yaml-file-icon.svg",
                      this, &LogWindow::extractYaml)
            ->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Y));
    }

    // process line of text where the cursor is
    auto text = textCursor().block().text().replace('\t', ' ').trimmed();
    auto url  = QRegularExpression(URL_REGEX).match(text);
    if (url.hasMatch()) {
        errorurl = url.captured(1);
        addMenuAction(menu, "Open &URL in Web Browser", ":/icons/help-browser.svg", this,
                      &LogWindow::openErrorUrl);
    }
    addMenuAction(menu, "&Jump to next warning or error", ":/icons/warning.svg", this,
                  &LogWindow::nextWarning)
        ->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_N));
    menu->addSeparator();
    addMenuAction(menu, "&Close Window", ":/icons/window-close.svg", this, &QWidget::close)
        ->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_W));
    addMenuAction(menu, "&Quit SPARTA-GUI", ":/icons/application-exit.svg", this, &LogWindow::quit)
        ->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Q));
    menu->exec(event->globalPos());
    delete menu;
}

// event filter to handle "Ambiguous shortcut override" issues
bool LogWindow::eventFilter(QObject *watched, QEvent *event)
{
    if (event->type() == QEvent::ShortcutOverride) {
        auto *keyEvent = dynamic_cast<QKeyEvent *>(event);
        if (!keyEvent) return QAbstractScrollArea::eventFilter(watched, event);
        if (keyEvent->modifiers().testFlag(Qt::ControlModifier) && keyEvent->key() == '/') {
            stopRun();
            event->accept();
            return true;
        }
        if (keyEvent->modifiers().testFlag(Qt::ControlModifier) && keyEvent->key() == 'W') {
            close();
            event->accept();
            return true;
        }
        // now that this window is a docked panel sharing the main window's
        // shortcut context, its own Ctrl+N/S/Q/Return would otherwise be
        // ambiguous with the identical main-window menu shortcuts
        if (keyEvent->modifiers().testFlag(Qt::ControlModifier) && keyEvent->key() == 'N') {
            nextWarning();
            event->accept();
            return true;
        }
        if (keyEvent->modifiers().testFlag(Qt::ControlModifier) && keyEvent->key() == 'S') {
            saveAs();
            event->accept();
            return true;
        }
        if (keyEvent->modifiers().testFlag(Qt::ControlModifier) && keyEvent->key() == 'Q') {
            quit();
            event->accept();
            return true;
        }
        if (keyEvent->modifiers().testFlag(Qt::ControlModifier) &&
            keyEvent->key() == Qt::Key_Return) {
            runBuffer();
            event->accept();
            return true;
        }
    }
    return QWidget::eventFilter(watched, event);
}

// Local Variables:
// c-basic-offset: 4
// End:
