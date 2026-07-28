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

// The rest of the chart window: the four ways data leaves it, the two dialogs
// that change how it looks, and the keys it claims for itself.
//
// test_chartanalysis.cpp took the post-process half; what stayed uncovered was
// everything either behind a QFileDialog (saveAs, exportDat, exportCsv,
// exportYaml, and chartsToPlotData underneath all three) or behind a dialog
// whose result is written back into the chart (changeStyle, referenceLines).
//
// The exports are worth more than their line count suggests.  They are the only
// path by which a run's numbers reach anything outside this application, and
// chartsToPlotData() beneath them reads *every* column while the window on
// screen shows one -- so an export that silently dropped the columns the user
// was not looking at would look correct in the window right up to the point the
// file was opened somewhere else.  That is what the column assertions below are
// for; a chart with two columns is used throughout so the difference shows.

#include "chartviewer.h"

#include "chartdialogs.h"
#include "constants.h"
#include "plotdata.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QImage>
#include <QKeyEvent>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QTemporaryDir>
#include <QTimer>

#include <cmath>

namespace {

/// Answers one file dialog with @p answer (an empty answer cancels it) and
/// records any message box that follows.
///
/// One reaper per interaction, as everywhere else: two alive at once both
/// answer the next modal and race.
class SaveTo : public QObject {
public:
    explicit SaveTo(QString path, int budgetMs = 8000) : answer(std::move(path)), left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &SaveTo::poll);
        timer.start();
    }
    int dialogs = 0;
    QStringList messages;
    [[nodiscard]] QString all() const { return messages.join(" | "); }

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            messages << box->windowTitle() + " " + box->text();
            box->accept();
            return;
        }
        if (auto *fd = qobject_cast<QFileDialog *>(m)) {
            ++dialogs;
            if (answer.isEmpty()) {
                static_cast<QDialog *>(fd)->reject();
            } else {
                fd->setDirectory(QFileInfo(answer).absolutePath());
                fd->selectFile(answer);
                static_cast<QDialog *>(fd)->accept();
            }
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
    }
    QTimer timer;
    QString answer;
    int left;
};

/// Fills in the chart-style dialog and accepts (or rejects) it.
class Restyle : public QObject {
public:
    ChartDisplayMode rawMode = ChartDisplayMode::Points;
    double rawWidth          = 5.0;
    double rawPoint          = 11.0;
    ChartDisplayMode proc    = ChartDisplayMode::LinesAndPoints;
    double procWidth         = 7.0;
    double procPoint         = 13.0;
    int legend               = static_cast<int>(LegendPos::TopRight);
    bool accept              = true;
    int dialogs              = 0;

    explicit Restyle(int budgetMs = 8000) : left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &Restyle::poll);
        timer.start();
    }

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        auto *dlg = qobject_cast<ChartStyleDialog *>(m);
        if (!dlg) {
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        ++dialogs;
        dlg->findChild<QComboBox *>("rawMode")->setCurrentIndex(static_cast<int>(rawMode));
        dlg->findChild<QDoubleSpinBox *>("rawWidth")->setValue(rawWidth);
        dlg->findChild<QDoubleSpinBox *>("rawPointSize")->setValue(rawPoint);
        dlg->findChild<QComboBox *>("procMode")->setCurrentIndex(static_cast<int>(proc));
        dlg->findChild<QDoubleSpinBox *>("procWidth")->setValue(procWidth);
        dlg->findChild<QDoubleSpinBox *>("procPointSize")->setValue(procPoint);
        auto *lg = dlg->findChild<QComboBox *>("legend");
        lg->setCurrentIndex(lg->findData(legend));
        if (accept)
            dlg->accept();
        else
            dlg->reject();
    }
    QTimer timer;
    int left;
};

/// Drives the reference-lines dialog: adds @p add rows, fills the first one in,
/// and sets the label style.
class RefLines : public QObject {
public:
    int add           = 1;
    double value      = 42.5;
    QString label     = "target";
    int orient        = 1; // horizontal
    double fontSize   = 14.5;
    int gap           = 9;
    bool boxed        = true;
    bool accept       = true;
    int dialogs       = 0;
    int rowsOnEntry   = -1; ///< how many rows the dialog already had when it opened

    explicit RefLines(int budgetMs = 8000) : left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &RefLines::poll);
        timer.start();
    }

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        auto *dlg = qobject_cast<RefLinesDialog *>(m);
        if (!dlg) {
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        ++dialogs;
        rowsOnEntry = dlg->lineCount();
        for (int i = 0; i < add; ++i) dlg->findChild<QPushButton *>("addLine")->click();
        // rows are named by their index at the time they were appended, so the
        // first row this reaper added is rowsOnEntry -- and when it added none
        // there is no row of its own to fill in
        if (add > 0) {
            const int n = rowsOnEntry;
            auto *o = dlg->findChild<QComboBox *>(QStringLiteral("orient%1").arg(n));
            auto *v = dlg->findChild<QDoubleSpinBox *>(QStringLiteral("value%1").arg(n));
            auto *l = dlg->findChild<QLineEdit *>(QStringLiteral("label%1").arg(n));
            if (o) o->setCurrentIndex(orient);
            if (v) v->setValue(value);
            if (l) l->setText(label);
        }
        dlg->findChild<QDoubleSpinBox *>("labelFont")->setValue(fontSize);
        dlg->findChild<QSpinBox *>("labelGap")->setValue(gap);
        dlg->findChild<QCheckBox *>("labelBoxed")->setChecked(boxed);
        if (accept)
            dlg->accept();
        else
            dlg->reject();
    }
    QTimer timer;
    int left;
};

class Export : public ::testing::Test {
protected:
    void SetUp() override
    {
        win = new ChartWindow(QStringLiteral("run1"), nullptr);
        win->resize(600, 400);
        // two columns on purpose: the window shows one at a time, the exports
        // must carry both
        PlotData d;
        d.setColumnNames({"Step", "Temp", "Press"});
        for (int i = 0; i < kRows; ++i)
            d.appendRow({double(i * 10), 300.0 + i, 1.5 * i});
        win->loadData(d, 0, {1, 2});
    }
    void TearDown() override { delete win; }

    static constexpr int kRows = 12;

    QString path(const QString &name) const { return dir.filePath(name); }

    /// invoke one of the window's private slots
    void call(const char *slot) const
    {
        QMetaObject::invokeMethod(win, slot);
        QCoreApplication::processEvents();
    }

    static QStringList linesOf(const QString &file)
    {
        QFile f(file);
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return {};
        return QString::fromUtf8(f.readAll()).split('\n', Qt::SkipEmptyParts);
    }

    ChartViewer *view() const { return win->findChild<ChartViewer *>("chartView"); }

    QTemporaryDir dir;
    ChartWindow *win = nullptr;
};

} // namespace

// ------------------------------------------------------------------ exports

TEST_F(Export, CsvCarriesEveryColumnAndEveryRow)
{
    SaveTo answer(path("out.csv"));
    call("exportCsv");

    ASSERT_EQ(answer.dialogs, 1) << "it did not ask where to write";
    const QStringList lines = linesOf(path("out.csv"));
    ASSERT_FALSE(lines.isEmpty()) << "nothing was written: " << answer.all().toStdString();
    EXPECT_EQ(lines.first(), "Step,Temp,Press")
        << "the header does not name the plotted columns";
    EXPECT_EQ(lines.size(), kRows + 1) << "a row was lost or a header counted twice";

    // the window shows one column; the file has to hold the other as well
    const QStringList last = lines.last().split(',');
    ASSERT_EQ(last.size(), 3);
    EXPECT_DOUBLE_EQ(last.at(0).toDouble(), 110.0);
    EXPECT_DOUBLE_EQ(last.at(1).toDouble(), 311.0);
    EXPECT_DOUBLE_EQ(last.at(2).toDouble(), 16.5) << "the column not on screen was not exported";
}

TEST_F(Export, DatIsGnuplotReadableAndNamesItsSource)
{
    SaveTo answer(path("out.dat"));
    call("exportDat");

    const QStringList lines = linesOf(path("out.dat"));
    ASSERT_FALSE(lines.isEmpty()) << "nothing was written: " << answer.all().toStdString();
    EXPECT_TRUE(lines.first().startsWith('#')) << "gnuplot data must open with a comment header";
    EXPECT_TRUE(lines.join('\n').contains("run1"))
        << "the export does not say which run it came from";

    int data = 0;
    for (const auto &l : lines)
        if (!l.trimmed().startsWith('#')) ++data;
    EXPECT_EQ(data, kRows) << "the number of data lines does not match the chart";
}

TEST_F(Export, YamlHoldsTheSameNumbers)
{
    SaveTo answer(path("out.yaml"));
    call("exportYaml");

    const QString text = linesOf(path("out.yaml")).join('\n');
    ASSERT_FALSE(text.isEmpty()) << "nothing was written: " << answer.all().toStdString();
    EXPECT_TRUE(text.contains("Temp")) << "the first column is missing from the YAML";
    EXPECT_TRUE(text.contains("Press")) << "the second column is missing from the YAML";
    EXPECT_TRUE(text.contains("311")) << "the last sample is missing from the YAML";
}

TEST_F(Export, CancellingAnExportWritesNothing)
{
    SaveTo answer{QString()};
    call("exportCsv");
    EXPECT_EQ(answer.dialogs, 1);
    EXPECT_TRUE(QDir(dir.path()).entryList(QDir::Files).isEmpty())
        << "a cancelled export left a file behind";
}

TEST_F(Export, AnUnwritableDestinationLeavesNoHalfFile)
{
    // writeExport() opens the file and gives up quietly if it cannot; what must
    // not happen is a truncated or empty file at the name the user chose
    SaveTo answer("/proc/definitely/not/writable/out.csv");
    call("exportCsv");
    EXPECT_FALSE(QFile::exists("/proc/definitely/not/writable/out.csv"));
}

TEST_F(Export, SaveAsWritesAPictureOfTheChart)
{
    SaveTo answer(path("chart.png"));
    call("saveAs");

    ASSERT_EQ(answer.dialogs, 1) << "it did not ask where to save";
    ASSERT_TRUE(QFile::exists(path("chart.png"))) << answer.all().toStdString();
    const QImage img(path("chart.png"));
    ASSERT_FALSE(img.isNull()) << "the file is not a readable image";
    EXPECT_GT(img.width(), 100) << "the picture is not the size of the chart";
    EXPECT_GT(img.height(), 100);
}

TEST_F(Export, TheDefaultNameFollowsTheRunAndTheColumn)
{
    // the user should not have to type a name; the offered one has to identify
    // both the run and which column is being written
    QString offered;
    QTimer poll;
    int left = 8000;
    QObject::connect(&poll, &QTimer::timeout, [&]() {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            poll.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        if (auto *fd = qobject_cast<QFileDialog *>(m)) {
            const QStringList sel = fd->selectedFiles();
            if (!sel.isEmpty()) offered = QFileInfo(sel.first()).fileName();
            static_cast<QDialog *>(fd)->reject();
        }
    });
    poll.setInterval(5);
    poll.start();
    call("saveAs");
    poll.stop();

    EXPECT_TRUE(offered.contains("run1")) << "offered: " << offered.toStdString();
    EXPECT_TRUE(offered.contains("Temp")) << "offered: " << offered.toStdString();
    EXPECT_TRUE(offered.endsWith(".png")) << "offered: " << offered.toStdString();
}

TEST_F(Export, AnEmptyChartExportsNothingAndDoesNotAsk)
{
    ChartWindow empty(QString(), nullptr);
    SaveTo answer(path("never.csv"));
    QMetaObject::invokeMethod(&empty, "exportCsv");
    QCoreApplication::processEvents();
    EXPECT_EQ(answer.dialogs, 0) << "it asked where to write a chart that has no data";
    EXPECT_FALSE(QFile::exists(path("never.csv")));
}

// ---------------------------------------------------------------- clipboard

TEST_F(Export, CopyPutsThePlotOnTheClipboard)
{
    auto *clip = QGuiApplication::clipboard();
    ASSERT_NE(clip, nullptr);
    clip->clear();
    call("copy");

    const QImage img = clip->image();
    ASSERT_FALSE(img.isNull()) << "nothing reached the clipboard";
    EXPECT_GT(img.width(), 100) << "what reached the clipboard is not the chart";
    EXPECT_GT(img.height(), 100);
}

// -------------------------------------------------------------------- style

TEST_F(Export, TheStyleDialogsChoicesReachTheChart)
{
    ASSERT_NE(view(), nullptr);
    Restyle r;
    call("changeStyle");

    ASSERT_EQ(r.dialogs, 1) << "the style dialog never appeared";
    EXPECT_EQ(view()->displayMode(), r.rawMode) << "the raw draw mode was not applied";
    EXPECT_DOUBLE_EQ(view()->displayWidth(), r.rawWidth);
    EXPECT_DOUBLE_EQ(view()->displayPointSize(), r.rawPoint);
    EXPECT_EQ(view()->smoothMode(), r.proc) << "the processed draw mode was not applied";
    EXPECT_DOUBLE_EQ(view()->smoothWidth(), r.procWidth);
    EXPECT_DOUBLE_EQ(view()->smoothPointSize(), r.procPoint);
}

TEST_F(Export, TheLegendPlacementIsRemembered)
{
    // the legend corner is window-wide rather than per-column, so it is the one
    // choice from that dialog that has to survive into the next session
    QSettings s;
    s.beginGroup(Keys::GROUP_CHARTS);
    s.remove(Keys::LEGEND);
    s.endGroup();

    Restyle r;
    r.legend = static_cast<int>(LegendPos::BottomLeft);
    call("changeStyle");
    ASSERT_EQ(r.dialogs, 1);

    QSettings back;
    back.beginGroup(Keys::GROUP_CHARTS);
    EXPECT_EQ(back.value(Keys::LEGEND, -1).toInt(), static_cast<int>(LegendPos::BottomLeft))
        << "the legend corner was not stored";
    back.endGroup();
}

TEST_F(Export, CancellingTheStyleDialogChangesNothing)
{
    ASSERT_NE(view(), nullptr);
    const auto mode  = view()->displayMode();
    const auto width = view()->displayWidth();

    Restyle r;
    r.accept   = false;
    r.rawWidth = 9.0;
    call("changeStyle");

    ASSERT_EQ(r.dialogs, 1);
    EXPECT_EQ(view()->displayMode(), mode) << "a cancelled dialog restyled the chart anyway";
    EXPECT_DOUBLE_EQ(view()->displayWidth(), width);
}

// ------------------------------------------------------------ reference lines

TEST_F(Export, AReferenceLineSurvivesIntoTheNextVisitToTheDialog)
{
    // refLines is private, so what says the line was kept is that the dialog
    // opens with it already in place the next time round -- which is also what
    // the user sees
    {
        RefLines add;
        call("referenceLines");
        ASSERT_EQ(add.dialogs, 1) << "the reference-lines dialog never appeared";
        EXPECT_EQ(add.rowsOnEntry, 0) << "a fresh chart already had reference lines";
    }
    RefLines again;
    again.add = 0;
    call("referenceLines");
    ASSERT_EQ(again.dialogs, 1);
    EXPECT_EQ(again.rowsOnEntry, 1) << "the line added a moment ago was not kept";
}

TEST_F(Export, CancellingTheReferenceLinesDialogKeepsTheOldSet)
{
    {
        RefLines add;
        call("referenceLines");
        ASSERT_EQ(add.dialogs, 1);
    }
    {
        RefLines discarded;
        discarded.add    = 2;
        discarded.accept = false;
        call("referenceLines");
        ASSERT_EQ(discarded.dialogs, 1);
    }
    RefLines check;
    check.add = 0;
    call("referenceLines");
    EXPECT_EQ(check.rowsOnEntry, 1) << "rejected rows were kept, or the old one was lost";
}

TEST_F(Export, TheLabelStyleIsStoredForEveryChartWindow)
{
    RefLines r;
    r.fontSize = 15.5;
    r.gap      = 7;
    r.boxed    = true;
    call("referenceLines");
    ASSERT_EQ(r.dialogs, 1);

    QSettings s;
    s.beginGroup(Keys::GROUP_CHARTS);
    EXPECT_DOUBLE_EQ(s.value(Keys::REFLABELSIZE).toDouble(), 15.5);
    EXPECT_EQ(s.value(Keys::REFLABELDIST).toInt(), 7);
    EXPECT_TRUE(s.value(Keys::REFLABELBOX).toBool());
    s.endGroup();
}

TEST_F(Export, AnEmptyChartHasNoReferenceLinesToEdit)
{
    ChartWindow empty(QString(), nullptr);
    RefLines r;
    QMetaObject::invokeMethod(&empty, "referenceLines");
    QCoreApplication::processEvents();
    EXPECT_EQ(r.dialogs, 0) << "reference lines were offered for a chart with nothing to mark";
}

// --------------------------------------------------------------------- keys

TEST_F(Export, CtrlCCopiesWithoutGoingThroughTheMenu)
{
    // the chart window is a docked panel sharing the main window's shortcut
    // context, so it claims these itself rather than letting Qt call the
    // ambiguous binding
    auto *clip = QGuiApplication::clipboard();
    ASSERT_NE(clip, nullptr);
    clip->clear();

    QKeyEvent key(QEvent::ShortcutOverride, 'C', Qt::ControlModifier);
    EXPECT_TRUE(QApplication::sendEvent(win, &key)) << "Ctrl+C was not claimed";
    EXPECT_TRUE(key.isAccepted());
    EXPECT_FALSE(clip->image().isNull()) << "Ctrl+C did not copy the chart";
}

TEST_F(Export, CtrlWClosesTheWindow)
{
    win->show();
    QKeyEvent key(QEvent::ShortcutOverride, 'W', Qt::ControlModifier);
    QApplication::sendEvent(win, &key);
    QCoreApplication::processEvents();
    EXPECT_TRUE(key.isAccepted());
    EXPECT_FALSE(win->isVisible()) << "Ctrl+W left the window open";
}

TEST_F(Export, CtrlQOnAStandaloneWindowClosesItRatherThanQuitting)
{
    // with no SpartaGui behind it there is no application to quit, and a Quit
    // that did nothing would strand the window
    win->show();
    QKeyEvent key(QEvent::ShortcutOverride, 'Q', Qt::ControlModifier);
    QApplication::sendEvent(win, &key);
    QCoreApplication::processEvents();
    EXPECT_TRUE(key.isAccepted());
    EXPECT_FALSE(win->isVisible());
}

TEST_F(Export, CtrlSlashIsClaimedEvenWithNothingToStop)
{
    QKeyEvent key(QEvent::ShortcutOverride, '/', Qt::ControlModifier);
    QApplication::sendEvent(win, &key);
    EXPECT_TRUE(key.isAccepted()) << "Ctrl+/ was left to the ambiguous main-window binding";
    EXPECT_TRUE(win->isVisible() || !win->isVisible()); // it must simply not crash
}

TEST_F(Export, AnOrdinaryKeyIsLeftAlone)
{
    QKeyEvent key(QEvent::ShortcutOverride, Qt::Key_A, Qt::NoModifier);
    QApplication::sendEvent(win, &key);
    EXPECT_FALSE(key.isAccepted()) << "the chart window swallowed a key it has no use for";
}

// Closing the window with the keyboard focus in one of its label fields.
//
// The snapshot viewer crashed on exactly this: hiding a widget moves the focus,
// the field that had it emits editingFinished(), and the member-function slot
// behind that is dispatched on an object whose derived destructor has already
// run.  The chart window's three axis-label fields are wired the same way.
TEST_F(Export, ClosingWithTheFocusInALabelFieldDoesNotCrash)
{
    win->show();
    QApplication::processEvents();

    auto *title = win->findChild<QLineEdit *>("chartTitle");
    ASSERT_NE(title, nullptr) << "no chart-title field to focus";
    title->setFocus();
    title->setText("something new");
    QApplication::processEvents();
    ASSERT_TRUE(title->hasFocus());

    delete win;
    win = nullptr;
    QApplication::processEvents();
    SUCCEED(); // getting here is the assertion
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_chartexport.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    QSettings().clear();
    return rc;
}

// Local Variables:
// c-basic-offset: 4
// End:
