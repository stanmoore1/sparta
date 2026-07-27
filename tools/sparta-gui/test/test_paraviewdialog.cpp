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

// The ParaView export dialog's conversion run: runConversion() and
// onProcessFinished().
//
// test_paraviewexport.cpp covers the pure part -- which arguments a set of
// settings turns into, and which of them are refused.  What was never run is
// the orchestration around it: locating the script, refusing to start without
// the tools, offering to replace stale output, launching pvpython in the right
// directory, and deciding afterwards whether the conversion worked.
//
// This is the hand-off to external analysis, so its failures leave the
// application looking fine and the data wrong somewhere else.  Reporting a
// non-zero exit as success is the worst of them: the user goes off to open a
// file that was never written, or worse, one left over from a previous run.
//
// ParaView is not installed here and does not need to be.  The tests supply a
// stub pvpython -- a script that records how it was called, writes the output
// it was asked for, and exits with a code the test chooses -- which exercises
// every one of those decisions.

#include "paraviewdialog.h"

#include "constants.h"
#include "paraviewexport.h"

#include <gtest/gtest.h>

#include <QAbstractButton>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSettings>
#include <QTemporaryDir>
#include <QTimer>

namespace {

/// Answers whatever modal appears and remembers what it said.
class Modals : public QObject {
public:
    explicit Modals(QMessageBox::StandardButton button = QMessageBox::No, int budgetMs = 15000) :
        button(button), left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &Modals::poll);
        timer.start();
    }
    QStringList seen;
    int boxes = 0;

    [[nodiscard]] bool said(const QString &needle) const
    {
        for (const auto &m : seen)
            if (m.contains(needle)) return true;
        return false;
    }
    [[nodiscard]] QString all() const { return seen.join(" | "); }

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
            ++boxes;
            seen << box->text() + " " + box->informativeText() + " " + box->detailedText();
            if (auto *b = box->button(button)) b->click();
            else box->accept();
            return;
        }
        if (auto *fd = qobject_cast<QFileDialog *>(m)) static_cast<QDialog *>(fd)->reject();
    }
    QTimer timer;
    QMessageBox::StandardButton button;
    int left;
};

class PvDialog : public ::testing::Test {
protected:
    void SetUp() override
    {
        QSettings().clear();
        startDir = QDir::currentPath();

        // findScriptsDir() walks up from the working directory looking for
        // tools/paraview/surf2paraview.py, so give it one to find
        ASSERT_TRUE(QDir(dir.path()).mkpath("tools/paraview"));
        for (const char *n : {"surf2paraview.py", "grid2paraview.py"})
            write(dir.filePath(QString("tools/paraview/") + n), "# stub\n");

        write(dir.filePath("in.surf"), "# a surface\n");
        QDir::setCurrent(dir.path());
    }

    void TearDown() override
    {
        QDir::setCurrent(startDir);
        QSettings().clear();
    }

    static void write(const QString &path, const QString &text)
    {
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly | QIODevice::Text));
        f.write(text.toUtf8());
    }

    static QString read(const QString &path)
    {
        QFile f(path);
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return {};
        return QString::fromUtf8(f.readAll());
    }

    /// A stand-in for pvpython: it records the directory it ran in and every
    /// argument it was given, optionally creates @p produces, and exits with
    /// @p exitCode.
    QString stubPvpython(int exitCode = 0, const QString &produces = QString())
    {
        const QString path = dir.filePath("pvpython");
        QString sh = "#!/bin/sh\n";
        sh += QString("pwd > %1\n").arg(dir.filePath("ran-in.txt"));
        sh += QString("printf '%s\\n' \"$@\" > ") + dir.filePath("argv.txt") + "\n";
        sh += "echo 'converting...'\n";
        if (!produces.isEmpty()) sh += QString("touch '%1'\n").arg(produces);
        sh += QString("exit %1\n").arg(exitCode);
        write(path, sh);
        QFile::setPermissions(path, QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner);
        return path;
    }

    static QStringList recordedArgs(const QString &path)
    {
        return read(path).split('\n', Qt::SkipEmptyParts);
    }

    template <class W> static W *ctl(QDialog &d, const char *name)
    {
        auto *w = d.findChild<W *>(QLatin1String(name));
        if (!w) ADD_FAILURE() << "no control named " << name;
        return w;
    }

    static QString logOf(QDialog &d)
    {
        auto *l = d.findChild<QPlainTextEdit *>("log");
        return l ? l->toPlainText() : QString();
    }

    template <class F> static bool waitFor(F done, int budgetMs = 15000)
    {
        QElapsedTimer clock;
        clock.start();
        while (!done() && clock.elapsed() < budgetMs)
            QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
        return done();
    }

    /// Set the dialog up for a surface conversion and press Convert; returns
    /// true once the run has finished (the button comes back).
    bool convert(QDialog &d, const QString &pvpython, const QString &output = "result")
    {
        ctl<QLineEdit>(d, "input")->setText(dir.filePath("in.surf"));
        ctl<QLineEdit>(d, "output")->setText(output);
        ctl<QLineEdit>(d, "pvpython")->setText(pvpython);
        auto *run = ctl<QPushButton>(d, "convert");
        QMetaObject::invokeMethod(&d, "runConversion");
        return waitFor([run] { return run->isEnabled() && run->text() == "Convert"; });
    }

    QTemporaryDir dir;
    QString startDir;
};

} // namespace

// ------------------------------------------------------------- refusing to run

TEST_F(PvDialog, WithNoInputFileItSaysSoAndStartsNothing)
{
    Modals modals;
    ParaViewExportDialog d(nullptr, dir.path());
    ctl<QLineEdit>(d, "pvpython")->setText(stubPvpython());
    QMetaObject::invokeMethod(&d, "runConversion");

    EXPECT_TRUE(modals.said("Cannot run the conversion")) << modals.all().toStdString();
    EXPECT_FALSE(QFile::exists(dir.filePath("argv.txt"))) << "pvpython was started anyway";
}

TEST_F(PvDialog, WithoutPvpythonItSaysWhereToGetItRatherThanFailingSilently)
{
    Modals modals;
    ParaViewExportDialog d(nullptr, dir.path());
    ctl<QLineEdit>(d, "input")->setText(dir.filePath("in.surf"));
    ctl<QLineEdit>(d, "output")->setText("result");
    ctl<QLineEdit>(d, "pvpython")->setText(dir.filePath("no-such-pvpython"));
    QMetaObject::invokeMethod(&d, "runConversion");

    EXPECT_TRUE(modals.said("pvpython was not found")) << modals.all().toStdString();
    EXPECT_TRUE(modals.said("Install ParaView")) << modals.all().toStdString();
}

TEST_F(PvDialog, WithoutTheConversionScriptItSaysSo)
{
    // no tools/paraview anywhere above the working directory
    QTemporaryDir bare;
    QDir::setCurrent(bare.path());

    Modals modals;
    ParaViewExportDialog d(nullptr, bare.path());
    ctl<QLineEdit>(d, "input")->setText(dir.filePath("in.surf"));
    ctl<QLineEdit>(d, "output")->setText("result");
    ctl<QLineEdit>(d, "pvpython")->setText(stubPvpython());
    QMetaObject::invokeMethod(&d, "runConversion");

    EXPECT_TRUE(modals.said("Could not find the conversion script")) << modals.all().toStdString();
    EXPECT_FALSE(QFile::exists(dir.filePath("argv.txt")));
}

// ---------------------------------------------------------------- running it

TEST_F(PvDialog, RunsTheScriptWithTheArgumentsTheSettingsProduce)
{
    // the arguments are what the external tool actually acts on; the pure
    // builder is tested next door, so this checks that what it built is what
    // was handed to the interpreter
    Modals modals;
    ParaViewExportDialog d(nullptr, dir.path());
    ASSERT_TRUE(convert(d, stubPvpython(0, dir.filePath("result.pvd"))))
        << "the conversion never finished: " << logOf(d).toStdString();

    const QStringList got = recordedArgs(dir.filePath("argv.txt"));
    ASSERT_FALSE(got.isEmpty()) << "pvpython was never started";

    ParaviewExport::Settings s;
    s.mode       = ParaviewExport::Mode::Surface;
    s.inputFile  = dir.filePath("in.surf");
    s.outputName = "result";
    const QStringList want =
        ParaviewExport::buildScriptArgs(s, dir.filePath("tools/paraview/surf2paraview.py"));
    EXPECT_EQ(got, want) << "the interpreter was given different arguments than were built";
}

TEST_F(PvDialog, RunsInTheInputFilesDirectorySoRelativeOutputLandsThere)
{
    // the scripts write their output relative to the working directory; running
    // them anywhere else scatters results where the user will not find them
    Modals modals;
    QDir(dir.path()).mkpath("deck");
    write(dir.filePath("deck/in.surf"), "# a surface\n");

    ParaViewExportDialog d(nullptr, dir.path());
    ctl<QLineEdit>(d, "input")->setText(dir.filePath("deck/in.surf"));
    ctl<QLineEdit>(d, "output")->setText("result");
    ctl<QLineEdit>(d, "pvpython")->setText(stubPvpython());
    auto *run = ctl<QPushButton>(d, "convert");
    QMetaObject::invokeMethod(&d, "runConversion");
    ASSERT_TRUE(waitFor([run] { return run->isEnabled(); }));

    const QString ranIn = read(dir.filePath("ran-in.txt")).trimmed();
    EXPECT_EQ(QFileInfo(ranIn).canonicalFilePath(),
              QFileInfo(dir.filePath("deck")).canonicalFilePath())
        << "the conversion ran in " << ranIn.toStdString();
}

TEST_F(PvDialog, TheModeChoosesWhichScriptIsRun)
{
    Modals modals;
    ParaViewExportDialog d(nullptr, dir.path());
    auto *mode = ctl<QComboBox>(d, "mode");
    ASSERT_NE(mode, nullptr);
    mode->setCurrentIndex(mode->findData(static_cast<int>(ParaviewExport::Mode::Grid)));

    ASSERT_TRUE(convert(d, stubPvpython())) << logOf(d).toStdString();
    const QStringList got = recordedArgs(dir.filePath("argv.txt"));
    ASSERT_FALSE(got.isEmpty());
    EXPECT_TRUE(got.first().endsWith("grid2paraview.py"))
        << "grid mode ran " << got.first().toStdString();
}

TEST_F(PvDialog, TheLogShowsWhatWasRunAndWhatItPrinted)
{
    Modals modals;
    ParaViewExportDialog d(nullptr, dir.path());
    ASSERT_TRUE(convert(d, stubPvpython(0, dir.filePath("result.pvd"))));

    const QString text = logOf(d);
    EXPECT_TRUE(text.contains("pvpython")) << text.toStdString();
    EXPECT_TRUE(text.contains("converting..."))
        << "the tool's own output did not reach the log:\n"
        << text.toStdString();
}

TEST_F(PvDialog, TheButtonSaysItIsBusyAndComesBackAfterwards)
{
    Modals modals;
    ParaViewExportDialog d(nullptr, dir.path());
    auto *run = ctl<QPushButton>(d, "convert");
    ASSERT_TRUE(convert(d, stubPvpython()));
    EXPECT_TRUE(run->isEnabled()) << "the dialog stayed busy after the run finished";
    EXPECT_EQ(run->text(), "Convert");
}

// --------------------------------------------------------------- the verdict

TEST_F(PvDialog, ASuccessfulConversionReportsWhereItWrote)
{
    Modals modals;
    ParaViewExportDialog d(nullptr, dir.path());
    ctl<QCheckBox>(d, "openAfter")->setChecked(false); // checked by default
    ASSERT_TRUE(convert(d, stubPvpython(0, dir.filePath("result.pvd"))));

    EXPECT_TRUE(logOf(d).contains("Done. Wrote")) << logOf(d).toStdString();
    EXPECT_TRUE(logOf(d).contains("result.pvd")) << logOf(d).toStdString();
    EXPECT_EQ(modals.boxes, 0) << "a good conversion complained: " << modals.all().toStdString();
}

TEST_F(PvDialog, AFailedConversionIsReportedAsFailedRatherThanDone)
{
    // the one that matters most: a non-zero exit reported as success sends the
    // user to open a file that was never written
    Modals modals;
    ParaViewExportDialog d(nullptr, dir.path());
    ASSERT_TRUE(convert(d, stubPvpython(3)));

    EXPECT_TRUE(logOf(d).contains("Conversion failed")) << logOf(d).toStdString();
    EXPECT_TRUE(logOf(d).contains("exit code 3")) << logOf(d).toStdString();
    EXPECT_FALSE(logOf(d).contains("Done. Wrote"))
        << "a failed conversion also claimed to be done";
    EXPECT_TRUE(modals.said("did not complete successfully")) << modals.all().toStdString();
}

TEST_F(PvDialog, AFailedConversionDoesNotLaunchParaView)
{
    Modals modals;
    ParaViewExportDialog d(nullptr, dir.path());
    ctl<QCheckBox>(d, "openAfter")->setChecked(true);
    ctl<QLineEdit>(d, "paraview")->setText(stubPvpython()); // an executable that exists
    ASSERT_TRUE(convert(d, stubPvpython(1)));

    EXPECT_FALSE(logOf(d).contains("Launched ParaView"))
        << "ParaView was opened on the output of a failed conversion";
}

TEST_F(PvDialog, AskingToOpenTheResultWithNoParaViewSaysWhereTheFileIs)
{
    Modals modals;
    ParaViewExportDialog d(nullptr, dir.path());
    ctl<QCheckBox>(d, "openAfter")->setChecked(true);
    ctl<QLineEdit>(d, "paraview")->setText(dir.filePath("no-such-paraview"));
    ASSERT_TRUE(convert(d, stubPvpython(0, dir.filePath("result.pvd"))));

    EXPECT_TRUE(modals.said("paraview executable was not found")) << modals.all().toStdString();
    EXPECT_TRUE(modals.said("result.pvd"))
        << "the user was not told where the file is: " << modals.all().toStdString();
}

TEST_F(PvDialog, NotAskingToOpenTheResultLeavesParaViewAlone)
{
    Modals modals;
    ParaViewExportDialog d(nullptr, dir.path());
    ctl<QCheckBox>(d, "openAfter")->setChecked(false);
    ctl<QLineEdit>(d, "paraview")->setText(dir.filePath("no-such-paraview"));
    ASSERT_TRUE(convert(d, stubPvpython(0, dir.filePath("result.pvd"))));

    EXPECT_EQ(modals.boxes, 0) << "it complained about a tool it was not asked to use: "
                               << modals.all().toStdString();
}

// ------------------------------------------------------------- stale output

TEST_F(PvDialog, ExistingOutputIsNotOverwrittenWithoutAsking)
{
    // the scripts refuse to overwrite, so the dialog clears the way first --
    // which means it deletes a file the user may still want
    write(dir.filePath("result.pvd"), "an earlier conversion\n");

    Modals no(QMessageBox::No);
    ParaViewExportDialog d(nullptr, dir.path());
    ctl<QLineEdit>(d, "input")->setText(dir.filePath("in.surf"));
    ctl<QLineEdit>(d, "output")->setText("result");
    ctl<QLineEdit>(d, "pvpython")->setText(stubPvpython());
    QMetaObject::invokeMethod(&d, "runConversion");

    EXPECT_TRUE(no.said("already exists")) << no.all().toStdString();
    EXPECT_EQ(read(dir.filePath("result.pvd")), "an earlier conversion\n")
        << "declining still destroyed the earlier output";
    EXPECT_FALSE(QFile::exists(dir.filePath("argv.txt"))) << "it converted anyway";
}

TEST_F(PvDialog, AgreeingToOverwriteClearsTheStaleOutputFirst)
{
    write(dir.filePath("result.pvd"), "an earlier conversion\n");

    Modals yes(QMessageBox::Yes);
    ParaViewExportDialog d(nullptr, dir.path());
    ASSERT_TRUE(convert(d, stubPvpython(0, dir.filePath("result.pvd"))));

    EXPECT_TRUE(yes.said("already exists")) << yes.all().toStdString();
    EXPECT_TRUE(QFile::exists(dir.filePath("argv.txt"))) << "it never ran";
    EXPECT_NE(read(dir.filePath("result.pvd")), "an earlier conversion\n")
        << "the stale output was left in place for the script to trip over";
}

TEST_F(PvDialog, TheToolPathsAreRememberedForNextTime)
{
    Modals modals;
    const QString pv = stubPvpython();
    ParaViewExportDialog d(nullptr, dir.path());
    ctl<QCheckBox>(d, "openAfter")->setChecked(false);
    ctl<QLineEdit>(d, "paraview")->setText(pv);
    ASSERT_TRUE(convert(d, pv));

    EXPECT_EQ(QSettings().value(Keys::PVPYTHON_PATH).toString(), pv)
        << "the interpreter has to be found again by hand every session";
    EXPECT_EQ(QSettings().value(Keys::PARAVIEW_PATH).toString(), pv);
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);

    static QTemporaryDir settingsDir;
    QCoreApplication::setOrganizationName("SPARTA-GUI test");
    QCoreApplication::setApplicationName(
        QString("test_paraviewdialog-%1").arg(QCoreApplication::applicationPid()));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settingsDir.path());

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
