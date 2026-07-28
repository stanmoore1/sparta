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

// The snapshot viewer's toolbar, its colour files, and the keys it claims.
//
// test_imageviewerinput.cpp covers steering the camera with the mouse.  This
// covers the row of buttons above the image -- the eight render toggles, the
// size fields, the camera buttons and reset -- plus loadColors/saveColors and
// the Alt- and Ctrl- keys the panel handles itself.
//
// Every one of those slots reads sender(), so none of them can be invoked
// directly: they do nothing unless a real button emitted the click.  That is
// why they had never run, and it is why each check here clicks the button by
// object name.
//
// What a click did is read back out of the `dump image` command the viewer puts
// on the clipboard, the same way the mouse tests do.  The command is the state
// the render actually uses and the form the user can paste into a deck, so a
// toggle that flipped the button but not the setting is told apart from one
// that did both -- which the screenshot suites cannot do.

#include "imageviewer.h"

#include "constants.h"
#include "dumpimagesettingsdialog.h"
#include "helpers.h"
#include "spartawrapper.h"
#include "viewerdisplay.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QClipboard>
#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QRegularExpression>
#include <QSettings>
#include <QSpinBox>
#include <QTemporaryDir>
#include <QTimer>

#include <cmath>

namespace {

const char *testLibrary()
{
    static const QByteArray env = qgetenv("SPARTA_PLUGIN_LIB");
    if (!env.isEmpty()) return env.constData();
#if defined(SPARTA_TEST_LIBRARY_PATH)
    return SPARTA_TEST_LIBRARY_PATH;
#else
    return "";
#endif
}

/// Dismisses anything modal.  Nothing but the colour-file cases should raise one.
class Modals : public QObject {
public:
    explicit Modals(int budgetMs = 20000) : left(budgetMs)
    {
        timer.setInterval(10);
        connect(&timer, &QTimer::timeout, this, &Modals::poll);
        timer.start();
    }
    QStringList seen;

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 10) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            seen << box->windowTitle() + " " + box->text();
            box->accept();
        } else if (auto *d = qobject_cast<QDialog *>(m)) {
            d->reject();
        }
    }
    QTimer timer;
    int left;
};

/// Answers one file dialog with a path (empty cancels) and records message boxes.
class PickFile : public QObject {
public:
    explicit PickFile(QString path, int budgetMs = 20000) :
        answer(std::move(path)), left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &PickFile::poll);
        timer.start();
    }
    int dialogs = 0;
    QStringList messages;
    [[nodiscard]] QString all() const { return messages.join(" | "); }
    [[nodiscard]] bool said(const QString &needle) const
    {
        for (const auto &m : messages)
            if (m.contains(needle)) return true;
        return false;
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

/// Ticks a checkbox in the settings dialog and accepts it.
class TickInSettings : public QObject {
public:
    explicit TickInSettings(QString control, int budgetMs = 20000) :
        name(std::move(control)), left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &TickInSettings::poll);
        timer.start();
    }
    int dialogs = 0;
    bool found  = false;

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        auto *dlg = qobject_cast<DumpImageSettingsDialog *>(m);
        if (!dlg) {
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        ++dialogs;
        if (auto *cb = dlg->findChild<QCheckBox *>(name)) {
            cb->setChecked(true);
            found = true;
        }
        dlg->accept();
    }
    QTimer timer;
    QString name;
    int left;
};

// A 3d box with a few particles: enough for the viewer to have a scene, and 3d
// so the rotate buttons are not disabled the way they are in two dimensions.
const char *const kDeck3d = "seed 12345\n"
                            "dimension 3\n"
                            "global gridcut 0.0 comm/sort yes\n"
                            "boundary r r r\n"
                            "create_box 0 10 0 10 0 10\n"
                            "create_grid 4 4 4\n"
                            "species ar.species Ar\n"
                            "mixture air Ar vstream 0.0 0.0 0.0\n"
                            "global nrho 1.0 fnum 1.0\n"
                            "create_particles air n 50\n"
                            "collide vss air ar.vss\n"
                            "run 0\n";

// The same box with a tetrahedron in it.  Without a surface the surf toggle is
// disabled -- correctly, there is nothing to draw -- so toggleSurf() can only be
// reached from a scene that has one.
const char *const kDeckSurf = "seed 12345\n"
                              "dimension 3\n"
                              "global gridcut 0.0 comm/sort yes\n"
                              "boundary o o o\n"
                              "create_box 0 10 0 10 0 10\n"
                              "create_grid 4 4 4\n"
                              "species ar.species Ar\n"
                              "mixture air Ar vstream 0.0 0.0 0.0\n"
                              "global nrho 1.0 fnum 1.0\n"
                              "read_surf data.tet\n"
                              "surf_collide diffuse1 diffuse 300.0 0.0\n"
                              "surf_modify all collide diffuse1\n"
                              "create_particles air n 50\n"
                              "collide vss air ar.vss\n"
                              "run 0\n";

const char *const kTet = "# a tetrahedron\n"
                         "\n"
                         "4 points\n"
                         "4 triangles\n"
                         "\n"
                         "Points\n"
                         "\n"
                         "1 3.0 3.0 3.0\n"
                         "2 7.0 3.0 3.0\n"
                         "3 5.0 7.0 3.0\n"
                         "4 5.0 5.0 7.0\n"
                         "\n"
                         "Triangles\n"
                         "\n"
                         "1 1 3 2\n"
                         "2 1 2 4\n"
                         "3 2 3 4\n"
                         "4 3 1 4\n";

const char *const kSpecies = "# ID, molwt, molmass, rotdof, rotrel, vibdof, vibrel, vibtemp, wt, q\n"
                             "Ar  40.00    6.63E-26  0    .0   0   .0    0.0    1.0      0.0\n";
const char *const kVss     = "# diameter, omega, tref, alpha\n"
                             "Ar   4.11e-10 0.81  273.15  1.4\n";

class ViewerButtons : public ::testing::Test {
protected:
    void SetUp() override
    {
        if (!*testLibrary()) GTEST_SKIP() << "no shared libsparta";
        QSettings settings;
        settings.clear();
        settings.setValue(Keys::PLUGIN_PATH, QString::fromLocal8Bit(testLibrary()));
        settings.sync();

        write("ar.species", kSpecies);
        write("ar.vss", kVss);
        write("data.tet", kTet);
        startDir = QDir::currentPath();
        QDir::setCurrent(dir.path());
    }

    void TearDown() override
    {
        delete viewer;
        viewer = nullptr;
        if (sparta) {
            sparta->close();
            delete sparta;
            sparta = nullptr;
        }
        QDir::setCurrent(startDir);
        QSettings().clear();
    }

    void write(const QString &name, const QString &text) const
    {
        QFile f(dir.filePath(name));
        EXPECT_TRUE(f.open(QIODevice::WriteOnly | QIODevice::Text));
        f.write(text.toUtf8());
    }

    /// A viewer over a live simulation with a box in it.
    ///
    /// The modal reaper lives only for the duration of the build.  Two reapers
    /// alive at once both answer the next modal and race, so a case that has to
    /// answer a file dialog itself cannot also be holding a catch-all.
    ImageViewer *build(const char *deck = kDeck3d)
    {
        Modals duringStartup;
        sparta = new SpartaWrapper;
        if (!sparta->loadLib(testLibrary())) return nullptr;
        char arg0[]  = "sparta";
        char *argv[] = {arg0, nullptr};
        sparta->open(1, argv);
        if (!sparta->isOpen()) return nullptr;
        sparta->commandsString(QString::fromLatin1(deck));

        viewer = new ImageViewer("in.test", sparta, nullptr);
        viewer->resize(420, 340);
        viewer->show();
        QApplication::processEvents();
        return viewer;
    }

    QPushButton *button(const char *name) const
    {
        return viewer->findChild<QPushButton *>(QLatin1String(name));
    }

    /// press a toolbar button by object name
    void press(const char *name) const
    {
        auto *b = button(name);
        ASSERT_NE(b, nullptr) << "no toolbar button named " << name;
        b->click();
        QApplication::processEvents();
    }

    /// The dump image command for the current view, via the clipboard.
    QString command() const
    {
        auto *clip = QGuiApplication::clipboard();
        if (!clip) return {};
        clip->clear();
        QMetaObject::invokeMethod(viewer, "cmdToClipboard");
        QApplication::processEvents();
        return clip->text();
    }

    QString movieCommand() const
    {
        auto *clip = QGuiApplication::clipboard();
        if (!clip) return {};
        clip->clear();
        QMetaObject::invokeMethod(viewer, "movieToClipboard");
        QApplication::processEvents();
        return clip->text();
    }

    /// true when @p keyword appears as its own word in the emitted command
    [[nodiscard]] bool emits(const QString &keyword) const
    {
        return command().contains(
            QRegularExpression(R"(\b)" + QRegularExpression::escape(keyword) + R"(\b)"));
    }

    static double numberAfter(const QString &cmd, const QString &keyword, int which,
                              double fallback)
    {
        const QRegularExpression word(R"(\b)" + QRegularExpression::escape(keyword) + R"(\b)");
        const int at = cmd.indexOf(word);
        if (at < 0) return fallback;
        const QStringList rest =
            cmd.mid(at + keyword.size()).trimmed().split(QRegularExpression(R"(\s+)"));
        if (which >= rest.size()) return fallback;
        bool ok        = false;
        const double v = rest.at(which).toDouble(&ok);
        return ok ? v : fallback;
    }

    // the defaults the builder omits (see ImageParams in dumpimage.h)
    static constexpr double kTheta = 60.0, kPhi = 30.0, kZoom = 1.0;

    double phi() const { return numberAfter(command(), "view", 1, kPhi); }
    double theta() const { return numberAfter(command(), "view", 0, kTheta); }
    double zoom() const { return numberAfter(command(), "zoom", 0, kZoom); }

    QString path(const QString &name) const { return dir.filePath(name); }

    QTemporaryDir dir;
    QString startDir;
    SpartaWrapper *sparta = nullptr;
    ImageViewer *viewer   = nullptr;
};

} // namespace

// ------------------------------------------------------------- render toggles

TEST_F(ViewerButtons, EveryRenderToggleFlipsTheSettingAndItsButtonTogether)
{
    // The button showing one thing while the render does another is the failure
    // that matters here: the picture would silently stop matching the toolbar.
    //
    // The command builder omits every setting still at its SPARTA default, so
    // the keyword is not simply present when the button is down.  Particles and
    // the box are drawn by default and only appear in the command when they are
    // switched *off*; the rest are off by default and appear when switched on.
    // Getting that polarity wrong in either direction is exactly the bug this
    // is looking for, so each case carries the polarity it expects.
    Modals modals;
    ASSERT_NE(build(), nullptr);

    struct {
        const char *name;
        const char *keyword;
        bool emittedWhenOn; ///< false: the keyword appears when the button is up
    } cases[] = {{"ssao", "ssao", true},       {"antialias", "fsaa", true},
                 {"particles", "particle", false}, {"grid", "grid", true},
                 {"box", "box", false},        {"axes", "axes", true}};

    for (const auto &c : cases) {
        auto *b = button(c.name);
        ASSERT_NE(b, nullptr) << "no button named " << c.name;
        const bool wasChecked = b->isChecked();
        EXPECT_EQ(emits(c.keyword), wasChecked == c.emittedWhenOn)
            << c.name << " started out of step with the command it produces: "
            << command().toStdString();

        press(c.name);
        EXPECT_NE(b->isChecked(), wasChecked) << c.name << " did not change state";
        EXPECT_EQ(emits(c.keyword), b->isChecked() == c.emittedWhenOn)
            << c.name << " changed the button but not the render setting: "
            << command().toStdString();

        press(c.name); // and back, so each case starts from the same place
        EXPECT_EQ(b->isChecked(), wasChecked) << c.name << " did not toggle back";
        EXPECT_EQ(emits(c.keyword), wasChecked == c.emittedWhenOn)
            << c.name << " left the render setting behind when it toggled back";
    }
}

TEST_F(ViewerButtons, TheSurfaceToggleIsOfferedOnlyWhenThereIsASurface)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    auto *b = button("surf");
    ASSERT_NE(b, nullptr);
    EXPECT_FALSE(b->isEnabled())
        << "a scene with no surfaces still offered to draw them, which SPARTA refuses";
}

TEST_F(ViewerButtons, TheSurfaceToggleDrawsTheSurfaceWhenThereIsOne)
{
    Modals modals;
    ASSERT_NE(build(kDeckSurf), nullptr);
    ASSERT_EQ(sparta->extractSetting("surf_exist"), 1) << "the deck did not read the surface";

    auto *b = button("surf");
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(b->isEnabled()) << "the surface toggle stayed disabled with a surface loaded";
    const bool wasOn = b->isChecked();

    press("surf");
    EXPECT_NE(b->isChecked(), wasOn);
    EXPECT_EQ(emits("surf"), b->isChecked())
        << "the button changed but the render did not: " << command().toStdString();

    press("surf");
    EXPECT_EQ(b->isChecked(), wasOn);
    EXPECT_EQ(emits("surf"), wasOn);
}

TEST_F(ViewerButtons, ShininessIsANumberRatherThanAFlag)
{
    // shiny carries a value, and the command omits it while it is at the SPARTA
    // default -- so the check is on the number, not on the keyword being there
    Modals modals;
    ASSERT_NE(build(), nullptr);
    auto *b = button("shiny");
    ASSERT_NE(b, nullptr);
    const bool wasOn      = b->isChecked();
    const double before   = numberAfter(command(), "shiny", 0, -1.0);

    press("shiny");
    EXPECT_NE(b->isChecked(), wasOn);
    const double after = numberAfter(command(), "shiny", 0, -1.0);
    EXPECT_NE(after, before) << "the shininess did not change: " << command().toStdString();
    if (b->isChecked()) EXPECT_GT(after, 0.0) << "shininess was switched on as zero";

    press("shiny");
    EXPECT_EQ(b->isChecked(), wasOn);
    EXPECT_DOUBLE_EQ(numberAfter(command(), "shiny", 0, -1.0), before);
}

TEST_F(ViewerButtons, TurningTheGridOnTakesTheCutPlanesOff)
{
    // grid volume rendering and grid cut planes are mutually exclusive in
    // SPARTA; asking for both is a command it refuses
    Modals modals;
    ASSERT_NE(build(), nullptr);
    if (button("grid")->isChecked()) press("grid");

    press("grid");
    const QString cmd = command();
    EXPECT_TRUE(cmd.contains(QRegularExpression(R"(\bgrid\b)"))) << cmd.toStdString();
    for (const char *plane : {"gridx", "gridy", "gridz"})
        EXPECT_FALSE(cmd.contains(QRegularExpression(QString(R"(\b%1\b)").arg(plane))))
            << plane << " survived turning the grid volume on: " << cmd.toStdString();
}

TEST_F(ViewerButtons, TurningTheGridOnTakesAnAlreadyEnabledCutPlaneOff)
{
    // the case that matters: a cut plane the user switched on in the settings
    // dialog, and then the grid volume from the toolbar.  SPARTA refuses a
    // command carrying both, so the toggle has to take the plane down
    ASSERT_NE(build(), nullptr);
    if (button("grid")->isChecked()) {
        Modals m;
        press("grid");
    }

    {
        TickInSettings plane("ivs.gridx");
        viewer->findChild<QPushButton *>("planes")->click();
        QApplication::processEvents();
        ASSERT_EQ(plane.dialogs, 1) << "the settings dialog never appeared";
        ASSERT_TRUE(plane.found) << "no gridx control in the Grid Planes tab";
    }
    Modals modals;
    ASSERT_TRUE(command().contains(QRegularExpression(R"(\bgridx\b)")))
        << "the cut plane was not switched on, so switching it off proves nothing: "
        << command().toStdString();

    press("grid");
    const QString cmd = command();
    EXPECT_TRUE(cmd.contains(QRegularExpression(R"(\bgrid\b)"))) << cmd.toStdString();
    EXPECT_FALSE(cmd.contains(QRegularExpression(R"(\bgridx\b)")))
        << "the grid volume and an x cut plane were asked for together: " << cmd.toStdString();
}

TEST_F(ViewerButtons, TurningTheGridOnGivesItAColourToUse)
{
    // "grid" with no colour source is not a command SPARTA accepts, so the
    // toggle has to supply one
    Modals modals;
    ASSERT_NE(build(), nullptr);
    if (button("grid")->isChecked()) press("grid");
    press("grid");
    EXPECT_TRUE(command().contains(QRegularExpression(R"(\bgrid\s+\S+)")))
        << "the grid was turned on without saying what to colour it by: "
        << command().toStdString();
}

// ----------------------------------------------------------------- image size

TEST_F(ViewerButtons, TheSizeFieldsReachTheRenderedImage)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);

    auto *x = viewer->findChild<QSpinBox *>("xsize");
    auto *y = viewer->findChild<QSpinBox *>("ysize");
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);

    x->setValue(640);
    emit x->editingFinished();
    QApplication::processEvents();
    y->setValue(480);
    emit y->editingFinished();
    QApplication::processEvents();

    const QString cmd = command();
    EXPECT_DOUBLE_EQ(numberAfter(cmd, "size", 0, -1), 640) << cmd.toStdString();
    EXPECT_DOUBLE_EQ(numberAfter(cmd, "size", 1, -1), 480) << cmd.toStdString();
}

// --------------------------------------------------------------- camera keys

TEST_F(ViewerButtons, TheRotateButtonsTurnTheCameraTenDegreesEachWay)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    const double phi0   = phi();
    const double theta0 = theta();

    press("rotleft");
    EXPECT_NEAR(phi(), phi0 - 10.0, 1e-6) << "rotate-left did not turn 10 degrees";
    press("rotright");
    EXPECT_NEAR(phi(), phi0, 1e-6) << "rotate-right did not undo rotate-left";

    press("rotup");
    const double up = theta();
    EXPECT_NE(up, theta0) << "rotate-up did not change the elevation";
    press("rotdown");
    EXPECT_NEAR(theta(), theta0, 1e-6) << "rotate-down did not undo rotate-up";
}

TEST_F(ViewerButtons, TheZoomButtonsStepByTenPercent)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    const double z0 = zoom();

    press("zoomin");
    EXPECT_NEAR(zoom(), z0 * 1.1, 1e-6) << "zoom in did not step by 10 percent";
    press("zoomout");
    EXPECT_NEAR(zoom(), z0, 1e-6) << "zoom out did not undo zoom in";
}

TEST_F(ViewerButtons, ResetPutsBackEveryThingTheToolbarChanged)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    const QString before = command();

    press("zoomin");
    press("rotleft");
    press("box");
    ASSERT_NE(command(), before) << "nothing was changed, so reset proves nothing";

    press("resetview");
    EXPECT_EQ(command(), before) << "reset left the view somewhere else:\n"
                                 << command().toStdString();
}

TEST_F(ViewerButtons, RecenterPutsTheCameraBackOnTheBoxCentre)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);

    // move the centre off the middle, then ask for it back
    press("recenter");
    const QString cmd = command();
    // the builder omits the centre when it is at the default 0.5 0.5 0.5, so
    // what says recenter worked is that it is no longer mentioned
    EXPECT_FALSE(cmd.contains(QRegularExpression(R"(\bcenter\s+d\b)")))
        << "the camera is still following a dynamic centre: " << cmd.toStdString();
}

// -------------------------------------------------------------- the clipboard

TEST_F(ViewerButtons, TheMovieCommandIsADumpMovieOfTheSameScene)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    press("box"); // something distinguishable in both

    const QString image = command();
    const QString movie = movieCommand();
    ASSERT_FALSE(movie.isEmpty());
    EXPECT_TRUE(movie.contains("dump") && movie.contains("movie"))
        << "the movie command is not a dump movie: " << movie.toStdString();
    EXPECT_TRUE(movie.contains(".mp4")) << movie.toStdString();
    EXPECT_NE(image, movie) << "the movie command is the image command verbatim";

    // the scene settings have to be the same in both, or the movie would not
    // show what the viewer is showing
    const bool boxOn = image.contains(QRegularExpression(R"(\bbox\b)"));
    EXPECT_EQ(boxOn, movie.contains(QRegularExpression(R"(\bbox\b)")))
        << "the movie does not carry the same scene as the image";
}

// ------------------------------------------------------------- colour files

TEST_F(ViewerButtons, ColoursAndLightsSurviveASaveAndLoad)
{
    ASSERT_NE(build(), nullptr);

    // put the lights somewhere recognisable first, so a load that quietly kept
    // the defaults is not mistaken for one that read the file
    {
        PickFile save(path("colors.json"));
        QMetaObject::invokeMethod(viewer, "saveColors");
        QApplication::processEvents();
        ASSERT_EQ(save.dialogs, 1) << "it did not ask where to save: " << save.all().toStdString();
    }
    ASSERT_TRUE(QFile::exists(path("colors.json")));

    // rewrite the file with values nothing else would produce
    QFile f(path("colors.json"));
    ASSERT_TRUE(f.open(QIODevice::ReadWrite));
    QJsonObject root = QJsonDocument::fromJson(f.readAll()).object();
    ASSERT_FALSE(root.value("colors").toArray().isEmpty())
        << "the saved file has no colours in it";
    QJsonObject lights;
    lights["ambient"] = 0.125;
    lights["key"]     = 0.375;
    lights["fill"]    = 0.625;
    lights["back"]    = 0.875;
    root["lights"]    = lights;
    QJsonArray colors;
    QJsonObject one;
    one["name"]  = "Ar";
    one["red"]   = 1.0;
    one["green"] = 0.0;
    one["blue"]  = 0.0;
    colors.append(one);
    root["colors"] = colors;
    f.resize(0);
    f.write(QJsonDocument(root).toJson());
    f.close();

    {
        PickFile load(path("colors.json"));
        QMetaObject::invokeMethod(viewer, "loadColors");
        QApplication::processEvents();
        ASSERT_EQ(load.dialogs, 1) << "it did not ask what to load";
        EXPECT_TRUE(load.messages.isEmpty())
            << "a file it had just written was refused: " << load.all().toStdString();
    }

    const QString cmd = command();
    EXPECT_NEAR(numberAfter(cmd, "lights", 0, -1), 0.125, 1e-6)
        << "the ambient light from the file did not reach the render: " << cmd.toStdString();
    EXPECT_NEAR(numberAfter(cmd, "lights", 1, -1), 0.375, 1e-6);
    EXPECT_NEAR(numberAfter(cmd, "lights", 2, -1), 0.625, 1e-6);
    EXPECT_NEAR(numberAfter(cmd, "lights", 3, -1), 0.875, 1e-6);
}

TEST_F(ViewerButtons, CancellingASaveWritesNothing)
{
    ASSERT_NE(build(), nullptr);
    PickFile cancel{QString()};
    QMetaObject::invokeMethod(viewer, "saveColors");
    QApplication::processEvents();
    EXPECT_EQ(cancel.dialogs, 1);
    EXPECT_FALSE(QFile::exists(path("colors.json")));
}

TEST_F(ViewerButtons, ALoadThatIsRefusedLeavesTheViewAlone)
{
    ASSERT_NE(build(), nullptr);
    write("notcolors.json", "{\"application\": \"something else\"}");
    const QString before = command();

    PickFile load(path("notcolors.json"));
    QMetaObject::invokeMethod(viewer, "loadColors");
    QApplication::processEvents();

    EXPECT_EQ(load.dialogs, 1);
    EXPECT_TRUE(load.said("Load Colors")) << "someone else's JSON was accepted without a word";
    EXPECT_EQ(command(), before) << "a refused file changed the view anyway";
}

// --------------------------------------------------------------------- keys

TEST_F(ViewerButtons, AltWAndAltHJumpToTheSizeFields)
{
    // the panel shares the main window's shortcut context, so these are handled
    // in its own event filter rather than as menu accelerators
    Modals modals;
    ASSERT_NE(build(), nullptr);
    auto *x = viewer->findChild<QSpinBox *>("xsize");
    auto *y = viewer->findChild<QSpinBox *>("ysize");
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);

    QKeyEvent h(QEvent::KeyPress, Qt::Key_H, Qt::AltModifier);
    QApplication::sendEvent(viewer, &h);
    QApplication::processEvents();
    ASSERT_TRUE(y->hasFocus()) << "Alt+H did not reach the height field";

    QKeyEvent w(QEvent::KeyPress, Qt::Key_W, Qt::AltModifier);
    QApplication::sendEvent(viewer, &w);
    QApplication::processEvents();
    EXPECT_TRUE(x->hasFocus()) << "Alt+W did not reach the width field";
    EXPECT_FALSE(y->hasFocus());
}

TEST_F(ViewerButtons, AltXOpensTheMixtureChooser)
{
    // Alt+X does not merely focus the combo, it drops its list open so the next
    // keystroke picks a mixture.  Focus is on the popup by then rather than on
    // the combo, so the list being up is what says it worked.
    //
    // This is also the collision test.  A button mnemonic is matched by Qt's
    // shortcut map before the panel's event filter is consulted, so a settings
    // button spelt "Bo&x" takes Alt-X away from the chooser -- and the key then
    // opens a dialog instead, which is what it used to do.
    Modals modals;
    ASSERT_NE(build(), nullptr);
    auto *box = viewer->findChild<QComboBox *>("mixture");
    ASSERT_NE(box, nullptr);
    ASSERT_FALSE(box->view()->isVisible());

    QKeyEvent x(QEvent::KeyPress, Qt::Key_X, Qt::AltModifier);
    QApplication::sendEvent(viewer, &x);
    QApplication::processEvents();
    EXPECT_TRUE(box->view()->isVisible()) << "Alt+X did not open the mixture chooser";
    EXPECT_TRUE(QApplication::activeModalWidget() == nullptr)
        << "Alt+X opened a dialog: a button mnemonic has taken the shortcut";
    box->hidePopup();
}

TEST_F(ViewerButtons, NoSettingsButtonTakesAKeyThePanelHandlesItself)
{
    // the general form of the bug above: any mnemonic on the settings row that
    // collides with Alt-W, Alt-H or Alt-X wins over the event filter silently
    Modals modals;
    ASSERT_NE(build(), nullptr);

    const QString claimed = "WHX";
    for (auto *b : viewer->findChildren<QPushButton *>()) {
        const QString text = b->text();
        const int amp      = [&]() {
            for (int i = 0; i + 1 < text.size(); ++i) {
                if (text.at(i) != '&') continue;
                if (text.at(i + 1) == '&') { ++i; continue; } // a literal ampersand
                return i;
            }
            return -1;
        }();
        if (amp < 0) continue;
        const QChar key = text.at(amp + 1).toUpper();
        EXPECT_FALSE(claimed.contains(key))
            << "the button \"" << text.toStdString() << "\" claims Alt-" << key.toLatin1()
            << ", which the panel handles itself";
    }
}

TEST_F(ViewerButtons, AnAltKeyWithNoMeaningHereClosesTheMixtureList)
{
    // any other Alt combination closes the popup and takes focus back to the
    // panel, so a stray accelerator cannot leave a list hanging open over the
    // image with no way to dismiss it
    Modals modals;
    ASSERT_NE(build(), nullptr);
    auto *box = viewer->findChild<QComboBox *>("mixture");
    ASSERT_NE(box, nullptr);

    QKeyEvent open(QEvent::KeyPress, Qt::Key_X, Qt::AltModifier);
    QApplication::sendEvent(viewer, &open);
    QApplication::processEvents();
    ASSERT_TRUE(box->view()->isVisible()) << "the list never opened, so closing it proves nothing";

    QKeyEvent stray(QEvent::KeyPress, Qt::Key_J, Qt::AltModifier);
    QApplication::sendEvent(viewer, &stray);
    QApplication::processEvents();
    EXPECT_FALSE(box->view()->isVisible()) << "a stray Alt key left the mixture list hanging open";
}

TEST_F(ViewerButtons, AltWClosesTheMixtureListOnItsWayToTheWidthField)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    auto *box = viewer->findChild<QComboBox *>("mixture");
    auto *x   = viewer->findChild<QSpinBox *>("xsize");
    ASSERT_NE(box, nullptr);
    ASSERT_NE(x, nullptr);

    QKeyEvent open(QEvent::KeyPress, Qt::Key_X, Qt::AltModifier);
    QApplication::sendEvent(viewer, &open);
    QApplication::processEvents();
    ASSERT_TRUE(box->view()->isVisible());

    QKeyEvent w(QEvent::KeyPress, Qt::Key_W, Qt::AltModifier);
    QApplication::sendEvent(viewer, &w);
    QApplication::processEvents();
    EXPECT_FALSE(box->view()->isVisible()) << "the mixture list stayed open over the image";
    EXPECT_TRUE(x->hasFocus()) << "Alt+W did not reach the width field";
}

TEST_F(ViewerButtons, TheMixtureChooserSelectsWhatItIsSetTo)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    auto *box = viewer->findChild<QComboBox *>("mixture");
    ASSERT_NE(box, nullptr);
    ASSERT_GE(box->count(), 2) << "the deck defines a mixture beyond 'all'";

    box->setCurrentText("air");
    QApplication::processEvents();
    EXPECT_TRUE(command().contains("air"))
        << "the chosen mixture did not reach the render: " << command().toStdString();
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_imageviewerbuttons.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    QSettings().clear();
    return rc;
}

// Local Variables:
// c-basic-offset: 4
// End:
