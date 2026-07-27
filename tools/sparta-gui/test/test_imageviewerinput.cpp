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

// Steering the snapshot with the mouse: ImageViewer::eventFilter().
//
// Drag to rotate, shift-drag to pan, wheel to zoom.  It is how the viewer is
// actually used -- the toolbar buttons exist, but nobody aims a camera by
// pressing "rotate left" eleven times -- and none of it had ever run under
// test.  The live suites drive the buttons and photograph the screen, which
// catches a crash but cannot say that a drag to the right turned the camera to
// the right rather than the left, or by how much.
//
// The view state is private, so every check here reads it back out of the
// `dump image` command the viewer emits for the clipboard.  That is the same
// state the render uses, and it is the form the user can paste into a deck, so
// a gesture that moves the camera and a gesture that only moves the picture are
// told apart.

#include "imageviewer.h"

#include "constants.h"
#include "helpers.h"
#include "spartawrapper.h"
#include "viewerdisplay.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QClipboard>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFont>
#include <QIcon>
#include <QLabel>
#include <QMessageBox>
#include <QMouseEvent>
#include <QRegularExpression>
#include <QSettings>
#include <QTemporaryDir>
#include <QTimer>
#include <QWheelEvent>

#include <cmath>
#include <memory>

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

/// Dismisses anything modal.  Nothing here should raise one.
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
            seen << box->text();
            box->accept();
        } else if (auto *d = qobject_cast<QDialog *>(m)) {
            d->reject();
        }
    }
    QTimer timer;
    int left;
};

// A 3d box with a few particles: enough for the viewer to have a scene, and 3d
// so the polar angle is not pinned the way it is in two dimensions.
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

const char *const kDeck2d = "seed 12345\n"
                            "dimension 2\n"
                            "global gridcut 0.0 comm/sort yes\n"
                            "boundary r r p\n"
                            "create_box 0 10 0 10 -0.5 0.5\n"
                            "create_grid 4 4 1\n"
                            "species ar.species Ar\n"
                            "mixture air Ar vstream 0.0 0.0 0.0\n"
                            "global nrho 1.0 fnum 1.0\n"
                            "create_particles air n 50\n"
                            "collide vss air ar.vss\n"
                            "run 0\n";

const char *const kSpecies = "# ID, molwt, molmass, rotdof, rotrel, vibdof, vibrel, vibtemp, wt, q\n"
                             "Ar  40.00    6.63E-26  0    .0   0   .0    0.0    1.0      0.0\n";
const char *const kVss     = "# diameter, omega, tref, alpha\n"
                             "Ar   4.11e-10 0.81  273.15  1.4\n";

class ViewerInput : public ::testing::Test {
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
    ImageViewer *build(const char *deck = kDeck3d)
    {
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

    QLabel *canvas() const
    {
        auto *d = viewer->findChild<ViewerDisplay *>();
        return d ? d->label() : nullptr;
    }

    /// The dump image command for the current view, via the clipboard.
    QString command()
    {
        auto *clip = QGuiApplication::clipboard();
        if (!clip) return {};
        clip->clear();
        QMetaObject::invokeMethod(viewer, "cmdToClipboard");
        QApplication::processEvents();
        return clip->text();
    }

    /// The nth number following @p keyword in the emitted command, or @p fallback
    /// when the keyword is absent.  The builder leaves out every setting that is
    /// still at its default, so "absent" means "the default" rather than "no
    /// such thing" -- and that is exactly the state a fresh viewer is in.
    static double numberAfter(const QString &cmd, const QString &keyword, int which,
                              double fallback)
    {
        const QRegularExpression word(R"(\b)" + QRegularExpression::escape(keyword) + R"(\b)");
        const int at = cmd.indexOf(word);
        if (at < 0) return fallback;
        const QStringList rest =
            cmd.mid(at + keyword.size()).trimmed().split(QRegularExpression(R"(\s+)"));
        if (which >= rest.size()) return fallback;
        bool ok = false;
        const double v = rest.at(which).toDouble(&ok);
        return ok ? v : fallback;
    }

    // the defaults the builder omits (see ImageParams in dumpimage.h)
    static constexpr double kTheta = 60.0, kPhi = 30.0, kZoom = 1.0, kCentre = 0.5;

    double phi() { return numberAfter(command(), "view", 1, kPhi); }
    double theta() { return numberAfter(command(), "view", 0, kTheta); }
    double zoom() { return numberAfter(command(), "zoom", 0, kZoom); }

    /// Send a press / move / release to the image, as a drag does.
    void drag(const QPoint &from, const QPoint &to, Qt::KeyboardModifiers mods = Qt::NoModifier)
    {
        auto *w = canvas();
        ASSERT_NE(w, nullptr);
        QMouseEvent press(QEvent::MouseButtonPress, from, w->mapToGlobal(from), Qt::LeftButton,
                          Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(w, &press);
        QMouseEvent move(QEvent::MouseMove, to, w->mapToGlobal(to), Qt::NoButton, Qt::LeftButton,
                         mods);
        QApplication::sendEvent(w, &move);
        QMouseEvent release(QEvent::MouseButtonRelease, to, w->mapToGlobal(to), Qt::LeftButton,
                            Qt::NoButton, Qt::NoModifier);
        QApplication::sendEvent(w, &release);
        QApplication::processEvents();
    }

    /// A move with the button held but no press on this widget: the button went
    /// down somewhere else and the pointer crossed the image.  (A move with no
    /// button at all never reaches the widget, so it cannot exercise anything.)
    void moveOver(const QPoint &to)
    {
        auto *w = canvas();
        ASSERT_NE(w, nullptr);
        QMouseEvent move(QEvent::MouseMove, to, w->mapToGlobal(to), Qt::NoButton, Qt::LeftButton,
                         Qt::NoModifier);
        QApplication::sendEvent(w, &move);
        QApplication::processEvents();
    }

    void wheel(int notches)
    {
        auto *w = canvas();
        ASSERT_NE(w, nullptr);
        const QPointF at(w->width() / 2.0, w->height() / 2.0);
        QWheelEvent ev(at, w->mapToGlobal(at.toPoint()), QPoint(), QPoint(0, 120 * notches),
                       Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
        QApplication::sendEvent(w, &ev);
        QApplication::processEvents();
    }

    QTemporaryDir dir;
    QString startDir;
    SpartaWrapper *sparta = nullptr;
    ImageViewer *viewer   = nullptr;
};

} // namespace

// ------------------------------------------------------------------ rotating

TEST_F(ViewerInput, DraggingRightTurnsTheCameraOneWayAndLeftTheOther)
{
    // the direction is the whole point: a sign error here makes the view move
    // opposite to the hand, which is the single most noticeable way a 3D
    // control can be wrong
    Modals modals;
    ASSERT_NE(build(), nullptr);
    const double before = phi();

    drag({100, 100}, {180, 100}); // 80 px to the right
    const double right = phi();
    EXPECT_GT(right, before)
        << "dragging right turned the camera the other way: " << before << " -> " << right;

    drag({180, 100}, {100, 100}); // and back again
    const double back = phi();
    EXPECT_NEAR(back, before, 1e-6)
        << "dragging back the same distance did not return the camera: " << before << " -> "
        << right << " -> " << back;
}

TEST_F(ViewerInput, TheRotationIsProportionalToHowFarTheMouseMoved)
{
    // a drag twice as long has to turn twice as far, or the control feels
    // unpredictable and small corrections are impossible
    Modals modals;
    ASSERT_NE(build(), nullptr);
    const double start = phi();

    drag({100, 100}, {140, 100}); // 40 px
    const double small = phi() - start;
    drag({140, 100}, {100, 100}); // back to where it began
    ASSERT_NEAR(phi(), start, 1e-6);

    drag({100, 100}, {180, 100}); // 80 px
    const double big = phi() - start;

    ASSERT_NE(small, 0.0);
    EXPECT_NEAR(big, 2 * small, 1e-6)
        << "a drag of twice the distance turned by " << big << " rather than " << 2 * small;
}

TEST_F(ViewerInput, DraggingVerticallyChangesTheElevation)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    const double before = theta();
    drag({100, 100}, {100, 160});
    EXPECT_NE(theta(), before) << "a vertical drag did not change the polar angle";
    EXPECT_DOUBLE_EQ(phi(), kPhi) << "a purely vertical drag also turned the azimuth";
}

TEST_F(ViewerInput, ATwoDimensionalViewNeverAsksForACameraAngle)
{
    // SPARTA forces view 0 0 for a 2d system, so emitting angles would produce a
    // command that contradicts what it will actually render.  The viewer also
    // leaves the polar angle alone while dragging in 2d, but that is invisible
    // from out here for the same reason -- this pins the part that is not.
    Modals modals;
    ASSERT_NE(build(kDeck2d), nullptr);
    ASSERT_EQ(sparta->extractSetting("dimension"), 2);

    drag({100, 100}, {100, 160});
    drag({100, 100}, {160, 100});
    EXPECT_FALSE(command().contains(" view "))
        << "a two-dimensional command carries camera angles SPARTA will ignore:\n"
        << command().toStdString();
}

TEST_F(ViewerInput, AMoveWithoutAPressOnTheImageDoesNotSteerIt)
{
    // the filter tracks a drag that began here, not the pointer: a button
    // pressed on another widget and dragged across the image must not take the
    // camera with it
    Modals modals;
    ASSERT_NE(build(), nullptr);
    const double before = phi();

    moveOver({200, 200});
    EXPECT_DOUBLE_EQ(phi(), before) << "the camera followed a drag that started elsewhere";
}

TEST_F(ViewerInput, AReleasedDragStopsSteering)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    drag({100, 100}, {140, 100});
    const double after = phi();

    // a move after the release: the drag is over, so this must be ignored
    moveOver({300, 100});
    EXPECT_DOUBLE_EQ(phi(), after) << "the camera kept turning after the button was released";
}

// -------------------------------------------------------------------- panning

TEST_F(ViewerInput, ShiftDraggingPansInsteadOfRotating)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    const double beforePhi = phi();
    const QString before   = command();

    drag({100, 100}, {160, 140}, Qt::ShiftModifier);

    EXPECT_DOUBLE_EQ(phi(), beforePhi) << "shift-dragging turned the camera as well as panning";
    EXPECT_NE(command(), before) << "shift-dragging changed nothing at all";
    EXPECT_TRUE(command().contains("center")) << command().toStdString();
}

TEST_F(ViewerInput, PanningStaysInsideTheBox)
{
    // the centre is a box fraction; letting it run past 0 or 1 aims the camera
    // outside the simulation entirely
    Modals modals;
    ASSERT_NE(build(), nullptr);
    for (int i = 0; i < 40; ++i) drag({300, 300}, {0, 0}, Qt::ShiftModifier);

    const QString cmd = command();
    const double cx = numberAfter(cmd, "center", 1, kCentre),
                 cy = numberAfter(cmd, "center", 2, kCentre);
    EXPECT_GE(cx, 0.0);
    EXPECT_LE(cx, 1.0);
    EXPECT_GE(cy, 0.0);
    EXPECT_LE(cy, 1.0);
}

// --------------------------------------------------------------------- zooming

TEST_F(ViewerInput, TheWheelZoomsInOneWayAndOutTheOther)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    const double before = zoom();

    wheel(1);
    const double in = zoom();
    EXPECT_GT(in, before) << "a wheel notch forward did not zoom in";

    wheel(-1);
    EXPECT_NEAR(zoom(), before, 1e-9) << "one notch back did not undo one notch forward";
}

TEST_F(ViewerInput, EachNotchZoomsByTheSameFactor)
{
    // a fixed factor per notch is what makes zooming feel linear on a log scale;
    // an additive step crawls when zoomed in and jumps when zoomed out
    Modals modals;
    ASSERT_NE(build(), nullptr);
    const double z0 = zoom();
    wheel(1);
    const double z1 = zoom();
    wheel(1);
    const double z2 = zoom();

    ASSERT_GT(z0, 0.0);
    EXPECT_NEAR(z1 / z0, z2 / z1, 1e-6)
        << "the second notch scaled by " << z2 / z1 << " and the first by " << z1 / z0;
}

TEST_F(ViewerInput, TheZoomCannotBeDrivenPastItsLimits)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    for (int i = 0; i < 200; ++i) wheel(1);
    const double high = zoom();
    EXPECT_TRUE(std::isfinite(high)) << "the zoom ran away to " << high;
    EXPECT_GT(high, 0.0);

    for (int i = 0; i < 400; ++i) wheel(-1);
    const double low = zoom();
    EXPECT_GT(low, 0.0) << "zooming out far enough turned the zoom to zero or negative";
    EXPECT_LT(low, high);
}

TEST_F(ViewerInput, AWheelEventWithNoNotchesChangesNothing)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    const double before = zoom();
    wheel(0);
    EXPECT_DOUBLE_EQ(zoom(), before);
}

// -------------------------------------------------------------- resetting it

TEST_F(ViewerInput, ResetViewUndoesEverythingTheMouseDid)
{
    Modals modals;
    ASSERT_NE(build(), nullptr);
    const QString original = command();

    drag({100, 100}, {180, 160});
    drag({100, 100}, {160, 140}, Qt::ShiftModifier);
    wheel(3);
    ASSERT_NE(command(), original) << "none of the gestures changed anything";

    QMetaObject::invokeMethod(viewer, "resetView");
    QApplication::processEvents();
    EXPECT_EQ(command(), original) << "the view did not come back to where it started";
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);

    GUI_MONOFONT = std::make_unique<QFont>("Monospace", -1, QFont::Normal);
    GUI_ALLFONT  = std::make_unique<QFont>("Arial", -1, QFont::Normal);
    GUI_MONOFONT->setStyleHint(QFont::Monospace, QFont::PreferQuality);
    GUI_MONOFONT->setFixedPitch(true);
    GUI_ALLFONT->setStyleHint(QFont::SansSerif, QFont::PreferQuality);
    Q_INIT_RESOURCE(spartagui);
    QIcon::setThemeSearchPaths(QStringList() << ":/icons");
    QIcon::setThemeName("spartagui");

    static QTemporaryDir settingsDir;
    QCoreApplication::setOrganizationName("SPARTA-GUI test");
    QCoreApplication::setApplicationName(
        QString("test_imageviewerinput-%1").arg(QCoreApplication::applicationPid()));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settingsDir.path());
    qputenv("XDG_DATA_HOME", settingsDir.path().toLocal8Bit());

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
