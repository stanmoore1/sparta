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

// The surface import wizard's SPARTA-facing half: boxGridCommands(),
// renderViaSparta(), runSpartaWatertight() and the ablation renders.
//
// test_stlimport.cpp covers the parsers, and test_stlimportwizard.cpp covers the
// pages on top of them with no simulator behind the wizard.  This covers what
// happens when there is one -- the domain the wizard builds for the surface it
// just imported, which is where every surface-based simulation starts.
//
// Its failures are the quiet kind.  A box that does not enclose the geometry, or
// a grid that is not the resolution the wizard displayed, still runs and still
// produces numbers.  So the assertions here are against the box SPARTA actually
// ended up with, read back through the library, and against the cell count
// SPARTA itself reported -- not against the command strings, which the builder
// tests already check and which cannot say whether SPARTA agreed.

#include "stlimportwizard.h"

#include "spartawrapper.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QHash>
#include <QDir>
#include <QFile>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QRegularExpression>
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

/// Dismisses anything modal and remembers what it said.  Nothing here should
/// need one; a dialog appearing is itself the finding.
class Modals : public QObject {
public:
    explicit Modals(int budgetMs = 30000) : left(budgetMs)
    {
        timer.setInterval(10);
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
        if ((left -= 10) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            else if (m) m->close();
            return;
        }
        if (!m) return;
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            ++boxes;
            seen << box->text() + " " + box->informativeText() + " " + box->detailedText();
            box->accept();
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
        else m->close();
    }
    QTimer timer;
    int left;
};

// ---------------------------------------------------------------- fixtures

/// A closed tetrahedron spanning the unit corner: bounds are exactly 0..1 in
/// every axis, so the padded box the wizard builds is arithmetic.
QString tetrahedron(double zscale = 1.0)
{
    struct Facet {
        double n[3], v[3][3];
    };
    const Facet facets[4] = {
        {{0, 0, -1}, {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}}},
        {{0, -1, 0}, {{0, 0, 0}, {1, 0, 0}, {0, 0, 1}}},
        {{-1, 0, 0}, {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}}},
        {{1, 1, 1}, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
    };
    QString out = "solid tetra\n";
    for (const auto &f : facets) {
        out += QString("  facet normal %1 %2 %3\n").arg(f.n[0]).arg(f.n[1]).arg(f.n[2]);
        out += "    outer loop\n";
        for (const auto &v : f.v)
            out += QString("      vertex %1 %2 %3\n").arg(v[0]).arg(v[1]).arg(v[2] * zscale);
        out += "    endloop\n  endfacet\n";
    }
    return out + "endsolid tetra\n";
}

/// A closed square as a SPARTA 2d surface file: four points, four lines.  An
/// STL is always triangles, so this is the only way to reach the wizard's
/// two-dimensional path.
QString squareSurf()
{
    return "# a unit square\n\n"
           "4 points\n"
           "4 lines\n\n"
           "Points\n\n"
           "1 0.0 0.0\n"
           "2 1.0 0.0\n"
           "3 1.0 1.0\n"
           "4 0.0 1.0\n\n"
           "Lines\n\n"
           "1 1 2\n"
           "2 2 3\n"
           "3 3 4\n"
           "4 4 1\n";
}

/// The same tetrahedron with one facet removed: an open surface, which is what
/// SPARTA's own watertight check has to reject.
QString openMesh()
{
    QString out = tetrahedron();
    return out.left(out.lastIndexOf("  facet normal")) + "endsolid tetra\n";
}

class WizardLive : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        if (!*testLibrary()) return;
        sparta = new SpartaWrapper;
        if (!sparta->loadLib(testLibrary())) {
            delete sparta;
            sparta = nullptr;
            return;
        }
        char arg0[]  = "sparta";
        char *argv[] = {arg0, nullptr};
        sparta->open(1, argv);
        if (!sparta->isOpen()) {
            delete sparta;
            sparta = nullptr;
        }
    }

    static void TearDownTestSuite()
    {
        if (sparta) sparta->close();
        delete sparta;
        sparta = nullptr;
    }

    void SetUp() override
    {
        if (!sparta) GTEST_SKIP() << "no shared libsparta: configure with -D SPARTA_TEST_LIBRARY=";
        startDir = QDir::currentPath();
        QDir::setCurrent(dir.path());
    }

    void TearDown() override { QDir::setCurrent(startDir); }

    QString writeStl(const QString &name, const QString &text) const
    {
        const QString p = dir.filePath(name);
        QFile f(p);
        EXPECT_TRUE(f.open(QIODevice::WriteOnly | QIODevice::Text));
        f.write(text.toUtf8());
        f.close();
        return p;
    }

    template <class W> static W *ctl(const QDialog &d, const char *name)
    {
        auto *w = d.findChild<W *>(QLatin1String(name));
        if (!w) ADD_FAILURE() << "no control named " << name;
        return w;
    }

    static QString diagnostics(const StlImportWizard &w)
    {
        auto *p = w.findChild<QPlainTextEdit *>("diagnostics");
        return p ? p->toPlainText() : QString();
    }

    /// The box SPARTA ended up with, read back through the library rather than
    /// from the command the wizard emitted.
    static bool boxFromSparta(double lo[3], double hi[3])
    {
        const auto *l = static_cast<const double *>(sparta->extractGlobal("boxlo"));
        const auto *h = static_cast<const double *>(sparta->extractGlobal("boxhi"));
        if (!l || !h) return false;
        for (int k = 0; k < 3; ++k) {
            lo[k] = l[k];
            hi[k] = h[k];
        }
        return true;
    }

    /// How many child cells SPARTA reported building, from its own output.
    static int cellsCreated(const QString &diag)
    {
        const QRegularExpression re(R"((\d+)\s+child grid cells)");
        const auto m = re.match(diag);
        return m.hasMatch() ? m.captured(1).toInt() : -1;
    }

    static SpartaWrapper *sparta;
    QTemporaryDir dir;
    QString startDir;
};

SpartaWrapper *WizardLive::sparta = nullptr;

} // namespace

// ------------------------------------------------------- the domain it builds

TEST_F(WizardLive, TheBoxSpartaEndsUpWithEnclosesTheGeometry)
{
    // the whole point of boxGridCommands(): a box that does not contain the
    // surface still runs, and every result from it is wrong
    Modals modals;
    StlImportWizard w(nullptr, sparta, writeStl("tetra.stl", tetrahedron()));
    ASSERT_TRUE(w.loaded());
    QMetaObject::invokeMethod(&w, "renderSpartaPreview");

    double lo[3], hi[3];
    ASSERT_TRUE(boxFromSparta(lo, hi)) << "SPARTA has no box after the preview render";

    // the mesh spans 0..1 in every axis, so the pad is 8% of 1
    const double pad = 0.08;
    for (int k = 0; k < 3; ++k) {
        EXPECT_LT(lo[k], 0.0) << "axis " << k << " starts inside the geometry";
        EXPECT_GT(hi[k], 1.0) << "axis " << k << " ends inside the geometry";
        EXPECT_NEAR(lo[k], -pad, 1e-6) << "axis " << k;
        EXPECT_NEAR(hi[k], 1.0 + pad, 1e-6) << "axis " << k;
    }
    EXPECT_EQ(sparta->extractSetting("dimension"), 3);
    EXPECT_EQ(sparta->extractSetting("surf_exist"), 1)
        << "the surface never reached SPARTA";
}

TEST_F(WizardLive, ThePaddingFollowsTheLargestExtentInAnyAxis)
{
    // the pad is a fraction of the biggest dimension, so a long thin object gets
    // the same clearance everywhere.  Sizing it from x and y alone looks correct
    // on anything roughly cubic and squeezes a tall object against the lid.
    Modals modals;
    StlImportWizard w(nullptr, sparta, writeStl("tall.stl", tetrahedron(4.0)));
    ASSERT_TRUE(w.loaded());
    QMetaObject::invokeMethod(&w, "renderSpartaPreview");

    double lo[3], hi[3];
    ASSERT_TRUE(boxFromSparta(lo, hi)) << diagnostics(w).toStdString();

    // the mesh is 1 x 1 x 4, so the pad is 8% of 4
    const double pad = 0.08 * 4.0;
    EXPECT_NEAR(lo[0], -pad, 1e-6) << "x was padded from the wrong extent";
    EXPECT_NEAR(hi[0], 1.0 + pad, 1e-6);
    EXPECT_NEAR(lo[2], -pad, 1e-6) << "the long axis was padded from the short ones";
    EXPECT_NEAR(hi[2], 4.0 + pad, 1e-6);
}

TEST_F(WizardLive, TheGridSpartaBuildsIsTheResolutionTheWizardShows)
{
    Modals modals;
    StlImportWizard w(nullptr, sparta, writeStl("tetra.stl", tetrahedron()));
    ASSERT_TRUE(w.loaded());

    auto *nx = ctl<QSpinBox>(w, "grid0");
    auto *ny = ctl<QSpinBox>(w, "grid1");
    auto *nz = ctl<QSpinBox>(w, "grid2");
    ASSERT_NE(nx, nullptr);
    ASSERT_NE(ny, nullptr);
    ASSERT_NE(nz, nullptr);
    nx->setValue(4);
    ny->setValue(5);
    nz->setValue(6);

    QMetaObject::invokeMethod(&w, "renderSpartaPreview");
    const int cells = cellsCreated(diagnostics(w));
    EXPECT_EQ(cells, 4 * 5 * 6)
        << "the wizard displayed 4x5x6 and SPARTA built " << cells << " cells:\n"
        << diagnostics(w).toStdString();
}

TEST_F(WizardLive, ChangingTheResolutionChangesWhatSpartaBuilds)
{
    // a spin box that is read once at construction and never again looks
    // identical in the interface and is wrong in every later render
    Modals modals;
    StlImportWizard w(nullptr, sparta, writeStl("tetra.stl", tetrahedron()));
    ASSERT_TRUE(w.loaded());
    for (const char *n : {"grid0", "grid1", "grid2"}) ctl<QSpinBox>(w, n)->setValue(2);

    QMetaObject::invokeMethod(&w, "renderSpartaPreview");
    const int small = cellsCreated(diagnostics(w));

    ctl<QPlainTextEdit>(w, "diagnostics")->clear();
    for (const char *n : {"grid0", "grid1", "grid2"}) ctl<QSpinBox>(w, n)->setValue(3);
    QMetaObject::invokeMethod(&w, "renderSpartaPreview");
    const int big = cellsCreated(diagnostics(w));

    EXPECT_EQ(small, 8);
    EXPECT_EQ(big, 27) << "the second render used the first render's resolution";
}

TEST_F(WizardLive, ATwoDimensionalSurfaceGetsATwoDimensionalDomain)
{
    // a 2d SPARTA run needs dimension 2 and the standard unit-thickness z slab;
    // getting that wrong makes the surface unusable rather than merely misplaced
    Modals modals;
    StlImportWizard w(nullptr, sparta, writeStl("square.surf", squareSurf()));
    ASSERT_TRUE(w.loaded()) << "the 2d surface file was not read";
    QMetaObject::invokeMethod(&w, "renderSpartaPreview");

    ASSERT_EQ(sparta->extractSetting("box_exist"), 1)
        << "SPARTA built no box: " << diagnostics(w).toStdString();
    EXPECT_EQ(sparta->extractSetting("dimension"), 2);

    double lo[3], hi[3];
    ASSERT_TRUE(boxFromSparta(lo, hi));
    EXPECT_NEAR(lo[2], -0.5, 1e-9) << "a 2d domain needs the standard z slab";
    EXPECT_NEAR(hi[2], 0.5, 1e-9);
    // and x/y still pad the geometry, which spans 0..1
    EXPECT_NEAR(lo[0], -0.08, 1e-6);
    EXPECT_NEAR(hi[1], 1.08, 1e-6);
}

TEST_F(WizardLive, ATwoDimensionalSurfaceGetsASingleLayerOfCells)
{
    // nz must be 1 in 2d whatever the z spin box says, or create_grid is refused
    Modals modals;
    StlImportWizard w(nullptr, sparta, writeStl("square.surf", squareSurf()));
    ASSERT_TRUE(w.loaded());
    ctl<QSpinBox>(w, "grid0")->setValue(4);
    ctl<QSpinBox>(w, "grid1")->setValue(5);
    ctl<QSpinBox>(w, "grid2")->setValue(7); // ignored in 2d

    QMetaObject::invokeMethod(&w, "renderSpartaPreview");
    EXPECT_EQ(cellsCreated(diagnostics(w)), 4 * 5)
        << "the z resolution leaked into a two-dimensional grid:\n"
        << diagnostics(w).toStdString();
}

// ------------------------------------------------- what SPARTA says about it

TEST_F(WizardLive, SpartaAcceptsAClosedSurfaceAndSaysSo)
{
    Modals modals;
    StlImportWizard w(nullptr, sparta, writeStl("closed.stl", tetrahedron()));
    ASSERT_TRUE(w.loaded());

    auto *verdict = ctl<QLabel>(w, "watertight");
    ASSERT_NE(verdict, nullptr);
    EXPECT_TRUE(verdict->text().contains("Watertight", Qt::CaseSensitive))
        << verdict->text().toStdString();
    EXPECT_FALSE(verdict->text().contains("Not watertight")) << verdict->text().toStdString();
}

TEST_F(WizardLive, SpartaRejectsAnOpenSurfaceAndTheVerdictSaysWhy)
{
    // the check that matters: read_surf refuses a leaking surface, and the
    // wizard has to report that rather than the optimistic preflight
    Modals modals;
    StlImportWizard w(nullptr, sparta, writeStl("open.stl", openMesh()));
    ASSERT_TRUE(w.loaded()) << "the open mesh should still parse";

    auto *verdict = ctl<QLabel>(w, "watertight");
    ASSERT_NE(verdict, nullptr);
    EXPECT_TRUE(verdict->text().contains("Not watertight"))
        << "an open surface was reported as usable: " << verdict->text().toStdString();
}

TEST_F(WizardLive, TheDiagnosticsRecordWhatSpartaActuallySaid)
{
    // without this the wizard is a black box: a render that fails leaves the
    // user with an error dialog and no way to find out what SPARTA objected to
    Modals modals;
    StlImportWizard w(nullptr, sparta, writeStl("tetra.stl", tetrahedron()));
    ASSERT_TRUE(w.loaded());
    QMetaObject::invokeMethod(&w, "renderSpartaPreview");

    const QString d = diagnostics(w);
    EXPECT_TRUE(d.contains("SPARTA preview render")) << d.toStdString();
    EXPECT_TRUE(d.contains("child grid cells"))
        << "SPARTA's own output did not reach the diagnostics pane:\n"
        << d.toStdString();
}

TEST_F(WizardLive, TheAuthoritativeRenderProducesAPicture)
{
    Modals modals;
    StlImportWizard w(nullptr, sparta, writeStl("tetra.stl", tetrahedron()));
    ASSERT_TRUE(w.loaded());

    auto *preview = ctl<QLabel>(w, "preview");
    ASSERT_NE(preview, nullptr);
    QMetaObject::invokeMethod(&w, "renderSpartaPreview");

    ASSERT_FALSE(preview->pixmap().isNull())
        << "no image reached the preview: " << diagnostics(w).toStdString();
    const QImage img = preview->pixmap().toImage();
    EXPECT_GT(img.width(), 0);

    // and it has to be a render of something.  The surface is drawn flat gray on
    // the background, so a correct picture has exactly two colours and a
    // substantial minority of pixels in the rarer one; a wrong camera or group
    // gives a single uniform colour.
    QHash<QRgb, int> colours;
    int sampled = 0;
    for (int y = 0; y < img.height(); y += 2)
        for (int x = 0; x < img.width(); x += 2) {
            ++colours[img.pixel(x, y)];
            ++sampled;
        }
    ASSERT_GE(colours.size(), 2) << "the preview is one flat colour; nothing was drawn";
    int rarest = sampled;
    for (auto it = colours.constBegin(); it != colours.constEnd(); ++it)
        rarest = qMin(rarest, it.value());
    EXPECT_GT(rarest * 200, sampled)
        << "the drawn surface covers under half a percent of the image; the camera "
           "is not looking at it";
}

TEST_F(WizardLive, WithNoSimulatorItSaysSoRatherThanRenderingNothing)
{
    Modals modals;
    StlImportWizard w(nullptr, nullptr, writeStl("tetra.stl", tetrahedron()));
    ASSERT_TRUE(w.loaded());
    QMetaObject::invokeMethod(&w, "renderSpartaPreview");

    EXPECT_TRUE(modals.said("SPARTA library is not loaded") || modals.said("render failed"))
        << "a render with no library reported: " << modals.all().toStdString();
}

TEST_F(WizardLive, ARenderLeavesNoFilesBehind)
{
    // it dumps into the temp directory and has to sweep up after itself, or a
    // long session fills the disk with frames nobody will look at
    Modals modals;
    const QStringList before = QDir(QDir::tempPath()).entryList({"sguiwiz*"}, QDir::Files);
    StlImportWizard w(nullptr, sparta, writeStl("tetra.stl", tetrahedron()));
    ASSERT_TRUE(w.loaded());
    QMetaObject::invokeMethod(&w, "renderSpartaPreview");

    const QStringList after = QDir(QDir::tempPath()).entryList({"sguiwiz*.ppm"}, QDir::Files);
    EXPECT_TRUE(after.isEmpty()) << "the render left " << after.join(", ").toStdString();
}

TEST_F(WizardLive, TheWizardIsStillUsableAfterARenderFails)
{
    // renderViaSparta() clears the instance and rebuilds it; a failure part way
    // through must not leave the wrapper in a state the next render inherits
    Modals modals;
    StlImportWizard bad(nullptr, sparta, writeStl("open.stl", openMesh()));
    ASSERT_TRUE(bad.loaded());
    QMetaObject::invokeMethod(&bad, "renderSpartaPreview");

    StlImportWizard good(nullptr, sparta, writeStl("tetra.stl", tetrahedron()));
    ASSERT_TRUE(good.loaded());
    QMetaObject::invokeMethod(&good, "renderSpartaPreview");
    EXPECT_FALSE(ctl<QLabel>(good, "preview")->pixmap().isNull())
        << "a good surface would not render after a bad one: "
        << diagnostics(good).toStdString();
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_stlwizardlive.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
