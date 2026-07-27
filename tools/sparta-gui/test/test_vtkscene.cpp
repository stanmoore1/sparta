// Tests for the 3D scene widget and the window that hosts it on its own
// (src/vtkscene.cpp), built only when SPARTA-GUI is configured with VTK.
//
// The 3D viewer had no tests at all. Its controls carried no tooltips, so they
// never got accessible names, so nothing could drive it from outside the
// process either -- it was the one window in the application with no coverage
// of any kind.
//
// What is checked here is the surface the rest of the program actually calls.
// The STL import wizard builds a polydata in memory, hands it over, colours it
// by a named field and shows the window; if any of that breaks, a user
// importing a leaking mesh loses the only view that shows them where the leaks
// are. That path is worth a test that does not need a display server.

#include "vtkscene.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QImage>
#include <QFile>
#include <QTemporaryDir>
#include <QDialog>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QSpinBox>
#include <QTest>
#include <QTimer>

#include "chartviewer.h"
#include "constants.h"
#include "plotdata.h"
#include "vtkfilters.h"

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCellType.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkDataArray.h>
#include <vtkImageData.h>

#include "gtest/gtest.h"

#include <cmath>

namespace {

// A two-triangle sheet with a per-cell "leak" scalar: the same shape the STL
// import wizard hands over when it reports a non-watertight surface.
vtkSmartPointer<vtkPolyData> leakySheet()
{
    auto pts = vtkSmartPointer<vtkPoints>::New();
    pts->InsertNextPoint(0.0, 0.0, 0.0);
    pts->InsertNextPoint(1.0, 0.0, 0.0);
    pts->InsertNextPoint(0.0, 1.0, 0.0);
    pts->InsertNextPoint(1.0, 1.0, 0.0);

    auto cells = vtkSmartPointer<vtkCellArray>::New();
    const vtkIdType t1[3] = {0, 1, 2};
    const vtkIdType t2[3] = {1, 3, 2};
    cells->InsertNextCell(3, t1);
    cells->InsertNextCell(3, t2);

    auto leak = vtkSmartPointer<vtkDoubleArray>::New();
    leak->SetName("leak");
    leak->InsertNextValue(0.0);
    leak->InsertNextValue(1.0);   // this one is the leaking element

    auto pd = vtkSmartPointer<vtkPolyData>::New();
    pd->SetPoints(pts);
    pd->SetPolys(cells);
    pd->GetCellData()->AddArray(leak);
    return pd;
}

// ---------------------------------------------------------------------------
// Looking at what was drawn
// ---------------------------------------------------------------------------
//
// Everything above this point checks bookkeeping: a layer was accepted, the
// list was emptied, a field name was tolerated. None of it would notice the
// scene rendering an empty grey rectangle, which is the failure a user of a 3D
// viewer actually experiences. The helpers below render the widget off-screen
// and read the pixels back, so the checks that follow can say where the
// geometry is and what colour it came out.

/**
 * @brief True if a render window can actually be created here
 *
 * VTK's "off-screen rendering" is off-screen in the sense that it does not
 * open a visible window; on Linux it is still vtkXOpenGLRenderWindow talking
 * GLX to an X server. With no DISPLAY it does not fail gracefully, it prints
 * "bad X server connection" and takes the process down, so the tests that
 * render have to check before they build one. The suite is run under Xvfb by
 * ctest; running the binary by hand without a display skips them and says so.
 */
bool canRender()
{
    return !qEnvironmentVariableIsEmpty("DISPLAY");
}

#define REQUIRE_RENDERER()                                                                    \
    if (!canRender()) GTEST_SKIP() << "VTK needs an X server even to render off-screen; "     \
                                      "run this under Xvfb (ctest does)"

/// Give the scene a size and let the layout reach the render window, which
/// only sizes its frame buffer from a resize event.
void present(QWidget &w, int width = 420, int height = 320)
{
    w.resize(width, height);
    w.show();
    QApplication::processEvents();
    QTest::qWait(50);
    QApplication::processEvents();
}

/// The render surface inside the scene, for tests that drive the camera.
VtkRenderArea *areaOf(VtkScene &scene)
{
    return scene.findChild<VtkRenderArea *>("vtkRenderArea");
}

/// Sum of the per-channel differences between two colours, 0..765.
int colorDistance(const QColor &a, const QColor &b)
{
    return std::abs(a.red() - b.red()) + std::abs(a.green() - b.green()) +
           std::abs(a.blue() - b.blue());
}

/**
 * @brief True if this pixel is geometry rather than background
 *
 * The scene's background is a vertical gradient, so there is no single
 * background colour to compare against: taking one corner as the reference
 * marks three quarters of an empty scene as "drawn". Along any one row the
 * gradient is constant, though, so each pixel is compared against the two
 * ends of its own row. Requiring it to differ from both means geometry that
 * happens to touch one edge does not make the whole row read as drawn.
 */
bool isDrawn(const QImage &img, int x, int y)
{
    const QColor c(img.pixel(x, y));
    const QColor left(img.pixel(0, y));
    const QColor right(img.pixel(img.width() - 1, y));
    return colorDistance(c, left) > 24 && colorDistance(c, right) > 24;
}

/// Fraction of the image covered by geometry: 0 means nothing was drawn.
double drawnFraction(const QImage &img)
{
    if (img.isNull() || img.width() < 3) return 0.0;
    int drawn = 0;
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x)
            if (isDrawn(img, x, y)) ++drawn;
    return double(drawn) / (img.width() * img.height());
}

/// Mean absolute per-channel difference between two renders, 0..255.
double imageDelta(const QImage &a, const QImage &b)
{
    if (a.isNull() || b.isNull() || a.size() != b.size()) return 255.0;
    double sum = 0.0;
    for (int y = 0; y < a.height(); ++y) {
        for (int x = 0; x < a.width(); ++x) {
            const QColor p(a.pixel(x, y)), q(b.pixel(x, y));
            sum += std::abs(p.red() - q.red()) + std::abs(p.green() - q.green()) +
                   std::abs(p.blue() - q.blue());
        }
    }
    return sum / (3.0 * a.width() * a.height());
}

/// Average colour of the drawn (non-background) pixels in a sub-rectangle,
/// given as fractions of the image. Background is ignored so the answer is
/// the colour of the geometry rather than of the space around it.
QColor drawnColorIn(const QImage &img, double x0, double y0, double x1, double y1)
{
    if (img.isNull() || img.width() < 3) return {};
    long r = 0, g = 0, b = 0, n = 0;
    for (int y = int(y0 * img.height()); y < int(y1 * img.height()); ++y) {
        for (int x = int(x0 * img.width()); x < int(x1 * img.width()); ++x) {
            if (!isDrawn(img, x, y)) continue;
            const QColor c(img.pixel(x, y));
            r += c.red();
            g += c.green();
            b += c.blue();
            ++n;
        }
    }
    if (n == 0) return {};
    return QColor(int(r / n), int(g / n), int(b / n));
}

/// A 3x3 grid of quads with a per-point "temp" ramp along x, as an
/// unstructured grid -- the shape "dump grid/vtk" produces.
vtkSmartPointer<vtkUnstructuredGrid> quadGrid()
{
    auto pts = vtkSmartPointer<vtkPoints>::New();
    for (int j = 0; j <= 3; ++j)
        for (int i = 0; i <= 3; ++i) pts->InsertNextPoint(i, j, 0.0);

    auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    grid->SetPoints(pts);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            const vtkIdType q[4] = {j * 4 + i, j * 4 + i + 1, (j + 1) * 4 + i + 1,
                                    (j + 1) * 4 + i};
            grid->InsertNextCell(VTK_QUAD, 4, q);
        }
    }

    auto temp = vtkSmartPointer<vtkDoubleArray>::New();
    temp->SetName("temp");
    for (int j = 0; j <= 3; ++j)
        for (int i = 0; i <= 3; ++i) temp->InsertNextValue(double(i));
    grid->GetPointData()->AddArray(temp);
    return grid;
}

TEST(VtkScene, StartsEmpty)
{
    VtkScene scene;
    EXPECT_FALSE(scene.hasContent());
}

TEST(VtkScene, AcceptsAnInMemoryDataset)
{
    VtkScene scene;
    scene.addDataset(leakySheet(), "surface (leaks in red)", VtkScene::Kind::Surface);
    EXPECT_TRUE(scene.hasContent());
}

TEST(VtkScene, ClearingRemovesEveryLayer)
{
    VtkScene scene;
    scene.addDataset(leakySheet(), "one", VtkScene::Kind::Surface);
    scene.addDataset(leakySheet(), "two", VtkScene::Kind::Surface);
    ASSERT_TRUE(scene.hasContent());
    scene.clearScene();
    EXPECT_FALSE(scene.hasContent());
}

// Colouring by a field that is not there must be ignored rather than fatal:
// the wizard only asks for "leak" when the mesh actually leaks.
TEST(VtkScene, ColouringByAnAbsentFieldIsHarmless)
{
    VtkScene scene;
    scene.addDataset(leakySheet(), "surface", VtkScene::Kind::Surface);
    scene.setColorField("no-such-field");
    EXPECT_TRUE(scene.hasContent());
}

TEST(VtkScene, ColouringByThePresentFieldKeepsTheScene)
{
    VtkScene scene;
    scene.addDataset(leakySheet(), "surface", VtkScene::Kind::Surface);
    scene.setColorField("leak");
    EXPECT_TRUE(scene.hasContent());
}

TEST(VtkScene, ReportsItselfAsAViewerSource)
{
    VtkScene scene;
    EXPECT_EQ(scene.sourceLabel().toStdString(), "3D");
    EXPECT_FALSE(scene.sourceTip().isEmpty());
    // an empty source has to say how to fill it, or it is just a blank pane
    EXPECT_FALSE(scene.emptyTip().isEmpty());
}

// This is the sequence StlImportWizard::showLeaksIn3D() performs verbatim.
// It is the reason the standalone window still exists after the viewers were
// consolidated: the wizard is a dialog over the main window, so its result
// must not be docked behind the dialog the user is still looking at.
TEST(SceneWindow, RunsTheLeakViewerSequence)
{
    SceneWindow win;
    EXPECT_FALSE(win.hasContent());

    win.clearScene();
    win.addDataset(leakySheet(), "surface (leaks in red)", SceneWindow::Kind::Surface);
    win.setColorField("leak");

    EXPECT_TRUE(win.hasContent());
    ASSERT_NE(win.scene(), nullptr);
    EXPECT_TRUE(win.scene()->hasContent());
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

TEST(VtkSceneRender, AnEmptySceneIsJustBackground)
{
    REQUIRE_RENDERER();
    VtkScene scene;
    present(scene);
    const QImage img = scene.currentImage();
    ASSERT_FALSE(img.isNull());
    EXPECT_GT(img.width(), 0);
    // the background is a vertical gradient, so "empty" is not one flat colour;
    // what it must not contain is geometry
    EXPECT_LT(drawnFraction(img), 0.02);
}

TEST(VtkSceneRender, ASurfaceActuallyReachesTheFrameBuffer)
{
    REQUIRE_RENDERER();
    VtkScene scene;
    present(scene);
    const QImage before = scene.currentImage();

    scene.addDataset(leakySheet(), "surface", VtkScene::Kind::Surface);
    QApplication::processEvents();
    const QImage after = scene.currentImage();

    ASSERT_FALSE(after.isNull());
    // resetView() frames the geometry, so it fills a good part of the view
    EXPECT_GT(drawnFraction(after), 0.10);
    EXPECT_GT(imageDelta(before, after), 1.0);
}

TEST(VtkSceneRender, TheGeometryIsFramedInTheMiddle)
{
    REQUIRE_RENDERER();
    // resetView() is supposed to point the camera at the data. If it did not,
    // the scene would render something -- just not where anyone can see it.
    VtkScene scene;
    present(scene);
    scene.addDataset(leakySheet(), "surface", VtkScene::Kind::Surface);
    QApplication::processEvents();
    const QImage img = scene.currentImage();
    ASSERT_FALSE(img.isNull());

    const QColor centre = drawnColorIn(img, 0.4, 0.4, 0.6, 0.6);
    EXPECT_TRUE(centre.isValid()) << "nothing drawn in the middle of the view";
    // and the very edge is still background
    EXPECT_LT(drawnFraction(img.copy(0, 0, img.width(), 4)), 0.05);
}

TEST(VtkSceneRender, ClearingTheSceneClearsThePicture)
{
    REQUIRE_RENDERER();
    VtkScene scene;
    present(scene);
    scene.addDataset(leakySheet(), "surface", VtkScene::Kind::Surface);
    QApplication::processEvents();
    ASSERT_GT(drawnFraction(scene.currentImage()), 0.10);

    scene.clearScene();
    QApplication::processEvents();
    EXPECT_LT(drawnFraction(scene.currentImage()), 0.02)
        << "the actor was dropped from the list but not from the renderer";
}

TEST(VtkSceneRender, ColouringBySurfaceLeakPaintsTheLeakingElementDifferently)
{
    REQUIRE_RENDERER();
    // This is the picture the STL import wizard exists to show: two triangles,
    // one flagged, and the flagged one must not look like its neighbour. The
    // sheet is two triangles of the unit square -- (0,0),(1,0),(0,1) with
    // leak=0 and (1,0),(1,1),(0,1) with leak=1 -- so with the default camera
    // (+x right, +y up) the unflagged one is the lower-left half of the view
    // and the flagged one the upper-right half.
    VtkScene scene;
    present(scene);
    scene.addDataset(leakySheet(), "surface", VtkScene::Kind::Surface);
    QApplication::processEvents();
    const QImage solid = scene.currentImage();

    scene.setColorField("leak");
    QApplication::processEvents();
    const QImage colored = scene.currentImage();
    ASSERT_FALSE(colored.isNull());

    EXPECT_GT(imageDelta(solid, colored), 2.0) << "colouring by a field changed nothing";

    const QColor low  = drawnColorIn(colored, 0.15, 0.55, 0.40, 0.85); // lower left
    const QColor high = drawnColorIn(colored, 0.60, 0.15, 0.85, 0.45); // upper right
    ASSERT_TRUE(low.isValid()) << "no geometry in the lower-left half";
    ASSERT_TRUE(high.isValid()) << "no geometry in the upper-right half";

    // the default map runs blue (low) to red (high)
    EXPECT_GT(low.blue(), low.red()) << "leak=0 element is not on the cold end";
    EXPECT_GT(high.red(), high.blue()) << "leak=1 element is not on the hot end";
}

TEST(VtkSceneRender, ChangingTheColorMapRepaints)
{
    REQUIRE_RENDERER();
    VtkScene scene;
    present(scene);
    scene.addDataset(quadGrid(), "grid", VtkScene::Kind::Grid);
    scene.setColorField("temp");
    QApplication::processEvents();
    const QImage rainbow = scene.currentImage();

    auto *cmap = scene.findChild<QComboBox *>();
    ASSERT_NE(cmap, nullptr);
    // find the colour-map combo rather than the field combo
    for (auto *c : scene.findChildren<QComboBox *>())
        if (c->count() > 0 && c->itemText(0) == "Rainbow") cmap = c;
    ASSERT_EQ(cmap->itemText(0), QString("Rainbow"));

    // Grayscale is the one map that cannot look like any of the others
    const int gray = cmap->findText("Grayscale");
    ASSERT_GE(gray, 0);
    cmap->setCurrentIndex(gray);
    QApplication::processEvents();
    const QImage grayscale = scene.currentImage();

    EXPECT_GT(imageDelta(rainbow, grayscale), 2.0);
    const QColor c = drawnColorIn(grayscale, 0.3, 0.3, 0.7, 0.7);
    ASSERT_TRUE(c.isValid());
    // grey means the channels agree; lighting keeps it from being exact
    EXPECT_LT(std::abs(c.red() - c.blue()), 40);
    EXPECT_LT(std::abs(c.red() - c.green()), 40);
}

TEST(VtkSceneRender, EdgesAndLegendChangeThePicture)
{
    REQUIRE_RENDERER();
    VtkScene scene;
    present(scene);
    scene.addDataset(quadGrid(), "grid", VtkScene::Kind::Grid);
    scene.setColorField("temp");
    QApplication::processEvents();
    const QImage plain = scene.currentImage();

    QCheckBox *edges = nullptr, *legend = nullptr;
    for (auto *b : scene.findChildren<QCheckBox *>()) {
        if (b->text() == "Edges") edges = b;
        if (b->text() == "Legend") legend = b;
    }
    ASSERT_NE(edges, nullptr);
    ASSERT_NE(legend, nullptr);

    edges->setChecked(!edges->isChecked());
    QApplication::processEvents();
    const QImage withEdges = scene.currentImage();
    EXPECT_GT(imageDelta(plain, withEdges), 0.5) << "the Edges box drew nothing";

    edges->setChecked(!edges->isChecked());
    QApplication::processEvents();
    EXPECT_LT(imageDelta(plain, scene.currentImage()), 0.5) << "unchecking Edges left them on";

    // the Legend box starts checked, so toggling it takes the scale bar away
    ASSERT_TRUE(legend->isChecked());
    legend->setChecked(false);
    QApplication::processEvents();
    const QImage noLegend = scene.currentImage();
    EXPECT_GT(imageDelta(plain, noLegend), 0.5) << "unchecking Legend removed nothing";

    // and it goes from the side of the view, where VTK puts it, not from over
    // the data -- a legend drawn across the middle would hide what it explains
    const int strip = plain.width() / 5;
    EXPECT_GT(imageDelta(plain.copy(plain.width() - strip, 0, strip, plain.height()),
                         noLegend.copy(plain.width() - strip, 0, strip, plain.height())),
              0.2)
        << "nothing changed at the edge, so the scale bar was not there";

    legend->setChecked(true);
    QApplication::processEvents();
    EXPECT_LT(imageDelta(plain, scene.currentImage()), 0.5) << "the legend did not come back";
}

TEST(VtkSceneRender, DraggingRotatesAndResetPutsItBack)
{
    REQUIRE_RENDERER();
    VtkScene scene;
    present(scene);
    scene.addDataset(quadGrid(), "grid", VtkScene::Kind::Grid);
    QApplication::processEvents();
    const QImage home = scene.currentImage();

    VtkRenderArea *area = areaOf(scene);
    ASSERT_NE(area, nullptr);

    const QPoint from(area->width() / 2, area->height() / 2);
    QTest::mousePress(area, Qt::LeftButton, Qt::NoModifier, from);
    QTest::mouseMove(area, from + QPoint(60, 25));
    QTest::mouseRelease(area, Qt::LeftButton, Qt::NoModifier, from + QPoint(60, 25));
    QApplication::processEvents();

    const QImage turned = scene.currentImage();
    EXPECT_GT(imageDelta(home, turned), 1.0) << "dragging did not move the camera";

    scene.resetView();
    QApplication::processEvents();
    EXPECT_LT(imageDelta(home, scene.currentImage()), 0.5)
        << "reset did not return to the framed view";
}

TEST(VtkSceneRender, TheWheelZooms)
{
    REQUIRE_RENDERER();
    VtkScene scene;
    present(scene);
    scene.addDataset(leakySheet(), "surface", VtkScene::Kind::Surface);
    QApplication::processEvents();
    const double before = drawnFraction(scene.currentImage());

    VtkRenderArea *area = areaOf(scene);
    ASSERT_NE(area, nullptr);
    const QPoint at(area->width() / 2, area->height() / 2);
    QWheelEvent wheel(QPointF(at), area->mapToGlobal(QPointF(at)), QPoint(), QPoint(0, 480),
                      Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
    QApplication::sendEvent(area, &wheel);
    QApplication::processEvents();

    // zooming in makes the geometry cover more of the view; the direction
    // matters, so this would catch an inverted wheel as well as a dead one
    EXPECT_GT(drawnFraction(scene.currentImage()), before * 1.1);
}

TEST(VtkSceneRender, ParticleAndSurfaceLayersDoNotLookAlike)
{
    REQUIRE_RENDERER();
    // the Kind argument picks a representation and a colour, and getting it
    // wrong means particle dumps render as a solid block
    VtkScene particles;
    present(particles);
    particles.addDataset(quadGrid(), "particles", VtkScene::Kind::Particles);
    QApplication::processEvents();

    VtkScene surface;
    present(surface);
    surface.addDataset(quadGrid(), "surface", VtkScene::Kind::Surface);
    QApplication::processEvents();

    const double dots = drawnFraction(particles.currentImage());
    const double solid = drawnFraction(surface.currentImage());
    EXPECT_GT(solid, dots * 2.0) << "points rendered as solidly as a surface";
    EXPECT_GT(dots, 0.0) << "no points drawn at all";
}

// ---------------------------------------------------------------------------
// Reading the files SPARTA writes
// ---------------------------------------------------------------------------
//
// addDatasetFile() is what "dump particle/vtk", "dump grid/vtk" and
// "dump surf/vtk" output goes through, and it had no test of any kind: the
// only covered entry point was the in-memory one the STL wizard uses.

TEST(VtkSceneFiles, ReadsXmlPolyData)
{
    REQUIRE_RENDERER();
    QTemporaryDir tmp;
    ASSERT_TRUE(tmp.isValid());
    const QString path = tmp.filePath("surf.vtp");

    auto w = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    w->SetFileName(path.toLocal8Bit().constData());
    w->SetInputData(leakySheet());
    ASSERT_EQ(w->Write(), 1);

    VtkScene scene;
    present(scene);
    QString err;
    ASSERT_TRUE(scene.addDatasetFile(path, "surf", VtkScene::Kind::Surface, &err)) << err.toStdString();
    QApplication::processEvents();
    EXPECT_TRUE(scene.hasContent());
    EXPECT_GT(drawnFraction(scene.currentImage()), 0.10) << "read but not drawn";

    // the field travelled with the file
    scene.setColorField("leak");
    QApplication::processEvents();
    const QColor high = drawnColorIn(scene.currentImage(), 0.60, 0.15, 0.85, 0.45);
    ASSERT_TRUE(high.isValid());
    EXPECT_GT(high.red(), high.blue());
}

TEST(VtkSceneFiles, ReadsXmlUnstructuredGrid)
{
    REQUIRE_RENDERER();
    QTemporaryDir tmp;
    ASSERT_TRUE(tmp.isValid());
    const QString path = tmp.filePath("grid.vtu");

    auto w = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    w->SetFileName(path.toLocal8Bit().constData());
    w->SetInputData(quadGrid());
    ASSERT_EQ(w->Write(), 1);

    VtkScene scene;
    present(scene);
    QString err;
    ASSERT_TRUE(scene.addDatasetFile(path, "grid", VtkScene::Kind::Grid, &err)) << err.toStdString();
    QApplication::processEvents();
    EXPECT_GT(drawnFraction(scene.currentImage()), 0.10);
}

TEST(VtkSceneFiles, ReadsTheLegacyFormat)
{
    REQUIRE_RENDERER();
    // .vtk is what a SPARTA build without the XML writers produces, and the
    // reader for it is a different class from the two above
    QTemporaryDir tmp;
    ASSERT_TRUE(tmp.isValid());
    const QString poly = tmp.filePath("surf.vtk");
    const QString grid = tmp.filePath("grid.vtk");

    auto pw = vtkSmartPointer<vtkPolyDataWriter>::New();
    pw->SetFileName(poly.toLocal8Bit().constData());
    pw->SetInputData(leakySheet());
    ASSERT_EQ(pw->Write(), 1);

    auto gw = vtkSmartPointer<vtkUnstructuredGridWriter>::New();
    gw->SetFileName(grid.toLocal8Bit().constData());
    gw->SetInputData(quadGrid());
    ASSERT_EQ(gw->Write(), 1);

    VtkScene scene;
    present(scene);
    QString err;
    EXPECT_TRUE(scene.addDatasetFile(poly, "surf", VtkScene::Kind::Surface, &err))
        << err.toStdString();
    EXPECT_TRUE(scene.addDatasetFile(grid, "grid", VtkScene::Kind::Grid, &err))
        << err.toStdString();
    QApplication::processEvents();
    EXPECT_GT(drawnFraction(scene.currentImage()), 0.10);
}

TEST(VtkSceneFiles, LayersStackRatherThanReplace)
{
    REQUIRE_RENDERER();
    QTemporaryDir tmp;
    ASSERT_TRUE(tmp.isValid());
    const QString path = tmp.filePath("surf.vtp");
    auto w = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    w->SetFileName(path.toLocal8Bit().constData());
    w->SetInputData(leakySheet());
    ASSERT_EQ(w->Write(), 1);

    VtkScene scene;
    present(scene);
    ASSERT_TRUE(scene.addDatasetFile(path, "one", VtkScene::Kind::Surface));
    scene.addDataset(quadGrid(), "two", VtkScene::Kind::Grid);
    QApplication::processEvents();
    // the grid is much larger than the sheet, so framing both has to widen the
    // view -- if the second load replaced the first, it would look identical
    // to loading the grid alone, which it does not need to prove; what it does
    // need to prove is that both are still there
    EXPECT_TRUE(scene.hasContent());
    scene.clearScene();
    EXPECT_FALSE(scene.hasContent());
}

TEST(VtkSceneFiles, RefusesAMissingFileWithAReason)
{
    REQUIRE_RENDERER();
    VtkScene scene;
    QString err;
    EXPECT_FALSE(scene.addDatasetFile("/nonexistent/nothing.vtu", "x",
                                      VtkScene::Kind::Grid, &err));
    EXPECT_FALSE(err.isEmpty()) << "failed silently, so the user is told nothing";
    EXPECT_FALSE(scene.hasContent());
}

TEST(VtkSceneFiles, RefusesAFileThatIsNotVtkData)
{
    REQUIRE_RENDERER();
    QTemporaryDir tmp;
    ASSERT_TRUE(tmp.isValid());
    const QString path = tmp.filePath("junk.vtu");
    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly));
    f.write("this is not a VTK file\n");
    f.close();

    VtkScene scene;
    QString err;
    EXPECT_FALSE(scene.addDatasetFile(path, "junk", VtkScene::Kind::Grid, &err));
    EXPECT_FALSE(scene.hasContent());
}

TEST(VtkSceneFiles, RefusesAnUnknownExtension)
{
    REQUIRE_RENDERER();
    QTemporaryDir tmp;
    ASSERT_TRUE(tmp.isValid());
    const QString path = tmp.filePath("data.txt");
    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly));
    f.write("1 2 3\n");
    f.close();

    VtkScene scene;
    QString err;
    EXPECT_FALSE(scene.addDatasetFile(path, "text", VtkScene::Kind::Generic, &err));
    EXPECT_FALSE(err.isEmpty());
}

TEST(SceneWindow, ClearingBetweenImportsLeavesNothingBehind)
{
    SceneWindow win;
    win.addDataset(leakySheet(), "first import", SceneWindow::Kind::Surface);
    ASSERT_TRUE(win.hasContent());

    // the wizard clears before every re-render, so a second look at a repaired
    // mesh must not still show the first one underneath
    win.clearScene();
    win.addDataset(leakySheet(), "second import", SceneWindow::Kind::Surface);
    EXPECT_TRUE(win.hasContent());
    win.clearScene();
    EXPECT_FALSE(win.hasContent());
}


// ---------------------------------------------------------------------------
// Probing and slicing
// ---------------------------------------------------------------------------
//
// The filters themselves are checked in test_vtkfilters.cpp against known
// fields.  What was never checked is the scene's use of them: which field it
// samples, where it puts the line, and what it reports back.  These are the
// only numbers the 3D viewer states as fact, and a probe that resolves the
// wrong cell says a plausible one with no way to notice.

/// A 11x11x11 grid over the unit cube whose scalar is exactly the x coordinate,
/// so anything sampled from it has an answer that can be written down.
vtkSmartPointer<vtkImageData> rampField(const char *name = "ramp")
{
    auto img = vtkSmartPointer<vtkImageData>::New();
    img->SetDimensions(11, 11, 11);
    img->SetOrigin(0.0, 0.0, 0.0);
    img->SetSpacing(0.1, 0.1, 0.1);

    auto a = vtkSmartPointer<vtkDoubleArray>::New();
    a->SetName(name);
    a->SetNumberOfTuples(img->GetNumberOfPoints());
    for (vtkIdType i = 0; i < img->GetNumberOfPoints(); ++i) {
        double p[3];
        img->GetPoint(i, p);
        a->SetTuple1(i, p[0]);
    }
    img->GetPointData()->AddArray(a);
    img->GetPointData()->SetScalars(a);
    return img;
}

/// Answers the QInputDialogs the filters put up, in order, and records any
/// message box instead of leaving it modal with nobody to press it.
class Answers : public QObject {
public:
    explicit Answers(QStringList replies, int budgetMs = 10000) :
        replies(std::move(replies)), left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &Answers::poll);
        timer.start();
    }
    QStringList messages;
    int dialogs = 0;

    [[nodiscard]] bool said(const QString &needle) const
    {
        for (const auto &m : messages)
            if (m.contains(needle)) return true;
        return false;
    }
    [[nodiscard]] QString all() const { return messages.join(" | "); }

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            else if (m) m->close();
            return;
        }
        if (!m) return;
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            messages << box->text() + " " + box->informativeText();
            box->accept();
            return;
        }
        if (auto *dlg = qobject_cast<QInputDialog *>(m)) {
            ++dialogs;
            if (replies.isEmpty()) {
                dlg->reject();
                return;
            }
            const QString reply = replies.takeFirst();
            if (reply.isNull()) {
                dlg->reject(); // the user pressed Cancel on this one
                return;
            }
            // the combo, spin and text variants each expose a different editor
            if (auto *combo = dlg->findChild<QComboBox *>()) combo->setCurrentText(reply);
            else if (auto *spin = dlg->findChild<QSpinBox *>()) spin->setValue(reply.toInt());
            else if (auto *dspin = dlg->findChild<QDoubleSpinBox *>()) dspin->setValue(reply.toDouble());
            else if (auto *edit = dlg->findChild<QLineEdit *>()) edit->setText(reply);
            dlg->accept();
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
    }
    QTimer timer;
    QStringList replies;
    int left;
};

/// The chart window a line probe opens, if it opened one.
ChartWindow *newestChart()
{
    ChartWindow *found = nullptr;
    for (auto *w : QApplication::topLevelWidgets())
        if (auto *c = qobject_cast<ChartWindow *>(w)) found = c;
    return found;
}

void closeCharts()
{
    // they carry WA_DeleteOnClose, so closing IS the delete
    for (auto *w : QApplication::topLevelWidgets())
        if (auto *c = qobject_cast<ChartWindow *>(w)) c->close();
    QApplication::processEvents();
}

QString statusOf(VtkScene &scene)
{
    auto *l = scene.findChild<QLabel *>("status");
    return l ? l->text() : QString();
}

} // namespace

// ---------------------------------------------------------------- line probe

TEST(SceneProbe, SamplesTheChosenFieldAlongTheDomainAndChartsIt)
{
    REQUIRE_RENDERER();
    closeCharts();
    VtkScene scene;
    scene.addDataset(rampField(), "ramp", VtkScene::Kind::Generic);
    present(scene);

    // the field name, then the sample count
    Answers answers({"ramp", "21"});
    QMetaObject::invokeMethod(&scene, "applyLineProbe");
    QApplication::processEvents();

    EXPECT_EQ(answers.dialogs, 2) << "the probe did not ask which field and how many samples";
    auto *chart = newestChart();
    ASSERT_NE(chart, nullptr) << "the probe produced no chart: " << answers.all().toStdString();
    EXPECT_TRUE(chart->windowTitle().contains("ramp"))
        << "the chart does not say what it sampled: " << chart->windowTitle().toStdString();

    // the default line runs along x through the centre, and the field IS x, so
    // the sampled values have to span the domain from 0 to 1 -- and so does the
    // arc length along the line
    auto *view = chart->findChild<ChartViewer *>();
    ASSERT_NE(view, nullptr);
    // getMinMax() returns the framing rectangle: y runs downwards and carries a
    // 5% margin, so the data range is recovered from the two corners
    const QRectF box = view->getMinMax();
    const double ylo = qMin(box.top(), box.bottom()), yhi = qMax(box.top(), box.bottom());
    const double ypad = Cfg::CHART_YPAD_FRACTION * (yhi - ylo) / (1 + 2 * Cfg::CHART_YPAD_FRACTION);

    EXPECT_NEAR(box.left(), 0.0, 1e-6) << "the line does not start at the domain edge";
    EXPECT_NEAR(box.right(), 1.0, 1e-6) << "the line does not reach the far edge";
    EXPECT_NEAR(ylo + ypad, 0.0, 1e-6) << "the sampled field does not start at its own minimum";
    EXPECT_NEAR(yhi - ypad, 1.0, 1e-6)
        << "the sampled values are not the field: a ramp from 0 to 1 came back as "
        << (ylo + ypad) << ".." << (yhi - ypad);
    closeCharts();
}

TEST(SceneProbe, TheSampleCountIsTheNumberOfPointsPlotted)
{
    REQUIRE_RENDERER();
    closeCharts();
    VtkScene scene;
    scene.addDataset(rampField(), "ramp", VtkScene::Kind::Generic);
    present(scene);

    Answers answers({"ramp", "7"});
    QMetaObject::invokeMethod(&scene, "applyLineProbe");
    QApplication::processEvents();

    // an independent probe of the same data along the same default line
    auto img = rampField();
    double b[6];
    img->GetBounds(b);
    double p1[3] = {b[0], 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])};
    double p2[3] = {b[1], 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])};
    auto line = VtkFilters::probeLine(img, p1, p2, 7);
    ASSERT_TRUE(line);
    EXPECT_EQ(line->GetNumberOfPoints(), 7);

    ASSERT_NE(newestChart(), nullptr);
    closeCharts();
}

TEST(SceneProbe, WithNoDataItSaysSoRatherThanProbingNothing)
{
    REQUIRE_RENDERER();
    closeCharts();
    VtkScene scene;
    present(scene);

    Answers answers({"ramp", "10"});
    QMetaObject::invokeMethod(&scene, "applyLineProbe");
    QApplication::processEvents();

    EXPECT_TRUE(answers.said("Load a dataset first")) << answers.all().toStdString();
    EXPECT_EQ(newestChart(), nullptr) << "a chart was opened with nothing to plot";
}

TEST(SceneProbe, ADatasetWithNoPointFieldIsRefused)
{
    // the leaky sheet carries its scalar on the cells, not the points, so there
    // is nothing for a point probe to interpolate
    REQUIRE_RENDERER();
    closeCharts();
    VtkScene scene;
    scene.addDataset(leakySheet(), "sheet", VtkScene::Kind::Surface);
    present(scene);

    Answers answers({"leak", "10"});
    QMetaObject::invokeMethod(&scene, "applyLineProbe");
    QApplication::processEvents();

    EXPECT_TRUE(answers.said("no point field")) << answers.all().toStdString();
    EXPECT_EQ(newestChart(), nullptr);
}

TEST(SceneProbe, CancellingTheFieldChoiceProbesNothing)
{
    REQUIRE_RENDERER();
    closeCharts();
    VtkScene scene;
    scene.addDataset(rampField(), "ramp", VtkScene::Kind::Generic);
    present(scene);

    Answers answers({QString()}); // Cancel on the first dialog
    QMetaObject::invokeMethod(&scene, "applyLineProbe");
    QApplication::processEvents();

    EXPECT_EQ(newestChart(), nullptr) << "cancelling still opened a chart";
}

// ----------------------------------------------------------------- cut plane

TEST(SceneProbe, ACutPlaneAddsTheSliceAsItsOwnLayer)
{
    REQUIRE_RENDERER();
    VtkScene scene;
    scene.addDataset(rampField(), "ramp", VtkScene::Kind::Generic);
    present(scene);
    const QString before = statusOf(scene);

    Answers answers({"X", "0.5"});
    QMetaObject::invokeMethod(&scene, "applyCutPlane");
    QApplication::processEvents();

    EXPECT_EQ(answers.dialogs, 2) << "the cut did not ask for an axis and a position";
    EXPECT_TRUE(statusOf(scene).startsWith("2 layer"))
        << "the slice did not become a layer: " << statusOf(scene).toStdString();
    EXPECT_NE(statusOf(scene), before);
}

TEST(SceneProbe, APlaneOutsideTheDataIsReportedRatherThanAddedEmpty)
{
    // QInputDialog clamps the position to the data bounds, so ask for a slice
    // of a dataset the plane cannot intersect at all: a two-triangle sheet
    // lying in z=0 has no thickness to cut across
    REQUIRE_RENDERER();
    VtkScene scene;
    scene.addDataset(leakySheet(), "sheet", VtkScene::Kind::Surface);
    present(scene);

    Answers answers({"Z", "0.0"});
    QMetaObject::invokeMethod(&scene, "applyCutPlane");
    QApplication::processEvents();

    // either it intersected (a degenerate slice) or it said it could not; what
    // it must never do is add a layer with nothing in it
    if (!answers.said("did not intersect"))
        EXPECT_TRUE(statusOf(scene).startsWith("2 layer"))
            << "a layer appeared that neither intersected nor was reported: "
            << statusOf(scene).toStdString();
}

TEST(SceneProbe, CancellingTheAxisLeavesTheSceneAlone)
{
    REQUIRE_RENDERER();
    VtkScene scene;
    scene.addDataset(rampField(), "ramp", VtkScene::Kind::Generic);
    present(scene);
    const QString before = statusOf(scene);

    Answers answers({QString()});
    QMetaObject::invokeMethod(&scene, "applyCutPlane");
    QApplication::processEvents();
    EXPECT_EQ(statusOf(scene), before) << "cancelling still changed the scene";
}

// --------------------------------------------------------------- point probe

TEST(SceneProbe, ClickingWithThePointProbeOnReadsTheFieldThere)
{
    // the picker turns a click into a world position and the probe reads the
    // field there; a wrong cell lookup reports a plausible number with nothing
    // to compare it against, so this compares against the field's own definition
    REQUIRE_RENDERER();
    VtkScene scene;
    scene.addDataset(rampField(), "ramp", VtkScene::Kind::Generic);
    present(scene, 400, 400);
    scene.resetView();
    QApplication::processEvents();

    QMetaObject::invokeMethod(&scene, "togglePointProbe", Q_ARG(bool, true));
    EXPECT_TRUE(statusOf(scene).contains("Point probe"))
        << "turning the probe on said nothing: " << statusOf(scene).toStdString();

    auto *area = areaOf(scene);
    ASSERT_NE(area, nullptr);
    const QPoint centre(area->width() / 2, area->height() / 2);
    QTest::mousePress(area, Qt::LeftButton, Qt::NoModifier, centre);
    QTest::mouseRelease(area, Qt::LeftButton, Qt::NoModifier, centre);
    QApplication::processEvents();

    const QString msg = statusOf(scene);
    if (msg.contains("no geometry under the cursor"))
        GTEST_SKIP() << "the camera is not looking at the data in this environment";

    ASSERT_TRUE(msg.contains("ramp=")) << "the probe reported no field: " << msg.toStdString();

    // the reading has to be the x coordinate it says it sampled at
    const QRegularExpression at(R"(\(([-\d.eE+]+),)");
    const QRegularExpression val(R"(ramp=([-\d.eE+]+))");
    const auto ma = at.match(msg), mv = val.match(msg);
    ASSERT_TRUE(ma.hasMatch()) << msg.toStdString();
    ASSERT_TRUE(mv.hasMatch()) << msg.toStdString();
    EXPECT_NEAR(mv.captured(1).toDouble(), ma.captured(1).toDouble(), 1e-3)
        << "the probe read " << mv.captured(1).toStdString() << " at x="
        << ma.captured(1).toStdString() << ", but the field is x";
}

TEST(SceneProbe, TurningThePointProbeOffStopsReadingOnClick)
{
    REQUIRE_RENDERER();
    VtkScene scene;
    scene.addDataset(rampField(), "ramp", VtkScene::Kind::Generic);
    present(scene, 400, 400);
    scene.resetView();

    QMetaObject::invokeMethod(&scene, "togglePointProbe", Q_ARG(bool, true));
    QMetaObject::invokeMethod(&scene, "togglePointProbe", Q_ARG(bool, false));
    const QString resting = statusOf(scene);

    auto *area = areaOf(scene);
    ASSERT_NE(area, nullptr);
    const QPoint centre(area->width() / 2, area->height() / 2);
    QTest::mousePress(area, Qt::LeftButton, Qt::NoModifier, centre);
    QTest::mouseRelease(area, Qt::LeftButton, Qt::NoModifier, centre);
    QApplication::processEvents();

    EXPECT_FALSE(statusOf(scene).contains("ramp="))
        << "a click still probed after the tool was turned off: "
        << statusOf(scene).toStdString();
    EXPECT_EQ(statusOf(scene), resting);
}

int main(int argc, char **argv)
{
    // These construct real widgets, so a QApplication has to exist first. The
    // offscreen platform keeps the suite runnable on a machine with no display,
    // which matters because the VTK build is the one least likely to be running
    // somewhere with a screen attached.
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
