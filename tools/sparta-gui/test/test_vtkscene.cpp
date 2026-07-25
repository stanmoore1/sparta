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
#include <QImage>

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

#include "gtest/gtest.h"

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

} // namespace

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
