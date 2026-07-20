// Unit tests for the pure ParaView-export command builders
// (src/paraviewexport.cpp).
//
// These verify the argument vectors passed to pvpython for surf2paraview.py
// and grid2paraview.py without launching ParaView: the option ordering,
// default-omission, glob-expanded result files, Exodus/chunk flags, the
// expected output path, and settings validation (checked against the scripts'
// argparse in sparta/tools/paraview/{surf,grid}2paraview.py).

#include "paraviewexport.h"

#include <QDir>
#include <QTemporaryFile>

#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace ParaviewExport;

namespace {

std::vector<std::string> toVec(const QStringList &list)
{
    std::vector<std::string> v;
    for (const auto &s : list) v.push_back(s.toStdString());
    return v;
}

} // namespace

TEST(ParaviewExport, ScriptNames)
{
    EXPECT_EQ(scriptName(Mode::Surface).toStdString(), "surf2paraview.py");
    EXPECT_EQ(scriptName(Mode::Grid).toStdString(), "grid2paraview.py");
}

TEST(ParaviewExport, SurfaceMinimalCommand)
{
    Settings s;
    s.mode = Mode::Surface;
    s.inputFile = "data.mir";
    s.outputName = "mir_surf";
    const auto c = toVec(buildScriptArgs(s, "/opt/sparta/tools/paraview/surf2paraview.py"));
    ASSERT_EQ(c.size(), 3u);
    EXPECT_EQ(c[0], "/opt/sparta/tools/paraview/surf2paraview.py");
    EXPECT_EQ(c[1], "data.mir");
    EXPECT_EQ(c[2], "mir_surf");
}

TEST(ParaviewExport, SurfaceWithResultsAndExodus)
{
    Settings s;
    s.mode = Mode::Surface;
    s.inputFile = "data.mir";
    s.outputName = "mir_surf";
    s.resultFiles = {"tmp_surf.1000", "tmp_surf.2000"};
    s.exodus = true;
    const auto c = toVec(buildScriptArgs(s, "surf2paraview.py"));
    // scriptPath, input, output, -r f1 f2, -e
    ASSERT_EQ(c.size(), 7u);
    EXPECT_EQ(c[3], "-r");
    EXPECT_EQ(c[4], "tmp_surf.1000");
    EXPECT_EQ(c[5], "tmp_surf.2000");
    EXPECT_EQ(c[6], "-e");
    // Exodus changes the expected output extension
    EXPECT_EQ(expectedOutput(s).toStdString(), "mir_surf.ex2");
}

TEST(ParaviewExport, GridDefaultChunksOmitted)
{
    Settings s;
    s.mode = Mode::Grid;
    s.inputFile = "mir.txt";
    s.outputName = "mir_grid";
    // all chunks at the script default of 100 -> no -x/-y/-z emitted
    const auto c = toVec(buildScriptArgs(s, "grid2paraview.py"));
    ASSERT_EQ(c.size(), 3u);
    EXPECT_EQ(c[1], "mir.txt");
    EXPECT_EQ(c[2], "mir_grid");
}

TEST(ParaviewExport, GridNonDefaultChunksEmittedInOrder)
{
    Settings s;
    s.mode = Mode::Grid;
    s.inputFile = "mir.txt";
    s.outputName = "mir_grid";
    s.xchunk = 50;
    s.ychunk = 100; // default -> omitted
    s.zchunk = 25;
    const auto c = toVec(buildScriptArgs(s, "grid2paraview.py"));
    // scriptPath, input, output, -x 50, -z 25   (y omitted)
    ASSERT_EQ(c.size(), 7u);
    EXPECT_EQ(c[3], "-x");
    EXPECT_EQ(c[4], "50");
    EXPECT_EQ(c[5], "-z");
    EXPECT_EQ(c[6], "25");
}

TEST(ParaviewExport, GridDoesNotEmitExodus)
{
    Settings s;
    s.mode = Mode::Grid;
    s.inputFile = "mir.txt";
    s.outputName = "mir_grid";
    s.exodus = true; // meaningless for grid; must be ignored
    const auto c = toVec(buildScriptArgs(s, "grid2paraview.py"));
    for (const auto &a : c) EXPECT_NE(a, "-e");
    EXPECT_EQ(expectedOutput(s).toStdString(), "mir_grid.pvd");
}

TEST(ParaviewExport, ExpectedOutputPvd)
{
    Settings s;
    s.mode = Mode::Surface;
    s.outputName = "out";
    EXPECT_EQ(expectedOutput(s).toStdString(), "out.pvd");
    s.mode = Mode::Grid;
    EXPECT_EQ(expectedOutput(s).toStdString(), "out.pvd");
}

TEST(ParaviewExport, ValidateRejectsEmptyAndMissing)
{
    QString err;
    Settings s;
    // no input
    EXPECT_FALSE(validate(s, err));
    EXPECT_FALSE(err.isEmpty());

    // input that does not exist
    s.inputFile = "/no/such/file.surf";
    s.outputName = "out";
    EXPECT_FALSE(validate(s, err));

    // a real file but empty output name
    QTemporaryFile tf;
    ASSERT_TRUE(tf.open());
    s.inputFile = tf.fileName();
    s.outputName = "";
    EXPECT_FALSE(validate(s, err));

    // valid
    s.outputName = "out";
    EXPECT_TRUE(validate(s, err));
    EXPECT_TRUE(err.isEmpty());
}

TEST(ParaviewExport, ValidateRejectsNonPositiveChunks)
{
    QTemporaryFile tf;
    ASSERT_TRUE(tf.open());
    QString err;
    Settings s;
    s.mode = Mode::Grid;
    s.inputFile = tf.fileName();
    s.outputName = "out";
    s.xchunk = 0;
    EXPECT_FALSE(validate(s, err));
    s.xchunk = 100;
    s.zchunk = -5;
    EXPECT_FALSE(validate(s, err));
    s.zchunk = 10;
    EXPECT_TRUE(validate(s, err));
}
