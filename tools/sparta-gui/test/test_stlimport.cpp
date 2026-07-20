// Unit tests for the pure STL / SPARTA-surface import core (src/stlimport.cpp).
//
// These exercise the converter and command builders without a GUI or a live
// SPARTA instance: STL/surf files are parsed into a SurfMesh, the watertight
// pre-check and command strings are validated against SPARTA syntax (verified
// against sparta/src/read_surf.cpp, surf.cpp, create_isurf.cpp and
// examples/explicit2implicit/in.exp2imp.sphere.3d).

#include "stlimport.h"

#include <QByteArray>
#include <QDataStream>
#include <QDir>
#include <QTemporaryFile>

#include "gtest/gtest.h"

#include <string>

using namespace StlImport;

namespace {

// locate the repo's data/ directory relative to this source tree so the
// apollo/orion fixtures resolve regardless of the build directory.
QString dataDir()
{
    // tools/sparta-gui/test/test_stlimport.cpp -> ../../../data
    QDir d(QStringLiteral(__FILE__));
    d.cdUp(); // test/
    d.cdUp(); // sparta-gui/
    d.cdUp(); // tools/
    d.cdUp(); // repo root
    return d.absoluteFilePath("data");
}

// write a minimal binary STL (one triangle) into a temp file, return its path
QString writeBinaryStlTri(QTemporaryFile &tf)
{
    tf.open();
    QDataStream ds(&tf);
    ds.setByteOrder(QDataStream::LittleEndian);
    ds.setFloatingPointPrecision(QDataStream::SinglePrecision);
    char header[80] = {0};
    tf.write(header, 80);
    ds << static_cast<quint32>(1);           // one triangle
    ds << 0.0f << 0.0f << 1.0f;              // normal (ignored)
    ds << 0.0f << 0.0f << 0.0f;              // v0
    ds << 1.0f << 0.0f << 0.0f;              // v1
    ds << 0.0f << 1.0f << 0.0f;              // v2
    ds << static_cast<quint16>(0);           // attribute
    tf.flush();
    return tf.fileName();
}

} // namespace

TEST(StlImport, ParseAsciiApolloMatchesSurfFixture)
{
    const QString stl = dataDir() + "/apollo.stl";
    const QString surf = dataDir() + "/sdata.apollo";

    SurfMesh mstl, msurf;
    QString err;
    ASSERT_TRUE(parseStl(stl, mstl, err)) << err.toStdString();
    ASSERT_TRUE(parseSurf(surf, msurf, err)) << err.toStdString();

    // stl2surf.py produced sdata.apollo from apollo.stl, so counts must match
    EXPECT_EQ(mstl.npoints(), 16721);
    EXPECT_EQ(mstl.nelements(), 33438);
    EXPECT_EQ(mstl.npoints(), msurf.npoints());
    EXPECT_EQ(mstl.nelements(), msurf.nelements());
    EXPECT_FALSE(mstl.is2d);
    EXPECT_FALSE(msurf.is2d);
    // extents populated
    EXPECT_LT(mstl.lo[0], mstl.hi[0]);
}

TEST(StlImport, BinaryAndAsciiAgree)
{
    // binary one-triangle STL
    QTemporaryFile bin("XXXXXX.stl");
    const QString binpath = writeBinaryStlTri(bin);
    SurfMesh mb;
    QString err;
    ASSERT_TRUE(parseStl(binpath, mb, err)) << err.toStdString();
    EXPECT_EQ(mb.nelements(), 1);
    EXPECT_EQ(mb.npoints(), 3);

    // equivalent ASCII STL
    QTemporaryFile asc("XXXXXX.stl");
    asc.open();
    asc.write("solid t\n"
              "facet normal 0 0 1\n outer loop\n"
              "  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n"
              " endloop\nendfacet\nendsolid t\n");
    asc.flush();
    SurfMesh ma;
    ASSERT_TRUE(parseStl(asc.fileName(), ma, err)) << err.toStdString();
    EXPECT_EQ(ma.nelements(), mb.nelements());
    EXPECT_EQ(ma.npoints(), mb.npoints());
}

TEST(StlImport, VertexDedupMergesSharedCorners)
{
    // two triangles sharing an edge -> 4 unique points, not 6
    QTemporaryFile asc("XXXXXX.stl");
    asc.open();
    asc.write("solid q\n"
              "facet normal 0 0 1\n outer loop\n"
              "  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n endloop\nendfacet\n"
              "facet normal 0 0 1\n outer loop\n"
              "  vertex 1 0 0\n  vertex 1 1 0\n  vertex 0 1 0\n endloop\nendfacet\n"
              "endsolid q\n");
    asc.flush();
    SurfMesh m;
    QString err;
    ASSERT_TRUE(parseStl(asc.fileName(), m, err)) << err.toStdString();
    EXPECT_EQ(m.nelements(), 2);
    EXPECT_EQ(m.npoints(), 4);
}

TEST(StlImport, WatertightDetectsOpenMeshAndLeakingTriangles)
{
    // single triangle -> three boundary (unmatched) edges, all on element 0
    SurfMesh m;
    m.points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    m.elems = {{0, 1, 2}};
    const WatertightReport r = checkWatertightPreflight(m);
    EXPECT_FALSE(r.watertight());
    EXPECT_EQ(r.unmatchedEdges, 3);
    EXPECT_TRUE(r.leakingElems.contains(0));
}

TEST(StlImport, WatertightPassesOnClosedTetrahedron)
{
    // a closed tetrahedron with consistent outward winding is watertight
    SurfMesh m;
    m.points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    // faces wound so every edge appears once in each direction
    m.elems = {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
    const WatertightReport r = checkWatertightPreflight(m);
    EXPECT_TRUE(r.watertight()) << "dup=" << r.duplicateEdges << " unmatched=" << r.unmatchedEdges;
    EXPECT_TRUE(r.leakingElems.isEmpty());
}

TEST(StlImport, BuildSurfFileLayoutAndTypeColumn)
{
    SurfMesh m;
    m.points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    m.elems = {{0, 1, 2}};

    const std::string plain = buildSurfFile(m, "test").toStdString();
    EXPECT_NE(plain.find("3 points"), std::string::npos);
    EXPECT_NE(plain.find("1 triangles"), std::string::npos);
    EXPECT_NE(plain.find("\nTriangles\n"), std::string::npos);
    EXPECT_NE(plain.find("1 1 2 3"), std::string::npos); // id v1 v2 v3

    QSet<int> bad;
    bad.insert(0);
    const std::string tagged = buildSurfFile(m, "test", bad).toStdString();
    EXPECT_NE(tagged.find("1 2 1 2 3"), std::string::npos); // id type(=2) v1 v2 v3
}

TEST(StlImport, ReadSurfCommandOmitsDefaultsAndOrders)
{
    StlImportSettings s;
    EXPECT_EQ(buildReadSurfCommand(s, "a.surf").toStdString(), "read_surf a.surf");

    s.useScale = true;
    s.scale[0] = s.scale[1] = s.scale[2] = 0.001;
    s.transKind = StlImportSettings::TransKind::Trans;
    s.trans[0] = 1;
    s.trans[1] = 2;
    s.trans[2] = 3;
    s.invert = true;
    // order must be: trans, scale, invert (transKind before scale before invert)
    EXPECT_EQ(buildReadSurfCommand(s, "a.surf").toStdString(),
              "read_surf a.surf trans 1 2 3 scale 0.001 0.001 0.001 invert");
}

TEST(StlImport, AblationCommandOrderAndSyntax)
{
    StlImportSettings s;
    s.mode = StlImportSettings::Mode::Implicit;
    s.isurfGroup = "all";
    s.ablateId = "fablate";
    s.thresh = 39.5;
    s.isurfMode = "voxel";
    s.nevery = 0;
    s.ablateScale = 0.2;
    s.ablateSource = "random";
    s.maxrandom = 0;

    const QStringList c = buildAblationCommands(s, "apollo.surf");
    ASSERT_EQ(c.size(), 4);
    EXPECT_EQ(c[0].toStdString(), "global surfs explicit");
    EXPECT_EQ(c[1].toStdString(), "read_surf apollo.surf");
    EXPECT_EQ(c[2].toStdString(), "fix fablate ablate all 0 0.2 random 0");
    EXPECT_EQ(c[3].toStdString(), "create_isurf all fablate 39.5 voxel");
}

TEST(StlImport, ThresholdValidation)
{
    EXPECT_TRUE(validThreshold(39.5));
    EXPECT_TRUE(validThreshold(150.5));
    EXPECT_FALSE(validThreshold(128.0)); // integer disallowed
    EXPECT_FALSE(validThreshold(0.0));
    EXPECT_FALSE(validThreshold(255.0));
    EXPECT_FALSE(validThreshold(300.0));
}
