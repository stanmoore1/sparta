// -*- c++ -*- /////////////////////////////////////////////////////////////////////////
// SPARTA-GUI - unit tests for the pure case-model core (Feature 8, canvas)
////////////////////////////////////////////////////////////////////////////////////////

#include "casemodel.h"

#include <gtest/gtest.h>

using namespace CaseModel;

namespace {

const char *kDeck =
    "# a small 2d case\n"
    "dimension 2\n"
    "boundary o p p\n"
    "create_box -1 1 -2 2 -0.5 0.5\n"
    "create_grid 20 20 1\n"
    "region hole cylinder z 0 0 0.3 INF INF\n"
    "read_surf data.circle group circle\n"
    "mixture air N2 O2 nrho 1.0e20 temp 300.0 vstream 100.0 0.0 0.0\n"
    "fix in emit/face air xlo xhi\n"
    "run 100\n";

TEST(CaseModel, ParsesBoxAndDimension)
{
    Model m = parse(kDeck);
    EXPECT_EQ(m.dimension, 2);
    ASSERT_TRUE(m.box.present);
    EXPECT_EQ(m.box.dimension, 2);
    EXPECT_DOUBLE_EQ(m.box.lo[0], -1.0);
    EXPECT_DOUBLE_EQ(m.box.hi[0], 1.0);
    EXPECT_DOUBLE_EQ(m.box.lo[1], -2.0);
    EXPECT_DOUBLE_EQ(m.box.hi[1], 2.0);
    EXPECT_DOUBLE_EQ(m.box.hi[2], 0.5);
}

TEST(CaseModel, ParsesBoundaryRegionSurfMixtureEmit)
{
    Model m = parse(kDeck);

    ASSERT_TRUE(m.boundary.present);
    EXPECT_EQ(m.boundary.spec[0], "o");
    EXPECT_EQ(m.boundary.spec[1], "p");
    EXPECT_EQ(m.boundary.spec[2], "p");

    ASSERT_EQ(m.regions.size(), 1);
    EXPECT_EQ(m.regions[0].id, "hole");
    EXPECT_EQ(m.regions[0].style, "cylinder");

    ASSERT_EQ(m.surfaces.size(), 1);
    EXPECT_EQ(m.surfaces[0].file, "data.circle");

    ASSERT_EQ(m.mixtures.size(), 1);
    EXPECT_EQ(m.mixtures[0].id, "air");
    EXPECT_EQ(m.mixtures[0].species, (QStringList{"N2", "O2"}));
    EXPECT_EQ(m.mixtures[0].nrho, "1.0e20");
    EXPECT_EQ(m.mixtures[0].temp, "300.0");
    EXPECT_EQ(m.mixtures[0].vstream, (QStringList{"100.0", "0.0", "0.0"}));

    ASSERT_EQ(m.emits.size(), 1);
    EXPECT_EQ(m.emits[0].id, "in");
    EXPECT_EQ(m.emits[0].mixture, "air");
    EXPECT_EQ(m.emits[0].faces, (QStringList{"xlo", "xhi"}));
}

TEST(CaseModel, PreservesEveryLineInOrder)
{
    Model m = parse(kDeck);
    // one Line per physical line (the deck ends with a newline -> trailing empty)
    EXPECT_EQ(m.lines.size(), QString(kDeck).split('\n').size());
    // the unrecognized command is still preserved verbatim
    bool sawGrid = false;
    for (const auto &ln : m.lines)
        if (ln.command == "create_grid") sawGrid = true;
    EXPECT_TRUE(sawGrid);
}

TEST(CaseModel, TokenizeStripsComments)
{
    EXPECT_EQ(tokenize("create_box 0 1 0 1 0 1  # the box"),
              (QStringList{"create_box", "0", "1", "0", "1", "0", "1"}));
    EXPECT_TRUE(tokenize("   # pure comment").isEmpty());
    EXPECT_TRUE(tokenize("").isEmpty());
}

TEST(CaseModel, SetBoundaryReplacesInPlaceMinimalDiff)
{
    QString out = setBoundary(kDeck, "r", "p", "p");
    // exactly one line changed
    QStringList a = QString(kDeck).split('\n');
    QStringList b = out.split('\n');
    ASSERT_EQ(a.size(), b.size());
    int diffs = 0, diffIdx = -1;
    for (int i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) { ++diffs; diffIdx = i; }
    EXPECT_EQ(diffs, 1);
    EXPECT_EQ(b[diffIdx], "boundary r p p");
    // and the model reflects it
    EXPECT_EQ(parse(out).boundary.spec[0], "r");
}

TEST(CaseModel, SetBoundaryInsertsBeforeBoxWhenAbsent)
{
    const char *deck = "create_box 0 1 0 1 0 1\nrun 0\n";
    QString out = setBoundary(deck, "p", "p", "p");
    Model m = parse(out);
    ASSERT_TRUE(m.boundary.present);
    ASSERT_TRUE(m.box.present);
    EXPECT_LT(m.boundary.sourceLine, m.box.sourceLine);  // boundary precedes the box
}

TEST(CaseModel, SetBoxExtentsRewritesCreateBox)
{
    const double lo[3] = {-5, -5, -0.5};
    const double hi[3] = {5, 5, 0.5};
    QString out = setBoxExtents(kDeck, lo, hi);
    Model m = parse(out);
    EXPECT_DOUBLE_EQ(m.box.lo[0], -5.0);
    EXPECT_DOUBLE_EQ(m.box.hi[0], 5.0);
    // no create_box -> unchanged
    EXPECT_EQ(setBoxExtents("run 0\n", lo, hi), QString("run 0\n"));
}

TEST(CaseModel, InsertEmitFaceAfterBoundary)
{
    QString out = insertEmitFace(kDeck, "in2", "air", {"ylo"});
    Model m = parse(out);
    bool found = false;
    for (const auto &e : m.emits)
        if (e.id == "in2" && e.mixture == "air" && e.faces == QStringList{"ylo"})
            found = true;
    EXPECT_TRUE(found);
    // inserted right after the boundary line
    EXPECT_GT(m.emits.size(), 1);
}

} // namespace
