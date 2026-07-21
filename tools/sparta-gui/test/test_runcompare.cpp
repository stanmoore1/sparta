// Unit tests for the pure run-comparison core (src/runcompare.cpp).

#include "runcompare.h"

#include "gtest/gtest.h"

using namespace RunCompare;

namespace {

int count(const QVector<DiffLine> &d, Op op)
{
    int n = 0;
    for (const auto &l : d)
        if (l.op == op) ++n;
    return n;
}

TEST(RunCompare, IdenticalHasNoChanges)
{
    const QStringList a = {"create_box 0 1 0 1 0 1", "run 100"};
    const auto d = diffLines(a, a);
    EXPECT_EQ(count(d, Op::Added), 0);
    EXPECT_EQ(count(d, Op::Removed), 0);
    EXPECT_EQ(count(d, Op::Context), 2);
    EXPECT_FALSE(decksDiffer(a.join('\n'), a.join('\n')));
}

TEST(RunCompare, OneLineChanged)
{
    const QStringList a = {"seed 12345", "run 100"};
    const QStringList b = {"seed 67890", "run 100"};
    const auto d = diffLines(a, b);
    EXPECT_EQ(count(d, Op::Removed), 1); // seed 12345
    EXPECT_EQ(count(d, Op::Added), 1);   // seed 67890
    EXPECT_EQ(count(d, Op::Context), 1); // run 100
    EXPECT_TRUE(decksDiffer(a.join('\n'), b.join('\n')));
}

TEST(RunCompare, InsertionAndDeletion)
{
    const QStringList a = {"a", "b", "c"};
    const QStringList b = {"a", "c", "d"}; // remove b, add d
    const auto d = diffLines(a, b);
    EXPECT_EQ(count(d, Op::Removed), 1); // b
    EXPECT_EQ(count(d, Op::Added), 1);   // d
    EXPECT_EQ(count(d, Op::Context), 2); // a, c
}

TEST(RunCompare, MetadataDelta)
{
    QMap<QString, QString> a{{"SPARTA version", "20250924"}, {"seed", "1"}, {"host", "n1"}};
    QMap<QString, QString> b{{"SPARTA version", "20250924"}, {"seed", "2"}, {"OS", "linux"}};
    const auto delta = diffMetadata(a, b);

    // union of keys, sorted
    EXPECT_EQ(delta.size(), 4); // OS, SPARTA version, host, seed
    int differing = 0;
    for (const auto &d : delta)
        if (d.differs()) ++differing;
    EXPECT_EQ(differing, 3); // seed (1 vs 2), host (n1 vs ""), OS ("" vs linux)

    for (const auto &d : delta) {
        if (d.key == "SPARTA version") EXPECT_FALSE(d.differs());
        if (d.key == "seed") { EXPECT_EQ(d.valueA, "1"); EXPECT_EQ(d.valueB, "2"); }
        if (d.key == "host") { EXPECT_EQ(d.valueA, "n1"); EXPECT_EQ(d.valueB, ""); }
    }
}

TEST(RunCompare, BuildHtmlContainsDiffAndMeta)
{
    RunArchive::RunRecord a, b;
    a.id = "runA"; b.id = "runB";
    a.deckText = "seed 1\nrun 100\n";
    b.deckText = "seed 2\nrun 100\n";
    a.metadata.insert("seed", "1");
    b.metadata.insert("seed", "2");
    const QString html = buildComparisonHtml(a, b);
    EXPECT_TRUE(html.contains("Run comparison"));
    EXPECT_TRUE(html.contains("Input deck diff"));
    EXPECT_TRUE(html.contains("seed 1"));   // removed line
    EXPECT_TRUE(html.contains("seed 2"));   // added line
    EXPECT_TRUE(html.contains("diffkey"));  // metadata highlight for the differing seed
}

} // namespace
