// Unit tests for the pure surface-quantity reduction core (src/surfreport.cpp).

#include "surfreport.h"

#include "gtest/gtest.h"

using namespace SurfReport;

namespace {

TEST(SurfReport, IntegratesForceMomentHeatFlux)
{
    const QStringList labels = {"fx", "fy", "fz", "tx", "ty", "tz", "etot", "press"};
    // two elements
    QVector<QVector<double>> rows = {
        {1.0, 2.0, 3.0, 0.5, 0.0, -0.5, 10.0, 100.0},
        {4.0, 5.0, 6.0, 0.5, 0.0, 0.5, 20.0, 200.0},
    };
    const Totals t = integrate(labels, rows);
    EXPECT_EQ(t.nsurf, 2);
    ASSERT_TRUE(t.hasForce);
    EXPECT_DOUBLE_EQ(t.force[0], 5.0);
    EXPECT_DOUBLE_EQ(t.force[1], 7.0);
    EXPECT_DOUBLE_EQ(t.force[2], 9.0);
    ASSERT_TRUE(t.hasMoment);
    EXPECT_DOUBLE_EQ(t.moment[0], 1.0);
    EXPECT_DOUBLE_EQ(t.moment[2], 0.0);
    ASSERT_TRUE(t.hasHeatFlux);
    EXPECT_DOUBLE_EQ(t.heatFlux, 30.0);
    // column sums
    EXPECT_DOUBLE_EQ(t.columnSum[labels.indexOf("press")], 300.0);
}

TEST(SurfReport, HeatFluxFromComponents)
{
    const QStringList labels = {"ke", "erot", "evib"};
    QVector<QVector<double>> rows = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const Totals t = integrate(labels, rows);
    ASSERT_TRUE(t.hasHeatFlux);
    EXPECT_DOUBLE_EQ(t.heatFlux, (1 + 4) + (2 + 5) + (3 + 6)); // ke+erot+evib summed
    EXPECT_FALSE(t.hasForce);
}

TEST(SurfReport, Distribution)
{
    const Distribution d = distribution({2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0});
    EXPECT_EQ(d.n, 8);
    EXPECT_DOUBLE_EQ(d.min, 2.0);
    EXPECT_DOUBLE_EQ(d.max, 9.0);
    EXPECT_DOUBLE_EQ(d.mean, 5.0);
    EXPECT_NEAR(d.stddev, 2.138090, 1e-5); // sample stddev
    EXPECT_EQ(distribution({}).n, 0);
}

TEST(SurfReport, ColumnAndCsv)
{
    const QStringList labels = {"a", "b"};
    QVector<QVector<double>> rows = {{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_EQ(column(rows, 1), (QVector<double>{2.0, 4.0}));
    const QString csv = toCsv(labels, rows);
    EXPECT_TRUE(csv.startsWith("element,a,b\n"));
    EXPECT_TRUE(csv.contains("0,1,2\n"));
    EXPECT_TRUE(csv.contains("1,3,4\n"));
}

TEST(SurfReport, ExpandColumnLabels)
{
    EXPECT_EQ(expandColumnLabels({"fx", "fy"}, 1), (QStringList{"fx", "fy"}));
    EXPECT_EQ(expandColumnLabels({"fx", "fy"}, 2),
              (QStringList{"fx_g1", "fx_g2", "fy_g1", "fy_g2"}));
}

} // namespace
