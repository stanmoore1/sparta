// Unit tests for the pure parametric-sweep model (src/sweepspec.cpp).
//
// No GUI, no SPARTA: variable expansion (list/range/linspace), cartesian and
// zip combination order/count, reducers, and error handling.

#include "sweepspec.h"

#include "gtest/gtest.h"

#include <cmath>
#include <string>
#include <vector>

using namespace Sweep;

namespace {
std::vector<std::string> vals(const QStringList &l)
{
    std::vector<std::string> v;
    for (const auto &s : l) v.push_back(s.toStdString());
    return v;
}
VarSweep listVar(const QString &n, const QStringList &vs)
{
    VarSweep v; v.name = n; v.kind = VarSweep::List; v.values = vs; return v;
}
} // namespace

TEST(SweepSpec, ExpandList)
{
    EXPECT_EQ(vals(listVar("a", {"1", "2", "3"}).expand()),
              (std::vector<std::string>{"1", "2", "3"}));
}

TEST(SweepSpec, ExpandRangeInclusiveEpsilon)
{
    VarSweep v; v.name = "a"; v.kind = VarSweep::Range;
    v.start = 1; v.stop = 5; v.step = 2;
    EXPECT_EQ(vals(v.expand()), (std::vector<std::string>{"1", "3", "5"}));

    // fractional step that would miss the endpoint without an epsilon
    VarSweep w; w.name = "b"; w.kind = VarSweep::Range;
    w.start = 0; w.stop = 1; w.step = 0.1;
    EXPECT_EQ(w.expand().size(), 11); // 0.0 .. 1.0 inclusive
}

TEST(SweepSpec, ExpandRangeDescending)
{
    VarSweep v; v.name = "a"; v.kind = VarSweep::Range;
    v.start = 3; v.stop = 1; v.step = -1;
    EXPECT_EQ(vals(v.expand()), (std::vector<std::string>{"3", "2", "1"}));
}

TEST(SweepSpec, ExpandLinspaceEndpoints)
{
    VarSweep v; v.name = "a"; v.kind = VarSweep::Linspace;
    v.start = 0; v.stop = 1; v.count = 5;
    const auto e = v.expand();
    ASSERT_EQ(e.size(), 5);
    EXPECT_EQ(e.first().toStdString(), "0");
    EXPECT_EQ(e.last().toStdString(), "1");

    VarSweep one; one.name = "b"; one.kind = VarSweep::Linspace;
    one.start = 7; one.stop = 99; one.count = 1;
    EXPECT_EQ(vals(one.expand()), (std::vector<std::string>{"7"}));
}

TEST(SweepSpec, CartesianOrderAndCount)
{
    SweepSpec s;
    s.vars << listVar("x", {"1", "2"}) << listVar("y", {"a", "b", "c"});
    s.combine = Combine::Cartesian;
    EXPECT_EQ(s.runCount(), 6);

    QString err;
    const auto combos = s.expand(&err);
    ASSERT_TRUE(err.isEmpty());
    ASSERT_EQ(combos.size(), 6);
    // last variable varies fastest: (1,a)(1,b)(1,c)(2,a)(2,b)(2,c)
    EXPECT_EQ(combos[0][0].second.toStdString(), "1");
    EXPECT_EQ(combos[0][1].second.toStdString(), "a");
    EXPECT_EQ(combos[1][1].second.toStdString(), "b");
    EXPECT_EQ(combos[3][0].second.toStdString(), "2");
    EXPECT_EQ(combos[3][1].second.toStdString(), "a");
    EXPECT_EQ(combos[5][1].second.toStdString(), "c");
}

TEST(SweepSpec, ZipEqualLength)
{
    SweepSpec s;
    s.vars << listVar("x", {"1", "2", "3"}) << listVar("y", {"10", "20", "30"});
    s.combine = Combine::Zip;
    EXPECT_EQ(s.runCount(), 3);
    QString err;
    const auto combos = s.expand(&err);
    ASSERT_TRUE(err.isEmpty());
    ASSERT_EQ(combos.size(), 3);
    EXPECT_EQ(combos[1][0].second.toStdString(), "2");
    EXPECT_EQ(combos[1][1].second.toStdString(), "20");
}

TEST(SweepSpec, ZipUnequalLengthErrors)
{
    SweepSpec s;
    s.vars << listVar("x", {"1", "2"}) << listVar("y", {"10", "20", "30"});
    s.combine = Combine::Zip;
    EXPECT_EQ(s.runCount(), 0);
    QString err;
    const auto combos = s.expand(&err);
    EXPECT_TRUE(combos.isEmpty());
    EXPECT_FALSE(err.isEmpty());
}

TEST(SweepSpec, EmptyAndUnnamedRejected)
{
    QString err;
    SweepSpec empty;
    EXPECT_TRUE(empty.expand(&err).isEmpty());
    EXPECT_FALSE(err.isEmpty());

    SweepSpec noname;
    noname.vars << listVar("", {"1"});
    EXPECT_TRUE(noname.expand(&err).isEmpty());
    EXPECT_FALSE(err.isEmpty());
}

TEST(SweepSpec, Reducers)
{
    const std::vector<double> s{3.0, 1.0, 4.0, 1.0, 5.0};
    EXPECT_DOUBLE_EQ(reduce(Reducer::Final, s), 5.0);
    EXPECT_DOUBLE_EQ(reduce(Reducer::Min, s), 1.0);
    EXPECT_DOUBLE_EQ(reduce(Reducer::Max, s), 5.0);
    EXPECT_DOUBLE_EQ(reduce(Reducer::Mean, s), 14.0 / 5.0);

    // Nothing sampled is not a reading of zero: every reducer has to say so,
    // or a run that produced no data lands in the table looking measured
    for (auto r : {Reducer::Final, Reducer::Min, Reducer::Max, Reducer::Mean})
        EXPECT_TRUE(std::isnan(reduce(r, {}))) << reducerName(r).toStdString();
}

TEST(EnsembleStats, KnownValues)
{
    // {2,4,4,4,5,5,7,9}: mean 5, sample sd sqrt(32/7)=2.13809, sem sd/sqrt(8)
    const EnsembleStats s = ensembleStats({2, 4, 4, 4, 5, 5, 7, 9}, 0.95);
    EXPECT_EQ(s.n, 8);
    EXPECT_NEAR(s.mean, 5.0, 1e-12);
    EXPECT_NEAR(s.stddev, 2.138090, 1e-5);
    EXPECT_NEAR(s.stderror, 0.755929, 1e-5);
    // t_{0.975,7} = 2.3646; approximation within 1%
    EXPECT_NEAR(s.ciHalf / s.stderror, 2.3646, 0.02);
}

TEST(EnsembleStats, SingleAndEmpty)
{
    const EnsembleStats one = ensembleStats({42.0});
    EXPECT_EQ(one.n, 1);
    EXPECT_NEAR(one.mean, 42.0, 1e-12);
    EXPECT_DOUBLE_EQ(one.stderror, 0.0);
    EXPECT_DOUBLE_EQ(one.ciHalf, 0.0);

    EXPECT_EQ(ensembleStats({}).n, 0);
}

TEST(EnsembleStats, LargeNApproachesNormal)
{
    // for large n the t critical value approaches z_{0.975} = 1.95996
    std::vector<double> v(1000);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(double(i));
    const EnsembleStats s = ensembleStats(v, 0.95);
    ASSERT_GT(s.stderror, 0.0);
    EXPECT_NEAR(s.ciHalf / s.stderror, 1.95996, 0.02);
}

TEST(SweepSpec, SeedForSequence)
{
    SweepSpec spec;
    spec.seedBase = 1000;
    EXPECT_EQ(spec.seedFor(0), 1000);
    EXPECT_EQ(spec.seedFor(5), 1005);
}
