// Unit tests for the post-processing analyses (src/analysis.cpp),
// exercised without a GUI.

#include "analysis.h"

#include "gtest/gtest.h"

#include <cmath>
#include <vector>

namespace {

TEST(Autocorrelation, ExactSmallCase)
{
    // y = [1,2,3], mean 2, deviations [-1,0,1], denom = 2
    //   lag0 = 2/2 = 1 ; lag1 = 0/2 = 0 ; lag2 = -1/2 = -0.5
    const std::vector<double> acf = autocorrelation({1.0, 2.0, 3.0}, 2);
    ASSERT_EQ(acf.size(), 3u);
    EXPECT_NEAR(acf[0], 1.0, 1.0e-12);
    EXPECT_NEAR(acf[1], 0.0, 1.0e-12);
    EXPECT_NEAR(acf[2], -0.5, 1.0e-12);
}

TEST(Autocorrelation, LagZeroIsOne)
{
    const std::vector<double> acf = autocorrelation({3.0, 1.0, 4.0, 1.0, 5.0, 9.0}, 3);
    ASSERT_FALSE(acf.empty());
    EXPECT_NEAR(acf[0], 1.0, 1.0e-12);
    for (double v : acf)
        EXPECT_LE(std::fabs(v), 1.0 + 1.0e-12);
}

TEST(Autocorrelation, ConstantSeriesIsEmpty)
{
    EXPECT_TRUE(autocorrelation({5.0, 5.0, 5.0, 5.0}, 2).empty());
}

TEST(Autocorrelation, TooShortIsEmpty)
{
    EXPECT_TRUE(autocorrelation({}, 5).empty());
    EXPECT_TRUE(autocorrelation({1.0}, 5).empty());
}

TEST(Autocorrelation, MaxlagClampedToLength)
{
    const std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};
    // non-positive maxlag clamps to n-1
    EXPECT_EQ(autocorrelation(y, 0).size(), 5u);
    // oversized maxlag clamps to n-1
    EXPECT_EQ(autocorrelation(y, 100).size(), 5u);
    // in-range maxlag is honored
    EXPECT_EQ(autocorrelation(y, 2).size(), 3u);
}

TEST(Autocorrelation, AlternatingIsAnticorrelated)
{
    std::vector<double> y(100);
    for (std::size_t i = 0; i < y.size(); ++i)
        y[i] = (i % 2 == 0) ? 1.0 : -1.0;

    const std::vector<double> acf = autocorrelation(y, 4);
    ASSERT_EQ(acf.size(), 5u);
    EXPECT_NEAR(acf[0], 1.0, 1.0e-12);
    EXPECT_LT(acf[1], 0.0); // odd lags: anticorrelated
    EXPECT_GT(acf[2], 0.0); // even lags: correlated
    EXPECT_LT(acf[3], 0.0);
}

// --- block-averaging uncertainty ------------------------------------------

TEST(BlockAverage, TooShortIsInvalid)
{
    EXPECT_FALSE(blockAverage({1.0, 2.0, 3.0}).valid);
    EXPECT_FALSE(blockAverage({}).valid);
}

TEST(BlockAverage, ConstantSeriesIsInvalid)
{
    EXPECT_FALSE(blockAverage(std::vector<double>(64, 7.0)).valid);
}

TEST(BlockAverage, WhiteNoiseStdErrMatchesNaive)
{
    // deterministic pseudo-white-noise (no correlation) -> block stderr should
    // agree with the naive s/sqrt(N) and tau_int ~ 0
    const int N = 4000;
    std::vector<double> y(N);
    unsigned long seed = 12345;
    double mean = 0.0, m2 = 0.0;
    for (int i = 0; i < N; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        const double u = double((seed >> 11) & 0xFFFFF) / double(0x100000); // [0,1)
        y[i] = u - 0.5;
    }
    for (double v : y) mean += v;
    mean /= N;
    for (double v : y) m2 += (v - mean) * (v - mean);
    const double naiveSem = std::sqrt(m2 / (N - 1) / N);

    const BlockStats bs = blockAverage(y, 0);
    ASSERT_TRUE(bs.valid);
    EXPECT_NEAR(bs.mean, mean, 1.0e-9);
    // uncorrelated: block SEM within ~30% of naive, tau_int small
    EXPECT_NEAR(bs.stderror, naiveSem, 0.35 * naiveSem);
    EXPECT_LT(bs.tauInt, 2.0);
    EXPECT_GT(bs.nEff, 0.4 * N);
}

TEST(BlockAverage, CorrelatedSeriesInflatesStdErr)
{
    // strongly correlated AR(1) with phi=0.9 -> block SEM >> naive SEM,
    // tau_int noticeably positive, nEff << N
    const int N = 4000;
    const double phi = 0.9;
    std::vector<double> y(N);
    unsigned long seed = 999;
    double prev = 0.0;
    for (int i = 0; i < N; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        const double u = double((seed >> 11) & 0xFFFFF) / double(0x100000) - 0.5;
        prev = phi * prev + u;
        y[i] = prev;
    }
    double mean = 0.0, m2 = 0.0;
    for (double v : y) mean += v;
    mean /= N;
    for (double v : y) m2 += (v - mean) * (v - mean);
    const double naiveSem = std::sqrt(m2 / (N - 1) / N);

    const BlockStats bs = blockAverage(y, 0);
    ASSERT_TRUE(bs.valid);
    EXPECT_GT(bs.stderror, 1.5 * naiveSem); // correlation inflates the error
    EXPECT_GT(bs.tauInt, 2.0);
    EXPECT_LT(bs.nEff, 0.5 * N);
}

// --- steady-state (burn-in) detection -------------------------------------

TEST(SteadyState, TooShortIsInvalid)
{
    EXPECT_FALSE(steadyStateCutoff({1.0, 2.0}).valid);
}

TEST(SteadyState, DetectsTransientThenPlateau)
{
    // a decaying transient for the first 200 samples settling to ~5.0, then a
    // stationary tail; the cutoff should land in/after the transient and the
    // retained mean should be close to 5.0
    const int N = 1000;
    std::vector<double> y(N);
    unsigned long seed = 7;
    for (int i = 0; i < N; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        const double noise = (double((seed >> 12) & 0xFFFF) / double(0x10000) - 0.5) * 0.1;
        const double transient = 20.0 * std::exp(-i / 40.0); // large early, ~0 by i~200
        y[i] = 5.0 + transient + noise;
    }
    const SteadyState ss = steadyStateCutoff(y);
    ASSERT_TRUE(ss.valid);
    EXPECT_GT(ss.cutoff, 30);         // discarded a chunk of the transient
    EXPECT_NEAR(ss.mean, 5.0, 0.2);   // retained mean near the plateau
    EXPECT_GT(ss.stderror, 0.0);
}

TEST(SteadyState, StationarySeriesKeepsMostData)
{
    // no transient -> cutoff should be small (near 0)
    const int N = 800;
    std::vector<double> y(N);
    unsigned long seed = 3;
    for (int i = 0; i < N; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        y[i] = 2.0 + (double((seed >> 12) & 0xFFFF) / double(0x10000) - 0.5) * 0.2;
    }
    const SteadyState ss = steadyStateCutoff(y);
    ASSERT_TRUE(ss.valid);
    EXPECT_LT(ss.cutoff, N / 4);
    EXPECT_NEAR(ss.mean, 2.0, 0.05);
}

} // namespace
