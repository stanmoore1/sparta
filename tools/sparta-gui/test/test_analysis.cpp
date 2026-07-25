// Unit tests for the post-processing analyses (src/analysis.cpp),
// exercised without a GUI.

#include "analysis.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Synthetic series with known statistics
// ---------------------------------------------------------------------------
//
// These analyses exist to answer "how uncertain is this average?" for output
// that is autocorrelated -- which DSMC thermo output always is. Checking that
// they do not crash on real data would say nothing about whether the number
// they print is right, so the series below are generated with properties that
// are known in closed form and the results are compared against those.
//
// The generator is a plain LCG rather than <random>: the standard
// distributions are not specified to produce the same values across
// implementations, and a test whose expected numbers depend on the standard
// library in use is a test that fails for someone else.

class Lcg {
public:
    explicit Lcg(std::uint64_t seed) : state(seed) {}
    /// uniform on [-0.5, 0.5), variance 1/12
    double uniform()
    {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return double((state >> 11) & 0xFFFFF) / double(0x100000) - 0.5;
    }

private:
    std::uint64_t state;
};

/**
 * @brief A first-order autoregressive series: y_t = phi*y_{t-1} + noise
 *
 * The reason for choosing AR(1) is that everything these analyses report has a
 * closed form for it:
 *   ACF(k)  = phi^k
 *   tau_int = phi / (1 - phi)          (sum of the ACF over positive lags)
 *   g       = 1 + 2*tau_int = (1+phi)/(1-phi)
 * so "the estimate is close to the truth" is a statement with a number in it.
 *
 * The first samples are discarded so the series starts already stationary;
 * without that the run-in is a transient and inflates the variance.
 */
std::vector<double> ar1(int n, double phi, std::uint64_t seed)
{
    Lcg rng(seed);
    double prev = 0.0;
    for (int i = 0; i < 2000; ++i) prev = phi * prev + rng.uniform(); // burn in
    std::vector<double> y;
    y.reserve(n);
    for (int i = 0; i < n; ++i) {
        prev = phi * prev + rng.uniform();
        y.push_back(prev);
    }
    return y;
}

/// The naive (correlation-blind) standard error of the mean, s/sqrt(N).
double naiveStdErr(const std::vector<double> &y)
{
    const double n    = double(y.size());
    const double mean = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double m2         = 0.0;
    for (double v : y) m2 += (v - mean) * (v - mean);
    return std::sqrt(m2 / (n - 1.0) / n);
}

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

// ---------------------------------------------------------------------------
// Autocorrelation against series whose ACF is known
// ---------------------------------------------------------------------------

// An independent, deliberately naive reimplementation of the documented
// estimator. If the fast version in analysis.cpp is ever rewritten, this says
// whether the numbers moved.
TEST(Autocorrelation, MatchesTheDocumentedEstimator)
{
    const std::vector<double> y = ar1(200, 0.5, 4242);
    const int maxlag            = 20;

    const double n    = double(y.size());
    const double mean = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double denom      = 0.0;
    for (double v : y) denom += (v - mean) * (v - mean);

    const std::vector<double> acf = autocorrelation(y, maxlag);
    ASSERT_EQ(acf.size(), std::size_t(maxlag + 1));
    for (int k = 0; k <= maxlag; ++k) {
        double num = 0.0;
        for (std::size_t i = 0; i + std::size_t(k) < y.size(); ++i)
            num += (y[i] - mean) * (y[i + k] - mean);
        EXPECT_NEAR(acf[std::size_t(k)], num / denom, 1.0e-12) << "at lag " << k;
    }
}

TEST(Autocorrelation, Ar1DecaysGeometrically)
{
    // ACF(k) = phi^k for an AR(1) process
    const double phi              = 0.7;
    const std::vector<double> y   = ar1(200000, phi, 20260725);
    const std::vector<double> acf = autocorrelation(y, 6);
    ASSERT_EQ(acf.size(), 7u);

    double expected = 1.0;
    for (int k = 0; k <= 6; ++k) {
        EXPECT_NEAR(acf[std::size_t(k)], expected, 0.05) << "at lag " << k;
        expected *= phi;
    }
}

TEST(Autocorrelation, PeriodicSeriesPeaksAtItsPeriod)
{
    // period 4: [1, 0, -1, 0] repeated. In phase at lags 0 and 4, exactly out
    // of phase at lag 2, and uncorrelated at the odd lags.
    std::vector<double> y;
    for (int i = 0; i < 400; ++i) {
        static const double cycle[4] = {1.0, 0.0, -1.0, 0.0};
        y.push_back(cycle[i % 4]);
    }
    const std::vector<double> acf = autocorrelation(y, 8);
    ASSERT_EQ(acf.size(), 9u);
    EXPECT_NEAR(acf[1], 0.0, 1.0e-12);
    EXPECT_NEAR(acf[3], 0.0, 1.0e-12);
    EXPECT_LT(acf[2], -0.9);          // anti-phase
    EXPECT_GT(acf[4], 0.9);           // back in phase one period later
    EXPECT_GT(acf[8], 0.9);
}

TEST(Autocorrelation, WhiteNoiseHasNoStructure)
{
    Lcg rng(31337);
    std::vector<double> y(50000);
    for (double &v : y) v = rng.uniform();

    const std::vector<double> acf = autocorrelation(y, 10);
    ASSERT_EQ(acf.size(), 11u);
    for (int k = 1; k <= 10; ++k)
        EXPECT_LT(std::fabs(acf[std::size_t(k)]), 0.02) << "at lag " << k;
}

TEST(Autocorrelation, LastLagIsTheSingleProductTerm)
{
    // At the largest lag only one pair contributes, so the value is fixed by
    // the two end points -- a closed form that catches an off-by-one in the
    // inner loop bound, which nothing else here would notice.
    const std::vector<double> y = {2.0, 4.0, 4.0, 10.0};
    const double mean           = 5.0;
    double denom                = 0.0;
    for (double v : y) denom += (v - mean) * (v - mean);

    const std::vector<double> acf = autocorrelation(y, 3);
    ASSERT_EQ(acf.size(), 4u);
    EXPECT_NEAR(acf[3], (2.0 - mean) * (10.0 - mean) / denom, 1.0e-12);
}

// ---------------------------------------------------------------------------
// Block averaging
// ---------------------------------------------------------------------------

TEST(BlockAverage, HonorsAnExplicitBlockCount)
{
    const std::vector<double> y = ar1(1000, 0.5, 11);
    EXPECT_EQ(blockAverage(y, 5).nblocks, 5);
    EXPECT_EQ(blockAverage(y, 40).nblocks, 40);
}

TEST(BlockAverage, DefaultBlockCountIsRootN)
{
    const std::vector<double> y = ar1(100, 0.5, 12);
    EXPECT_EQ(blockAverage(y, 0).nblocks, 10);   // floor(sqrt(100))
    EXPECT_EQ(blockAverage(y, 1).nblocks, 10);   // one block cannot give an error
    EXPECT_EQ(blockAverage(y, -7).nblocks, 10);  // nonsense falls back too
}

TEST(BlockAverage, BlockCountIsClampedSoEveryBlockHasTwoSamples)
{
    const std::vector<double> y = ar1(100, 0.5, 13);
    // asking for more blocks than the series can fill would leave blocks of a
    // single sample, whose "variance of the block means" is just the variance
    EXPECT_EQ(blockAverage(y, 1000).nblocks, 50);
    EXPECT_EQ(blockAverage(y, 100).nblocks, 50);
}

TEST(BlockAverage, MeanAndVarianceDescribeTheWholeSeries)
{
    // 10 samples in 3 blocks: the blocks can only cover 9 of them. The mean
    // and variance reported still have to be those of everything the caller
    // handed over, or "the mean of this series" quietly means something else.
    const std::vector<double> y = {1, 2, 3, 4, 5, 6, 7, 8, 9, 100};

    const double n    = double(y.size());
    const double mean = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double m2         = 0.0;
    for (double v : y) m2 += (v - mean) * (v - mean);

    const BlockStats bs = blockAverage(y, 3);
    ASSERT_TRUE(bs.valid);
    EXPECT_EQ(bs.nblocks, 3);
    EXPECT_NEAR(bs.mean, mean, 1.0e-12);
    EXPECT_NEAR(bs.variance, m2 / (n - 1.0), 1.0e-12);
    // the trailing sample really is the one left out of the blocking
    EXPECT_NE(bs.mean, std::accumulate(y.begin(), y.begin() + 9, 0.0) / 9.0);
}

TEST(BlockAverage, StdErrorIsTheErrorOfTheBlockMeans)
{
    // recomputed here from the definition, so this pins the number rather than
    // its order of magnitude
    const std::vector<double> y = ar1(1000, 0.6, 14);
    const int nblocks           = 25;
    const int L                 = int(y.size()) / nblocks;
    const int used              = L * nblocks;

    std::vector<double> bmean;
    for (int b = 0; b < nblocks; ++b) {
        double m = 0.0;
        for (int i = 0; i < L; ++i) m += y[std::size_t(b * L + i)];
        bmean.push_back(m / L);
    }
    const double usedMean =
        std::accumulate(y.begin(), y.begin() + used, 0.0) / double(used);
    double varBlocks = 0.0;
    for (double m : bmean) varBlocks += (m - usedMean) * (m - usedMean);
    varBlocks /= double(nblocks - 1);

    const BlockStats bs = blockAverage(y, nblocks);
    ASSERT_TRUE(bs.valid);
    EXPECT_NEAR(bs.stderror, std::sqrt(varBlocks / nblocks), 1.0e-12);
}

TEST(BlockAverage, Ar1RecoversTheKnownCorrelationTime)
{
    // tau_int = phi/(1-phi) = 4 and g = (1+phi)/(1-phi) = 9 for phi = 0.8, so
    // the correlation-aware error should be sqrt(9) = 3x the naive one.
    const double phi            = 0.8;
    const std::vector<double> y = ar1(200000, phi, 777);
    const BlockStats bs         = blockAverage(y, 0);
    ASSERT_TRUE(bs.valid);

    EXPECT_NEAR(bs.tauInt, phi / (1.0 - phi), 0.8);                 // 4.0
    EXPECT_NEAR(bs.stderror / naiveStdErr(y), 3.0, 0.4);
    // N_eff = N/g: 200000/9 ~ 22000
    EXPECT_NEAR(bs.nEff, double(y.size()) / 9.0, 0.2 * y.size() / 9.0);
}

TEST(BlockAverage, EffectiveCountIsTheSampleCountOverTheInefficiency)
{
    // whatever tau_int comes out as, N_eff has to be consistent with it
    for (double phi : {0.0, 0.3, 0.6, 0.9}) {
        const std::vector<double> y = ar1(20000, phi, 99);
        const BlockStats bs         = blockAverage(y, 0);
        ASSERT_TRUE(bs.valid) << "phi = " << phi;
        const double g    = 1.0 + 2.0 * bs.tauInt;
        const int used    = (int(y.size()) / bs.nblocks) * bs.nblocks;
        EXPECT_NEAR(bs.nEff * g, double(used), 0.02 * used) << "phi = " << phi;
        EXPECT_LE(bs.nEff, double(used) + 1.0) << "phi = " << phi;
        EXPECT_GE(bs.nEff, 1.0) << "phi = " << phi;
    }
}

TEST(BlockAverage, AntiCorrelatedSeriesBeatsTheNaiveError)
{
    // an alternating series averages out faster than independent samples, so
    // the honest error is *smaller* than s/sqrt(N) and tau_int floors at zero
    Lcg rng(2024);
    std::vector<double> y(4000);
    for (std::size_t i = 0; i < y.size(); ++i)
        y[i] = ((i % 2) ? -1.0 : 1.0) + 0.01 * rng.uniform();

    const BlockStats bs = blockAverage(y, 0);
    ASSERT_TRUE(bs.valid);
    EXPECT_LT(bs.stderror, naiveStdErr(y));
    EXPECT_GE(bs.tauInt, 0.0);   // never reported as negative
    EXPECT_NEAR(bs.tauInt, 0.0, 1.0e-9);
}

TEST(BlockAverage, ShortestAnalyzableSeries)
{
    // four samples is the documented minimum: two blocks of two
    const BlockStats bs = blockAverage({1.0, 2.0, 3.0, 5.0}, 0);
    ASSERT_TRUE(bs.valid);
    EXPECT_EQ(bs.nblocks, 2);
    EXPECT_NEAR(bs.mean, 2.75, 1.0e-12);
    // blocks are [1,2] -> 1.5 and [3,5] -> 4.0; their sample variance is
    // (1.5-2.75)^2 + (4-2.75)^2 = 3.125, and SEM = sqrt(3.125/2)
    EXPECT_NEAR(bs.stderror, std::sqrt(3.125 / 2.0), 1.0e-12);
}

TEST(BlockAverage, AlmostConstantSeriesIsStillAnalyzed)
{
    // exactly constant is rejected (nothing to say about its error); a series
    // with any spread at all must not be
    std::vector<double> y(64, 7.0);
    y[10] += 1.0e-9;
    const BlockStats bs = blockAverage(y, 0);
    EXPECT_TRUE(bs.valid);
    EXPECT_GT(bs.variance, 0.0);
}

// ---------------------------------------------------------------------------
// Steady-state (burn-in) detection
// ---------------------------------------------------------------------------

TEST(SteadyState, CutsAtAStepChange)
{
    // the cleanest case there is: pure noise around 0 for 200 samples, then
    // pure noise around 10. Retaining anything before the step keeps a huge
    // variance, so the minimum is at the step itself.
    Lcg rng(5);
    std::vector<double> y;
    for (int i = 0; i < 1000; ++i)
        y.push_back((i < 200 ? 0.0 : 10.0) + 0.05 * rng.uniform());

    const SteadyState ss = steadyStateCutoff(y);
    ASSERT_TRUE(ss.valid);
    EXPECT_NEAR(ss.cutoff, 200, 5);
    EXPECT_NEAR(ss.mean, 10.0, 0.01);
}

TEST(SteadyState, NeverDiscardsMoreThanHalfTheSeries)
{
    // a transient that runs past the half-way point: the search is capped, so
    // the answer is the cap rather than a cutoff that throws the data away
    std::vector<double> y;
    for (int i = 0; i < 1000; ++i) y.push_back(i < 700 ? 100.0 - 0.1 * i : 30.0);

    const SteadyState ss = steadyStateCutoff(y);
    ASSERT_TRUE(ss.valid);
    EXPECT_LE(ss.cutoff, 500);
    EXPECT_GE(ss.cutoff, 0);
}

TEST(SteadyState, ConstantSeriesKeepsEverythingAndHasNoError)
{
    // blockAverage rejects a constant series, so this exercises the fallback
    // path -- which used to be the only way to get a wrong answer here
    const std::vector<double> y(500, 3.5);
    const SteadyState ss = steadyStateCutoff(y);
    ASSERT_TRUE(ss.valid);
    EXPECT_EQ(ss.cutoff, 0);
    EXPECT_NEAR(ss.mean, 3.5, 1.0e-12);
    EXPECT_NEAR(ss.stderror, 0.0, 1.0e-12);
}

TEST(SteadyState, FourSamplesIsEnough)
{
    EXPECT_FALSE(steadyStateCutoff({1.0, 2.0, 3.0}).valid);
    EXPECT_TRUE(steadyStateCutoff({1.0, 2.0, 3.0, 4.0}).valid);
}

TEST(SteadyState, ReportedMeanIsTheMeanOfWhatWasKept)
{
    const std::vector<double> y = ar1(2000, 0.7, 6);
    const SteadyState ss        = steadyStateCutoff(y);
    ASSERT_TRUE(ss.valid);

    const std::vector<double> tail(y.begin() + ss.cutoff, y.end());
    const BlockStats bs = blockAverage(tail, 0);
    ASSERT_TRUE(bs.valid);
    EXPECT_NEAR(ss.mean, bs.mean, 1.0e-12);
    EXPECT_NEAR(ss.stderror, bs.stderror, 1.0e-12);
}

TEST(SteadyState, DecayingTransientLeavesThePlateauMean)
{
    // the shape DSMC output actually has: an exponential approach to a value
    Lcg rng(8);
    std::vector<double> y;
    for (int i = 0; i < 2000; ++i)
        y.push_back(5.0 + 20.0 * std::exp(-i / 40.0) + 0.05 * rng.uniform());

    const SteadyState ss = steadyStateCutoff(y);
    ASSERT_TRUE(ss.valid);
    EXPECT_GT(ss.cutoff, 20);            // the transient was cut
    EXPECT_NEAR(ss.mean, 5.0, 0.02);     // and what is left is the plateau
    EXPECT_LT(ss.stderror, 0.01);
}

// ---------------------------------------------------------------------------
// Restricting a fit to a sub-range
// ---------------------------------------------------------------------------

TEST(RestrictToXRange, KeepsTheInclusiveWindow)
{
    std::vector<double> x = {0, 1, 2, 3, 4, 5};
    std::vector<double> y = {0, 10, 20, 30, 40, 50};
    EXPECT_TRUE(restrictToXRange(x, y, 1.0, 3.0));
    EXPECT_EQ(x, (std::vector<double>{1, 2, 3}));
    EXPECT_EQ(y, (std::vector<double>{10, 20, 30}));
}

TEST(RestrictToXRange, DropsBothCoordinatesTogether)
{
    // the failure this guards against does not change the number of points,
    // it changes which y belongs to which x
    std::vector<double> x = {0, 5, 1, 6, 2};
    std::vector<double> y = {0, 50, 10, 60, 20};
    ASSERT_TRUE(restrictToXRange(x, y, 0.0, 2.0));
    ASSERT_EQ(x.size(), y.size());
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_DOUBLE_EQ(y[i], 10.0 * x[i]);
}

TEST(RestrictToXRange, RefusesToLeaveTooFewPoints)
{
    const std::vector<double> x0 = {0, 1, 2, 3};
    const std::vector<double> y0 = {0, 1, 2, 3};
    std::vector<double> x = x0, y = y0;
    // only one point in range: the caller is told, and nothing is dropped, so
    // an unchecked call still fits the full series instead of a single point
    EXPECT_FALSE(restrictToXRange(x, y, 0.5, 1.5));
    EXPECT_EQ(x, x0);
    EXPECT_EQ(y, y0);

    // nothing in range at all
    x = x0;
    y = y0;
    EXPECT_FALSE(restrictToXRange(x, y, 10.0, 20.0));
    EXPECT_EQ(x, x0);
}

TEST(RestrictToXRange, RejectsAnInvertedOrEmptyRange)
{
    const std::vector<double> x0 = {0, 1, 2, 3};
    std::vector<double> x = x0, y = x0;
    EXPECT_FALSE(restrictToXRange(x, y, 3.0, 1.0));
    EXPECT_EQ(x, x0);
    EXPECT_FALSE(restrictToXRange(x, y, 2.0, 2.0));
    EXPECT_EQ(x, x0);
}

TEST(RestrictToXRange, RejectsMismatchedLengths)
{
    std::vector<double> x = {0, 1, 2};
    std::vector<double> y = {0, 1};
    EXPECT_FALSE(restrictToXRange(x, y, 0.0, 2.0));
    EXPECT_EQ(x.size(), 3u);
}

TEST(RestrictToXRange, AWholeRangeChangesNothing)
{
    const std::vector<double> x0 = {0, 1, 2, 3};
    std::vector<double> x = x0, y = x0;
    EXPECT_TRUE(restrictToXRange(x, y, -1.0, 99.0));
    EXPECT_EQ(x, x0);
}

} // namespace
