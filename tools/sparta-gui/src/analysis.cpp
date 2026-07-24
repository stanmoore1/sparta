// -*- c++ -*- /////////////////////////////////////////////////////////////////////////
// SPARTA-GUI - A Graphical Tool to Learn and Explore the SPARTA DSMC Simulation Software
//
// Copyright (c) 2023, 2024, 2025, 2026  Axel Kohlmeyer
//
// Documentation: https://sparta.github.io/sparta-gui/
// Contact: akohlmey@gmail.com
//
// This software is distributed under the GNU General Public License version 2 or later.
////////////////////////////////////////////////////////////////////////////////////////

#include "analysis.h"

#include <cmath>

std::vector<double> autocorrelation(const std::vector<double> &y, int maxlag)
{
    const int n = static_cast<int>(y.size());
    if (n < 2) return {};

    if ((maxlag <= 0) || (maxlag >= n)) maxlag = n - 1;

    // mean
    double mean = 0.0;
    for (double v : y)
        mean += v;
    mean /= static_cast<double>(n);

    // total variance (denominator); zero for a constant series
    double denom = 0.0;
    for (double v : y) {
        const double d = v - mean;
        denom += d * d;
    }
    if (denom <= 0.0) return {};

    std::vector<double> acf(maxlag + 1, 0.0);
    for (int k = 0; k <= maxlag; ++k) {
        double num = 0.0;
        for (int i = 0; i + k < n; ++i)
            num += (y[i] - mean) * (y[i + k] - mean);
        acf[k] = num / denom;
    }
    return acf;
}

BlockStats blockAverage(const std::vector<double> &y, int nblocks)
{
    BlockStats s;
    const int n = static_cast<int>(y.size());
    if (n < 4) return s;

    // default block count ~ sqrt(N), clamped so each block has >= 2 samples
    if (nblocks <= 1) nblocks = static_cast<int>(std::floor(std::sqrt(double(n))));
    if (nblocks < 2) nblocks = 2;
    if (nblocks > n / 2) nblocks = n / 2;

    const int L = n / nblocks;          // block length (equal-length blocks)
    const int used = L * nblocks;       // samples covered by whole blocks

    // Reported mean and variance describe the whole series the caller passed
    // in.  Blocking needs equal-length blocks, so it can only cover the first
    // `used` samples; reporting the blocked subset's mean instead would
    // silently ignore up to nblocks-1 trailing samples, which is not what a
    // caller asking for "the mean of this series" expects.
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += y[i];
    mean /= n;
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = y[i] - mean;
        var += d * d;
    }
    if (var <= 0.0) return s;            // constant series
    const double sampleVar = var / (n - 1);

    // The block-derived quantities below (block variance, statistical
    // inefficiency, standard error) must all be computed consistently over the
    // blocked subset, so they use its own mean and variance.
    double usedMean = 0.0;
    for (int i = 0; i < used; ++i) usedMean += y[i];
    usedMean /= used;
    double usedVar = 0.0;
    for (int i = 0; i < used; ++i) {
        const double d = y[i] - usedMean;
        usedVar += d * d;
    }
    if (usedVar <= 0.0) return s;        // constant over the blocked subset
    const double usedSampleVar = usedVar / (used - 1);

    // block means and their sample variance
    double varBlocks = 0.0;
    std::vector<double> bmean(nblocks, 0.0);
    for (int b = 0; b < nblocks; ++b) {
        double m = 0.0;
        for (int i = 0; i < L; ++i) m += y[b * L + i];
        bmean[b] = m / L;
    }
    for (int b = 0; b < nblocks; ++b) {
        const double d = bmean[b] - usedMean;
        varBlocks += d * d;
    }
    varBlocks /= (nblocks - 1);         // sample variance of the block means

    // standard error of the mean from the block means
    const double sem = std::sqrt(varBlocks / nblocks);

    // statistical inefficiency g = L * var(blockMeans) / sampleVar ~ 1 + 2*tau
    const double g = L * varBlocks / usedSampleVar;

    s.valid    = true;
    s.mean     = mean;
    s.variance = sampleVar;
    s.stderror = sem;
    s.tauInt   = std::max(0.0, 0.5 * (g - 1.0));
    s.nEff     = (g > 0.0) ? std::min(double(used), used / g) : double(used);
    if (s.nEff < 1.0) s.nEff = 1.0;
    s.nblocks  = nblocks;
    return s;
}

SteadyState steadyStateCutoff(const std::vector<double> &y)
{
    SteadyState s;
    const int n = static_cast<int>(y.size());
    if (n < 4) return s;

    // suffix sums so each candidate cutoff is O(1): S1[d]=sum_{i>=d} y, S2 likewise for y^2
    std::vector<double> S1(n + 1, 0.0), S2(n + 1, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        S1[i] = S1[i + 1] + y[i];
        S2[i] = S2[i + 1] + y[i] * y[i];
    }

    // MSER: minimize SEM^2 = var_retained / m^2 over cutoffs in the first half
    int best = 0;
    double bestStat = 1e300;
    const int dmax = n / 2;
    for (int d = 0; d <= dmax; ++d) {
        const int m = n - d;
        if (m < 2) break;
        const double mean = S1[d] / m;
        double var = S2[d] - S1[d] * S1[d] / m;   // sum of squared deviations
        if (var < 0.0) var = 0.0;                 // guard rounding
        const double stat = var / (double(m) * double(m));
        if (stat < bestStat) { bestStat = stat; best = d; }
    }

    // characterize the retained tail with a correlation-aware standard error
    const std::vector<double> tail(y.begin() + best, y.end());
    const BlockStats bs = blockAverage(tail, 0);

    s.valid  = true;
    s.cutoff = best;
    if (bs.valid) {
        s.mean     = bs.mean;
        s.stderror = bs.stderror;
    } else {
        const int m = n - best;
        s.mean     = S1[best] / m;
        s.stderror = std::sqrt(std::max(0.0, bestStat));
    }
    return s;
}

// Local Variables:
// c-basic-offset: 4
// End:
