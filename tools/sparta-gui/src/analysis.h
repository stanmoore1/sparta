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

#ifndef ANALYSIS_H
#define ANALYSIS_H

// Small, self-contained (Qt-free) post-processing analyses on a data series.
// Pure functions on std::vector<double> so they can be unit-tested without a
// GUI and reused by the chart post-processing dialog.

#include <cstddef>
#include <vector>

/**
 * @brief Normalized autocorrelation function (ACF) of a data series
 * @param y      Input samples (assumed equally spaced)
 * @param maxlag Largest lag to compute; values <= 0 or >= y.size() are
 *               clamped to y.size()-1
 * @return ACF values for lags 0..maxlag (length maxlag+1), normalized so that
 *         the lag-0 value is 1; an empty vector if the input has fewer than
 *         two samples or zero variance (a constant series)
 *
 * Uses the standard biased estimator
 * @f$ \mathrm{ACF}(k) = \frac{\sum_{i=0}^{N-1-k}(y_i-\bar y)(y_{i+k}-\bar y)}
 * {\sum_{i=0}^{N-1}(y_i-\bar y)^2} @f$.
 */
std::vector<double> autocorrelation(const std::vector<double> &y, int maxlag);

/**
 * @brief Block-averaging (batch-means) uncertainty of a correlated series.
 *
 * DSMC thermo output is autocorrelated, so the naive standard error
 * @f$s/\sqrt{N}@f$ underestimates the true uncertainty of the mean.  Splitting
 * the series into @p nblocks contiguous blocks and taking the standard error of
 * the block means gives a correlation-aware estimate, and the ratio of the two
 * yields the statistical inefficiency @f$g=1+2\tau_\mathrm{int}@f$.
 */
struct BlockStats {
    bool valid    = false; ///< false if the series was too short to analyze
    double mean   = 0.0;   ///< grand mean of the series
    double variance = 0.0; ///< sample variance of the samples (N-1 normalization)
    double stderror = 0.0; ///< block-averaged standard error of the mean
    double tauInt = 0.0;   ///< integrated autocorrelation time (in samples), >= 0
    double nEff   = 0.0;   ///< effective number of independent samples, in [1, N]
    int nblocks   = 0;     ///< number of blocks actually used
};

/**
 * @brief Estimate the mean and its uncertainty by block averaging.
 * @param y       input samples (equally spaced)
 * @param nblocks number of blocks; if <= 1, a default ~sqrt(N) is used
 * @return statistics; @c valid is false when @p y has fewer than 4 samples or
 *         is constant
 */
BlockStats blockAverage(const std::vector<double> &y, int nblocks = 0);

/**
 * @brief Steady-state (burn-in) cutoff via the Marginal Standard Error Rule.
 *
 * MSER picks the truncation point @f$d@f$ that minimizes the standard error of
 * the mean of the retained tail @f$y_d..y_{N-1}@f$, i.e. it discards the initial
 * transient while keeping as much stationary data as possible.  The search is
 * restricted to the first half of the series so at least half the samples are
 * retained.
 */
struct SteadyState {
    bool valid     = false; ///< false if the series was too short
    int cutoff     = 0;     ///< index of the first retained sample (burn-in length)
    double mean    = 0.0;   ///< mean of the retained (post-cutoff) samples
    double stderror = 0.0;  ///< block-averaged standard error of the retained mean
};

/**
 * @brief Detect the burn-in cutoff of a time series (MSER).
 * @param y input samples (equally spaced)
 * @return the cutoff index and the post-cutoff mean +/- standard error
 */
SteadyState steadyStateCutoff(const std::vector<double> &y);

/**
 * @brief Keep only the samples whose abscissa lies in [@p xmin, @p xmax].
 *
 * The fitting analyses let the user fit a sub-range of a chart rather than the
 * whole of it -- the interesting part of a DSMC run is rarely the start-up
 * transient.  Both vectors are shortened in place, and in step: dropping a
 * point from one and not the other silently pairs each x with its neighbour's
 * y, which fits a plausible-looking curve to data that does not exist.
 *
 * @param x     abscissa, shortened in place
 * @param y     ordinate, shortened in place alongside @p x
 * @param xmin  lower bound, inclusive
 * @param xmax  upper bound, inclusive
 * @param minPoints smallest useful result; below it nothing is dropped
 * @return true if a restriction was applied; false if the range was empty or
 *         inverted, or kept fewer than @p minPoints points -- in which case
 *         @p x and @p y are left untouched, so a caller that ignores the
 *         result still fits the full data rather than nothing at all
 */
bool restrictToXRange(std::vector<double> &x, std::vector<double> &y, double xmin, double xmax,
                      std::size_t minPoints = 2);

#endif

// Local Variables:
// c-basic-offset: 4
// End:
