/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Pure, GUI-free model for a parametric sweep (DOE study): a set of SPARTA
   index variables each varied over a list/range/linspace, expanded into the
   ordered list of per-run variable assignments, plus the reducer that turns a
   run's sampled series into one tabulated number.  No Qt widgets, no SPARTA,
   so it is unit-tested in isolation like dumpimage/stlimport.
------------------------------------------------------------------------- */

#ifndef SWEEPSPEC_H
#define SWEEPSPEC_H

#include <QList>
#include <QPair>
#include <QString>
#include <QStringList>

#include <vector>

namespace Sweep {

/** @brief One variable's sweep definition. */
struct VarSweep {
    QString name;                       ///< index-variable name to override per run
    enum Kind { List, Range, Linspace } kind = List;
    QStringList values;                 ///< List: explicit values
    double start = 0.0;                 ///< Range/Linspace: first value
    double stop = 0.0;                  ///< Range/Linspace: last value
    double step = 1.0;                  ///< Range: increment
    int count = 0;                      ///< Linspace: number of points

    /** @brief Expand to concrete value strings (pure). */
    QStringList expand() const;
};

/** @brief How multiple variables' value lists are combined into runs. */
enum class Combine { Cartesian, Zip };

/** @brief How a run's sampled series is reduced to one tabulated value. */
enum class Reducer { Final, Min, Max, Mean };

QString reducerName(Reducer r);

/** @brief A full sweep: the variables, how to combine them, and what to record. */
struct SweepSpec {
    QList<VarSweep> vars;
    Combine combine = Combine::Cartesian;
    QStringList quantities;             ///< thermo keywords to tabulate
    QList<Reducer> reducers;            ///< parallel to quantities (Final if short)

    // --- ensemble (replicate) options ---
    int replicates = 1;                 ///< runs per sweep point with distinct seeds (>=1)
    QString seedVariable;               ///< index variable set to a fresh seed each replicate
    int seedBase = 12345;               ///< replicate k (0-based) uses seedBase + k
    double ciLevel = 0.95;              ///< confidence level for the reported interval

    /** @brief Seed value for replicate @p k (0-based). */
    int seedFor(int k) const { return seedBase + k; }

    /**
     * @brief The ordered list of runs, each a list of (name,value) assignments.
     *
     * Cartesian = odometer over each variable's expansion (last variable varies
     * fastest); Zip = index-wise across equal-length expansions. On an error
     * (no variables, empty expansion, or mismatched Zip lengths) sets @p err
     * (when non-null) and returns an empty list.
     */
    QList<QList<QPair<QString, QString>>> expand(QString *err = nullptr) const;

    /** @brief Number of runs expand() would produce (0 on an invalid spec). */
    int runCount() const;

    /** @brief Reducer for quantity column @p i (Final when reducers is short). */
    Reducer reducerFor(int i) const;
};

/** @brief Reduce a sampled series to one value; 0.0 for an empty series. */
double reduce(Reducer r, const std::vector<double> &samples);

/**
 * @brief Aggregated statistics over an ensemble of replicate runs.
 *
 * Each replicate contributes one already-reduced scalar (e.g. the final or
 * mean value of a thermo quantity); ensembleStats() turns those into a mean
 * with a standard error and a Student-t confidence interval, the standard way
 * to report a DSMC result computed from N independent-seed replicates.
 */
struct EnsembleStats {
    int n           = 0;    ///< number of replicates
    double mean     = 0.0;  ///< sample mean
    double stddev   = 0.0;  ///< sample standard deviation (N-1)
    double stderror = 0.0;  ///< standard error of the mean = stddev/sqrt(n)
    double ciHalf   = 0.0;  ///< half-width of the confidence interval (t-based)
    double ciLevel  = 0.95; ///< confidence level the interval was built for
};

/**
 * @brief Mean +/- standard error and t-CI across replicate values.
 * @param values   one reduced scalar per replicate
 * @param ciLevel  two-sided confidence level (e.g. 0.95)
 * @return statistics; for a single value stddev/stderror/ciHalf are 0
 */
EnsembleStats ensembleStats(const std::vector<double> &values, double ciLevel = 0.95);

} // namespace Sweep

#endif // SWEEPSPEC_H
