/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Pure, GUI-free model for a parametric sweep (DOE study): a set of SPARTA
   index variables each varied over a list/range/linspace, expanded into the
   ordered list of per-run variable assignments, plus the reducer that turns a
   run's sampled series into one tabulated number.  No Qt widgets, no SPARTA,
   so it is unit-tested in isolation like dumpimage/stlimport/schedulerspec.
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

} // namespace Sweep

#endif // SWEEPSPEC_H
