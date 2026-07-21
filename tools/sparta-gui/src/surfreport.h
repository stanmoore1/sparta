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

#ifndef SURFREPORT_H
#define SURFREPORT_H

// Pure, GUI-free reduction of per-surface-element data (from a `compute surf` or
// `fix ave/surf`) into engineering quantities: integrated force and moment
// vectors, total heat flux, per-column sums, and per-element distributions, plus
// a CSV of the raw per-element rows.  It operates on an already-extracted array
// (the GUI shell reads it from the SPARTA library via extractCompute/extractFix
// and passes it in with the column labels), so the reduction is fully
// unit-testable without Qt widgets or a running SPARTA instance.

#include <QString>
#include <QStringList>
#include <QVector>

namespace SurfReport {

/** @brief Integrated / summed quantities over all surface elements. */
struct Totals {
    int nsurf = 0;             ///< number of surface elements
    QStringList labels;        ///< column labels (parallel to columnSum)
    QVector<double> columnSum; ///< sum of each column over all elements

    bool hasForce = false;     ///< fx/fy/fz columns were present
    double force[3] = {0, 0, 0};

    bool hasMoment = false;    ///< tx/ty/tz columns were present
    double moment[3] = {0, 0, 0};

    bool hasHeatFlux = false;  ///< etot (or ke [+ erot + evib]) columns were present
    double heatFlux = 0.0;
};

/**
 * @brief Sum and integrate per-element rows into @ref Totals.
 * @param labels one label per column (e.g. "fx","fy","fz","press",...)
 * @param rows   one entry per surface element, each of length labels.size()
 *
 * Recognizes the force triple (fx,fy,fz), the moment/torque triple (tx,ty,tz)
 * and a heat-flux column (etot, else ke summed with erot/evib when present) by
 * label, and sums every column.
 */
Totals integrate(const QStringList &labels, const QVector<QVector<double>> &rows);

/** @brief Extract column @p c across all rows (missing entries skipped). */
QVector<double> column(const QVector<QVector<double>> &rows, int c);

/** @brief Simple descriptive statistics of a value list. */
struct Distribution {
    int n = 0;
    double min = 0.0, max = 0.0, mean = 0.0, stddev = 0.0;
};

/** @brief min/max/mean/sample-stddev of @p values (n<1 -> zeros). */
Distribution distribution(const QVector<double> &values);

/** @brief CSV text: a header of @p labels then one line per element row. */
QString toCsv(const QStringList &labels, const QVector<QVector<double>> &rows);

/**
 * @brief Column labels for a `compute surf` value list over @p ngroup groups.
 * @param values the per-surf value keywords (fx, press, ...)
 * @param ngroup number of mixture groups the compute tabulates
 *
 * For a single group the labels are the values verbatim; for multiple groups
 * each value is suffixed with the 1-based group index (value-major ordering,
 * matching SPARTA's per-surf array layout).
 */
QStringList expandColumnLabels(const QStringList &values, int ngroup);

} // namespace SurfReport

#endif // SURFREPORT_H

// Local Variables:
// c-basic-offset: 4
// End:
