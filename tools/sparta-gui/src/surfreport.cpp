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

#include "surfreport.h"

#include <cmath>

namespace SurfReport {

namespace {

// index of a label in the header, or -1
int find(const QStringList &labels, const QString &name)
{
    return labels.indexOf(name);
}

} // namespace

Totals integrate(const QStringList &labels, const QVector<QVector<double>> &rows)
{
    Totals t;
    t.labels = labels;
    t.nsurf  = rows.size();
    const int nc = labels.size();
    t.columnSum.fill(0.0, nc);

    for (const auto &row : rows)
        for (int c = 0; c < nc && c < row.size(); ++c)
            t.columnSum[c] += row.at(c);

    // force triple
    const int ix = find(labels, "fx"), iy = find(labels, "fy"), iz = find(labels, "fz");
    if (ix >= 0 && iy >= 0 && iz >= 0) {
        t.hasForce = true;
        t.force[0] = t.columnSum[ix];
        t.force[1] = t.columnSum[iy];
        t.force[2] = t.columnSum[iz];
    }

    // moment/torque triple
    const int tx = find(labels, "tx"), ty = find(labels, "ty"), tz = find(labels, "tz");
    if (tx >= 0 && ty >= 0 && tz >= 0) {
        t.hasMoment = true;
        t.moment[0] = t.columnSum[tx];
        t.moment[1] = t.columnSum[ty];
        t.moment[2] = t.columnSum[tz];
    }

    // heat flux: prefer the pre-summed etot, else ke (+ erot + evib when present)
    const int ie = find(labels, "etot");
    if (ie >= 0) {
        t.hasHeatFlux = true;
        t.heatFlux    = t.columnSum[ie];
    } else {
        const int ike = find(labels, "ke");
        if (ike >= 0) {
            t.hasHeatFlux = true;
            t.heatFlux    = t.columnSum[ike];
            const int ir = find(labels, "erot");
            const int iv = find(labels, "evib");
            if (ir >= 0) t.heatFlux += t.columnSum[ir];
            if (iv >= 0) t.heatFlux += t.columnSum[iv];
        }
    }

    return t;
}

QVector<double> column(const QVector<QVector<double>> &rows, int c)
{
    QVector<double> v;
    v.reserve(rows.size());
    for (const auto &row : rows)
        if (c >= 0 && c < row.size()) v.push_back(row.at(c));
    return v;
}

Distribution distribution(const QVector<double> &values)
{
    Distribution d;
    d.n = values.size();
    if (d.n < 1) return d;

    d.min = d.max = values.first();
    double sum = 0.0;
    for (double v : values) {
        if (v < d.min) d.min = v;
        if (v > d.max) d.max = v;
        sum += v;
    }
    d.mean = sum / d.n;
    if (d.n >= 2) {
        double ss = 0.0;
        for (double v : values) {
            const double dv = v - d.mean;
            ss += dv * dv;
        }
        d.stddev = std::sqrt(ss / (d.n - 1));
    }
    return d;
}

QString toCsv(const QStringList &labels, const QVector<QVector<double>> &rows)
{
    QString out;
    out += "element," + labels.join(',') + '\n';
    for (int i = 0; i < rows.size(); ++i) {
        out += QString::number(i);
        for (double v : rows.at(i)) out += ',' + QString::number(v, 'g', 10);
        out += '\n';
    }
    return out;
}

QStringList expandColumnLabels(const QStringList &values, int ngroup)
{
    if (ngroup <= 1) return values;
    QStringList out;
    for (const QString &v : values)
        for (int g = 1; g <= ngroup; ++g)
            out << QString("%1_g%2").arg(v).arg(g);
    return out;
}

} // namespace SurfReport
