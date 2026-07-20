/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Pure parametric-sweep expansion + reducers.  See sweepspec.h.
------------------------------------------------------------------------- */

#include "sweepspec.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace Sweep {

QStringList VarSweep::expand() const
{
    switch (kind) {
    case List:
        return values;
    case Linspace: {
        QStringList out;
        if (count <= 0) return out;
        if (count == 1) { out << QString::number(start, 'g', 12); return out; }
        const double d = (stop - start) / double(count - 1);
        for (int i = 0; i < count; ++i)
            out << QString::number(start + d * i, 'g', 12);
        return out;
    }
    case Range:
    default: {
        QStringList out;
        if (step == 0.0) return out;
        // inclusive of stop within a floating-point epsilon
        const double eps = std::abs(step) * 1e-9;
        if (step > 0.0)
            for (double v = start; v <= stop + eps; v += step)
                out << QString::number(v, 'g', 12);
        else
            for (double v = start; v >= stop - eps; v += step)
                out << QString::number(v, 'g', 12);
        return out;
    }
    }
}

QString reducerName(Reducer r)
{
    switch (r) {
    case Reducer::Min:  return "min";
    case Reducer::Max:  return "max";
    case Reducer::Mean: return "mean";
    case Reducer::Final:
    default:            return "final";
    }
}

Reducer SweepSpec::reducerFor(int i) const
{
    return (i >= 0 && i < reducers.size()) ? reducers.at(i) : Reducer::Final;
}

int SweepSpec::runCount() const
{
    if (vars.isEmpty()) return 0;
    QList<int> sizes;
    for (const auto &v : vars) {
        const int n = v.expand().size();
        if (n == 0) return 0;
        sizes << n;
    }
    if (combine == Combine::Zip) {
        const int n = sizes.first();
        for (int s : sizes) if (s != n) return 0;
        return n;
    }
    long long total = 1;
    for (int s : sizes) total *= s;
    return int(total);
}

QList<QList<QPair<QString, QString>>> SweepSpec::expand(QString *err) const
{
    auto fail = [&](const QString &m) {
        if (err) *err = m;
        return QList<QList<QPair<QString, QString>>>{};
    };
    if (err) err->clear();
    if (vars.isEmpty()) return fail("No variables to sweep.");

    QList<QStringList> expansions;
    for (const auto &v : vars) {
        if (v.name.trimmed().isEmpty()) return fail("A swept variable has no name.");
        const QStringList e = v.expand();
        if (e.isEmpty()) return fail(QString("Variable '%1' expands to no values.").arg(v.name));
        expansions << e;
    }

    QList<QList<QPair<QString, QString>>> out;

    if (combine == Combine::Zip) {
        const int n = expansions.first().size();
        for (const auto &e : expansions)
            if (e.size() != n)
                return fail("Zip requires every variable to have the same number of values.");
        for (int i = 0; i < n; ++i) {
            QList<QPair<QString, QString>> combo;
            for (int v = 0; v < vars.size(); ++v)
                combo << qMakePair(vars.at(v).name, expansions.at(v).at(i));
            out << combo;
        }
        return out;
    }

    // Cartesian product via an odometer; the last variable varies fastest.
    QList<int> idx(vars.size(), 0);
    while (true) {
        QList<QPair<QString, QString>> combo;
        for (int v = 0; v < vars.size(); ++v)
            combo << qMakePair(vars.at(v).name, expansions.at(v).at(idx.at(v)));
        out << combo;

        int pos = vars.size() - 1;
        while (pos >= 0) {
            if (++idx[pos] < expansions.at(pos).size()) break;
            idx[pos] = 0;
            --pos;
        }
        if (pos < 0) break;
    }
    return out;
}

double reduce(Reducer r, const std::vector<double> &s)
{
    if (s.empty()) return 0.0;
    switch (r) {
    case Reducer::Min:  return *std::min_element(s.begin(), s.end());
    case Reducer::Max:  return *std::max_element(s.begin(), s.end());
    case Reducer::Mean: return std::accumulate(s.begin(), s.end(), 0.0) / double(s.size());
    case Reducer::Final:
    default:            return s.back();
    }
}

} // namespace Sweep

// Local Variables:
// c-basic-offset: 4
// End:
