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
    // No samples is not a measurement of zero.  A run whose quantity never
    // appeared -- a misspelt keyword, or a run too short for the stats poller to
    // catch a line -- has to reach the table as "n/a"; a 0.0 here would sit in
    // the results indistinguishable from a real reading, and would drag an
    // ensemble mean down with it.
    if (s.empty()) return std::nan("");
    switch (r) {
    case Reducer::Min:  return *std::min_element(s.begin(), s.end());
    case Reducer::Max:  return *std::max_element(s.begin(), s.end());
    case Reducer::Mean: return std::accumulate(s.begin(), s.end(), 0.0) / double(s.size());
    case Reducer::Final:
    default:            return s.back();
    }
}

namespace {

// Inverse standard-normal CDF (Acklam's rational approximation), |err| < 1.2e-9.
double invNormalCdf(double p)
{
    if (p <= 0.0) return -1e18;
    if (p >= 1.0) return 1e18;
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                               -2.759285104469687e+02, 1.383577518672690e+02,
                               -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                               -1.556989798598866e+02, 6.680131188771972e+01,
                               -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                               -2.400758277161838e+00, -2.549732539343734e+00,
                               4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                               2.445134137142996e+00, 3.754408661907416e+00};
    const double plow = 0.02425, phigh = 1.0 - 0.02425;
    if (p < plow) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > phigh) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

// Student-t quantile via the Cornish-Fisher expansion from the normal quantile;
// adequate for the small replicate counts used in ensemble reporting.
double tQuantile(double p, int dof)
{
    const double z = invNormalCdf(p);
    if (dof <= 0) return z;
    const double g1 = (z * z * z + z) / 4.0;
    const double g2 = (5.0 * std::pow(z, 5) + 16.0 * z * z * z + 3.0 * z) / 96.0;
    const double g3 = (3.0 * std::pow(z, 7) + 19.0 * std::pow(z, 5) + 17.0 * z * z * z -
                       15.0 * z) / 384.0;
    const double n = dof;
    return z + g1 / n + g2 / (n * n) + g3 / (n * n * n);
}

} // namespace

EnsembleStats ensembleStats(const std::vector<double> &values, double ciLevel)
{
    EnsembleStats s;
    s.n       = static_cast<int>(values.size());
    s.ciLevel = ciLevel;
    if (s.n == 0) return s;

    double mean = 0.0;
    for (double v : values) mean += v;
    mean /= s.n;
    s.mean = mean;
    if (s.n < 2) return s;   // no spread from a single replicate

    double ss = 0.0;
    for (double v : values) {
        const double d = v - mean;
        ss += d * d;
    }
    s.stddev   = std::sqrt(ss / (s.n - 1));
    s.stderror = s.stddev / std::sqrt(double(s.n));
    const double tcrit = tQuantile(0.5 * (1.0 + ciLevel), s.n - 1);
    s.ciHalf   = tcrit * s.stderror;
    return s;
}

} // namespace Sweep

// Local Variables:
// c-basic-offset: 4
// End:
