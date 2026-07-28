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

#include "surfreportdialog.h"

#include "casemodel.h"    // reuse tokenize()
#include "helpers.h"
#include "spartawrapper.h"
#include "surfreport.h"

#include <QComboBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QTextStream>
#include <QVBoxLayout>

#include <cmath>

using SurfReport::Distribution;
using SurfReport::Totals;

SurfReportDialog::SurfReportDialog(QWidget *parent, SpartaWrapper *sparta,
                                   const QString &deckText)
    : QDialog(parent), sparta_(sparta), deckText_(deckText)
{
    setWindowTitle("Surface Quantities Report");
    resize(560, 520);

    auto *outer = new QVBoxLayout(this);

    auto *intro = new QLabel(
        "Integrate per-surface-element data (force, moment, heat flux, pressure, "
        "shear) from a <b>compute surf</b> or <b>fix ave/surf</b> in the running "
        "simulation.", this);
    intro->setWordWrap(true);
    outer->addWidget(intro);

    auto *form = new QFormLayout;
    source_ = new QComboBox(this);
    source_->setObjectName("source");
    source_->setAccessibleName("Per-surface compute or fix to report on");
    // per-surf sources: computes and fixes (the user picks; a non-surf pick is
    // reported as such when computed)
    const int nc = sparta_->idCount("compute");
    for (int i = 0; i < nc; ++i) source_->addItem("c_" + sparta_->idName("compute", i));
    const int nf = sparta_->idCount("fix");
    for (int i = 0; i < nf; ++i) source_->addItem("f_" + sparta_->idName("fix", i));
    form->addRow("Source:", source_);

    labels_ = new QLineEdit(this);
    labels_->setObjectName("labels");
    labels_->setAccessibleName("Column labels for the per-surface array");
    labels_->setPlaceholderText("comma-separated value labels, e.g. fx, fy, fz, etot");
    labels_->setToolTip("Column labels for the per-surf array; auto-filled from the "
                        "deck when possible.  fx/fy/fz -> force, tx/ty/tz -> moment, "
                        "etot (or ke) -> heat flux.");
    form->addRow("Value labels:", labels_);
    outer->addLayout(form);

    auto *btnRow = new QHBoxLayout;
    auto *computeBtn = new QPushButton("Compute Report", this);
    computeBtn->setObjectName("compute");
    csvBtn_ = new QPushButton("Export CSV...", this);
    csvBtn_->setObjectName("csv");
    csvBtn_->setEnabled(false);
    btnRow->addWidget(computeBtn);
    btnRow->addWidget(csvBtn_);
    btnRow->addStretch();
    outer->addLayout(btnRow);

    report_ = new QPlainTextEdit(this);
    report_->setObjectName("report");
    report_->setAccessibleName("Surface quantities report");
    report_->setReadOnly(true);
    report_->setLineWrapMode(QPlainTextEdit::NoWrap);
    outer->addWidget(report_, 1);

    connect(source_, &QComboBox::currentTextChanged, this, &SurfReportDialog::onSourceChanged);
    connect(computeBtn, &QPushButton::clicked, this, &SurfReportDialog::computeReport);
    connect(csvBtn_, &QPushButton::clicked, this, &SurfReportDialog::exportCsv);

    if (source_->count() > 0) onSourceChanged();
    else report_->setPlainText("No computes or fixes are defined in the running simulation.");
}

// Parse the deck to recover the value keywords a compute/fix tabulates, so the
// per-surf array columns can be labeled.  Best-effort: computes are read
// directly; a fix ave/surf that averages c_ID[*] resolves to that compute's
// values.  The user can always edit the labels field.
QStringList SurfReportDialog::deriveLabels(const QString &source) const
{
    if (source.size() < 3) return {};
    const QChar kind = source.at(0);        // 'c' or 'f'
    const QString id = source.mid(2);       // strip "c_"/"f_"
    // Per-surf array columns are values x mixture-groups; the group count is not
    // introspectable, so we label one column per value (correct for a
    // single-group mixture, the common case) and let the user extend the labels
    // field for multi-group mixtures.
    const int ngroup = 1;

    const QStringList lines = deckText_.split('\n');

    auto computeValues = [&](const QString &cid) -> QStringList {
        for (const QString &ln : lines) {
            const QStringList t = CaseModel::tokenize(ln);
            if (t.size() >= 6 && t.at(0) == "compute" && t.at(1) == cid && t.at(2) == "surf") {
                // compute ID surf group mix v1 v2 ... [norm flag]
                QStringList vals;
                for (int i = 5; i < t.size(); ++i) {
                    if (t.at(i) == "norm") break;
                    vals << t.at(i);
                }
                return vals;
            }
        }
        return {};
    };

    if (kind == 'c') {
        return SurfReport::expandColumnLabels(computeValues(id), ngroup);
    }

    // fix ave/surf: fix ID ave/surf group Nevery Nrepeat Nfreq value1 ...
    for (const QString &ln : lines) {
        const QStringList t = CaseModel::tokenize(ln);
        if (t.size() >= 7 && t.at(0) == "fix" && t.at(1) == id && t.at(2) == "ave/surf") {
            QStringList out;
            for (int i = 7; i < t.size(); ++i) {
                const QString v = t.at(i);
                if (v == "ave" || v == "one" || v == "running" || v == "window") break;
                // c_cid[*] / c_cid[N] / c_cid
                if (v.startsWith("c_")) {
                    QString cid = v.mid(2);
                    int col = -1;
                    const int lb = cid.indexOf('[');
                    if (lb >= 0) {
                        const QString idx = cid.mid(lb + 1).chopped(1); // inside [...]
                        cid = cid.left(lb);
                        if (idx != "*") col = idx.toInt();
                    }
                    const QStringList vals = SurfReport::expandColumnLabels(computeValues(cid), ngroup);
                    if (col > 0 && col <= vals.size()) out << vals.at(col - 1);
                    else out << vals;
                } else {
                    out << v;
                }
            }
            return out;
        }
    }
    return {};
}

void SurfReportDialog::onSourceChanged()
{
    labels_->setText(deriveLabels(source_->currentText()).join(", "));
}

void SurfReportDialog::computeReport()
{
    report_->clear();
    csvBtn_->setEnabled(false);
    lastRows_.clear();
    lastLabels_.clear();

    if (sparta_->extractSetting("surf_exist") != 1) {
        report_->setPlainText("No surfaces exist in the running simulation "
                              "(run a deck that reads a surface first).");
        return;
    }

    const QString source = source_->currentText();
    if (source.size() < 3) return;
    const bool isFix = source.startsWith("f_");
    const QString id = source.mid(2);

    QStringList labels;
    for (const QString &l : labels_->text().split(',', Qt::SkipEmptyParts))
        labels << l.trimmed();
    if (labels.isEmpty()) {
        report_->setPlainText("Enter the value labels for this source's columns "
                              "(could not auto-derive them from the deck).");
        return;
    }
    const int ncol = labels.size();

    int nrow = sparta_->extractSetting("nlocal_surf");
    if (nrow <= 0) nrow = sparta_->extractSetting("nsurf");
    if (nrow <= 0) {
        report_->setPlainText("The simulation reports zero surface elements.");
        return;
    }

    // Read the per-surf data.  A `compute surf` / `fix ave/surf` yields an ARRAY
    // whose columns are values x mixture-groups, so we try the array form first
    // and only fall back to a vector for genuinely single-column per-surf sources.
    // We read exactly `ncol` (label count) columns; since the deck-derived labels
    // list one entry per value (<= the actual value x group column count), this
    // never reads past the array's real width.
    QVector<QVector<double>> rows;
    rows.reserve(nrow);
    const int style = SpartaWrapper::SURF_STYLE;

    auto *a = static_cast<double **>(
        isFix ? sparta_->extractFix(id, style, SpartaWrapper::ARRAY_TYPE, nrow, ncol)
              : sparta_->extractCompute(id, style, SpartaWrapper::ARRAY_TYPE));
    if (a) {
        for (int i = 0; i < nrow; ++i) {
            QVector<double> r(ncol);
            for (int j = 0; j < ncol; ++j) r[j] = a[i][j];
            rows.push_back(r);
        }
    } else if (ncol == 1) {
        auto *v = static_cast<double *>(
            isFix ? sparta_->extractFix(id, style, SpartaWrapper::VECTOR_TYPE, nrow, 1)
                  : sparta_->extractCompute(id, style, SpartaWrapper::VECTOR_TYPE));
        if (!v) {
            report_->setPlainText(QString("'%1' is not a readable per-surface compute/fix.")
                                      .arg(source));
            return;
        }
        for (int i = 0; i < nrow; ++i) rows.push_back({v[i]});
    } else {
        report_->setPlainText(
            QString("'%1' did not return a per-surface array.\n"
                    "Check that it is a per-surf compute/fix and that the value labels "
                    "match its columns (values x mixture groups).")
                .arg(source));
        return;
    }

    const Totals t = SurfReport::integrate(labels, rows);
    lastLabels_ = labels;
    lastRows_   = rows;
    csvBtn_->setEnabled(true);

    // format the textual report
    QString out;
    out += QString("Source: %1   at timestep %2\n")
               .arg(source)
               .arg(qint64(sparta_->getThermo("step")));
    out += QString("Surface elements: %1\n\n").arg(t.nsurf);

    // A `compute surf` accumulates its tallies over a run and keeps them until
    // the next setup clears them -- and creating an image is a setup: the
    // render issues `run 0 pre yes post no` against the live instance, which
    // discards them.  So a report taken after a picture has been drawn reads
    // back as all zeros, and a table of zeros looks exactly like a simulation
    // in which nothing ever hit the surface.  The two are worth telling apart,
    // and the reader cannot do it from the numbers.  A `fix ave/surf` keeps its
    // own averaged copy and is unaffected, which is the way out.
    if (!isFix) {
        bool allZero = true;
        for (const QVector<double> &r : rows) {
            for (double v : r)
                if (v != 0.0) { allZero = false; break; }
            if (!allZero) break;
        }
        if (allZero)
            out += QString(
                       "Note: '%1' read back as all zeros. Either nothing struck the\n"
                       "surface, or the tallies have been cleared since the run -- "
                       "creating\nan image re-runs setup, which does that. A `fix ave/surf` "
                       "over this\ncompute keeps its own averaged copy; report on that "
                       "instead, or\nre-run the deck.\n\n")
                       .arg(source);
    }

    if (t.hasForce)
        out += QString("Integrated force   Fx=%1  Fy=%2  Fz=%3   |F|=%4\n")
                   .arg(t.force[0], 0, 'g', 6).arg(t.force[1], 0, 'g', 6)
                   .arg(t.force[2], 0, 'g', 6)
                   .arg(std::sqrt(t.force[0] * t.force[0] + t.force[1] * t.force[1] +
                                  t.force[2] * t.force[2]), 0, 'g', 6);
    if (t.hasMoment)
        out += QString("Integrated moment  Mx=%1  My=%2  Mz=%3\n")
                   .arg(t.moment[0], 0, 'g', 6).arg(t.moment[1], 0, 'g', 6)
                   .arg(t.moment[2], 0, 'g', 6);
    if (t.hasHeatFlux)
        out += QString("Total heat flux    Q=%1\n").arg(t.heatFlux, 0, 'g', 6);
    if (t.hasForce || t.hasMoment || t.hasHeatFlux) out += '\n';

    out += "Per-column summary:\n";
    out += QString("  %1  %2  %3  %4  %5  %6\n")
               .arg("column", -14).arg("sum", 14).arg("mean", 14)
               .arg("min", 14).arg("max", 14).arg("stddev", 14);
    for (int c = 0; c < labels.size(); ++c) {
        const Distribution d = SurfReport::distribution(SurfReport::column(rows, c));
        out += QString("  %1  %2  %3  %4  %5  %6\n")
                   .arg(labels.at(c), -14)
                   .arg(t.columnSum.value(c), 14, 'g', 6)
                   .arg(d.mean, 14, 'g', 6)
                   .arg(d.min, 14, 'g', 6)
                   .arg(d.max, 14, 'g', 6)
                   .arg(d.stddev, 14, 'g', 6);
    }
    report_->setPlainText(out);
}

void SurfReportDialog::exportCsv()
{
    if (lastRows_.isEmpty()) return;
    const QString fn = QFileDialog::getSaveFileName(this, "Export Per-Element CSV",
                                                    "surf_report.csv", "CSV files (*.csv)");
    if (fn.isEmpty()) return;
    QFile f(fn);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) return;
    QTextStream ts(&f);
    ts << SurfReport::toCsv(lastLabels_, lastRows_);
}
