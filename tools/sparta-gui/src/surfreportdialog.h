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

#ifndef SURFREPORTDIALOG_H
#define SURFREPORTDIALOG_H

// GUI shell over surfreport.{h,cpp} (Feature 3): pick a per-surf compute or
// fix ave/surf from the running simulation, read its per-element array from the
// SPARTA library (extractCompute/extractFix), and report the integrated force /
// moment / heat flux, per-column sums and per-element distributions, with CSV
// export.  It is the first consumer of the wrapper's long-wired but unused
// per-surf extraction path.  Requires a running simulation with surfaces
// (extractSetting("surf_exist") == 1).

#include <QDialog>
#include <QStringList>
#include <QVector>

class QComboBox;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;

class SpartaWrapper;

class SurfReportDialog : public QDialog {
    Q_OBJECT

public:
    /// @param sparta running instance; @param deckText current editor deck (for
    /// auto-deriving value labels of the selected compute/fix).
    SurfReportDialog(QWidget *parent, SpartaWrapper *sparta, const QString &deckText);

private slots:
    void onSourceChanged();
    void computeReport();
    void exportCsv();

private:
    QStringList deriveLabels(const QString &source) const; // value labels from the deck

    SpartaWrapper *sparta_;
    QString deckText_;

    QComboBox *source_    = nullptr; ///< c_<id> / f_<id> per-surf sources
    QLineEdit *labels_    = nullptr; ///< editable value labels (auto-filled)
    QPushButton *csvBtn_  = nullptr;
    QPlainTextEdit *report_ = nullptr;

    // last computed data, retained for CSV export
    QStringList lastLabels_;
    QVector<QVector<double>> lastRows_;
};

#endif // SURFREPORTDIALOG_H

// Local Variables:
// c-basic-offset: 4
// End:
