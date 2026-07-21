/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Parametric sweep UI + driver.  SweepController chains local in-process
   runs (reusing SpartaGui::doRun and the variable-injection seam) one at a
   time, recording a chosen thermo quantity per parameter combination.
   SweepResultsModel tabulates the results; SweepPanel is the docked control.

   The pure expansion/reducers live in sweepspec.{h,cpp}; this is the
   GUI-coupled glue.  It respects the single-instance in-process constraint
   by only advancing from SpartaGui::runFinished (one run in flight at a time).
------------------------------------------------------------------------- */

#ifndef SWEEPPANEL_H
#define SWEEPPANEL_H

#include "sweepspec.h"

#include <QAbstractTableModel>
#include <QObject>
#include <QVector>
#include <QWidget>

#include <vector>

class SpartaGui;
class SpartaWrapper;
class QComboBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QRadioButton;
class QSpinBox;
class QTableView;
class QTableWidget;

// ---------------------------------------------------------------------------

/** @brief Live-filling results table: swept-variable columns + quantity columns. */
class SweepResultsModel : public QAbstractTableModel {
    Q_OBJECT
public:
    explicit SweepResultsModel(QObject *parent = nullptr);
    void reset(const QStringList &headers);
    void addRow(const QStringList &cells, bool ok);
    void clearRows();

    int rowCount(const QModelIndex &parent = {}) const override;
    int columnCount(const QModelIndex &parent = {}) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation o, int role = Qt::DisplayRole) const override;

    QStringList headers() const { return headers_; }
    const QVector<QStringList> &rows() const { return rows_; }

private:
    QStringList headers_;
    QVector<QStringList> rows_;
    QVector<bool> ok_;
};

// ---------------------------------------------------------------------------

/** @brief Drives a sweep as a queue of sequential in-process runs. */
class SweepController : public QObject {
    Q_OBJECT
public:
    SweepController(SpartaGui *gui, SpartaWrapper *sparta, SweepResultsModel *model,
                    QObject *parent = nullptr);

    void start(const Sweep::SweepSpec &spec);
    void stop();
    bool active() const { return active_; }

signals:
    void progress(int done, int total);
    void finished(bool completed);

private slots:
    void onRunFinished(bool success);
    void onSample();

private:
    void launchNext();
    double readThermo(const QString &keyword) const; // current cached value, NaN if absent

    SpartaGui *gui_;
    SpartaWrapper *sparta_;
    SweepResultsModel *model_;
    Sweep::SweepSpec spec_;
    QList<QList<QPair<QString, QString>>> combos_;
    int index_ = -1;
    int repIndex_ = 0;                       // replicate within the current sweep point
    bool active_ = false;
    bool stopRequested_ = false;
    QVector<std::vector<double>> samples_;    // per quantity, this run
    QVector<std::vector<double>> repVals_;    // per quantity, reduced value of each replicate
};

// ---------------------------------------------------------------------------

/** @brief Docked panel to define, run, and tabulate a parametric sweep. */
class SweepPanel : public QWidget {
    Q_OBJECT
public:
    SweepPanel(QWidget *parent, SpartaGui *gui, SpartaWrapper *sparta);

private slots:
    void addVariableRow();
    void removeVariableRow();
    void refreshVariables();
    void startSweep();
    void onProgress(int done, int total);
    void onFinished(bool completed);
    void exportCsv();
    void chartResults();

private:
    bool buildSpec(Sweep::SweepSpec &spec, QString &err) const;

    SpartaGui *gui_;
    SpartaWrapper *sparta_;
    SweepResultsModel *model_;
    SweepController *controller_;

    QTableWidget *varTable_ = nullptr;
    QRadioButton *cartesian_ = nullptr;
    QLineEdit *quantities_ = nullptr;
    QComboBox *reducer_ = nullptr;
    QSpinBox *replicates_ = nullptr;
    QLineEdit *seedVar_ = nullptr;
    QSpinBox *seedBase_ = nullptr;
    QPushButton *startBtn_ = nullptr;
    QProgressBar *progress_ = nullptr;
    QLabel *status_ = nullptr;
    QTableView *results_ = nullptr;
    QStringList discovered_;
};

#endif // SWEEPPANEL_H
