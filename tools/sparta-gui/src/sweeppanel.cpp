/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Parametric sweep UI + driver.  See sweeppanel.h.
------------------------------------------------------------------------- */

#include "sweeppanel.h"

#include "chartviewer.h"
#include "helpers.h"
#include "plotdata.h"
#include "spartagui.h"
#include "spartawrapper.h"

#include <QComboBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QSpinBox>
#include <QTableView>
#include <QTableWidget>
#include <QTextStream>
#include <QVBoxLayout>

#include <cmath>

using namespace Sweep;

// ===========================================================================
// SweepResultsModel
// ===========================================================================

SweepResultsModel::SweepResultsModel(QObject *parent) : QAbstractTableModel(parent) {}

void SweepResultsModel::reset(const QStringList &headers)
{
    beginResetModel();
    headers_ = headers;
    rows_.clear();
    ok_.clear();
    endResetModel();
}

void SweepResultsModel::addRow(const QStringList &cells, bool ok)
{
    beginInsertRows({}, rows_.size(), rows_.size());
    rows_.append(cells);
    ok_.append(ok);
    endInsertRows();
}

void SweepResultsModel::clearRows()
{
    beginResetModel();
    rows_.clear();
    ok_.clear();
    endResetModel();
}

int SweepResultsModel::rowCount(const QModelIndex &p) const { return p.isValid() ? 0 : rows_.size(); }
int SweepResultsModel::columnCount(const QModelIndex &p) const
{
    return p.isValid() ? 0 : headers_.size();
}

QVariant SweepResultsModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid() || index.row() >= rows_.size()) return {};
    if (role == Qt::DisplayRole) {
        const QStringList &r = rows_.at(index.row());
        return index.column() < r.size() ? r.at(index.column()) : QString();
    }
    if (role == Qt::ForegroundRole && index.row() < ok_.size() && !ok_.at(index.row()))
        return QColor(Qt::red);
    return {};
}

QVariant SweepResultsModel::headerData(int section, Qt::Orientation o, int role) const
{
    if (role != Qt::DisplayRole) return {};
    if (o == Qt::Horizontal)
        return section < headers_.size() ? headers_.at(section) : QString();
    return section + 1;
}

// ===========================================================================
// SweepController
// ===========================================================================

SweepController::SweepController(SpartaGui *gui, SpartaWrapper *sparta,
                                 SweepResultsModel *model, QObject *parent)
    : QObject(parent), gui_(gui), sparta_(sparta), model_(model)
{
}

void SweepController::start(const SweepSpec &spec)
{
    QString err;
    combos_ = spec.expand(&err);
    if (combos_.isEmpty()) {
        emit finished(false);
        return;
    }
    spec_ = spec;
    if (spec_.replicates < 1) spec_.replicates = 1;
    samples_.resize(spec_.quantities.size());
    repVals_.resize(spec_.quantities.size());

    // results headers: swept variables, then quantity(reducer).  With replicates,
    // each quantity becomes a mean plus a standard-error column.
    const bool ensemble = spec_.replicates > 1;
    QStringList headers;
    for (const auto &v : spec_.vars) headers << v.name;
    for (int q = 0; q < spec_.quantities.size(); ++q) {
        const QString base = QString("%1 (%2)").arg(spec_.quantities.at(q),
                                                    reducerName(spec_.reducerFor(q)));
        if (ensemble) {
            headers << base + " mean";
            headers << base + " +/-SE";
        } else {
            headers << base;
        }
    }
    model_->reset(headers);

    index_ = -1;
    repIndex_ = 0;
    active_ = true;
    stopRequested_ = false;
    connect(gui_, &SpartaGui::runFinished, this, &SweepController::onRunFinished);
    connect(gui_, &SpartaGui::thermoSampled, this, &SweepController::onSample);
    emit progress(0, combos_.size() * spec_.replicates);
    launchNext();
}

void SweepController::stop()
{
    if (!active_) return;
    stopRequested_ = true;
    gui_->stopRun(); // cooperative; the ensuing runFinished ends the sweep
}

void SweepController::launchNext()
{
    // advance the (sweep point, replicate) cursor: replicates run back-to-back,
    // then the sweep point advances
    if (index_ < 0) {
        index_ = 0;
        repIndex_ = 0;
    } else if (++repIndex_ >= spec_.replicates) {
        repIndex_ = 0;
        ++index_;
    }

    const int total = combos_.size() * spec_.replicates;
    emit progress(index_ * spec_.replicates + repIndex_, total);

    if (index_ >= combos_.size()) {
        active_ = false;
        disconnect(gui_, &SpartaGui::runFinished, this, &SweepController::onRunFinished);
        disconnect(gui_, &SpartaGui::thermoSampled, this, &SweepController::onSample);
        emit finished(!stopRequested_);
        return;
    }

    if (repIndex_ == 0)
        for (auto &r : repVals_) r.clear();   // a fresh sweep point
    for (auto &s : samples_) s.clear();

    // build this run's variable assignment: the swept values plus, for an
    // ensemble, a distinct seed for this replicate
    auto assign = combos_.at(index_);
    if (spec_.replicates > 1 && !spec_.seedVariable.isEmpty())
        assign.append({spec_.seedVariable, QString::number(spec_.seedFor(repIndex_))});
    gui_->setRunVariables(assign);
    gui_->runBuffer();
}

double SweepController::readThermo(const QString &keyword) const
{
    const int num = sparta_->lastThermoAs<int>("num", 0);
    for (int i = 0; i < num; ++i) {
        if (sparta_->lastThermoString("keyword", i) == keyword) {
            const int dt = sparta_->lastThermoAs<int>("type", i);
            if (dt == 0) return sparta_->lastThermoAs<int>("data", i);
            if (dt == 2) return sparta_->lastThermoAs<double>("data", i);
            if (dt == 4) return static_cast<double>(sparta_->lastThermoAs<int64_t>("data", i));
            return 0.0;
        }
    }
    return std::nan("");
}

void SweepController::onSample()
{
    if (!active_ || !sparta_->isRunning()) return;
    sparta_->lastThermo("lock", 0);
    for (int q = 0; q < spec_.quantities.size(); ++q) {
        const double v = readThermo(spec_.quantities.at(q));
        if (!std::isnan(v)) samples_[q].push_back(v);
    }
    sparta_->lastThermo("unlock", 0);
}

void SweepController::onRunFinished(bool success)
{
    if (!active_) return;
    if (index_ < 0 || index_ >= combos_.size()) return;

    // reduce this run to one value per quantity and stash it for the replicate set
    for (int q = 0; q < spec_.quantities.size(); ++q) {
        const Reducer red = spec_.reducerFor(q);
        const double val = (red == Reducer::Final) ? readThermo(spec_.quantities.at(q))
                                                   : reduce(red, samples_[q]);
        if (!std::isnan(val)) repVals_[q].push_back(val);
    }

    // emit a results row once the sweep point's replicates are all done
    const bool pointDone = (repIndex_ + 1 >= spec_.replicates);
    if (pointDone) {
        QStringList cells;
        for (const auto &kv : combos_.at(index_)) cells << kv.second;
        for (int q = 0; q < spec_.quantities.size(); ++q) {
            if (spec_.replicates > 1) {
                const EnsembleStats es = ensembleStats(repVals_[q], spec_.ciLevel);
                cells << (es.n > 0 ? QString::number(es.mean, 'g', 10) : QString("n/a"));
                cells << (es.n > 1 ? QString::number(es.stderror, 'g', 6) : QString("-"));
            } else {
                const double v = repVals_[q].empty() ? std::nan("") : repVals_[q].back();
                cells << (std::isnan(v) ? QString("n/a") : QString::number(v, 'g', 10));
            }
        }
        model_->addRow(cells, success && !stopRequested_);
    }

    if (stopRequested_) {
        active_ = false;
        disconnect(gui_, &SpartaGui::runFinished, this, &SweepController::onRunFinished);
        disconnect(gui_, &SpartaGui::thermoSampled, this, &SweepController::onSample);
        emit finished(false);
        return;
    }
    launchNext();
}

// ===========================================================================
// SweepPanel
// ===========================================================================

SweepPanel::SweepPanel(QWidget *parent, SpartaGui *gui, SpartaWrapper *sparta)
    : QWidget(parent), gui_(gui), sparta_(sparta)
{
    model_ = new SweepResultsModel(this);
    controller_ = new SweepController(gui_, sparta_, model_, this);
    connect(controller_, &SweepController::progress, this, &SweepPanel::onProgress);
    connect(controller_, &SweepController::finished, this, &SweepPanel::onFinished);

    auto *outer = new QVBoxLayout(this);
    outer->setContentsMargins(4, 4, 4, 4);

    auto *intro = new QLabel(
        "Vary index variables over ranges and tabulate a thermo quantity per run. "
        "Each combination runs the editor buffer in-process, one at a time.", this);
    intro->setWordWrap(true);
    outer->addWidget(intro);

    // variables table
        // Named after what they hold.  Counting widgets by type instead is a
    // trap here: the variable-name cells are editable combo boxes, and each
    // of those embeds a QLineEdit that turns up in findChildren<QLineEdit*>
    // ahead of the panel's own fields.
varTable_ = new QTableWidget(0, 3, this);
    varTable_->setObjectName("varTable");
    varTable_->setAccessibleName("Swept variables");
    varTable_->setHorizontalHeaderLabels({"Variable", "Type", "Specification"});
    varTable_->horizontalHeader()->setStretchLastSection(true);
    varTable_->verticalHeader()->setVisible(false);
    varTable_->setMaximumHeight(150);
    outer->addWidget(varTable_);

    auto *vrow = new QHBoxLayout;
    auto *addBtn = new QPushButton("Add Variable", this);
    auto *rmBtn = new QPushButton("Remove", this);
    auto *refBtn = new QPushButton("Detect from Deck", this);
    vrow->addWidget(addBtn);
    vrow->addWidget(rmBtn);
    vrow->addWidget(refBtn);
    vrow->addStretch();
    cartesian_ = new QRadioButton("Cartesian product", this);
    cartesian_->setObjectName("cartesian");
    cartesian_->setAccessibleName("Cartesian product of the variables");
    auto *zip = new QRadioButton("Zip (parallel)", this);
    zip->setObjectName("zip");
    zip->setAccessibleName("Pair the variables elementwise");
    cartesian_->setChecked(true);
    vrow->addWidget(cartesian_);
    vrow->addWidget(zip);
    outer->addLayout(vrow);

    // quantities + reducer
    auto *qrow = new QHBoxLayout;
    qrow->addWidget(new QLabel("Tabulate:", this));
    quantities_ = new QLineEdit(this);
    quantities_->setObjectName("quantities");
    quantities_->setAccessibleName("Thermo quantities to tabulate");
    quantities_->setPlaceholderText("thermo keywords, comma-separated (e.g. Np, c_temp)");
    qrow->addWidget(quantities_, 1);
    qrow->addWidget(new QLabel("Reduce:", this));
    reducer_ = new QComboBox(this);
    reducer_->setObjectName("reducer");
    reducer_->setAccessibleName("How each run is reduced to one number");
    reducer_->addItem("final value", static_cast<int>(Reducer::Final));
    reducer_->addItem("minimum", static_cast<int>(Reducer::Min));
    reducer_->addItem("maximum", static_cast<int>(Reducer::Max));
    reducer_->addItem("mean", static_cast<int>(Reducer::Mean));
    qrow->addWidget(reducer_);
    outer->addLayout(qrow);

    // ensemble (replicate) controls: run each sweep point N times with a fresh
    // seed and report mean +/- standard error
    auto *erow2 = new QHBoxLayout;
    erow2->addWidget(new QLabel("Replicates:", this));
    replicates_ = new QSpinBox(this);
    replicates_->setObjectName("replicates");
    replicates_->setAccessibleName("Runs per combination");
    replicates_->setRange(1, 1000);
    replicates_->setValue(1);
    replicates_->setToolTip("Runs per sweep point, each with a distinct seed "
                            "(1 = no ensemble). Results become mean +/- std. error.");
    erow2->addWidget(replicates_);
    erow2->addWidget(new QLabel("Seed variable:", this));
    seedVar_ = new QLineEdit(this);
    seedVar_->setObjectName("seedVar");
    seedVar_->setAccessibleName("Variable holding the per-replicate seed");
    seedVar_->setPlaceholderText("e.g. seed  (referenced as ${seed} in the deck)");
    erow2->addWidget(seedVar_, 1);
    erow2->addWidget(new QLabel("Base seed:", this));
    seedBase_ = new QSpinBox(this);
    seedBase_->setObjectName("seedBase");
    seedBase_->setAccessibleName("First seed value");
    seedBase_->setRange(1, 2000000000);
    seedBase_->setValue(12345);
    erow2->addWidget(seedBase_);
    outer->addLayout(erow2);

    // run controls
    auto *crow = new QHBoxLayout;
    startBtn_ = new QPushButton("Run Sweep", this);
    startBtn_->setObjectName("startSweep");
    startBtn_->setAccessibleName("Run or stop the sweep");
    crow->addWidget(startBtn_);
    progress_ = new QProgressBar(this);
    progress_->setObjectName("progress");
    progress_->setAccessibleName("Sweep progress");
    progress_->setObjectName("progress");
    progress_->setAccessibleName("Sweep progress");
    crow->addWidget(progress_, 1);
    status_ = new QLabel("Idle.", this);
    status_->setObjectName("status");
    status_->setAccessibleName("Sweep status");
    crow->addWidget(status_);
    outer->addLayout(crow);

    // results
    results_ = new QTableView(this);
    results_->setModel(model_);
    results_->horizontalHeader()->setStretchLastSection(true);
    outer->addWidget(results_, 1);

    auto *erow = new QHBoxLayout;
    auto *csvBtn = new QPushButton("Export CSV...", this);
    auto *chartBtn = new QPushButton("Chart Results", this);
    erow->addStretch();
    erow->addWidget(csvBtn);
    erow->addWidget(chartBtn);
    outer->addLayout(erow);

    connect(addBtn, &QPushButton::clicked, this, &SweepPanel::addVariableRow);
    connect(rmBtn, &QPushButton::clicked, this, &SweepPanel::removeVariableRow);
    connect(refBtn, &QPushButton::clicked, this, &SweepPanel::refreshVariables);
    connect(startBtn_, &QPushButton::clicked, this, &SweepPanel::startSweep);
    connect(csvBtn, &QPushButton::clicked, this, &SweepPanel::exportCsv);
    connect(chartBtn, &QPushButton::clicked, this, &SweepPanel::chartResults);

    refreshVariables();
}

void SweepPanel::refreshVariables()
{
    discovered_.clear();
    // the panel is constructible without a main window (the results model and
    // the spec builder need neither), so ask only when there is one to ask
    const auto vars = gui_ ? gui_->discoverVariables() : QList<QPair<QString, QString>>{};
    for (const auto &kv : vars)
        if (!kv.first.isEmpty()) discovered_ << kv.first;
    if (varTable_->rowCount() == 0 && !discovered_.isEmpty()) addVariableRow();
}

void SweepPanel::addVariableRow()
{
    const int r = varTable_->rowCount();
    varTable_->insertRow(r);

    auto *nameCombo = new QComboBox(varTable_);
    nameCombo->setEditable(true);
    nameCombo->addItems(discovered_);
    varTable_->setCellWidget(r, 0, nameCombo);

    auto *typeCombo = new QComboBox(varTable_);
    typeCombo->addItems({"List", "Range", "Linspace"});
    varTable_->setCellWidget(r, 1, typeCombo);

    varTable_->setItem(r, 2, new QTableWidgetItem(""));
    // hint the spec format
    varTable_->item(r, 2)->setToolTip(
        "List: 1, 2, 3   |   Range: start:stop:step   |   Linspace: start:stop:count");
}

void SweepPanel::removeVariableRow()
{
    const int r = varTable_->currentRow();
    if (r >= 0) varTable_->removeRow(r);
}

bool SweepPanel::buildSpec(SweepSpec &spec, QString &err) const
{
    spec.vars.clear();
    for (int r = 0; r < varTable_->rowCount(); ++r) {
        auto *nameCombo = qobject_cast<QComboBox *>(varTable_->cellWidget(r, 0));
        auto *typeCombo = qobject_cast<QComboBox *>(varTable_->cellWidget(r, 1));
        auto *specItem = varTable_->item(r, 2);
        if (!nameCombo || !typeCombo) continue;
        const QString name = nameCombo->currentText().trimmed();
        const QString spectext = specItem ? specItem->text().trimmed() : QString();
        if (name.isEmpty()) continue;

        VarSweep v;
        v.name = name;
        const QString type = typeCombo->currentText();
        if (type == "List") {
            v.kind = VarSweep::List;
            for (const QString &p : spectext.split(',', Qt::SkipEmptyParts))
                v.values << p.trimmed();
            if (v.values.isEmpty()) { err = QString("Variable '%1': empty value list.").arg(name); return false; }
        } else {
            const QStringList parts = spectext.split(':');
            if (parts.size() != 3) {
                err = QString("Variable '%1': use start:stop:%2.")
                          .arg(name, type == "Range" ? "step" : "count");
                return false;
            }
            v.start = parts.at(0).toDouble();
            v.stop = parts.at(1).toDouble();
            if (type == "Range") { v.kind = VarSweep::Range; v.step = parts.at(2).toDouble(); }
            else { v.kind = VarSweep::Linspace; v.count = parts.at(2).toInt(); }
        }
        spec.vars << v;
    }
    if (spec.vars.isEmpty()) { err = "Add at least one variable to sweep."; return false; }

    spec.combine = cartesian_->isChecked() ? Combine::Cartesian : Combine::Zip;
    for (const QString &q : quantities_->text().split(',', Qt::SkipEmptyParts))
        spec.quantities << q.trimmed();
    if (spec.quantities.isEmpty()) { err = "Enter at least one thermo quantity to tabulate."; return false; }
    const auto red = static_cast<Reducer>(reducer_->currentData().toInt());
    for (int i = 0; i < spec.quantities.size(); ++i) spec.reducers << red;

    // ensemble options
    spec.replicates   = replicates_->value();
    spec.seedVariable = seedVar_->text().trimmed();
    spec.seedBase     = seedBase_->value();
    if (spec.replicates > 1 && spec.seedVariable.isEmpty()) {
        err = "Enter a seed variable name to run replicates with distinct seeds "
              "(and reference it as ${name} in the deck).";
        return false;
    }

    QString experr;
    if (spec.expand(&experr).isEmpty()) { err = experr; return false; }
    return true;
}

void SweepPanel::startSweep()
{
    if (controller_->active()) { controller_->stop(); return; }
    if (sparta_ && sparta_->isRunning()) {
        QMessageBox::warning(this, "Parametric Sweep",
                             "A simulation is already running; stop it first.");
        return;
    }
    SweepSpec spec;
    QString err;
    if (!buildSpec(spec, err)) {
        QMessageBox::warning(this, "Parametric Sweep", err);
        return;
    }
    startBtn_->setText("Stop Sweep");
    controller_->start(spec);
}

void SweepPanel::onProgress(int done, int total)
{
    progress_->setRange(0, total);
    progress_->setValue(done);
    status_->setText(QString("Run %1 / %2").arg(qMin(done + 1, total)).arg(total));
}

void SweepPanel::onFinished(bool completed)
{
    startBtn_->setText("Run Sweep");
    status_->setText(completed ? "Sweep complete." : "Sweep stopped.");
    progress_->setValue(progress_->maximum());
}

void SweepPanel::exportCsv()
{
    if (model_->rows().isEmpty()) {
        QMessageBox::information(this, "Export CSV", "No results to export yet.");
        return;
    }
    const QString fn = QFileDialog::getSaveFileName(this, "Export Sweep Results",
                                                    "sweep.csv", "CSV files (*.csv)");
    if (fn.isEmpty()) return;
    QFile f(fn);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) return;
    QTextStream ts(&f);
    ts << model_->headers().join(',') << '\n';
    for (const auto &row : model_->rows()) ts << row.join(',') << '\n';
}

void SweepPanel::chartResults()
{
    const auto &rows = model_->rows();
    const QStringList headers = model_->headers();
    if (rows.isEmpty() || headers.size() < 2) {
        QMessageBox::information(this, "Chart Results", "Not enough numeric results to chart.");
        return;
    }
    // keep rows whose every cell is numeric; chart columns 1.. vs column 0
    const int ncol = headers.size();
    std::vector<std::vector<double>> cols(ncol);
    for (const auto &row : rows) {
        if (row.size() < ncol) continue;
        bool allNum = true;
        std::vector<double> vals(ncol);
        for (int c = 0; c < ncol; ++c) {
            bool ok = false;
            vals[c] = row.at(c).toDouble(&ok);
            if (!ok) { allNum = false; break; }
        }
        if (!allNum) continue;
        for (int c = 0; c < ncol; ++c) cols[c].push_back(vals[c]);
    }
    if (cols[0].empty()) {
        QMessageBox::information(this, "Chart Results",
                                 "No fully-numeric rows to chart (the x variable must be numeric).");
        return;
    }
    PlotData data;
    data.setColumnNames(headers);
    for (int c = 0; c < ncol; ++c) data.addColumn(headers.at(c), cols[c]);
    QList<int> ycols;
    // plot the value/mean columns; skip the ensemble standard-error columns so
    // they don't appear as spurious flat lines (they annotate the mean instead)
    for (int c = 1; c < ncol; ++c)
        if (!headers.at(c).endsWith("+/-SE")) ycols << c;

    auto *win = new ChartWindow("Sweep Results", nullptr);
    win->setAttribute(Qt::WA_DeleteOnClose);
    win->loadData(data, 0, ycols);
    win->show();
}

// Local Variables:
// c-basic-offset: 4
// End:
