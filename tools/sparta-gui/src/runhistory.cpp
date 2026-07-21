/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Run history / provenance controller + docked panel.  See runhistory.h.
------------------------------------------------------------------------- */

#include "runhistory.h"

#include "runcompare.h"

#include <QItemSelectionModel>

#include <algorithm>

#include "constants.h"

#include <QCheckBox>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QMessageBox>
#include <QPageSize>
#include <QPdfWriter>
#include <QPushButton>
#include <QSettings>
#include <QStandardPaths>
#include <QTableView>
#include <QTextDocument>
#include <QUrl>
#include <QVBoxLayout>

using RunArchive::RunRecord;

// ===========================================================================
// HistoryModel
// ===========================================================================

HistoryModel::HistoryModel(RunHistory *hist, QObject *parent)
    : QAbstractTableModel(parent), hist_(hist)
{
}

int HistoryModel::rowCount(const QModelIndex &p) const { return p.isValid() ? 0 : hist_->count(); }
int HistoryModel::columnCount(const QModelIndex &p) const { return p.isValid() ? 0 : NCols; }

QVariant HistoryModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid() || index.row() >= hist_->count() || role != Qt::DisplayRole) return {};
    const RunRecord &r = hist_->at(index.row());
    switch (index.column()) {
    case ColTime:   return r.timestamp;
    case ColDeck:   return r.deckName;
    case ColStatus: return r.status;
    case ColImages: return r.imageFiles.size();
    default:        return {};
    }
}

QVariant HistoryModel::headerData(int section, Qt::Orientation o, int role) const
{
    if (role != Qt::DisplayRole || o != Qt::Horizontal) return {};
    switch (section) {
    case ColTime:   return "Finished";
    case ColDeck:   return "Deck";
    case ColStatus: return "Status";
    case ColImages: return "Images";
    default:        return {};
    }
}

void HistoryModel::refresh()
{
    beginResetModel();
    endResetModel();
}

// ===========================================================================
// RunHistory
// ===========================================================================

RunHistory::RunHistory(QObject *parent) : QObject(parent)
{
    model_ = new HistoryModel(this, this);
    load();
}

QString RunHistory::baseDir() const
{
    const QString d = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/history";
    QDir().mkpath(d);
    return d;
}

QString RunHistory::indexFile() const { return baseDir() + "/runs.json"; }
QString RunHistory::recordDir(int row) const
{
    return RunArchive::runArchiveDir(baseDir(), records_.at(row).id);
}

void RunHistory::archive(const RunRecord &recIn, const QStringList &images)
{
    RunRecord rec = recIn;
    const QString dir = RunArchive::runArchiveDir(baseDir(), rec.id);
    QDir().mkpath(dir);

    // copy the images into the archive and rewrite the record's paths to the copies
    QStringList archived;
    int n = 0;
    for (const QString &src : images) {
        if (!QFileInfo::exists(src)) continue;
        const QString dst = QString("%1/image_%2_%3")
                                .arg(dir).arg(n++, 3, 10, QChar('0'))
                                .arg(QFileInfo(src).fileName());
        if (QFile::copy(src, dst)) archived << dst;
    }
    rec.imageFiles = archived;

    records_.prepend(rec);
    save();
    model_->refresh();
    emit changed();
    emit message(QString("Archived run '%1' (%2 image(s)).").arg(rec.deckName).arg(archived.size()));
}

QString RunHistory::writeReportHtml(int row)
{
    if (row < 0 || row >= records_.size()) return {};
    const RunRecord &r = records_.at(row);
    QMap<QString, QByteArray> imgs;
    for (const QString &p : r.imageFiles) {
        QFile f(p);
        if (f.open(QIODevice::ReadOnly)) imgs.insert(p, f.readAll());
    }
    const QString html = RunArchive::buildRunReportHtml(r, imgs);
    const QString path = recordDir(row) + "/report.html";
    QFile out(path);
    if (!out.open(QIODevice::WriteOnly | QIODevice::Text)) return {};
    out.write(html.toUtf8());
    return path;
}

QString RunHistory::writeComparisonHtml(int rowA, int rowB)
{
    if (rowA < 0 || rowA >= records_.size() || rowB < 0 || rowB >= records_.size())
        return {};
    const RunRecord &a = records_.at(rowA);
    const RunRecord &b = records_.at(rowB);

    auto loadImages = [](const RunRecord &r) {
        QMap<QString, QByteArray> imgs;
        for (const QString &p : r.imageFiles) {
            QFile f(p);
            if (f.open(QIODevice::ReadOnly)) imgs.insert(p, f.readAll());
        }
        return imgs;
    };

    const QString html =
        RunCompare::buildComparisonHtml(a, b, loadImages(a), loadImages(b));
    const QString path = recordDir(rowA) + "/compare.html";
    QFile out(path);
    if (!out.open(QIODevice::WriteOnly | QIODevice::Text)) return {};
    out.write(html.toUtf8());
    return path;
}

QString RunHistory::writeReportPdf(int row)
{
    if (row < 0 || row >= records_.size()) return {};
    const RunRecord &r = records_.at(row);
    QMap<QString, QByteArray> imgs;
    for (const QString &p : r.imageFiles) {
        QFile f(p);
        if (f.open(QIODevice::ReadOnly)) imgs.insert(p, f.readAll());
    }
    const QString html = RunArchive::buildRunReportHtml(r, imgs);
    const QString path = recordDir(row) + "/report.pdf";

    QTextDocument doc;
    doc.setHtml(html);
    QPdfWriter writer(path);
    writer.setPageSize(QPageSize(QPageSize::A4));
    doc.setPageSize(QSizeF(writer.width(), writer.height()));
    doc.print(&writer);
    return QFileInfo::exists(path) ? path : QString();
}

void RunHistory::removeRecord(int row)
{
    if (row < 0 || row >= records_.size()) return;
    QDir(recordDir(row)).removeRecursively();
    records_.removeAt(row);
    save();
    model_->refresh();
    emit changed();
}

void RunHistory::save() const
{
    QJsonArray arr;
    for (const auto &r : records_) arr.append(r.toJson());
    QFile f(indexFile());
    if (f.open(QIODevice::WriteOnly))
        f.write(QJsonDocument(arr).toJson(QJsonDocument::Indented));
}

void RunHistory::load()
{
    QFile f(indexFile());
    if (!f.open(QIODevice::ReadOnly)) return;
    const auto arr = QJsonDocument::fromJson(f.readAll()).array();
    for (const auto &v : arr) records_.append(RunRecord::fromJson(v.toObject()));
}

// ===========================================================================
// HistoryPanel
// ===========================================================================

HistoryPanel::HistoryPanel(QWidget *parent, RunHistory *hist) : QWidget(parent), hist_(hist)
{
    auto *outer = new QVBoxLayout(this);
    outer->setContentsMargins(4, 4, 4, 4);

    auto *tb = new QHBoxLayout;
    auto *htmlBtn = new QPushButton("Report (HTML)", this);
    auto *pdfBtn = new QPushButton("Report (PDF)", this);
    compareBtn_ = new QPushButton("Compare (2)", this);
    compareBtn_->setToolTip("Select exactly two runs to diff their decks, metadata and images");
    compareBtn_->setEnabled(false);
    auto *openBtn = new QPushButton("Open Folder", this);
    auto *delBtn = new QPushButton("Delete", this);
    tb->addWidget(htmlBtn);
    tb->addWidget(pdfBtn);
    tb->addWidget(compareBtn_);
    tb->addWidget(openBtn);
    tb->addWidget(delBtn);
    tb->addStretch();
    outer->addLayout(tb);

    table_ = new QTableView(this);
    table_->setModel(hist_->model());
    table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    // allow selecting two runs for a comparison
    table_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    table_->horizontalHeader()->setStretchLastSection(true);
    table_->verticalHeader()->setVisible(false);
    outer->addWidget(table_, 1);

    auto *archiveChk = new QCheckBox(
        "Archive finished runs here automatically (deck, log, thermo, images)", this);
    archiveChk->setChecked(QSettings().value(Keys::ARCHIVE_RUNS, false).toBool());
    connect(archiveChk, &QCheckBox::toggled, this,
            [](bool on) { QSettings().setValue(Keys::ARCHIVE_RUNS, on); });
    outer->addWidget(archiveChk);

    connect(htmlBtn, &QPushButton::clicked, this, &HistoryPanel::reportHtml);
    connect(pdfBtn, &QPushButton::clicked, this, &HistoryPanel::reportPdf);
    connect(compareBtn_, &QPushButton::clicked, this, &HistoryPanel::compareSelected);
    connect(openBtn, &QPushButton::clicked, this, &HistoryPanel::openFolder);
    connect(delBtn, &QPushButton::clicked, this, &HistoryPanel::deleteSelected);
    // enable Compare only when exactly two runs are selected
    connect(table_->selectionModel(), &QItemSelectionModel::selectionChanged, this,
            [this]() { compareBtn_->setEnabled(selectedRows().size() == 2); });
}

int HistoryPanel::selectedRow() const
{
    const auto rows = table_->selectionModel()->selectedRows();
    return rows.isEmpty() ? -1 : rows.first().row();
}

QList<int> HistoryPanel::selectedRows() const
{
    QList<int> out;
    for (const auto &idx : table_->selectionModel()->selectedRows()) out << idx.row();
    std::sort(out.begin(), out.end());
    return out;
}

void HistoryPanel::compareSelected()
{
    const QList<int> rows = selectedRows();
    if (rows.size() != 2) {
        QMessageBox::information(this, "Compare Runs", "Select exactly two runs to compare.");
        return;
    }
    // the table lists newest first; compare older (A) against newer (B)
    const QString p = hist_->writeComparisonHtml(rows.at(1), rows.at(0));
    if (!p.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(p));
    else QMessageBox::warning(this, "Compare Runs", "Could not write the comparison report.");
}

void HistoryPanel::reportHtml()
{
    const int r = selectedRow();
    if (r < 0) return;
    const QString p = hist_->writeReportHtml(r);
    if (!p.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(p));
    else QMessageBox::warning(this, "Report", "Could not write the HTML report.");
}

void HistoryPanel::reportPdf()
{
    const int r = selectedRow();
    if (r < 0) return;
    const QString p = hist_->writeReportPdf(r);
    if (!p.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(p));
    else QMessageBox::warning(this, "Report", "Could not write the PDF report.");
}

void HistoryPanel::openFolder()
{
    const int r = selectedRow();
    if (r >= 0) QDesktopServices::openUrl(QUrl::fromLocalFile(hist_->recordDir(r)));
}

void HistoryPanel::deleteSelected()
{
    const int r = selectedRow();
    if (r < 0) return;
    if (QMessageBox::question(this, "Delete Run",
                              "Delete this archived run and its files?") == QMessageBox::Yes)
        hist_->removeRecord(r);
}

// Local Variables:
// c-basic-offset: 4
// End:
