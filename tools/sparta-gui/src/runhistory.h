/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Run history / provenance: RunHistory archives finished runs (deck, log,
   thermo, images, metadata) under the app data dir and regenerates a
   self-contained HTML/PDF report on demand.  HistoryModel/HistoryPanel are
   the docked table view over the archive.  Archiving is opt-in (off by
   default) so nothing is copied unless the user enables it.

   The pure record + HTML builder live in runarchive.{h,cpp}; this is the
   stateful GUI glue.
------------------------------------------------------------------------- */

#ifndef RUNHISTORY_H
#define RUNHISTORY_H

#include "runarchive.h"

#include <QAbstractTableModel>
#include <QList>
#include <QObject>
#include <QWidget>

class RunHistory;
class QTableView;

/** @brief Table model over a RunHistory's archived records. */
class HistoryModel : public QAbstractTableModel {
    Q_OBJECT
public:
    enum Column { ColTime, ColDeck, ColStatus, ColImages, NCols };
    explicit HistoryModel(RunHistory *hist, QObject *parent = nullptr);
    int rowCount(const QModelIndex &parent = {}) const override;
    int columnCount(const QModelIndex &parent = {}) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation o, int role = Qt::DisplayRole) const override;
    void refresh();

private:
    RunHistory *hist_;
};

class RunHistory : public QObject {
    Q_OBJECT
public:
    explicit RunHistory(QObject *parent = nullptr);

    /** @brief Archive a finished run: copy @p images into the archive and record it. */
    void archive(const RunArchive::RunRecord &rec, const QStringList &images);

    int count() const { return records_.size(); }
    const RunArchive::RunRecord &at(int row) const { return records_.at(row); }
    HistoryModel *model() { return model_; }

    /** @brief Write (and return the path to) an HTML report for record @p row. */
    QString writeReportHtml(int row);
    /** @brief Write an HTML comparison of records @p rowA and @p rowB (path, or empty). */
    QString writeComparisonHtml(int rowA, int rowB);
    /** @brief Write (and return the path to) a PDF report for record @p row (empty on failure). */
    QString writeReportPdf(int row);
    /** @brief Local archive directory for record @p row. */
    QString recordDir(int row) const;
    void removeRecord(int row);

signals:
    void changed();
    void message(const QString &text);

private:
    QString baseDir() const;
    QString indexFile() const;
    void load();
    void save() const;

    QList<RunArchive::RunRecord> records_;
    HistoryModel *model_ = nullptr;
};

/** @brief Docked panel listing archived runs with report/open/delete actions. */
class HistoryPanel : public QWidget {
    Q_OBJECT
public:
    HistoryPanel(QWidget *parent, RunHistory *hist);

private slots:
    void reportHtml();
    void reportPdf();
    void compareSelected();
    void openFolder();
    void deleteSelected();

private:
    /** @brief Explain an empty table (archiving off vs simply no runs yet) */
    void updateHint();

    int selectedRow() const;
    QList<int> selectedRows() const;
    RunHistory *hist_;
    QTableView *table_ = nullptr;
    class QPushButton *compareBtn_ = nullptr;
    class QLabel *hint_ = nullptr;   ///< Shown only while the table is empty
};

#endif // RUNHISTORY_H
