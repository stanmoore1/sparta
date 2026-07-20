/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   RemoteJobManager: the controller for remote/cluster execution.  Owns the
   job list and connection profiles, drives all ssh/scp/rsync work through
   asynchronous QProcess (never blocking the UI), persists jobs and profiles
   as JSON in the app config dir, and reattaches to still-running jobs on
   relaunch.  RemoteJobsModel exposes the job list to a QTableView.

   The pure command construction lives in schedulerspec.{h,cpp}; this file is
   the stateful, GUI-adjacent glue.  It never touches SpartaWrapper, so remote
   submission is independent of the single-instance in-process run.
------------------------------------------------------------------------- */

#ifndef REMOTEJOBMANAGER_H
#define REMOTEJOBMANAGER_H

#include "remotejob.h"

#include <QAbstractTableModel>
#include <QHash>
#include <QList>
#include <QMap>
#include <QObject>
#include <QTimer>

class QProcess;
class RemoteJobManager;

/** @brief Read-only table model over a RemoteJobManager's job list. */
class RemoteJobsModel : public QAbstractTableModel {
    Q_OBJECT
public:
    enum Column { ColName, ColProfile, ColScheduler, ColState, ColJobId, ColSubmitted, NCols };
    explicit RemoteJobsModel(RemoteJobManager *mgr, QObject *parent = nullptr);

    int rowCount(const QModelIndex &parent = {}) const override;
    int columnCount(const QModelIndex &parent = {}) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation o, int role = Qt::DisplayRole) const override;

    void refreshRow(int row);
    void refreshAll();

private:
    RemoteJobManager *mgr_;
};

class RemoteJobManager : public QObject {
    Q_OBJECT
public:
    explicit RemoteJobManager(QObject *parent = nullptr);
    ~RemoteJobManager() override;

    // --- profiles ---
    QList<Remote::ConnectionProfile> profiles() const { return profiles_.values(); }
    Remote::ConnectionProfile profile(const QString &name) const { return profiles_.value(name); }
    void saveProfile(const Remote::ConnectionProfile &p); // insert or update by name
    void removeProfile(const QString &name);

    // --- jobs (read side, for the model) ---
    int jobCount() const { return jobs_.size(); }
    const Remote::RemoteJob &jobAt(int row) const { return jobs_.at(row); }
    int rowForId(const QString &localId) const;

    RemoteJobsModel *model() { return model_; }

    // --- actions ---
    /** @brief Stage files, submit, and begin tracking. Fills localId/remoteRunDir. */
    void submit(Remote::RemoteJob draft, const QStringList &localFilesToStage);
    void cancel(const QString &localId);
    void resubmit(const QString &localId, const QStringList &localFilesToStage);
    void pullArtifacts(const QString &localId);
    void removeJob(const QString &localId);

signals:
    void jobsAboutToChange();
    void jobsChanged();
    void jobUpdated(const QString &localId);
    void logChunk(const QString &localId, const QString &text);
    void message(const QString &text); // human-readable status line

private slots:
    void onPollTick();

private:
    // async step helpers (each starts one QProcess and chains on finished)
    void stageAndSubmit(const QString &localId, const QStringList &localFiles);
    void stepRsyncUp(const QString &localId, const QStringList &localFiles);
    void stepSubmit(const QString &localId);
    void startTail(const QString &localId);
    void stopTail(const QString &localId);

    QProcess *newProc(const QString &localId);
    void failJob(const QString &localId, const QString &err);
    void setState(const QString &localId, Remote::JobState st);
    Remote::RemoteJob *job(const QString &localId);

    void load();
    void saveJobs() const;
    void saveProfiles() const;
    QString jobsFile() const;
    QString profilesFile() const;
    static QString sshExe();
    static QString scpOrRsync(bool &isRsync);
    QString controlPath(const Remote::RemoteJob &j) const;

    QList<Remote::RemoteJob> jobs_;
    QMap<QString, Remote::ConnectionProfile> profiles_;
    RemoteJobsModel *model_ = nullptr;
    QTimer pollTimer_;
    QHash<QString, QProcess *> tails_;     // localId -> long-lived tail process
    QHash<QString, bool> pollInFlight_;    // localId -> a poll is running
};

#endif // REMOTEJOBMANAGER_H
