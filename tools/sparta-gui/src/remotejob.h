/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Plain-data models for remote/cluster execution: connection profiles,
   per-submission parameters, and the persisted job record.  Qt-Core only
   (no widgets, no SPARTA), so the models and the schedulerspec.{h,cpp}
   builders that consume them can be unit-tested in isolation, mirroring
   paraviewexport.{h,cpp} / stlimport.{h,cpp}.
------------------------------------------------------------------------- */

#ifndef REMOTEJOB_H
#define REMOTEJOB_H

#include <QDateTime>
#include <QString>
#include <QStringList>

class QJsonObject;

namespace Remote {

/** @brief Supported HPC batch schedulers. */
enum class Scheduler { Slurm, PBS, Flux };

/** @brief String name of a scheduler (for persistence / UI). */
QString schedulerName(Scheduler s);
/** @brief Parse a scheduler name; defaults to Slurm on an unknown value. */
Scheduler schedulerFromName(const QString &name);

/**
 * @brief A reusable SSH/cluster connection profile.
 *
 * Authentication is always key-based: the GUI shells out to the user's own
 * ssh/scp/rsync and never handles, stores, or transmits a password.
 */
struct ConnectionProfile {
    QString name;                    ///< user-visible profile name
    QString host;                    ///< ssh host
    QString user;                    ///< ssh user
    int port = 22;                   ///< ssh port
    QString remoteWorkdir;           ///< base directory on the cluster
    Scheduler scheduler = Scheduler::Slurm;
    QString launchCmd = "srun";      ///< parallel launcher prefix (srun / mpirun ...)
    QString spartaExe = "spa_";      ///< remote SPARTA executable
    QStringList moduleLoads;         ///< "module load ..." lines to emit
    QString batchTemplate;           ///< generic override; empty => scheduler preset

    QJsonObject toJson() const;
    static ConnectionProfile fromJson(const QJsonObject &o);
};

/** @brief Per-submission knobs substituted into the batch template. */
struct SubmitParams {
    QString jobName = "sparta";
    int nodes = 1;
    int ntasks = 1;
    QString walltime = "01:00:00";
    QString account;                 ///< optional charge account
    QString queue;                   ///< partition (Slurm) / queue (PBS/Flux)
    QString inputDeck;               ///< remote-relative SPARTA input file name
    QStringList extraDirectives;     ///< passthrough scheduler directives

    QJsonObject toJson() const;
    static SubmitParams fromJson(const QJsonObject &o);
};

/** @brief Lifecycle state of a submitted job. */
enum class JobState {
    Draft, Staging, Submitted, Queued, Running, Completing,
    Completed, Failed, Cancelled, Unknown
};

QString jobStateName(JobState s);
JobState jobStateFromName(const QString &name);
/** @brief True once the job has reached a terminal state (no more polling). */
bool isTerminal(JobState s);

/** @brief One submitted job; the unit persisted across sessions. */
struct RemoteJob {
    QString localId;                 ///< GUID we assign, stable across restart
    QString remoteJobId;             ///< scheduler id captured at submit
    QString profileName;             ///< FK into the profile store
    SubmitParams params;
    JobState state = JobState::Draft;
    QString remoteRunDir;            ///< remoteWorkdir/<localId>
    QString localRunDir;             ///< where artifacts are pulled
    QStringList stagedFiles;         ///< deck + data files pushed up
    QDateTime submittedAt;
    QDateTime updatedAt;
    QString lastError;

    QJsonObject toJson() const;
    static RemoteJob fromJson(const QJsonObject &o);
};

} // namespace Remote

#endif // REMOTEJOB_H
