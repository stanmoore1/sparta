/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   RemoteJobManager implementation.  See remotejobmanager.h.
------------------------------------------------------------------------- */

#include "remotejobmanager.h"

#include "helpers.h"
#include "schedulerspec.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QStandardPaths>
#include <QUuid>

using namespace Remote;

namespace {

constexpr int POLL_INTERVAL_MS = 15000;

QString shq(const QString &s)
{
    QString out = s;
    out.replace('\'', "'\\''");
    return '\'' + out + '\'';
}

// ssh option string (no target) for rsync's -e, derived from the same options
// SchedulerSpec::sshBase() builds so auth/timeout behavior matches.
QString sshDashE(const ConnectionProfile &p, const QString &cp)
{
    QStringList opts = SchedulerSpec::sshBase(p, cp);
    if (!opts.isEmpty()) opts.removeLast(); // drop user@host
    return "ssh " + opts.join(' ');
}

QString configDir()
{
    QString d = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
    QDir().mkpath(d);
    return d;
}

} // namespace

// ===========================================================================
// RemoteJobsModel
// ===========================================================================

RemoteJobsModel::RemoteJobsModel(RemoteJobManager *mgr, QObject *parent)
    : QAbstractTableModel(parent), mgr_(mgr)
{
}

int RemoteJobsModel::rowCount(const QModelIndex &parent) const
{
    return parent.isValid() ? 0 : mgr_->jobCount();
}

int RemoteJobsModel::columnCount(const QModelIndex &parent) const
{
    return parent.isValid() ? 0 : NCols;
}

QVariant RemoteJobsModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid() || index.row() >= mgr_->jobCount()) return {};
    if (role != Qt::DisplayRole && role != Qt::ToolTipRole) return {};
    const RemoteJob &j = mgr_->jobAt(index.row());
    switch (index.column()) {
    case ColName:      return j.params.jobName;
    case ColProfile:   return j.profileName;
    case ColScheduler: return schedulerName(mgr_->profile(j.profileName).scheduler);
    case ColState:     return jobStateName(j.state);
    case ColJobId:     return j.remoteJobId;
    case ColSubmitted: return j.submittedAt.isValid()
                              ? j.submittedAt.toString("yyyy-MM-dd hh:mm") : QString();
    default:           return {};
    }
}

QVariant RemoteJobsModel::headerData(int section, Qt::Orientation o, int role) const
{
    if (role != Qt::DisplayRole || o != Qt::Horizontal) return {};
    switch (section) {
    case ColName:      return "Job";
    case ColProfile:   return "Profile";
    case ColScheduler: return "Scheduler";
    case ColState:     return "State";
    case ColJobId:     return "Job ID";
    case ColSubmitted: return "Submitted";
    default:           return {};
    }
}

void RemoteJobsModel::refreshRow(int row)
{
    if (row < 0 || row >= mgr_->jobCount()) return;
    emit dataChanged(index(row, 0), index(row, NCols - 1));
}

void RemoteJobsModel::refreshAll()
{
    beginResetModel();
    endResetModel();
}

// ===========================================================================
// RemoteJobManager
// ===========================================================================

RemoteJobManager::RemoteJobManager(QObject *parent) : QObject(parent)
{
    model_ = new RemoteJobsModel(this, this);
    load();
    connect(&pollTimer_, &QTimer::timeout, this, &RemoteJobManager::onPollTick);
    pollTimer_.setInterval(POLL_INTERVAL_MS);
    pollTimer_.start();
    // reattach: an immediate poll resolves the state of any job left running
    QTimer::singleShot(0, this, &RemoteJobManager::onPollTick);
}

RemoteJobManager::~RemoteJobManager()
{
    for (auto *t : tails_)
        if (t) { t->kill(); t->waitForFinished(200); }
}

QString RemoteJobManager::sshExe()
{
    const QString e = findExe("ssh");
    return e.isEmpty() ? "ssh" : e;
}

QString RemoteJobManager::scpOrRsync(bool &isRsync)
{
    const QString r = findExe("rsync");
    if (!r.isEmpty()) { isRsync = true; return r; }
    isRsync = false;
    const QString s = findExe("scp");
    return s.isEmpty() ? "scp" : s;
}

QString RemoteJobManager::controlPath(const RemoteJob &j) const
{
    // one multiplexed connection per host, under the runtime/temp dir
    const ConnectionProfile p = profiles_.value(j.profileName);
    QString base = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    return base + "/sparta-gui-ssh-" + p.user + "-" + p.host + ".sock";
}

RemoteJob *RemoteJobManager::job(const QString &localId)
{
    for (auto &j : jobs_)
        if (j.localId == localId) return &j;
    return nullptr;
}

int RemoteJobManager::rowForId(const QString &localId) const
{
    for (int i = 0; i < jobs_.size(); ++i)
        if (jobs_.at(i).localId == localId) return i;
    return -1;
}

QProcess *RemoteJobManager::newProc(const QString &)
{
    auto *p = new QProcess(this);
    return p;
}

void RemoteJobManager::setState(const QString &localId, JobState st)
{
    RemoteJob *j = job(localId);
    if (!j || j->state == st) return;
    const JobState prev = j->state;
    j->state = st;
    j->updatedAt = QDateTime::currentDateTime();
    if (st == JobState::Running && prev != JobState::Running) startTail(localId);
    if (isTerminal(st)) { stopTail(localId); }
    saveJobs();
    const int row = rowForId(localId);
    if (row >= 0) model_->refreshRow(row);
    emit jobUpdated(localId);
}

void RemoteJobManager::failJob(const QString &localId, const QString &err)
{
    RemoteJob *j = job(localId);
    if (!j) return;
    j->lastError = err;
    emit message(QString("Job %1: %2").arg(j->params.jobName, err));
    setState(localId, JobState::Failed);
}

// --- submit pipeline -------------------------------------------------------

void RemoteJobManager::submit(RemoteJob draft, const QStringList &localFilesToStage)
{
    const ConnectionProfile p = profiles_.value(draft.profileName);
    if (p.host.isEmpty()) { emit message("Submit failed: unknown or empty profile."); return; }

    if (draft.localId.isEmpty())
        draft.localId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    draft.remoteRunDir = p.remoteWorkdir + "/sparta-gui-jobs/" + draft.localId;
    draft.localRunDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) +
                        "/remote_runs/" + draft.localId;
    QDir().mkpath(draft.localRunDir);
    draft.state = JobState::Staging;
    draft.submittedAt = QDateTime::currentDateTime();
    draft.updatedAt = draft.submittedAt;
    draft.stagedFiles = localFilesToStage;

    emit jobsAboutToChange();
    jobs_.prepend(draft);
    emit jobsChanged();
    model_->refreshAll();
    saveJobs();

    stageAndSubmit(draft.localId, localFilesToStage);
}

void RemoteJobManager::stageAndSubmit(const QString &localId, const QStringList &localFiles)
{
    RemoteJob *j = job(localId);
    if (!j) return;
    const ConnectionProfile p = profiles_.value(j->profileName);
    const QString cp = controlPath(*j);

    // step A: mkdir the run dir and write job.sh from the rendered template
    const QString script = SchedulerSpec::renderScript(p, j->params);
    const QString cmd = QString("mkdir -p %1 && cat > %2")
                            .arg(shq(j->remoteRunDir), shq(j->remoteRunDir + "/job.sh"));
    QProcess *proc = newProc(localId);
    connect(proc, &QProcess::started, proc, [proc, script]() {
        proc->write(script.toUtf8());
        proc->closeWriteChannel();
    });
    connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
            [this, localId, proc](int code, QProcess::ExitStatus st) {
                const QString err = QString::fromLocal8Bit(proc->readAllStandardError());
                proc->deleteLater();
                if (st != QProcess::NormalExit || code != 0) {
                    failJob(localId, "staging (mkdir/script) failed: " + err.trimmed());
                    return;
                }
                RemoteJob *jj = job(localId);
                if (jj) stepRsyncUp(localId, jj->stagedFiles);
            });
    proc->start(sshExe(), SchedulerSpec::sshBase(p, cp) << cmd);
}

void RemoteJobManager::stepRsyncUp(const QString &localId, const QStringList &localFiles)
{
    RemoteJob *j = job(localId);
    if (!j) return;
    const ConnectionProfile p = profiles_.value(j->profileName);
    const QString cp = controlPath(*j);
    const QString target = QString("%1@%2:%3/")
                               .arg(p.user, p.host, j->remoteRunDir);

    if (localFiles.isEmpty()) { stepSubmit(localId); return; }

    bool isRsync = false;
    const QString tool = scpOrRsync(isRsync);
    QStringList args;
    if (isRsync) {
        args << "-az" << "-e" << sshDashE(p, cp);
        args << localFiles << target;
    } else {
        // scp -P port file... user@host:dir/
        if (p.port != 22 && p.port > 0) args << "-P" << QString::number(p.port);
        args << "-o" << "BatchMode=yes" << localFiles << target;
    }
    QProcess *proc = newProc(localId);
    connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
            [this, localId, proc](int code, QProcess::ExitStatus st) {
                const QString err = QString::fromLocal8Bit(proc->readAllStandardError());
                proc->deleteLater();
                if (st != QProcess::NormalExit || code != 0) {
                    failJob(localId, "file staging (rsync/scp) failed: " + err.trimmed());
                    return;
                }
                stepSubmit(localId);
            });
    proc->start(tool, args);
}

void RemoteJobManager::stepSubmit(const QString &localId)
{
    RemoteJob *j = job(localId);
    if (!j) return;
    const ConnectionProfile p = profiles_.value(j->profileName);
    const QString cp = controlPath(*j);
    setState(localId, JobState::Submitted);

    QProcess *proc = newProc(localId);
    connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
            [this, localId, proc, p](int code, QProcess::ExitStatus st) {
                const QString out = QString::fromLocal8Bit(proc->readAllStandardOutput());
                const QString err = QString::fromLocal8Bit(proc->readAllStandardError());
                proc->deleteLater();
                if (st != QProcess::NormalExit || code != 0) {
                    failJob(localId, "submit failed: " + err.trimmed());
                    return;
                }
                const QString id = SchedulerSpec::parseSubmitId(p.scheduler, out);
                RemoteJob *jj = job(localId);
                if (!jj) return;
                if (id.isEmpty()) {
                    failJob(localId, "could not parse job id from: " + out.trimmed());
                    return;
                }
                jj->remoteJobId = id;
                emit message(QString("Submitted job %1 (id %2).").arg(jj->params.jobName, id));
                setState(localId, JobState::Queued);
            });
    proc->start(sshExe(),
                SchedulerSpec::submitArgs(p, j->remoteRunDir + "/job.sh", cp));
}

// --- polling ---------------------------------------------------------------

void RemoteJobManager::onPollTick()
{
    for (const RemoteJob &jc : jobs_) {
        const QString id = jc.localId;
        if (jc.remoteJobId.isEmpty()) continue;
        if (isTerminal(jc.state) || jc.state == JobState::Staging || jc.state == JobState::Draft)
            continue;
        if (pollInFlight_.value(id, false)) continue;

        const ConnectionProfile p = profiles_.value(jc.profileName);
        if (p.host.isEmpty()) continue;
        const QString cp = controlPath(jc);
        pollInFlight_[id] = true;

        QProcess *proc = newProc(id);
        connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
                [this, id, proc, p](int code, QProcess::ExitStatus st) {
                    const QString out = QString::fromLocal8Bit(proc->readAllStandardOutput());
                    proc->deleteLater();
                    pollInFlight_[id] = false;
                    if (st != QProcess::NormalExit) return; // transient; retry next tick
                    const JobState ns = SchedulerSpec::parsePollState(p.scheduler, out);
                    RemoteJob *jj = job(id);
                    if (!jj) return;
                    const JobState prev = jj->state;
                    if (ns != prev && ns != JobState::Unknown) {
                        setState(id, ns);
                        if (isTerminal(ns) && ns == JobState::Completed)
                            pullArtifacts(id);
                    }
                });
        proc->start(sshExe(), SchedulerSpec::pollArgs(p, jc.remoteJobId, cp));
    }
}

// --- tail ------------------------------------------------------------------

void RemoteJobManager::startTail(const QString &localId)
{
    if (tails_.contains(localId)) return;
    RemoteJob *j = job(localId);
    if (!j) return;
    const ConnectionProfile p = profiles_.value(j->profileName);
    const QString cp = controlPath(*j);
    // SPARTA writes log.sparta in its run dir; tail -F waits for it to appear
    const QString cmd = QString("tail -n +1 -F %1").arg(shq(j->remoteRunDir + "/log.sparta"));
    auto *proc = newProc(localId);
    proc->setProcessChannelMode(QProcess::MergedChannels);
    connect(proc, &QProcess::readyReadStandardOutput, this, [this, localId, proc]() {
        emit logChunk(localId, QString::fromLocal8Bit(proc->readAllStandardOutput()));
    });
    proc->start(sshExe(), SchedulerSpec::sshBase(p, cp) << cmd);
    tails_.insert(localId, proc);
}

void RemoteJobManager::stopTail(const QString &localId)
{
    if (auto *t = tails_.take(localId)) {
        t->kill();
        t->deleteLater();
    }
}

// --- artifacts / cancel / resubmit ----------------------------------------

void RemoteJobManager::pullArtifacts(const QString &localId)
{
    RemoteJob *j = job(localId);
    if (!j) return;
    const ConnectionProfile p = profiles_.value(j->profileName);
    const QString cp = controlPath(*j);
    QDir().mkpath(j->localRunDir);
    const QString source = QString("%1@%2:%3/").arg(p.user, p.host, j->remoteRunDir);

    bool isRsync = false;
    const QString tool = scpOrRsync(isRsync);
    QStringList args;
    if (isRsync) {
        args << "-az" << "-e" << sshDashE(p, cp) << source << (j->localRunDir + "/");
    } else {
        if (p.port != 22 && p.port > 0) args << "-P" << QString::number(p.port);
        args << "-o" << "BatchMode=yes" << "-r" << source << j->localRunDir;
    }
    auto *proc = newProc(localId);
    connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
            [this, localId, proc](int code, QProcess::ExitStatus st) {
                const QString err = QString::fromLocal8Bit(proc->readAllStandardError());
                proc->deleteLater();
                RemoteJob *jj = job(localId);
                if (!jj) return;
                if (st != QProcess::NormalExit || code != 0)
                    emit message("Pull failed for " + jj->params.jobName + ": " + err.trimmed());
                else
                    emit message(QString("Pulled artifacts for %1 into %2")
                                     .arg(jj->params.jobName, jj->localRunDir));
            });
    proc->start(tool, args);
}

void RemoteJobManager::cancel(const QString &localId)
{
    RemoteJob *j = job(localId);
    if (!j || j->remoteJobId.isEmpty()) return;
    const ConnectionProfile p = profiles_.value(j->profileName);
    const QString cp = controlPath(*j);
    auto *proc = newProc(localId);
    connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
            [this, localId, proc](int, QProcess::ExitStatus) {
                proc->deleteLater();
                setState(localId, JobState::Cancelled);
            });
    proc->start(sshExe(), SchedulerSpec::cancelArgs(p, j->remoteJobId, cp));
}

void RemoteJobManager::resubmit(const QString &localId, const QStringList &localFilesToStage)
{
    RemoteJob *j = job(localId);
    if (!j) return;
    RemoteJob draft;
    draft.profileName = j->profileName;
    draft.params = j->params;
    const QStringList files = localFilesToStage.isEmpty() ? j->stagedFiles : localFilesToStage;
    submit(draft, files);
}

void RemoteJobManager::removeJob(const QString &localId)
{
    const int row = rowForId(localId);
    if (row < 0) return;
    stopTail(localId);
    emit jobsAboutToChange();
    jobs_.removeAt(row);
    emit jobsChanged();
    model_->refreshAll();
    saveJobs();
}

// --- profiles --------------------------------------------------------------

void RemoteJobManager::saveProfile(const ConnectionProfile &p)
{
    profiles_.insert(p.name, p);
    saveProfiles();
}

void RemoteJobManager::removeProfile(const QString &name)
{
    profiles_.remove(name);
    saveProfiles();
}

// --- persistence -----------------------------------------------------------

QString RemoteJobManager::jobsFile() const { return configDir() + "/remote_jobs.json"; }
QString RemoteJobManager::profilesFile() const { return configDir() + "/remote_profiles.json"; }

void RemoteJobManager::saveJobs() const
{
    QJsonArray arr;
    for (const auto &j : jobs_) arr.append(j.toJson());
    QFile f(jobsFile());
    if (f.open(QIODevice::WriteOnly))
        f.write(QJsonDocument(arr).toJson(QJsonDocument::Indented));
}

void RemoteJobManager::saveProfiles() const
{
    QJsonArray arr;
    for (const auto &p : profiles_) arr.append(p.toJson());
    QFile f(profilesFile());
    if (f.open(QIODevice::WriteOnly))
        f.write(QJsonDocument(arr).toJson(QJsonDocument::Indented));
}

void RemoteJobManager::load()
{
    QFile pf(profilesFile());
    if (pf.open(QIODevice::ReadOnly)) {
        const auto arr = QJsonDocument::fromJson(pf.readAll()).array();
        for (const auto &v : arr) {
            const auto p = ConnectionProfile::fromJson(v.toObject());
            if (!p.name.isEmpty()) profiles_.insert(p.name, p);
        }
    }
    QFile jf(jobsFile());
    if (jf.open(QIODevice::ReadOnly)) {
        const auto arr = QJsonDocument::fromJson(jf.readAll()).array();
        for (const auto &v : arr) jobs_.append(RemoteJob::fromJson(v.toObject()));
    }
}

// Local Variables:
// c-basic-offset: 4
// End:
