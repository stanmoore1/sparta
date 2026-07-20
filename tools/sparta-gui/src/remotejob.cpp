/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Serialization + enum helpers for the remote-execution data models.
   See remotejob.h.
------------------------------------------------------------------------- */

#include "remotejob.h"

#include <QJsonArray>
#include <QJsonObject>

namespace Remote {

QString schedulerName(Scheduler s)
{
    switch (s) {
    case Scheduler::PBS:  return "pbs";
    case Scheduler::Flux: return "flux";
    case Scheduler::Slurm:
    default:              return "slurm";
    }
}

Scheduler schedulerFromName(const QString &name)
{
    const QString n = name.trimmed().toLower();
    if (n == "pbs" || n == "torque") return Scheduler::PBS;
    if (n == "flux")                 return Scheduler::Flux;
    return Scheduler::Slurm;
}

QString jobStateName(JobState s)
{
    switch (s) {
    case JobState::Draft:      return "Draft";
    case JobState::Staging:    return "Staging";
    case JobState::Submitted:  return "Submitted";
    case JobState::Queued:     return "Queued";
    case JobState::Running:    return "Running";
    case JobState::Completing: return "Completing";
    case JobState::Completed:  return "Completed";
    case JobState::Failed:     return "Failed";
    case JobState::Cancelled:  return "Cancelled";
    case JobState::Unknown:
    default:                   return "Unknown";
    }
}

JobState jobStateFromName(const QString &name)
{
    const QString n = name.trimmed();
    if (n == "Draft")      return JobState::Draft;
    if (n == "Staging")    return JobState::Staging;
    if (n == "Submitted")  return JobState::Submitted;
    if (n == "Queued")     return JobState::Queued;
    if (n == "Running")    return JobState::Running;
    if (n == "Completing") return JobState::Completing;
    if (n == "Completed")  return JobState::Completed;
    if (n == "Failed")     return JobState::Failed;
    if (n == "Cancelled")  return JobState::Cancelled;
    return JobState::Unknown;
}

bool isTerminal(JobState s)
{
    return s == JobState::Completed || s == JobState::Failed ||
           s == JobState::Cancelled;
}

// --- ConnectionProfile ---------------------------------------------------

QJsonObject ConnectionProfile::toJson() const
{
    QJsonObject o;
    o["name"] = name;
    o["host"] = host;
    o["user"] = user;
    o["port"] = port;
    o["remoteWorkdir"] = remoteWorkdir;
    o["scheduler"] = schedulerName(scheduler);
    o["launchCmd"] = launchCmd;
    o["spartaExe"] = spartaExe;
    o["moduleLoads"] = QJsonArray::fromStringList(moduleLoads);
    o["batchTemplate"] = batchTemplate;
    return o;
}

ConnectionProfile ConnectionProfile::fromJson(const QJsonObject &o)
{
    ConnectionProfile p;
    p.name = o["name"].toString();
    p.host = o["host"].toString();
    p.user = o["user"].toString();
    p.port = o["port"].toInt(22);
    p.remoteWorkdir = o["remoteWorkdir"].toString();
    p.scheduler = schedulerFromName(o["scheduler"].toString());
    p.launchCmd = o["launchCmd"].toString("srun");
    p.spartaExe = o["spartaExe"].toString("spa_");
    for (const auto &v : o["moduleLoads"].toArray()) p.moduleLoads << v.toString();
    p.batchTemplate = o["batchTemplate"].toString();
    return p;
}

// --- SubmitParams --------------------------------------------------------

QJsonObject SubmitParams::toJson() const
{
    QJsonObject o;
    o["jobName"] = jobName;
    o["nodes"] = nodes;
    o["ntasks"] = ntasks;
    o["walltime"] = walltime;
    o["account"] = account;
    o["queue"] = queue;
    o["inputDeck"] = inputDeck;
    o["extraDirectives"] = QJsonArray::fromStringList(extraDirectives);
    return o;
}

SubmitParams SubmitParams::fromJson(const QJsonObject &o)
{
    SubmitParams p;
    p.jobName = o["jobName"].toString("sparta");
    p.nodes = o["nodes"].toInt(1);
    p.ntasks = o["ntasks"].toInt(1);
    p.walltime = o["walltime"].toString("01:00:00");
    p.account = o["account"].toString();
    p.queue = o["queue"].toString();
    p.inputDeck = o["inputDeck"].toString();
    for (const auto &v : o["extraDirectives"].toArray()) p.extraDirectives << v.toString();
    return p;
}

// --- RemoteJob -----------------------------------------------------------

QJsonObject RemoteJob::toJson() const
{
    QJsonObject o;
    o["localId"] = localId;
    o["remoteJobId"] = remoteJobId;
    o["profileName"] = profileName;
    o["params"] = params.toJson();
    o["state"] = jobStateName(state);
    o["remoteRunDir"] = remoteRunDir;
    o["localRunDir"] = localRunDir;
    o["stagedFiles"] = QJsonArray::fromStringList(stagedFiles);
    o["submittedAt"] = submittedAt.toString(Qt::ISODate);
    o["updatedAt"] = updatedAt.toString(Qt::ISODate);
    o["lastError"] = lastError;
    return o;
}

RemoteJob RemoteJob::fromJson(const QJsonObject &o)
{
    RemoteJob j;
    j.localId = o["localId"].toString();
    j.remoteJobId = o["remoteJobId"].toString();
    j.profileName = o["profileName"].toString();
    j.params = SubmitParams::fromJson(o["params"].toObject());
    j.state = jobStateFromName(o["state"].toString());
    j.remoteRunDir = o["remoteRunDir"].toString();
    j.localRunDir = o["localRunDir"].toString();
    for (const auto &v : o["stagedFiles"].toArray()) j.stagedFiles << v.toString();
    j.submittedAt = QDateTime::fromString(o["submittedAt"].toString(), Qt::ISODate);
    j.updatedAt = QDateTime::fromString(o["updatedAt"].toString(), Qt::ISODate);
    j.lastError = o["lastError"].toString();
    return j;
}

} // namespace Remote

// Local Variables:
// c-basic-offset: 4
// End:
