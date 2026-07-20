/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Pure builders + parsers for remote submission.  See schedulerspec.h.
------------------------------------------------------------------------- */

#include "schedulerspec.h"

#include <QFileInfo>
#include <QRegularExpression>

namespace {

using SchedulerSpec::ConnectionProfile;
using Remote::Scheduler;

// single-quote a token for a POSIX remote shell: wrap in '...', and turn any
// embedded single quote into the '\'' escape sequence.
QString shq(const QString &s)
{
    QString out = s;
    out.replace('\'', "'\\''");
    return '\'' + out + '\'';
}

QString submitTool(Scheduler s)
{
    switch (s) {
    case Scheduler::PBS:  return "qsub";
    case Scheduler::Flux: return "flux batch";
    case Scheduler::Slurm:
    default:              return "sbatch";
    }
}

} // namespace

namespace SchedulerSpec {

QString defaultTemplate(Scheduler s)
{
    switch (s) {
    case Scheduler::PBS:
        return QStringLiteral(
            "#!/bin/bash\n"
            "#PBS -N ${JOBNAME}\n"
            "#PBS -l nodes=${NODES}:ppn=${NTASKS}\n"
            "#PBS -l walltime=${WALLTIME}\n"
            "#PBS -A ${ACCOUNT}\n"
            "#PBS -q ${QUEUE}\n"
            "cd \"$PBS_O_WORKDIR\"\n"
            "${MODULES}\n"
            "${LAUNCH} ${SPARTAEXE} -in ${INPUT}\n");
    case Scheduler::Flux:
        return QStringLiteral(
            "#!/bin/bash\n"
            "# flux: --job-name=${JOBNAME}\n"
            "# flux: -N ${NODES}\n"
            "# flux: -n ${NTASKS}\n"
            "# flux: -t ${WALLTIME}\n"
            "# flux: --setattr=system.bank=${ACCOUNT}\n"
            "# flux: --queue=${QUEUE}\n"
            "${MODULES}\n"
            "${LAUNCH} ${SPARTAEXE} -in ${INPUT}\n");
    case Scheduler::Slurm:
    default:
        return QStringLiteral(
            "#!/bin/bash\n"
            "#SBATCH --job-name=${JOBNAME}\n"
            "#SBATCH --nodes=${NODES}\n"
            "#SBATCH --ntasks=${NTASKS}\n"
            "#SBATCH --time=${WALLTIME}\n"
            "#SBATCH --account=${ACCOUNT}\n"
            "#SBATCH --partition=${QUEUE}\n"
            "${MODULES}\n"
            "${LAUNCH} ${SPARTAEXE} -in ${INPUT}\n");
    }
}

QString renderScript(const ConnectionProfile &p, const SubmitParams &sp)
{
    const QString tmpl = p.batchTemplate.isEmpty() ? defaultTemplate(p.scheduler)
                                                    : p.batchTemplate;

    const QString launch = p.launchCmd.trimmed();
    const QString exe    = p.spartaExe.trimmed();

    QStringList out;
    const QStringList lines = tmpl.split('\n');
    for (const QString &raw : lines) {
        // ${MODULES}: expand to one line per "module load ..." entry, or drop
        // the line entirely when there are none.
        if (raw.contains("${MODULES}")) {
            for (const QString &m : p.moduleLoads) {
                const QString mm = m.trimmed();
                if (!mm.isEmpty()) out << mm;
            }
            continue;
        }
        // drop directive lines whose only variable is an unset optional
        if (raw.contains("${ACCOUNT}") && sp.account.trimmed().isEmpty()) continue;
        if (raw.contains("${QUEUE}") && sp.queue.trimmed().isEmpty()) continue;

        QString line = raw;
        line.replace("${JOBNAME}", sp.jobName);
        line.replace("${NODES}", QString::number(sp.nodes));
        line.replace("${NTASKS}", QString::number(sp.ntasks));
        line.replace("${WALLTIME}", sp.walltime);
        line.replace("${ACCOUNT}", sp.account);
        line.replace("${QUEUE}", sp.queue);
        line.replace("${LAUNCH}", launch);
        line.replace("${SPARTAEXE}", exe);
        line.replace("${INPUT}", sp.inputDeck);
        out << line;
    }

    // extra passthrough directives, inserted after the shebang
    if (!sp.extraDirectives.isEmpty() && !out.isEmpty()) {
        QStringList merged;
        merged << out.first();
        for (const QString &d : sp.extraDirectives)
            if (!d.trimmed().isEmpty()) merged << d.trimmed();
        for (int i = 1; i < out.size(); ++i) merged << out.at(i);
        out = merged;
    }

    QString text = out.join('\n');
    if (!text.endsWith('\n')) text += '\n';
    return text;
}

QStringList sshBase(const ConnectionProfile &p, const QString &controlPath)
{
    QStringList a;
    if (p.port != 22 && p.port > 0) a << "-p" << QString::number(p.port);
    // key-only, fail-fast: never prompt for a password, never hang.
    a << "-o" << "BatchMode=yes" << "-o" << "ConnectTimeout=10";
    if (!controlPath.isEmpty()) {
        a << "-o" << "ControlMaster=auto"
          << "-o" << "ControlPersist=60s"
          << "-o" << ("ControlPath=" + controlPath);
    }
    a << (p.user.isEmpty() ? p.host : p.user + "@" + p.host);
    return a;
}

QString submitCommand(const ConnectionProfile &p, const QString &remoteScriptPath)
{
    const QString dir  = QFileInfo(remoteScriptPath).path();
    const QString base = QFileInfo(remoteScriptPath).fileName();
    return QStringLiteral("cd %1 && %2 %3").arg(shq(dir), submitTool(p.scheduler), shq(base));
}

QString pollCommand(Scheduler s, const QString &remoteJobId)
{
    switch (s) {
    case Scheduler::PBS:
        return QStringLiteral("qstat -f %1 2>/dev/null").arg(shq(remoteJobId));
    case Scheduler::Flux:
        return QStringLiteral("flux jobs -no {status} %1 2>/dev/null").arg(shq(remoteJobId));
    case Scheduler::Slurm:
    default:
        return QStringLiteral("squeue -h -j %1 -o %T 2>/dev/null").arg(shq(remoteJobId));
    }
}

QString cancelCommand(Scheduler s, const QString &remoteJobId)
{
    switch (s) {
    case Scheduler::PBS:  return QStringLiteral("qdel %1").arg(shq(remoteJobId));
    case Scheduler::Flux: return QStringLiteral("flux cancel %1").arg(shq(remoteJobId));
    case Scheduler::Slurm:
    default:              return QStringLiteral("scancel %1").arg(shq(remoteJobId));
    }
}

QStringList submitArgs(const ConnectionProfile &p, const QString &remoteScriptPath,
                       const QString &controlPath)
{
    return sshBase(p, controlPath) << submitCommand(p, remoteScriptPath);
}

QStringList pollArgs(const ConnectionProfile &p, const QString &remoteJobId,
                     const QString &controlPath)
{
    return sshBase(p, controlPath) << pollCommand(p.scheduler, remoteJobId);
}

QStringList cancelArgs(const ConnectionProfile &p, const QString &remoteJobId,
                       const QString &controlPath)
{
    return sshBase(p, controlPath) << cancelCommand(p.scheduler, remoteJobId);
}

QString parseSubmitId(Scheduler s, const QString &submitStdout)
{
    const QString text = submitStdout.trimmed();
    if (text.isEmpty()) return {};

    if (s == Scheduler::Slurm) {
        // "Submitted batch job 12345"
        static const QRegularExpression re(R"(Submitted batch job\s+(\d+))");
        const auto m = re.match(text);
        if (m.hasMatch()) return m.captured(1);
        // fall back to a trailing integer anywhere
        static const QRegularExpression re2(R"((\d+)\s*$)");
        const auto m2 = re2.match(text);
        return m2.hasMatch() ? m2.captured(1) : QString();
    }

    // PBS ("123.head") and Flux (fXXXX): first token of the first non-empty line
    for (const QString &line : text.split('\n')) {
        const QString t = line.trimmed();
        if (t.isEmpty()) continue;
        return t.section(QRegularExpression("\\s+"), 0, 0);
    }
    return {};
}

JobState parsePollState(Scheduler s, const QString &pollStdout)
{
    const QString text = pollStdout.trimmed();
    if (text.isEmpty()) return JobState::Completed; // left the queue

    if (s == Scheduler::PBS) {
        // find "job_state = X" in qstat -f output
        static const QRegularExpression re(R"(job_state\s*=\s*(\w))");
        const auto m = re.match(text);
        if (!m.hasMatch()) return JobState::Unknown;
        const QChar c = m.captured(1).at(0).toUpper();
        switch (c.toLatin1()) {
        case 'Q': case 'H': case 'W': case 'T': return JobState::Queued;
        case 'R': return JobState::Running;
        case 'E': return JobState::Completing;
        case 'C': return JobState::Completed;
        case 'F': return JobState::Failed;
        default:  return JobState::Unknown;
        }
    }

    const QString tok = text.split('\n').first().trimmed()
                            .section(QRegularExpression("\\s+"), 0, 0).toUpper();

    if (s == Scheduler::Flux) {
        if (tok == "SCHED" || tok == "DEPEND" || tok == "PRIORITY" || tok == "HELD")
            return JobState::Queued;
        if (tok == "RUN")      return JobState::Running;
        if (tok == "CLEANUP")  return JobState::Completing;
        if (tok == "INACTIVE") return JobState::Completed;
        return JobState::Unknown;
    }

    // Slurm %T
    if (tok == "PENDING" || tok == "CONFIGURING" || tok == "RESV_DEL_HOLD")
        return JobState::Queued;
    if (tok == "RUNNING")    return JobState::Running;
    if (tok == "COMPLETING") return JobState::Completing;
    if (tok == "COMPLETED")  return JobState::Completed;
    if (tok == "CANCELLED")  return JobState::Cancelled;
    if (tok == "FAILED" || tok == "NODE_FAIL" || tok == "TIMEOUT" ||
        tok == "OUT_OF_MEMORY" || tok == "BOOT_FAIL" || tok == "DEADLINE")
        return JobState::Failed;
    return JobState::Unknown;
}

} // namespace SchedulerSpec

// Local Variables:
// c-basic-offset: 4
// End:
