/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Pure, GUI-free builders for remote/cluster execution: render the batch
   submission script from a template and build the exact argv vectors for
   the ssh/submit/poll/cancel commands per scheduler, plus parse the
   scheduler's submit and poll output.  No Qt widgets, no QProcess, no
   network -- everything here is a pure function of the settings, so it is
   unit-tested exactly like paraviewexport.{h,cpp}.
------------------------------------------------------------------------- */

#ifndef SCHEDULERSPEC_H
#define SCHEDULERSPEC_H

#include "remotejob.h"

#include <QString>
#include <QStringList>

namespace SchedulerSpec {

using Remote::ConnectionProfile;
using Remote::JobState;
using Remote::Scheduler;
using Remote::SubmitParams;

/**
 * @brief Default batch-script template for a scheduler.
 *
 * Contains the directive block (`#SBATCH` / `#PBS` / `# flux:`) and body with
 * `${...}` placeholders substituted by renderScript(): NODES, NTASKS, WALLTIME,
 * ACCOUNT, QUEUE, JOBNAME, MODULES, LAUNCH, SPARTAEXE, INPUT.
 */
QString defaultTemplate(Scheduler s);

/**
 * @brief Render the batch script for a submission.
 *
 * Uses `profile.batchTemplate` when non-empty, otherwise defaultTemplate() for
 * the profile's scheduler, then substitutes every `${...}` placeholder from the
 * profile + params. Lines whose only content was an empty optional (e.g. an
 * absent ACCOUNT directive) are dropped.
 */
QString renderScript(const ConnectionProfile &p, const SubmitParams &sp);

/**
 * @brief ssh argument vector up to and including the `user@host` target.
 *
 * The returned list is the args for QProcess(findExe("ssh"), args); the caller
 * appends the remote command as one trailing argument. Includes key-only,
 * fail-fast options: BatchMode=yes, ConnectTimeout, and (when @p controlPath is
 * non-empty) ControlMaster/ControlPersist so repeated polls reuse one
 * connection. Never enables password auth or auto-accepts host keys.
 */
QStringList sshBase(const ConnectionProfile &p, const QString &controlPath = QString());

/** @brief Full ssh argv to submit @p remoteScriptPath under the scheduler. */
QStringList submitArgs(const ConnectionProfile &p, const QString &remoteScriptPath,
                       const QString &controlPath = QString());

/** @brief Full ssh argv to poll the state of @p remoteJobId. */
QStringList pollArgs(const ConnectionProfile &p, const QString &remoteJobId,
                     const QString &controlPath = QString());

/** @brief Full ssh argv to cancel @p remoteJobId. */
QStringList cancelArgs(const ConnectionProfile &p, const QString &remoteJobId,
                       const QString &controlPath = QString());

/** @brief The remote command string the submit ssh runs (exposed for tests). */
QString submitCommand(const ConnectionProfile &p, const QString &remoteScriptPath);
/** @brief The remote command string the poll ssh runs. */
QString pollCommand(Scheduler s, const QString &remoteJobId);
/** @brief The remote command string the cancel ssh runs. */
QString cancelCommand(Scheduler s, const QString &remoteJobId);

/**
 * @brief Extract the scheduler job id from submit stdout.
 *
 * Slurm: the trailing integer of "Submitted batch job N". PBS: the first token
 * (e.g. "123.head"). Flux: the first non-empty token (the fXXXX jobid).
 * Returns an empty string when nothing parses.
 */
QString parseSubmitId(Scheduler s, const QString &submitStdout);

/**
 * @brief Map poll stdout to a JobState.
 *
 * Empty output means the job has left the queue -> Completed. Recognizes the
 * per-scheduler running/pending/completing tokens; unknown non-empty output
 * yields JobState::Unknown.
 */
JobState parsePollState(Scheduler s, const QString &pollStdout);

} // namespace SchedulerSpec

#endif // SCHEDULERSPEC_H
