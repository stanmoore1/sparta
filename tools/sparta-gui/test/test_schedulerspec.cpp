// Unit tests for the pure remote-execution builders/parsers
// (src/schedulerspec.cpp) and the data-model serialization (src/remotejob.cpp).
//
// No network, no QProcess: every function is a pure function of the settings.
// Command argv, batch-script rendering, submit/poll parsing, and JSON
// round-trips are asserted for Slurm, PBS, and Flux.

#include "remotejob.h"
#include "schedulerspec.h"

#include <QJsonObject>

#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace Remote;
using namespace SchedulerSpec;

namespace {

std::vector<std::string> toVec(const QStringList &l)
{
    std::vector<std::string> v;
    for (const auto &s : l) v.push_back(s.toStdString());
    return v;
}

ConnectionProfile slurmProfile()
{
    ConnectionProfile p;
    p.name = "cluster";
    p.host = "hpc.example.edu";
    p.user = "alice";
    p.port = 22;
    p.remoteWorkdir = "/scratch/alice/sparta";
    p.scheduler = Scheduler::Slurm;
    p.launchCmd = "srun";
    p.spartaExe = "spa_";
    return p;
}

} // namespace

TEST(SchedulerSpec, SchedulerNameRoundTrip)
{
    for (auto s : {Scheduler::Slurm, Scheduler::PBS, Scheduler::Flux})
        EXPECT_EQ(schedulerFromName(schedulerName(s)), s);
    EXPECT_EQ(schedulerFromName("torque"), Scheduler::PBS);
    EXPECT_EQ(schedulerFromName("bogus"), Scheduler::Slurm); // default
}

TEST(SchedulerSpec, RenderScriptSubstitutesAndDropsEmptyOptionals)
{
    ConnectionProfile p = slurmProfile();
    p.moduleLoads = {"module load sparta", "module load openmpi"};
    SubmitParams sp;
    sp.jobName = "flow3d";
    sp.nodes = 2;
    sp.ntasks = 64;
    sp.walltime = "02:30:00";
    sp.inputDeck = "in.flow";
    // account + queue intentionally empty -> those directive lines dropped

    const QString script = renderScript(p, sp);
    const std::string s = script.toStdString();

    EXPECT_NE(s.find("#SBATCH --job-name=flow3d"), std::string::npos);
    EXPECT_NE(s.find("#SBATCH --nodes=2"), std::string::npos);
    EXPECT_NE(s.find("#SBATCH --ntasks=64"), std::string::npos);
    EXPECT_NE(s.find("#SBATCH --time=02:30:00"), std::string::npos);
    EXPECT_NE(s.find("module load sparta"), std::string::npos);
    EXPECT_NE(s.find("module load openmpi"), std::string::npos);
    EXPECT_NE(s.find("srun spa_ -in in.flow"), std::string::npos);
    // no leftover placeholders
    EXPECT_EQ(s.find("${"), std::string::npos);
    // empty account/queue lines were dropped
    EXPECT_EQ(s.find("--account="), std::string::npos);
    EXPECT_EQ(s.find("--partition="), std::string::npos);
}

TEST(SchedulerSpec, RenderScriptKeepsAccountAndQueueWhenSet)
{
    ConnectionProfile p = slurmProfile();
    SubmitParams sp;
    sp.account = "PROJ123";
    sp.queue = "batch";
    sp.inputDeck = "in.x";
    const std::string s = renderScript(p, sp).toStdString();
    EXPECT_NE(s.find("#SBATCH --account=PROJ123"), std::string::npos);
    EXPECT_NE(s.find("#SBATCH --partition=batch"), std::string::npos);
}

TEST(SchedulerSpec, RenderScriptPerSchedulerDirectivePrefix)
{
    SubmitParams sp;
    sp.inputDeck = "in.x";
    ConnectionProfile p;
    p.spartaExe = "spa_";
    p.launchCmd = "srun";

    p.scheduler = Scheduler::PBS;
    EXPECT_NE(renderScript(p, sp).toStdString().find("#PBS -N"), std::string::npos);
    p.scheduler = Scheduler::Flux;
    EXPECT_NE(renderScript(p, sp).toStdString().find("# flux:"), std::string::npos);
    p.scheduler = Scheduler::Slurm;
    EXPECT_NE(renderScript(p, sp).toStdString().find("#SBATCH"), std::string::npos);
}

TEST(SchedulerSpec, RenderScriptUserTemplateOverrides)
{
    ConnectionProfile p = slurmProfile();
    p.batchTemplate = "#!/bin/sh\nMYRUN ${SPARTAEXE} ${INPUT}\n";
    SubmitParams sp;
    sp.inputDeck = "deck.in";
    const std::string s = renderScript(p, sp).toStdString();
    EXPECT_NE(s.find("MYRUN spa_ deck.in"), std::string::npos);
    EXPECT_EQ(s.find("#SBATCH"), std::string::npos); // preset not used
}

TEST(SchedulerSpec, SshBasePortAndOptions)
{
    ConnectionProfile p = slurmProfile();
    // default port 22 -> no -p (respect ~/.ssh/config)
    auto a = toVec(sshBase(p));
    ASSERT_FALSE(a.empty());
    EXPECT_EQ(a.back(), "alice@hpc.example.edu");
    EXPECT_NE(std::find(a.begin(), a.end(), "BatchMode=yes"), a.end());
    EXPECT_EQ(std::find(a.begin(), a.end(), "-p"), a.end());

    p.port = 2222;
    auto b = toVec(sshBase(p));
    ASSERT_GE(b.size(), 2u);
    EXPECT_EQ(b[0], "-p");
    EXPECT_EQ(b[1], "2222");
}

TEST(SchedulerSpec, SshBaseControlMasterWhenPathGiven)
{
    ConnectionProfile p = slurmProfile();
    auto a = toVec(sshBase(p, "/tmp/cm-sock"));
    EXPECT_NE(std::find(a.begin(), a.end(), "ControlMaster=auto"), a.end());
    EXPECT_NE(std::find(a.begin(), a.end(), "ControlPath=/tmp/cm-sock"), a.end());
    // absent by default
    auto b = toVec(sshBase(p));
    EXPECT_EQ(std::find(b.begin(), b.end(), "ControlMaster=auto"), b.end());
}

TEST(SchedulerSpec, SubmitPollCancelCommandsPerScheduler)
{
    EXPECT_EQ(pollCommand(Scheduler::Slurm, "123").toStdString(),
              "squeue -h -j '123' -o %T 2>/dev/null");
    EXPECT_EQ(pollCommand(Scheduler::PBS, "123.head").toStdString(),
              "qstat -f '123.head' 2>/dev/null");
    EXPECT_EQ(pollCommand(Scheduler::Flux, "f42").toStdString(),
              "flux jobs -no {status} 'f42' 2>/dev/null");

    EXPECT_EQ(cancelCommand(Scheduler::Slurm, "123").toStdString(), "scancel '123'");
    EXPECT_EQ(cancelCommand(Scheduler::PBS, "123").toStdString(), "qdel '123'");
    EXPECT_EQ(cancelCommand(Scheduler::Flux, "f42").toStdString(), "flux cancel 'f42'");

    ConnectionProfile p = slurmProfile();
    EXPECT_EQ(submitCommand(p, "/scratch/alice/sparta/abc/job.sh").toStdString(),
              "cd '/scratch/alice/sparta/abc' && sbatch 'job.sh'");
    p.scheduler = Scheduler::Flux;
    EXPECT_EQ(submitCommand(p, "/w/x/job.sh").toStdString(),
              "cd '/w/x' && flux batch 'job.sh'");
}

TEST(SchedulerSpec, SubmitArgsFullVector)
{
    ConnectionProfile p = slurmProfile();
    auto a = toVec(submitArgs(p, "/scratch/alice/sparta/j1/job.sh"));
    ASSERT_GE(a.size(), 2u);
    EXPECT_EQ(a.back(), "cd '/scratch/alice/sparta/j1' && sbatch 'job.sh'");
    EXPECT_EQ(a[a.size() - 2], "alice@hpc.example.edu");
}

TEST(SchedulerSpec, ParseSubmitId)
{
    EXPECT_EQ(parseSubmitId(Scheduler::Slurm, "Submitted batch job 987654").toStdString(),
              "987654");
    EXPECT_EQ(parseSubmitId(Scheduler::PBS, "12345.headnode\n").toStdString(),
              "12345.headnode");
    EXPECT_EQ(parseSubmitId(Scheduler::Flux, "f4Tms9kP2\n").toStdString(), "f4Tms9kP2");
    EXPECT_TRUE(parseSubmitId(Scheduler::Slurm, "").toStdString().empty());
}

TEST(SchedulerSpec, ParsePollStateSlurm)
{
    EXPECT_EQ(parsePollState(Scheduler::Slurm, "PENDING"), JobState::Queued);
    EXPECT_EQ(parsePollState(Scheduler::Slurm, "RUNNING"), JobState::Running);
    EXPECT_EQ(parsePollState(Scheduler::Slurm, "COMPLETING"), JobState::Completing);
    EXPECT_EQ(parsePollState(Scheduler::Slurm, "FAILED"), JobState::Failed);
    EXPECT_EQ(parsePollState(Scheduler::Slurm, "CANCELLED"), JobState::Cancelled);
    EXPECT_EQ(parsePollState(Scheduler::Slurm, ""), JobState::Completed); // gone
    EXPECT_EQ(parsePollState(Scheduler::Slurm, "WEIRD"), JobState::Unknown);
}

TEST(SchedulerSpec, ParsePollStatePBS)
{
    EXPECT_EQ(parsePollState(Scheduler::PBS, "    job_state = Q"), JobState::Queued);
    EXPECT_EQ(parsePollState(Scheduler::PBS, "  job_state = R\n exec_host = n1"),
              JobState::Running);
    EXPECT_EQ(parsePollState(Scheduler::PBS, "job_state = E"), JobState::Completing);
    EXPECT_EQ(parsePollState(Scheduler::PBS, "job_state = C"), JobState::Completed);
    EXPECT_EQ(parsePollState(Scheduler::PBS, ""), JobState::Completed);
}

TEST(SchedulerSpec, ParsePollStateFlux)
{
    EXPECT_EQ(parsePollState(Scheduler::Flux, "SCHED"), JobState::Queued);
    EXPECT_EQ(parsePollState(Scheduler::Flux, "RUN"), JobState::Running);
    EXPECT_EQ(parsePollState(Scheduler::Flux, "CLEANUP"), JobState::Completing);
    EXPECT_EQ(parsePollState(Scheduler::Flux, "INACTIVE"), JobState::Completed);
    EXPECT_EQ(parsePollState(Scheduler::Flux, ""), JobState::Completed);
}

TEST(SchedulerSpec, ProfileJsonRoundTrip)
{
    ConnectionProfile p = slurmProfile();
    p.moduleLoads = {"module load a", "module load b"};
    p.batchTemplate = "custom ${INPUT}";
    p.scheduler = Scheduler::Flux;
    const ConnectionProfile r = ConnectionProfile::fromJson(p.toJson());
    EXPECT_EQ(r.name.toStdString(), p.name.toStdString());
    EXPECT_EQ(r.host.toStdString(), p.host.toStdString());
    EXPECT_EQ(r.port, p.port);
    EXPECT_EQ(r.scheduler, Scheduler::Flux);
    EXPECT_EQ(r.moduleLoads, p.moduleLoads);
    EXPECT_EQ(r.batchTemplate.toStdString(), p.batchTemplate.toStdString());
}

TEST(SchedulerSpec, JobJsonRoundTrip)
{
    RemoteJob j;
    j.localId = "abc-123";
    j.remoteJobId = "555";
    j.profileName = "cluster";
    j.state = JobState::Running;
    j.remoteRunDir = "/scratch/j/abc-123";
    j.stagedFiles = {"in.flow", "data.surf"};
    j.params.nodes = 4;
    j.params.inputDeck = "in.flow";
    const RemoteJob r = RemoteJob::fromJson(j.toJson());
    EXPECT_EQ(r.localId.toStdString(), "abc-123");
    EXPECT_EQ(r.remoteJobId.toStdString(), "555");
    EXPECT_EQ(r.state, JobState::Running);
    EXPECT_EQ(r.stagedFiles.size(), 2);
    EXPECT_EQ(r.params.nodes, 4);
}

TEST(SchedulerSpec, IsTerminal)
{
    EXPECT_TRUE(isTerminal(JobState::Completed));
    EXPECT_TRUE(isTerminal(JobState::Failed));
    EXPECT_TRUE(isTerminal(JobState::Cancelled));
    EXPECT_FALSE(isTerminal(JobState::Running));
    EXPECT_FALSE(isTerminal(JobState::Queued));
}
