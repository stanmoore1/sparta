// -*- c++ -*- /////////////////////////////////////////////////////////////////////////
// SPARTA-GUI - A Graphical Tool to Learn and Explore the SPARTA DSMC Simulation Software
//
// Copyright (c) 2023, 2024, 2025, 2026  Axel Kohlmeyer
//
// This software is distributed under the GNU General Public License version 2 or later.
////////////////////////////////////////////////////////////////////////////////////////

#include "inputcheck.h"

#include <gtest/gtest.h>

using namespace InputCheck;

namespace {

// A representative (small) SPARTA vocabulary for the tests.
Context makeContext()
{
    Context ctx;
    ctx.commands = {"seed",       "dimension",  "global",     "boundary",   "create_box",
                    "create_grid", "balance_grid", "species",  "mixture",   "read_surf",
                    "surf_collide", "surf_modify", "collide",  "create_particles",
                    "fix",        "compute",    "dump",       "dump_modify", "region",
                    "variable",   "timestep",   "run",        "stats",      "include",
                    "read_grid",  "read_restart", "read_particles", "react", "surf_react",
                    "python"};
    ctx.styles["fix"]          = {"emit/face", "emit/surf", "ave/grid", "ave/surf"};
    ctx.styles["compute"]      = {"grid", "surf", "boundary", "thermal/grid"};
    ctx.styles["dump"]         = {"image", "grid", "surf", "particle"};
    ctx.styles["collide"]      = {"vss", "vhs", "hs"};
    ctx.styles["surf_collide"] = {"diffuse", "specular", "vanish"};
    ctx.styles["region"]       = {"block", "sphere", "cylinder"};
    // doc-derived command specs (command minArgs variadic) for the tested commands
    ctx.commandSpecs = parseSyntaxTable(
        "seed 1 0\n"
        "dimension 1 0\n"
        "global 0 1\n"
        "boundary 3 0\n"
        "create_box 6 0\n"
        "create_grid 3 1\n"
        "collide 1 1\n"
        "compute 2 1\n"
        "fix 2 1\n"
        "dump 5 1\n"
        "region 2 1\n"
        "variable 2 1\n"
        "timestep 1 0\n"
        "run 1 1\n"
        "read_surf 1 1\n"
        "include 1 0\n"
        "print 1 1\n");
    ctx.checkVocabulary = true;
    ctx.checkReferences = true;
    return ctx;
}

int countCode(const QList<Diagnostic> &d, const QString &code)
{
    int n = 0;
    for (const auto &x : d)
        if (x.code == code) ++n;
    return n;
}

} // namespace

TEST(InputCheck, CleanDeckHasNoDiagnostics)
{
    Context ctx = makeContext();
    const QString deck =
        "# a clean deck\n"
        "seed            12345\n"
        "dimension       3\n"
        "global          nrho 1.0 fnum 0.1\n"
        "boundary        o r r\n"
        "create_box      -2 2 -2 2 -2 2\n"
        "create_grid     10 10 10\n"
        "collide         vss air air.vss\n"
        "compute         g grid all all n\n"
        "fix             in emit/face air xlo\n"
        "run             100\n";
    const auto d = checkDeckText(deck, ctx);
    EXPECT_TRUE(d.isEmpty()) << (d.isEmpty() ? "" : d.first().message.toStdString());
}

TEST(InputCheck, UnknownCommandIsError)
{
    Context ctx = makeContext();
    const auto d = checkDeckText("dimensionn 3\n", ctx);
    ASSERT_EQ(countCode(d, "unknown-command"), 1);
    EXPECT_EQ(d.first().severity, Severity::Error);
    EXPECT_EQ(d.first().line, 1);
}

TEST(InputCheck, UnknownStyleAtSlotTwoAndSlotOne)
{
    Context ctx = makeContext();
    // fix style lives at token index 2
    auto d1 = checkDeckText("fix in emit/bogus air xlo\n", ctx);
    EXPECT_EQ(countCode(d1, "unknown-style"), 1);
    // collide style lives at token index 1
    auto d2 = checkDeckText("collide bogus air air.vss\n", ctx);
    EXPECT_EQ(countCode(d2, "unknown-style"), 1);
    // a valid style produces nothing
    auto d3 = checkDeckText("fix in emit/face air xlo\n", ctx);
    EXPECT_EQ(countCode(d3, "unknown-style"), 0);
}

TEST(InputCheck, AcceleratorSuffixStyleAccepted)
{
    Context ctx = makeContext();
    const auto d = checkDeckText("collide vss/kk air air.vss\n", ctx);
    EXPECT_EQ(countCode(d, "unknown-style"), 0);
}

TEST(InputCheck, MissingStyleArgumentIsTooFewArgs)
{
    Context ctx = makeContext();
    const auto d = checkDeckText("fix onlyid\n", ctx);
    EXPECT_EQ(countCode(d, "too-few-args"), 1);
}

TEST(InputCheck, DocSpecTooFewArgs)
{
    Context ctx = makeContext();
    // create_box needs 6 args; only 5 given
    const auto d = checkDeckText("create_box -2 2 -2 2 -2\n", ctx);
    EXPECT_EQ(countCode(d, "too-few-args"), 1);
    EXPECT_EQ(d.first().severity, Severity::Error);
}

TEST(InputCheck, DocSpecTooManyArgsForExactArity)
{
    Context ctx = makeContext();
    // boundary takes exactly 3 args (variadic=0); 4 given
    const auto d = checkDeckText("boundary o r r r\n", ctx);
    EXPECT_EQ(countCode(d, "too-many-args"), 1);
    // a variadic command never trips too-many
    const auto ok = checkDeckText("run 100 start 0 stop 1000 pre no\n", ctx);
    EXPECT_EQ(countCode(ok, "too-many-args"), 0);
}

TEST(InputCheck, MissingFileWarnsWhenProbeSaysAbsent)
{
    Context ctx = makeContext();
    ctx.fileExists = [](const QString &f) { return f == QStringLiteral("there.surf"); };
    auto absent = checkDeckText("read_surf missing.surf\n", ctx);
    EXPECT_EQ(countCode(absent, "missing-file"), 1);
    EXPECT_EQ(absent.first().severity, Severity::Warning);
    auto present = checkDeckText("read_surf there.surf\n", ctx);
    EXPECT_EQ(countCode(present, "missing-file"), 0);
    // globbed / variable names are not probed
    auto glob = checkDeckText("read_surf part.*.surf\n", ctx);
    EXPECT_EQ(countCode(glob, "missing-file"), 0);
}

TEST(InputCheck, UndefinedVariableReferences)
{
    Context ctx = makeContext();
    // $x, ${name}, v_name all flagged when undefined
    auto d = checkDeckText("run ${nsteps}\nfix f emit/face air xlo\ntimestep $t\n", ctx);
    EXPECT_EQ(countCode(d, "undefined-variable"), 2);
    // once defined, no warning; forward reference is fine (order-insensitive)
    auto ok = checkDeckText("run ${nsteps}\nvariable nsteps index 100\n", ctx);
    EXPECT_EQ(countCode(ok, "undefined-variable"), 0);
    // the injected gui_run variable is always considered defined
    auto guirun = checkDeckText("print \"run ${gui_run}\"\n", ctx);
    EXPECT_EQ(countCode(guirun, "undefined-variable"), 0);
}

TEST(InputCheck, UndefinedComputeAndFixReferences)
{
    Context ctx = makeContext();
    auto d = checkDeckText("dump d grid all 100 tmp.grid c_missing[1] f_gone\n", ctx);
    EXPECT_EQ(countCode(d, "undefined-compute"), 1);
    EXPECT_EQ(countCode(d, "undefined-fix"), 1);
    auto ok = checkDeckText(
        "compute g grid all all n\nfix a ave/grid all 1 1 1 c_g\n"
        "dump d grid all 100 tmp.grid c_g[1] f_a\n",
        ctx);
    EXPECT_EQ(countCode(ok, "undefined-compute"), 0);
    EXPECT_EQ(countCode(ok, "undefined-fix"), 0);
}

TEST(InputCheck, CommentsAreIgnored)
{
    Context ctx = makeContext();
    // a fully commented bad command and a trailing comment must not trip checks
    auto d = checkDeckText("# boguscmd 1 2 3\nrun 100   # trailing $undefined c_nope\n", ctx);
    EXPECT_TRUE(d.isEmpty());
}

TEST(InputCheck, LineContinuationJoined)
{
    Context ctx = makeContext();
    // a valid command split across lines with '&' must validate as one line
    auto d = checkDeckText("fix in &\n    emit/face air xlo\n", ctx);
    EXPECT_TRUE(d.isEmpty());
    // ... and an unknown style is still caught across the continuation
    auto bad = checkDeckText("fix in &\n    emit/bogus air xlo\n", ctx);
    EXPECT_EQ(countCode(bad, "unknown-style"), 1);
    EXPECT_EQ(bad.first().line, 1); // reported at the start of the logical line
}

TEST(InputCheck, VariableExpandedCommandOrStyleSkipped)
{
    Context ctx = makeContext();
    // a variable-expanded command name is not flagged as unknown
    auto d1 = checkDeckText("${mycmd} 1 2 3\n", ctx);
    EXPECT_EQ(countCode(d1, "unknown-command"), 0);
    // a variable-expanded style is not flagged as unknown
    auto d2 = checkDeckText("variable s index vss\ncollide ${s} air air.vss\n", ctx);
    EXPECT_EQ(countCode(d2, "unknown-style"), 0);
}

TEST(InputCheck, TripleQuotedPythonBlockIgnored)
{
    Context ctx = makeContext();
    // the embedded Python of a "python ... here \"\"\" ... \"\"\"" command must
    // not be validated as SPARTA commands
    const QString deck =
        "variable foo index 1\n"
        "python truncate return v_foo input 1 iv_arg format fi here \"\"\"\n"
        "def truncate(x):\n"
        "  return int(x)\n"
        "\"\"\"\n"
        "run 100\n";
    const auto d = checkDeckText(deck, ctx);
    EXPECT_EQ(countCode(d, "unknown-command"), 0);
    EXPECT_TRUE(d.isEmpty()) << (d.isEmpty() ? "" : d.first().message.toStdString());
}

TEST(InputCheck, ColumnPointsAtOffendingToken)
{
    Context ctx = makeContext();
    auto d = checkDeckText("fix in emit/bogus air xlo\n", ctx);
    ASSERT_EQ(d.size(), 1);
    // "emit/bogus" starts at column 8 (1-based) in "fix in emit/bogus ..."
    EXPECT_EQ(d.first().column, 8);
}
