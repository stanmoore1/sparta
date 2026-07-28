// -*- c++ -*- /////////////////////////////////////////////////////////////////////////
// SPARTA-GUI - A Graphical Tool to Learn and Explore the SPARTA DSMC Simulation Software
//
// Copyright (c) 2023, 2024, 2025, 2026  Axel Kohlmeyer
//
// Documentation: https://sparta.github.io/sparta-gui/
// Contact: akohlmey@gmail.com
//
// This software is distributed under the GNU General Public License version 2 or later.
////////////////////////////////////////////////////////////////////////////////////////

// Loading the SPARTA library, and what the wrapper does when that fails.
//
// Every other suite uses a library that loads.  This one uses the ones that do
// not: a file that is not there, a file that is not an ELF object, a truncated
// one, and a good one loaded on top of an open instance.  None of it had ever
// run, and it is the part of the wrapper where a mistake is not a wrong number
// but a jump into freed or absent memory.
//
// The reason it is worth a suite of its own: in plugin mode every library call
// goes through a function table, and that table is absent long before any
// instance exists -- from application start until the user has chosen a
// library.  A call that is not guarded, or guarded on the instance alone,
// dereferences the absent table to find the function and jumps through it.
// Writing these found two such calls.
//
// Two safeguards keep the instance and the table from disagreeing, and mutation
// testing shows they are redundant with each other: loadLib() closes the open
// instance before releasing the old table, and isOpen() requires both handles.
// Removing either alone leaves every check here passing, because the other
// still holds.  That is worth knowing rather than mistaking for a coverage gap
// -- and it is not where the reachable crashes were.  Those were the calls with
// no guard at all, which this suite does catch.

#include "spartawrapper.h"

#include "constants.h"

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QTemporaryDir>

namespace {

const char *testLibrary()
{
    static const QByteArray env = qgetenv("SPARTA_PLUGIN_LIB");
    if (!env.isEmpty()) return env.constData();
#if defined(SPARTA_TEST_LIBRARY_PATH)
    return SPARTA_TEST_LIBRARY_PATH;
#else
    return "";
#endif
}

#define REQUIRE_LIBRARY() \
    if (!*testLibrary() || !QFile::exists(testLibrary())) GTEST_SKIP() << "no shared libsparta"

class WrapperLoad : public ::testing::Test {
protected:
    /// A copy of the real library with its tail cut off: a plausible result of
    /// an interrupted download, and the case the ELF check exists for.
    QString truncatedLibrary(double keep = 0.6) const
    {
        const QString out = dir.filePath("libtruncated.so");
        QFile in(testLibrary());
        if (!in.open(QIODevice::ReadOnly)) return {};
        const QByteArray all = in.readAll();
        in.close();
        QFile f(out);
        if (!f.open(QIODevice::WriteOnly)) return {};
        f.write(all.left(static_cast<int>(all.size() * keep)));
        f.close();
        return out;
    }

    QString write(const QString &name, const QByteArray &bytes) const
    {
        const QString p = dir.filePath(name);
        QFile f(p);
        EXPECT_TRUE(f.open(QIODevice::WriteOnly));
        f.write(bytes);
        f.close();
        return p;
    }

    /// Touch every guarded entry point.  With nothing loaded these must all
    /// return their empty value rather than calling through a null table.
    static void pokeEverything(SpartaWrapper &w)
    {
        w.version();
        w.extractSetting("dimension");
        w.extractGlobal("boxlo");
        w.extractVariable("x");
        w.extractCompute("1", SpartaWrapper::SURF_STYLE, SpartaWrapper::ARRAY_TYPE);
        w.extractFix("1", SpartaWrapper::SURF_STYLE, SpartaWrapper::ARRAY_TYPE, 0, 0);
        w.getThermo("step");
        w.lastThermoAs<int>("num", 0);
        w.lastThermoString("keyword", 0);
        w.isRunning();
        w.hasError();
        w.idCount("compute");
        w.idName("compute", 0);
        w.styleCount("compute");
        w.styleName("compute", 0);
        w.command("print hello");
        w.commandsString("print hello\n");
        w.forceTimeout();
        w.close();
    }

    QTemporaryDir dir;
};

} // namespace

// -------------------------------------------------------------- bad libraries

TEST_F(WrapperLoad, AFileThatIsNotThereIsRefused)
{
    SpartaWrapper w;
    EXPECT_FALSE(w.loadLib(dir.filePath("no-such-library.so")));
    EXPECT_FALSE(w.isOpen());
}

TEST_F(WrapperLoad, AFileThatIsNotALibraryIsRefused)
{
    SpartaWrapper w;
    EXPECT_FALSE(w.loadLib(write("plain.so", "#!/bin/sh\necho not a library\n")));
    EXPECT_FALSE(w.isOpen());
}

TEST_F(WrapperLoad, AnEmptyFileIsRefused)
{
    SpartaWrapper w;
    EXPECT_FALSE(w.loadLib(write("empty.so", QByteArray())));
    EXPECT_FALSE(w.isOpen());
}

TEST_F(WrapperLoad, ATruncatedLibraryIsRejectedBeforeItReachesTheLoader)
{
    // handing a truncated object to dlopen() takes the process down inside the
    // dynamic linker, where there is nothing to catch: the ELF headers are
    // checked first precisely so this returns false instead of crashing
    REQUIRE_LIBRARY();
    const QString bad = truncatedLibrary();
    ASSERT_FALSE(bad.isEmpty());
    ASSERT_TRUE(QFile::exists(bad));

    SpartaWrapper w;
    EXPECT_FALSE(w.loadLib(bad)) << "a truncated library was accepted";
    EXPECT_FALSE(w.isOpen());
}

TEST_F(WrapperLoad, AnElfHeaderWithNoBodyIsRejected)
{
    // just the magic and enough of a header to look like an object: the segment
    // table promises content that is not in the file
    REQUIRE_LIBRARY();
    QFile in(testLibrary());
    ASSERT_TRUE(in.open(QIODevice::ReadOnly));
    const QByteArray head = in.read(4096);
    in.close();

    SpartaWrapper w;
    EXPECT_FALSE(w.loadLib(write("headonly.so", head)));
    EXPECT_FALSE(w.isOpen());
}

// ------------------------------------------------------- nothing loaded at all

TEST_F(WrapperLoad, WithNoLibraryEveryCallIsHarmless)
{
    // the guards are the only thing between an unconfigured application and a
    // jump through a null function table; the window calls several of these
    // before the user has ever chosen a library
    SpartaWrapper w;
    ASSERT_FALSE(w.isOpen());
    EXPECT_NO_FATAL_FAILURE(pokeEverything(w));

    EXPECT_EQ(w.version(), 0);
    EXPECT_EQ(w.extractSetting("dimension"), 0);
    EXPECT_EQ(w.extractGlobal("boxlo"), nullptr);
    EXPECT_FALSE(w.isRunning());
    EXPECT_EQ(w.idCount("compute"), 0);
}

TEST_F(WrapperLoad, OpeningWithNoLibraryLoadedDoesNotOpenAnything)
{
    // open() cannot use isOpen() as its guard -- that is false in exactly this
    // case -- so it carries its own, and this is what keeps it there
    SpartaWrapper w;
    char arg0[]  = "sparta";
    char *argv[] = {arg0, nullptr};
    EXPECT_NO_FATAL_FAILURE(w.open(1, argv));
    EXPECT_FALSE(w.isOpen()) << "an instance was opened with no library to open it through";
}

// ------------------------------------------------------------ a good library

TEST_F(WrapperLoad, TheRealLibraryLoadsAndReportsAVersionNewEnough)
{
    REQUIRE_LIBRARY();
    SpartaWrapper w;
    ASSERT_TRUE(w.loadLib(testLibrary())) << "the configured library was rejected";
    EXPECT_TRUE(w.hasPlugin());

    char arg0[]  = "sparta";
    char *argv[] = {arg0, nullptr};
    w.open(1, argv);
    ASSERT_TRUE(w.isOpen());
    EXPECT_GE(w.version(), 20000101) << "the version is not a YYYYMMDD date";
    w.close();
    EXPECT_FALSE(w.isOpen()) << "closing left the wrapper claiming to be open";
}

TEST_F(WrapperLoad, ARejectedLibraryClosesTheInstanceRatherThanLeavingItHalfAlive)
{
    // the user points preferences at a library that is refused while a
    // simulation instance is open.  That instance belongs to the library being
    // swapped out, so it has to be closed rather than left recorded: every
    // later call would otherwise be made through a table that has been freed.
    REQUIRE_LIBRARY();
    SpartaWrapper w;
    ASSERT_TRUE(w.loadLib(testLibrary()));
    char arg0[]  = "sparta";
    char *argv[] = {arg0, nullptr};
    w.open(1, argv);
    ASSERT_TRUE(w.isOpen());

    EXPECT_FALSE(w.loadLib(write("rubbish.so", "not a library at all\n")))
        << "a file that is not a library was accepted";
    EXPECT_FALSE(w.isOpen())
        << "the wrapper still claims to be open through a function table it has freed";
    EXPECT_NO_FATAL_FAILURE(pokeEverything(w));
}

TEST_F(WrapperLoad, TheLibraryCanBeLoadedAgainAfterARejectedOne)
{
    // and the application has to recover: pointing preferences back at a good
    // library must work rather than leaving the session permanently broken
    REQUIRE_LIBRARY();
    SpartaWrapper w;
    ASSERT_TRUE(w.loadLib(testLibrary()));
    ASSERT_FALSE(w.loadLib(write("rubbish2.so", "still not a library\n")));
    ASSERT_TRUE(w.loadLib(testLibrary())) << "a good library was refused after a bad one";

    char arg0[]  = "sparta";
    char *argv[] = {arg0, nullptr};
    w.open(1, argv);
    EXPECT_TRUE(w.isOpen());
    EXPECT_GT(w.version(), 0);
    w.close();
}

TEST_F(WrapperLoad, OnlyOneInstanceIsEverOpen)
{
    REQUIRE_LIBRARY();
    SpartaWrapper w;
    ASSERT_TRUE(w.loadLib(testLibrary()));
    char arg0[]  = "sparta";
    char *argv[] = {arg0, nullptr};
    w.open(1, argv);
    ASSERT_TRUE(w.isOpen());

    // a second open must be ignored rather than leaking the first instance
    EXPECT_NO_FATAL_FAILURE(w.open(1, argv));
    EXPECT_TRUE(w.isOpen());
    w.close();
    EXPECT_FALSE(w.isOpen());
}

// ------------------------------------------------------------ the port's stubs

TEST_F(WrapperLoad, ThePortsConstantStubsAnswerWithoutASimulator)
{
    // extract_pair, extract_atom and the GPU/OpenMP queries have no SPARTA
    // equivalent and were kept as constants so the upstream call sites still
    // compile.  They must answer the same way with and without a library --
    // anything else means one of them started reaching for a simulator.
    SpartaWrapper w;
    EXPECT_EQ(w.extractPair("cutoff"), nullptr);
    EXPECT_EQ(w.extractAtom("x"), nullptr);
    EXPECT_FALSE(w.hasGpuDevice());
    EXPECT_FALSE(w.configHasCurlSupport());
    EXPECT_FALSE(w.configHasPackage("OPENMP"));

    if (!*testLibrary() || !QFile::exists(testLibrary())) return;
    ASSERT_TRUE(w.loadLib(testLibrary()));
    char arg0[]  = "sparta";
    char *argv[] = {arg0, nullptr};
    w.open(1, argv);
    ASSERT_TRUE(w.isOpen());

    EXPECT_EQ(w.extractPair("cutoff"), nullptr);
    EXPECT_EQ(w.extractAtom("x"), nullptr);
    EXPECT_FALSE(w.hasGpuDevice());
    EXPECT_FALSE(w.configHasCurlSupport());
    EXPECT_FALSE(w.configHasPackage("OPENMP"));
    w.close();
}

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
