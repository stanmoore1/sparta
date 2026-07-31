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

// The bundled lookup tables.
//
// Four hand-maintained tables drive things the user reaches for by name: which
// documentation page a command opens, which computes and fixes the image viewer
// offers to colour by, which commands the syntax highlighter knows, and which
// commands the GUI handles itself instead of passing to the library. Nothing
// checked any of them, and every one is a list of strings that has to agree
// with something outside the file -- a page in the SPARTA manual, a style the
// library was built with. They drift silently: an entry that no longer resolves
// shows up as a help link that opens nothing, or a colour-by choice that makes
// the render fail, and only for the person who happens to pick that entry.

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSet>
#include <QString>
#include <QStringList>
#include <QRegularExpression>
#include <QTextStream>

#include "spartawrapper.h"

namespace {

// The SPARTA documentation sources, beside the GUI in the same checkout. The
// manual is Sphinx, and its HTML is built rather than committed, so what the
// checkout has to compare against is doc/src/<page>.rst -- one reST source per
// rendered <page>.html.
QString docDir()
{
#if defined(SPARTA_DOC_DIR)
    return QString(SPARTA_DOC_DIR);
#else
    return QString();
#endif
}

// This manual's own reST sources.
QString guiDocDir()
{
#if defined(SPARTA_GUI_DOC_DIR)
    return QString(SPARTA_GUI_DOC_DIR);
#else
    return QString();
#endif
}

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

// Non-empty, non-comment lines of a bundled resource, whitespace trimmed.
QStringList lines(const QString &resource)
{
    QStringList out;
    QFile f(resource);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return out;
    QTextStream in(&f);
    while (!in.atEnd()) {
        const QString line = in.readLine().trimmed();
        if (line.isEmpty() || line.startsWith('#')) continue;
        out << line;
    }
    return out;
}

} // namespace

// ------------------------------------------------------------- help_index

TEST(Resources, HelpIndexIsWellFormed)
{
    const QStringList rows = lines(":/help_index.table");
    ASSERT_GT(rows.size(), 100) << "help_index.table is empty or did not load from the resource "
                                   "bundle -- every F1 lookup would open nothing";

    // Each row is <page.html> <command> [<style>]. The key is the whole
    // command, style included: "fix" and "fix ablate" are different entries
    // pointing at different pages, and treating the columns as separate names
    // makes them look like duplicates of each other.
    QSet<QString> commands;
    for (const QString &row : rows) {
        const QStringList cols = row.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        ASSERT_GE(cols.size(), 2) << "not a page/command row: " << row.toStdString();
        ASSERT_LE(cols.size(), 3) << "more than a command and a style: " << row.toStdString();
        EXPECT_TRUE(cols[0].endsWith(".html"))
            << row.toStdString() << " does not name an HTML page";

        const QString key = cols.mid(1).join(' ');
        EXPECT_FALSE(commands.contains(key))
            << key.toStdString() << " is listed twice, so which page it opens depends on "
                                    "which row is read last";
        commands.insert(key);
    }
}

TEST(Resources, EveryHelpPageExists)
{
    if (docDir().isEmpty()) GTEST_SKIP() << "no documentation tree configured";
    const QDir doc(docDir());
    ASSERT_TRUE(doc.exists()) << docDir().toStdString();

    const QStringList rows = lines(":/help_index.table");
    ASSERT_FALSE(rows.isEmpty());

    QStringList missing;
    for (const QString &row : rows) {
        const QString page = row.section(QRegularExpression("\\s+"), 0, 0);
        QString source     = page;
        source.replace(QRegularExpression("\\.html$"), ".rst");
        if (!QFileInfo::exists(doc.filePath(source))) missing << page;
    }
    EXPECT_TRUE(missing.isEmpty())
        << missing.size() << " help pages do not exist, so those commands' help opens nothing: "
        << missing.join(", ").toStdString();
}

// This manual links into the SPARTA manual by URL, which is not something
// either Sphinx build can check: the two are separate projects, so a page that
// gets renamed or dropped on the SPARTA side leaves a link here that still
// builds and still looks fine, and only 404s for the reader who clicks it.
// Both trees are in the same checkout, so the link target can be checked
// against the reST source it will be rendered from.
TEST(Resources, EveryLinkIntoTheSpartaManualNamesAPageThatExists)
{
    if (docDir().isEmpty() || guiDocDir().isEmpty()) GTEST_SKIP() << "no documentation trees";
    const QDir doc(docDir());
    const QDir guidoc(guiDocDir());
    ASSERT_TRUE(doc.exists()) << docDir().toStdString();
    ASSERT_TRUE(guidoc.exists()) << guiDocDir().toStdString();

    const QRegularExpression link("https://sparta\\.github\\.io/doc/([A-Za-z0-9_]+)\\.html");
    QStringList dead;
    int checked = 0;
    for (const QString &name : guidoc.entryList({"*.rst"}, QDir::Files)) {
        QFile f(guidoc.filePath(name));
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) continue;
        QTextStream in(&f);
        int lineno = 0;
        while (!in.atEnd()) {
            const QString line = in.readLine();
            ++lineno;
            auto it = link.globalMatch(line);
            while (it.hasNext()) {
                const QString page = it.next().captured(1);
                ++checked;
                if (!QFileInfo::exists(doc.filePath(page + ".rst")))
                    dead << QString("%1:%2: %3.html").arg(name).arg(lineno).arg(page);
            }
        }
    }
    ASSERT_GT(checked, 0) << "found no links into the SPARTA manual at all, so this checks nothing";
    EXPECT_TRUE(dead.isEmpty())
        << dead.size() << " links point at pages the SPARTA manual does not have: "
        << dead.join(", ").toStdString();
}

// ------------------------------------------------------------ image_style

TEST(Resources, ImageStyleTableIsWellFormed)
{
    const QStringList rows = lines(":/image_style.table");
    ASSERT_FALSE(rows.isEmpty()) << "image_style.table did not load -- the image viewer would "
                                    "offer nothing to colour by";

    for (const QString &row : rows) {
        const QStringList cols = row.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        ASSERT_EQ(cols.size(), 2) << "not a <kind> <style> row: " << row.toStdString();
        EXPECT_TRUE(cols[0] == "compute" || cols[0] == "fix")
            << row.toStdString() << ": only computes and fixes produce per-element data";
    }
}

// The table is a catalogue of every style that supports dump image, not a
// description of one build: fft/grid needs the optional FFT package, the /kk
// variants need KOKKOS, and the viewer filters the list against the running
// instance. So what has to hold for every entry, in every build, is that the
// viewer's Help button has somewhere to send the user.
TEST(Resources, EveryImageStyleIsDocumented)
{
    QSet<QString> documented;
    for (const QString &row : lines(":/help_index.table")) {
        const QStringList cols = row.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (cols.size() >= 2) documented.insert(cols.mid(1).join(' '));
    }
    ASSERT_FALSE(documented.isEmpty());

    QStringList undocumented;
    for (const QString &row : lines(":/image_style.table"))
        if (!documented.contains(row)) undocumented << row;

    EXPECT_TRUE(undocumented.isEmpty())
        << undocumented.size() << " colour-by choices have no help page: "
        << undocumented.join(", ").toStdString();
}

// And of the styles this particular build does have, the viewer must be
// offering them -- an entry spelled differently from the library's own name
// never matches, so the choice silently disappears from the dialog.
TEST(Resources, TheStylesThisBuildHasAreSpelledAsTheLibrarySpellsThem)
{
    if (!*testLibrary()) GTEST_SKIP() << "no shared libsparta to ask";

    SpartaWrapper sparta;
    ASSERT_TRUE(sparta.loadLib(testLibrary())) << "could not load " << testLibrary();
    char arg0[]  = "sparta";
    char *argv[] = {arg0, nullptr};
    sparta.open(1, argv);
    ASSERT_TRUE(sparta.isOpen()) << "could not start a SPARTA instance from " << testLibrary();

    QSet<QString> have;
    for (const char *kind : {"compute", "fix"}) {
        const int n = sparta.styleCount(kind);
        for (int i = 0; i < n; ++i)
            have.insert(QString(kind) + " " + sparta.styleName(kind, i));
    }
    ASSERT_FALSE(have.isEmpty()) << "the library reported no compute or fix styles at all";

    int matched = 0;
    for (const QString &row : lines(":/image_style.table"))
        if (have.contains(row)) ++matched;

    // Not all of them: this build has neither the FFT package nor KOKKOS. But
    // if none matched, the two sides disagree about how a style is written and
    // the viewer would offer nothing at all.
    EXPECT_GT(matched, 0) << "not one entry of image_style.table names a style this SPARTA has; "
                             "the table and the library disagree about style names";
}

// -------------------------------------------------------- internal commands

TEST(Resources, InternalCommandsAreNotAlsoLibraryCommands)
{
    const QStringList internal = lines(":/sparta_internal_commands.txt");
    ASSERT_FALSE(internal.isEmpty());

    QSet<QString> seen;
    for (const QString &cmd : internal) {
        EXPECT_FALSE(cmd.contains(' ')) << cmd.toStdString() << " is not a single command name";
        EXPECT_FALSE(seen.contains(cmd)) << cmd.toStdString() << " is listed twice";
        seen.insert(cmd);
    }
}

// ------------------------------------------------------- command_syntax

TEST(Resources, CommandSyntaxTableIsWellFormed)
{
    const QStringList rows = lines(":/command_syntax.table");
    ASSERT_FALSE(rows.isEmpty()) << "command_syntax.table did not load";

    QSet<QString> seen;
    for (const QString &row : rows) {
        const QString cmd = row.section(QRegularExpression("\\s+"), 0, 0);
        EXPECT_FALSE(cmd.isEmpty()) << row.toStdString();
        EXPECT_FALSE(seen.contains(cmd))
            << cmd.toStdString() << " has two syntax lines; the second is unreachable";
        seen.insert(cmd);
    }
}

// Every command the syntax table describes should also have somewhere to send
// the user for the full story.
TEST(Resources, EverySyntaxEntryHasHelp)
{
    QSet<QString> documented;
    for (const QString &row : lines(":/help_index.table")) {
        const QStringList cols = row.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        for (int i = 1; i < cols.size(); ++i)
            documented.insert(cols[i]);
    }
    ASSERT_FALSE(documented.isEmpty());

    QStringList orphans;
    for (const QString &row : lines(":/command_syntax.table")) {
        const QString cmd = row.section(QRegularExpression("\\s+"), 0, 0);
        if (!documented.contains(cmd)) orphans << cmd;
    }
    EXPECT_TRUE(orphans.isEmpty())
        << orphans.size() << " commands have a syntax hint but no help page: "
        << orphans.join(", ").toStdString();
}
