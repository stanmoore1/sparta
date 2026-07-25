// Unit tests for the archived-run store (src/runhistory.cpp).
//
// RunArchive, which renders a record into a report, already had tests. The
// store around it did not: archiving a finished run, copying its images
// somewhere they will survive the working directory being cleaned, writing the
// index back out, reloading it in the next session, and deleting a record and
// its files. That is the part with a filesystem underneath it, so it is the
// part where a failure loses a user's data rather than mis-drawing a page.
//
// Every test runs against its own AppDataLocation so nothing touches a real
// history, and so a test cannot see what an earlier one archived.

#include "runhistory.h"

#include "runarchive.h"

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTest>
#include <QSignalSpy>
#include <QStandardPaths>
#include <QTemporaryDir>

#include "gtest/gtest.h"

namespace {

/// Redirects QStandardPaths::AppDataLocation for the lifetime of the object,
/// so each test archives into an empty directory of its own.
class ScratchAppData {
public:
    ScratchAppData()
    {
        // XDG_DATA_HOME, not QStandardPaths::setTestModeEnabled(): test mode
        // answers with a fixed ~/.qttest path and ignores the environment, so
        // every test would have shared one history and each would have seen
        // what the ones before it archived.
        previous_ = qgetenv("XDG_DATA_HOME");
        dir_      = new QTemporaryDir;
        qputenv("XDG_DATA_HOME", dir_->path().toLocal8Bit());
    }
    ~ScratchAppData()
    {
        if (previous_.isEmpty()) qunsetenv("XDG_DATA_HOME");
        else qputenv("XDG_DATA_HOME", previous_);
        delete dir_;
    }
    QString path() const { return dir_->path(); }

private:
    QTemporaryDir *dir_ = nullptr;
    QByteArray previous_;
};

RunArchive::RunRecord makeRecord(const QString &id, const QString &deck = "in.circle")
{
    RunArchive::RunRecord r;
    r.id         = id;
    r.timestamp  = "2026-07-25T12:00:00";
    r.deckName   = deck;
    r.deckText   = "units si\nrun 100\n";
    r.logText    = "SPARTA (24 Sep 2025)\nstep  np\n0  0\n";
    r.thermoYaml = "keywords: [step, np]\ndata:\n  - [0, 0]\n";
    r.workDir    = "/tmp";
    r.status     = "ok";
    r.metadata.insert("nprocs", "1");
    return r;
}

/// A tiny PNG on disk, to stand in for a rendered frame.
QString makeImage(const QString &path)
{
    QImage img(4, 4, QImage::Format_RGB32);
    img.fill(Qt::red);
    img.save(path, "PNG");
    return path;
}

TEST(RunHistory, StartsEmpty)
{
    ScratchAppData scratch;
    RunHistory hist;
    EXPECT_EQ(hist.count(), 0);
    ASSERT_NE(hist.model(), nullptr);
    EXPECT_EQ(hist.model()->rowCount(), 0);
}

TEST(RunHistory, ArchivingARunRecordsIt)
{
    ScratchAppData scratch;
    RunHistory hist;
    QSignalSpy changed(&hist, &RunHistory::changed);

    hist.archive(makeRecord("run-1"), {});
    EXPECT_EQ(hist.count(), 1);
    EXPECT_EQ(hist.at(0).deckName, QString("in.circle"));
    EXPECT_EQ(changed.count(), 1) << "the panel is only refreshed by this signal";
    EXPECT_EQ(hist.model()->rowCount(), 1);
}

TEST(RunHistory, NewestRunComesFirst)
{
    // the panel shows the list top-down and a user looking for "the run I just
    // did" should not have to scroll to the bottom of a long history
    ScratchAppData scratch;
    RunHistory hist;
    hist.archive(makeRecord("run-1", "first.in"), {});
    hist.archive(makeRecord("run-2", "second.in"), {});
    ASSERT_EQ(hist.count(), 2);
    EXPECT_EQ(hist.at(0).deckName, QString("second.in"));
    EXPECT_EQ(hist.at(1).deckName, QString("first.in"));
}

TEST(RunHistory, ImagesAreCopiedIntoTheArchive)
{
    // The point of archiving is that the record outlives the run. Images that
    // stayed where the run wrote them would be gone the moment the working
    // directory was cleaned, and the report would show broken frames.
    ScratchAppData scratch;
    QTemporaryDir work;
    ASSERT_TRUE(work.isValid());
    const QString a = makeImage(work.filePath("frame.0000.png"));
    const QString b = makeImage(work.filePath("frame.0001.png"));

    RunHistory hist;
    hist.archive(makeRecord("run-1"), {a, b});
    ASSERT_EQ(hist.count(), 1);

    const QStringList kept = hist.at(0).imageFiles;
    ASSERT_EQ(kept.size(), 2);
    for (const QString &f : kept) {
        EXPECT_TRUE(QFileInfo::exists(f)) << f.toStdString();
        EXPECT_FALSE(f.startsWith(work.path()))
            << "the record still points into the run's own directory";
        EXPECT_TRUE(f.startsWith(hist.recordDir(0)))
            << "the copy did not land in this record's archive directory";
    }

    // and deleting the originals leaves the archive intact
    QFile::remove(a);
    QFile::remove(b);
    for (const QString &f : kept) EXPECT_TRUE(QFileInfo::exists(f));
}

TEST(RunHistory, MissingImagesAreSkippedRatherThanRecorded)
{
    // a frame the run promised but never wrote must not become a dead entry
    // that the report then fails to inline
    ScratchAppData scratch;
    QTemporaryDir work;
    const QString good = makeImage(work.filePath("there.png"));

    RunHistory hist;
    hist.archive(makeRecord("run-1"), {good, work.filePath("missing.png")});
    ASSERT_EQ(hist.count(), 1);
    EXPECT_EQ(hist.at(0).imageFiles.size(), 1);
}

TEST(RunHistory, ReloadsWhatWasArchived)
{
    // the index is written to disk on every archive; the next session builds
    // its list from that file alone
    ScratchAppData scratch;
    {
        RunHistory hist;
        hist.archive(makeRecord("run-1", "alpha.in"), {});
        hist.archive(makeRecord("run-2", "beta.in"), {});
        ASSERT_EQ(hist.count(), 2);
    }

    RunHistory reopened;
    ASSERT_EQ(reopened.count(), 2) << "the index was not written, or not read back";
    EXPECT_EQ(reopened.at(0).deckName, QString("beta.in"));
    EXPECT_EQ(reopened.at(1).deckName, QString("alpha.in"));
    // the embedded content survives the round trip, not just the file names
    EXPECT_EQ(reopened.at(0).logText, QString("SPARTA (24 Sep 2025)\nstep  np\n0  0\n"));
    EXPECT_EQ(reopened.at(0).status, QString("ok"));
    EXPECT_EQ(reopened.at(0).metadata.value("nprocs"), QString("1"));
}

TEST(RunHistory, DeletingARecordRemovesItsFilesToo)
{
    ScratchAppData scratch;
    QTemporaryDir work;
    const QString img = makeImage(work.filePath("frame.png"));

    RunHistory hist;
    hist.archive(makeRecord("run-1"), {img});
    ASSERT_EQ(hist.count(), 1);
    const QString dir = hist.recordDir(0);
    ASSERT_TRUE(QDir(dir).exists());

    hist.removeRecord(0);
    EXPECT_EQ(hist.count(), 0);
    EXPECT_FALSE(QDir(dir).exists())
        << "the record went from the list but its files stayed on disk";

    // and the deletion is persistent
    RunHistory reopened;
    EXPECT_EQ(reopened.count(), 0);
}

TEST(RunHistory, DeletingAnOutOfRangeRowIsIgnored)
{
    ScratchAppData scratch;
    RunHistory hist;
    hist.archive(makeRecord("run-1"), {});
    hist.removeRecord(-1);
    hist.removeRecord(7);
    EXPECT_EQ(hist.count(), 1);
}

TEST(RunHistory, WritesAReportForARecord)
{
    ScratchAppData scratch;
    QTemporaryDir work;
    const QString img = makeImage(work.filePath("frame.png"));

    RunHistory hist;
    hist.archive(makeRecord("run-1"), {img});

    const QString path = hist.writeReportHtml(0);
    ASSERT_FALSE(path.isEmpty());
    ASSERT_TRUE(QFileInfo::exists(path));

    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::ReadOnly));
    const QString html = QString::fromUtf8(f.readAll());
    EXPECT_TRUE(html.contains("in.circle")) << "the report does not name the deck it is about";
    EXPECT_TRUE(html.contains("units si")) << "the input deck is not in the report";
    // images are inlined, so the report is one file that can be sent to someone
    EXPECT_TRUE(html.contains("data:image")) << "the frame was linked rather than embedded";
}

TEST(RunHistory, RefusesAReportForARowThatIsNotThere)
{
    ScratchAppData scratch;
    RunHistory hist;
    EXPECT_TRUE(hist.writeReportHtml(0).isEmpty());
    EXPECT_TRUE(hist.writeReportHtml(-1).isEmpty());
}

TEST(RunHistory, ComparesTwoRuns)
{
    ScratchAppData scratch;
    RunHistory hist;
    hist.archive(makeRecord("run-1", "alpha.in"), {});
    hist.archive(makeRecord("run-2", "beta.in"), {});

    const QString path = hist.writeComparisonHtml(0, 1);
    ASSERT_FALSE(path.isEmpty());
    ASSERT_TRUE(QFileInfo::exists(path));

    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::ReadOnly));
    const QString html = QString::fromUtf8(f.readAll());
    EXPECT_TRUE(html.contains("alpha.in"));
    EXPECT_TRUE(html.contains("beta.in")) << "a comparison that names only one side";
}

TEST(RunHistory, RefusesToCompareARowThatIsNotThere)
{
    ScratchAppData scratch;
    RunHistory hist;
    hist.archive(makeRecord("run-1"), {});
    EXPECT_TRUE(hist.writeComparisonHtml(0, 5).isEmpty());
    EXPECT_TRUE(hist.writeComparisonHtml(-1, 0).isEmpty());
}

// ---------------------------------------------------------------------------
// The table model the panel shows
// ---------------------------------------------------------------------------

TEST(HistoryModel, ShowsOneRowPerRecordAndTheDocumentedColumns)
{
    ScratchAppData scratch;
    QTemporaryDir work;
    const QString img = makeImage(work.filePath("frame.png"));

    RunHistory hist;
    hist.archive(makeRecord("run-1", "alpha.in"), {img});

    HistoryModel *m = hist.model();
    ASSERT_NE(m, nullptr);
    ASSERT_EQ(m->rowCount(), 1);
    ASSERT_EQ(m->columnCount(), int(HistoryModel::NCols));

    EXPECT_EQ(m->data(m->index(0, HistoryModel::ColDeck)).toString(), QString("alpha.in"));
    EXPECT_EQ(m->data(m->index(0, HistoryModel::ColStatus)).toString(), QString("ok"));
    EXPECT_EQ(m->data(m->index(0, HistoryModel::ColImages)).toInt(), 1);
    EXPECT_FALSE(m->data(m->index(0, HistoryModel::ColTime)).toString().isEmpty());

    for (int c = 0; c < HistoryModel::NCols; ++c)
        EXPECT_FALSE(m->headerData(c, Qt::Horizontal).toString().isEmpty())
            << "column " << c << " has no header, so the table shows a bare number";
}

TEST(HistoryModel, AnInvalidIndexYieldsNothing)
{
    ScratchAppData scratch;
    RunHistory hist;
    hist.archive(makeRecord("run-1"), {});
    HistoryModel *m = hist.model();
    EXPECT_FALSE(m->data(m->index(5, 0)).isValid());
    EXPECT_FALSE(m->data(QModelIndex()).isValid());
}

} // namespace

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("SPARTA-GUI-tests");
    QCoreApplication::setApplicationName("runhistory-tests");
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
