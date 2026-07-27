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
//
// Integrity checking of downloaded files: the SHA256SUMS lookup, the local
// hash, and the decision between them.
//
// This is the only code in the application that decides whether a file fetched
// off the network may be kept, and none of it had ever been executed.  A parser
// that fails to find the entry, or a comparison that always agrees, downgrades
// the check to nothing at all while still reporting success -- which is exactly
// what a broken integrity check looks like from outside.
//
// The tests run against file:// URLs, so the real fetch-parse-compare path runs
// end to end with no network and no server: QNetworkAccessManager treats a
// local directory as the "remote" one, SHA256SUMS and all.

#include "urldownloader.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QCryptographicHash>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QMessageBox>
#include <QTemporaryDir>
#include <QTimer>
#include <QUrl>

namespace {

/// Dismisses the mismatch dialog and remembers what it said.
class Modals : public QObject {
public:
    explicit Modals(int budgetMs = 10000) : left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &Modals::poll);
        timer.start();
    }
    QStringList messages;
    int boxes = 0;

    [[nodiscard]] bool said(const QString &needle) const
    {
        for (const auto &m : messages)
            if (m.contains(needle)) return true;
        return false;
    }
    [[nodiscard]] QString all() const { return messages.join(" | "); }

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            else if (m) m->close();
            return;
        }
        if (!m) return;
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            ++boxes;
            messages << box->text() + " " + box->informativeText() + " " + box->detailedText();
            box->accept();
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
        else m->close();
    }
    QTimer timer;
    int left;
};

class Checksum : public ::testing::Test {
protected:
    /// write @p text to @p name in the "remote" directory and return its URL
    QString publish(const QString &name, const QByteArray &bytes) const
    {
        const QString path = dir.filePath(name);
        QFile f(path);
        EXPECT_TRUE(f.open(QIODevice::WriteOnly));
        f.write(bytes);
        f.close();
        return QUrl::fromLocalFile(path).toString();
    }

    QString path(const QString &name) const { return dir.filePath(name); }

    static QString sha256(const QByteArray &bytes)
    {
        return QCryptographicHash::hash(bytes, QCryptographicHash::Sha256).toHex().toLower();
    }

    /// publish a SHA256SUMS beside the payload, with the given body
    void publishSums(const QByteArray &body) const { publish("SHA256SUMS", body); }

    QTemporaryDir dir;
};

const QByteArray kPayload = "in.circle\nrun 100\n";

} // namespace

// -------------------------------------------------------------- the local hash

TEST_F(Checksum, HashesAFileToItsKnownSha256)
{
    // the empty string's SHA-256 is a published constant, so this pins the
    // algorithm and the encoding rather than merely agreeing with itself
    publish("empty.bin", QByteArray());
    EXPECT_EQ(URLDownloader::getLocalChecksum(path("empty.bin")),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    publish("abc.bin", "abc");
    EXPECT_EQ(URLDownloader::getLocalChecksum(path("abc.bin")),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST_F(Checksum, ADifferentByteGivesADifferentHash)
{
    publish("a.bin", "the payload");
    publish("b.bin", "the payloae"); // one byte apart
    EXPECT_NE(URLDownloader::getLocalChecksum(path("a.bin")),
              URLDownloader::getLocalChecksum(path("b.bin")));
}

TEST_F(Checksum, AFileThatCannotBeReadHasNoHash)
{
    EXPECT_TRUE(URLDownloader::getLocalChecksum(path("does-not-exist")).isEmpty());
    QDir(dir.path()).mkdir("adirectory");
    EXPECT_TRUE(URLDownloader::getLocalChecksum(path("adirectory")).isEmpty());
}

// ------------------------------------------------------------ the remote entry

TEST_F(Checksum, FindsTheEntryForTheFileBeingDownloaded)
{
    const QString url = publish("payload.tar.gz", kPayload);
    publishSums(QString("%1  other.tar.gz\n%2  payload.tar.gz\n%3  more.tar.gz\n")
                    .arg(sha256("other"), sha256(kPayload), sha256("more"))
                    .toUtf8());

    URLDownloader dl;
    EXPECT_EQ(dl.getRemoteChecksum(url), sha256(kPayload))
        << "the wrong line of SHA256SUMS was matched";
}

TEST_F(Checksum, AcceptsTheSpacingsRealSha256sumsFilesUse)
{
    // "<hash>  <name>" (binary), "<hash> <name>", "*<name>" and "./<name>" are
    // all produced by sha256sum and its relatives
    struct Case {
        const char *label;
        QString line;
    };
    const QString h = sha256(kPayload);
    const QList<Case> cases{
        {"two spaces", h + "  payload.tar.gz"},
        {"one space", h + " payload.tar.gz"},
        {"star prefix", h + "  *payload.tar.gz"},
        {"dot-slash prefix", h + "  ./payload.tar.gz"},
        {"trailing whitespace", h + "  payload.tar.gz   "},
    };

    for (const auto &c : cases) {
        const QString url = publish("payload.tar.gz", kPayload);
        publishSums(("# a comment line\n\n" + c.line + "\n").toUtf8());
        URLDownloader dl;
        EXPECT_EQ(dl.getRemoteChecksum(url), h) << c.label;
    }
}

TEST_F(Checksum, TheHashComesBackLowercaseHoweverItWasWritten)
{
    const QString url = publish("payload.tar.gz", kPayload);
    publishSums((sha256(kPayload).toUpper() + "  payload.tar.gz\n").toUtf8());

    URLDownloader dl;
    EXPECT_EQ(dl.getRemoteChecksum(url), sha256(kPayload))
        << "an uppercase SHA256SUMS entry would never compare equal";
}

TEST_F(Checksum, AFileWithNoEntryHasNoRemoteChecksum)
{
    const QString url = publish("payload.tar.gz", kPayload);
    publishSums((sha256("something else") + "  a-different-file.tar.gz\n").toUtf8());

    URLDownloader dl;
    EXPECT_TRUE(dl.getRemoteChecksum(url).isEmpty())
        << "an entry for another file was accepted as this one's";
}

TEST_F(Checksum, ANameThatMerelyContainsTheFilenameIsNotAMatch)
{
    // "not-payload.tar.gz" ends with the name we want; a substring match here
    // would hand back a hash belonging to a different file entirely
    const QString url = publish("payload.tar.gz", kPayload);
    publishSums((sha256("impostor") + "  not-payload.tar.gz\n").toUtf8());

    URLDownloader dl;
    EXPECT_TRUE(dl.getRemoteChecksum(url).isEmpty())
        << "a different file's entry matched on a suffix";
}

TEST_F(Checksum, NoSumsFileAtAllMeansNoRemoteChecksum)
{
    const QString url = publish("payload.tar.gz", kPayload);
    URLDownloader dl;
    EXPECT_TRUE(dl.getRemoteChecksum(url).isEmpty());
}

TEST_F(Checksum, MalformedLinesAreSkippedRatherThanMisparsed)
{
    const QString url = publish("payload.tar.gz", kPayload);
    publishSums(QString("garbage-with-no-space\n"
                        "\n"
                        "   \n"
                        "%1  payload.tar.gz\n")
                    .arg(sha256(kPayload))
                    .toUtf8());

    URLDownloader dl;
    EXPECT_EQ(dl.getRemoteChecksum(url), sha256(kPayload))
        << "a malformed line stopped the parse before the real entry";
}

TEST_F(Checksum, AUrlWithNoPathHasNoChecksumToLookUp)
{
    URLDownloader dl;
    EXPECT_TRUE(dl.getRemoteChecksum("payload.tar.gz").isEmpty());
}

// ------------------------------------------------------------ the whole check

// download() is what actually consumes verifyChecksum(): it fetches, writes,
// verifies, and deletes the file if the hash does not match.  Driving it over
// file:// URLs exercises that decision without a network.
TEST_F(Checksum, AMatchingHashKeepsTheDownloadedFile)
{
    Modals modals;
    const QString url = publish("payload.tar.gz", kPayload);
    publishSums((sha256(kPayload) + "  payload.tar.gz\n").toUtf8());

    URLDownloader dl;
    const QString out = path("downloaded.tar.gz");
    ASSERT_TRUE(dl.download(url, out)) << dl.errorString().toStdString();
    EXPECT_TRUE(QFile::exists(out));
    EXPECT_EQ(URLDownloader::getLocalChecksum(out), sha256(kPayload));
    EXPECT_EQ(modals.boxes, 0) << "a good download complained: " << modals.all().toStdString();
}

TEST_F(Checksum, AMismatchedHashIsRefusedAndTheFileIsDeleted)
{
    // the reason all of this exists: a file whose contents are not what the
    // publisher signed must not be left on disk for something else to open
    Modals modals;
    const QString url = publish("payload.tar.gz", kPayload);
    publishSums((sha256("what the publisher meant to ship") + "  payload.tar.gz\n").toUtf8());

    URLDownloader dl;
    const QString out = path("downloaded.tar.gz");
    EXPECT_FALSE(dl.download(url, out)) << "a file failing its checksum was accepted";
    EXPECT_FALSE(QFile::exists(out)) << "the rejected file was left on disk";
    EXPECT_TRUE(dl.errorString().contains("checksum mismatch")) << dl.errorString().toStdString();
    EXPECT_TRUE(modals.said("checksum")) << "the user was not told why: " << modals.all().toStdString();
    EXPECT_TRUE(modals.said("payload.tar.gz")) << modals.all().toStdString();
}

TEST_F(Checksum, TheMismatchDialogShowsBothHashes)
{
    Modals modals;
    const QString url = publish("payload.tar.gz", kPayload);
    const QString wrong = sha256("not this");
    publishSums((wrong + "  payload.tar.gz\n").toUtf8());

    URLDownloader dl;
    dl.download(url, path("downloaded.tar.gz"));
    EXPECT_TRUE(modals.said(wrong)) << "the expected hash was not shown: " << modals.all().toStdString();
    EXPECT_TRUE(modals.said(sha256(kPayload)))
        << "the actual hash was not shown: " << modals.all().toStdString();
}

TEST_F(Checksum, WithNoSumsFilePublishedTheDownloadIsKept)
{
    // deliberate policy: a publisher who ships no SHA256SUMS cannot be checked,
    // and refusing every such download would break the feature outright.  It is
    // pinned here so that changing it is a decision rather than an accident.
    Modals modals;
    const QString url = publish("payload.tar.gz", kPayload);

    URLDownloader dl;
    const QString out = path("downloaded.tar.gz");
    EXPECT_TRUE(dl.download(url, out)) << dl.errorString().toStdString();
    EXPECT_TRUE(QFile::exists(out));
    EXPECT_EQ(modals.boxes, 0) << modals.all().toStdString();
}

TEST_F(Checksum, ASourceThatDoesNotExistIsAnErrorNotAnEmptyFile)
{
    Modals modals;
    URLDownloader dl;
    const QString out = path("downloaded.tar.gz");
    EXPECT_FALSE(dl.download(QUrl::fromLocalFile(path("no-such-file")).toString(), out));
    EXPECT_FALSE(QFile::exists(out)) << "a failed download left a file behind";
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_checksum.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
