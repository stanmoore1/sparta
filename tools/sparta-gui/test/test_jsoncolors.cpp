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

// The species colour file: what "Save Colors" writes and "Load Colors" reads.
//
// It is the one thing the image viewer persists that a user can hand to
// someone else, so it has a header identifying what it is and a revision to
// refuse a format it does not understand.  Both ends go through a file dialog,
// which is why neither had been checked -- driving that from a timer is all it
// takes.

#include "imageviewer_internal.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QTemporaryDir>
#include <QTimer>

namespace {

// Answers the file dialog, and records what any message box said.
class Answer : public QObject {
public:
    explicit Answer(QString path = QString(), int budgetMs = 3000) :
        answer(std::move(path)), left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &Answer::poll);
        timer.start();
    }

    QString answer;
    int fileDialogs = 0;
    QStringList messages;

    [[nodiscard]] bool said(const QString &needle) const
    {
        for (const auto &m : messages)
            if (m.contains(needle)) return true;
        return false;
    }

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            // out of budget: dismiss whatever is still up rather than leave it
            // modal with nobody to answer it, which hangs the run
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            else if (m) m->close();
            return;
        }
        if (!m) return;
        if (auto *fd = qobject_cast<QFileDialog *>(m)) {
            ++fileDialogs;
            if (answer.isEmpty()) {
                static_cast<QDialog *>(fd)->reject();
            } else {
                fd->setDirectory(QFileInfo(answer).absolutePath());
                fd->selectFile(answer);
                static_cast<QDialog *>(fd)->accept();
            }
            return;
        }
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            messages << box->text() + "\n" + box->informativeText();
            box->accept();
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
        else m->close();
    }

    QTimer timer;
    int left;
};

class Colors : public ::testing::Test {
protected:
    QString write(const QString &name, const QString &text) const
    {
        const QString p = dir.filePath(name);
        QFile f(p);
        EXPECT_TRUE(f.open(QIODevice::WriteOnly | QIODevice::Text));
        f.write(text.toUtf8());
        f.close();
        return p;
    }

    /// a well-formed colours file with the given body fields spliced in
    QString goodFile(const QString &name, const QString &extra = QString()) const
    {
        return write(name, QString(R"({"application":"SPARTA","format":"colors","revision":1,
             "colors":[{"name":"N","red":1.0,"green":0.0,"blue":0.0},
                       {"name":"O","red":0.0,"green":0.5,"blue":1.0}],
             "lights":{"ambient":0.1,"key":0.8,"fill":0.4,"back":0.7}%1})")
                         .arg(extra));
    }

    static QJsonObject load(const QString &path)
    {
        Answer answer(path);
        auto obj = loadJsonColors(nullptr);
        QCoreApplication::processEvents();
        return obj;
    }

    QTemporaryDir dir;
};

} // namespace

// ---------------------------------------------------------------- reading

TEST_F(Colors, ReadsAWellFormedFile)
{
    const QJsonObject obj = load(goodFile("good.json"));
    ASSERT_FALSE(obj.isEmpty()) << "a valid colours file was rejected";

    const QJsonArray colors = obj["colors"].toArray();
    ASSERT_EQ(colors.size(), 2);
    EXPECT_EQ(colors.at(0).toObject().value("name").toString(), "N");
    EXPECT_DOUBLE_EQ(colors.at(1).toObject().value("blue").toDouble(), 1.0);

    const QJsonObject lights = obj["lights"].toObject();
    EXPECT_DOUBLE_EQ(lights.value("key").toDouble(), 0.8);
}

TEST_F(Colors, CancellingReadsNothing)
{
    Answer answer; // cancel
    const QJsonObject obj = loadJsonColors(nullptr);
    QCoreApplication::processEvents();
    EXPECT_EQ(answer.fileDialogs, 1);
    EXPECT_TRUE(obj.isEmpty());
}

TEST_F(Colors, AFileThatIsNotJsonIsRefusedWithItsReason)
{
    const QString path = write("garbage.json", "this is not JSON at all\n");
    Answer answer(path);
    const QJsonObject obj = loadJsonColors(nullptr);
    QCoreApplication::processEvents();

    EXPECT_TRUE(obj.isEmpty()) << "malformed JSON was accepted";
    EXPECT_TRUE(answer.said("Invalid JSON colors file")) << answer.messages.join(" | ").toStdString();
}

TEST_F(Colors, JsonThatIsNotAnObjectIsRefused)
{
    const QString path = write("array.json", "[1, 2, 3]\n");
    Answer answer(path);
    const QJsonObject obj = loadJsonColors(nullptr);
    QCoreApplication::processEvents();
    EXPECT_TRUE(obj.isEmpty());
    EXPECT_TRUE(answer.said("Invalid JSON colors file"));
}

TEST_F(Colors, SomeoneElsesJsonIsRefusedRatherThanMisread)
{
    // valid JSON, but not a SPARTA colours file: without the header check this
    // would come back as an empty colour list and silently reset every species
    const QString path = write("other.json", R"({"application":"something","format":"colors"})");
    Answer answer(path);
    const QJsonObject obj = loadJsonColors(nullptr);
    QCoreApplication::processEvents();

    EXPECT_TRUE(obj.isEmpty());
    EXPECT_TRUE(answer.said("is not a SPARTA colors file"))
        << answer.messages.join(" | ").toStdString();
}

TEST_F(Colors, TheRightApplicationWithTheWrongFormatIsStillRefused)
{
    const QString path = write("wrongfmt.json",
                               R"({"application":"SPARTA","format":"preferences","revision":1})");
    Answer answer(path);
    EXPECT_TRUE(loadJsonColors(nullptr).isEmpty());
    QCoreApplication::processEvents();
    EXPECT_TRUE(answer.said("is not a SPARTA colors file"));
}

TEST_F(Colors, AFutureRevisionIsRefusedRatherThanGuessedAt)
{
    const QString path = write("future.json",
                               R"({"application":"SPARTA","format":"colors","revision":7,
                                   "colors":[{"name":"N","red":1,"green":0,"blue":0}]})");
    Answer answer(path);
    const QJsonObject obj = loadJsonColors(nullptr);
    QCoreApplication::processEvents();

    EXPECT_TRUE(obj.isEmpty()) << "a file from a newer version was read anyway";
    EXPECT_TRUE(answer.said("incompatible revision 7"))
        << answer.messages.join(" | ").toStdString();
}

TEST_F(Colors, SomethingThatIsNotAFileIsRefused)
{
    // a directory: it exists, so the open dialog will hand it over, and it
    // cannot be read as a file -- which is the branch that reports why
    QDir(dir.path()).mkdir("adirectory.json");
    Answer answer(dir.filePath("adirectory.json"));
    const QJsonObject obj = loadJsonColors(nullptr);
    QCoreApplication::processEvents();
    EXPECT_TRUE(obj.isEmpty()) << "a directory was read as a colours file";
}

// ---------------------------------------------------------------- writing

TEST_F(Colors, WritesAFileItCanReadBack)
{
    QJsonArray colors;
    for (const auto &pair : {std::pair<const char *, double>{"N", 0.25},
                             std::pair<const char *, double>{"O", 0.75}}) {
        QJsonObject c;
        c["name"]  = pair.first;
        c["red"]   = pair.second;
        c["green"] = 0.5;
        c["blue"]  = 1.0 - pair.second;
        colors.append(c);
    }
    QJsonObject lights;
    lights["ambient"] = 0.1;
    lights["key"]     = 0.9;
    lights["fill"]    = 0.45;
    lights["back"]    = 0.8;

    const QString path = dir.filePath("written.json");
    {
        Answer answer(path);
        saveJsonColors(nullptr, colors, lights);
        QCoreApplication::processEvents();
        EXPECT_EQ(answer.fileDialogs, 1) << "Save Colors did not ask where to save";
    }
    ASSERT_TRUE(QFile::exists(path)) << "nothing was written";

    const QJsonObject back = load(path);
    ASSERT_FALSE(back.isEmpty()) << "the file it just wrote could not be read back";
    EXPECT_EQ(back["application"].toString(), "SPARTA");
    EXPECT_EQ(back["format"].toString(), "colors");
    EXPECT_EQ(back["revision"].toInt(), 1) << "the revision written is not the one read";

    const QJsonArray got = back["colors"].toArray();
    ASSERT_EQ(got.size(), 2);
    EXPECT_EQ(got.at(0).toObject().value("name").toString(), "N");
    EXPECT_DOUBLE_EQ(got.at(0).toObject().value("red").toDouble(), 0.25);
    EXPECT_DOUBLE_EQ(back["lights"].toObject().value("fill").toDouble(), 0.45);
}

TEST_F(Colors, CancellingTheSaveWritesNothing)
{
    Answer answer; // cancel
    saveJsonColors(nullptr, QJsonArray{}, QJsonObject{});
    QCoreApplication::processEvents();
    EXPECT_EQ(answer.fileDialogs, 1);
    EXPECT_TRUE(QDir(dir.path()).entryList(QDir::Files).isEmpty());
}

TEST_F(Colors, SavingSomewhereUnwritableSaysSo)
{
    Answer answer("/proc/definitely/not/writable/colors.json");
    saveJsonColors(nullptr, QJsonArray{}, QJsonObject{});
    QCoreApplication::processEvents();
    EXPECT_TRUE(answer.said("Could not open") || answer.said("Save"))
        << answer.messages.join(" | ").toStdString();
}

TEST_F(Colors, AnEmptyColourListStillProducesAValidFile)
{
    const QString path = dir.filePath("empty.json");
    {
        Answer answer(path);
        saveJsonColors(nullptr, QJsonArray{}, QJsonObject{});
        QCoreApplication::processEvents();
    }
    ASSERT_TRUE(QFile::exists(path));

    // it must still carry the header, or it cannot be read back at all
    const QJsonObject back = load(path);
    EXPECT_FALSE(back.isEmpty()) << "a file with no colours in it lost its header too";
    EXPECT_EQ(back["format"].toString(), "colors");
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_jsoncolors.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
