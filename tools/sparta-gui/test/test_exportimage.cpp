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

// Saving a snapshot to a file: exportImage().
//
// Both viewers and the chart window hand their picture to this, and it is the
// only way a rendered frame leaves the application.  It had never run.
//
// The interesting part is the fallback.  Qt writes a handful of formats; for
// anything else the image goes to a temporary PNG which ImageMagick converts to
// the name the user typed.
//
// Two of the three failure branches around that turn out to be unreachable on a
// Linux box with ImageMagick installed, which is worth writing down rather than
// leaving as an apparent gap.  findExe() searches /usr/bin whatever PATH says,
// so "the converter is absent" cannot be produced; and ImageMagick answers an
// extension it does not know by falling back to its own default format and
// exiting zero, so "the conversion failed" needs a destination that cannot be
// written at all -- where there is then nothing left to clean up either.  What
// is reachable is covered below.

#include "helpers.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QColor>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QImage>
#include <QMessageBox>
#include <QProcess>
#include <QTemporaryDir>
#include <QTimer>

namespace {

bool haveImageMagick()
{
    QProcess p;
    p.start("convert", {"-version"});
    if (p.waitForFinished(5000) && p.exitCode() == 0) return true;
    QProcess q;
    q.start("magick", {"-version"});
    return q.waitForFinished(5000) && q.exitCode() == 0;
}

/// Answers the save dialog with a path, and records any message box.
class SaveTo : public QObject {
public:
    explicit SaveTo(QString path, int budgetMs = 20000) :
        answer(std::move(path)), left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &SaveTo::poll);
        timer.start();
    }
    int dialogs = 0;
    QStringList messages;

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
            return;
        }
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            messages << box->windowTitle() + " " + box->text() + " " + box->informativeText() +
                        " " + box->detailedText();
            box->accept();
            return;
        }
        if (auto *fd = qobject_cast<QFileDialog *>(m)) {
            ++dialogs;
            if (answer.isEmpty()) {
                static_cast<QDialog *>(fd)->reject();
            } else {
                fd->setDirectory(QFileInfo(answer).absolutePath());
                fd->selectFile(answer);
                static_cast<QDialog *>(fd)->accept();
            }
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
    }
    QTimer timer;
    QString answer;
    int left;
};

class Export : public ::testing::Test {
protected:
    /// A picture with a recognisable colour, so a file that came back is the
    /// image that went in rather than a blank of the right size.
    static QImage picture(int w = 40, int h = 30)
    {
        QImage img(w, h, QImage::Format_RGB32);
        img.fill(QColor(17, 200, 90));
        return img;
    }

    static bool looksLikeThePicture(const QString &path)
    {
        const QImage back(path);
        if (back.isNull()) return false;
        const QColor c = back.pixelColor(back.width() / 2, back.height() / 2);
        // allow for a lossy format on the way out
        return std::abs(c.red() - 17) < 24 && std::abs(c.green() - 200) < 24 &&
               std::abs(c.blue() - 90) < 24;
    }

    QString path(const QString &name) const { return dir.filePath(name); }

    QTemporaryDir dir;
};

} // namespace

// ------------------------------------------------------------- what Qt writes

TEST_F(Export, WritesAPngQtCanSaveItself)
{
    SaveTo answer(path("shot.png"));
    QImage img = picture();
    exportImage(nullptr, &img, "Snapshot");
    QApplication::processEvents();

    EXPECT_EQ(answer.dialogs, 1) << "it did not ask where to save";
    ASSERT_TRUE(QFile::exists(path("shot.png"))) << "nothing was written";
    EXPECT_TRUE(looksLikeThePicture(path("shot.png")))
        << "the file is not the image that was exported";
    EXPECT_TRUE(answer.messages.isEmpty()) << "a good save complained: " << answer.all().toStdString();
}

TEST_F(Export, WritesTheOtherFormatsQtSupports)
{
    for (const char *ext : {"bmp", "jpg", "ppm"}) {
        const QString p = path(QString("shot.%1").arg(ext));
        SaveTo answer(p);
        QImage img = picture();
        exportImage(nullptr, &img, "Snapshot");
        QApplication::processEvents();
        ASSERT_TRUE(QFile::exists(p)) << ext << " was not written";
        EXPECT_TRUE(looksLikeThePicture(p)) << ext << " does not hold the exported image";
    }
}

TEST_F(Export, CancellingTheDialogWritesNothing)
{
    SaveTo answer{QString()};
    QImage img = picture();
    exportImage(nullptr, &img, "Snapshot");
    QApplication::processEvents();

    EXPECT_EQ(answer.dialogs, 1);
    EXPECT_TRUE(QDir(dir.path()).entryList(QDir::Files).isEmpty());
    EXPECT_TRUE(answer.messages.isEmpty())
        << "cancelling produced an error about a save that was never attempted: "
        << answer.all().toStdString();
}

TEST_F(Export, WithNoImageItDoesNotEvenAsk)
{
    SaveTo answer(path("never.png"));
    exportImage(nullptr, nullptr, "Snapshot");
    QApplication::processEvents();

    EXPECT_EQ(answer.dialogs, 0) << "it asked where to save an image it does not have";
    EXPECT_FALSE(QFile::exists(path("never.png")));
}

// -------------------------------------------------- what ImageMagick writes

TEST_F(Export, ConvertsAFormatQtCannotWriteItself)
{
    // Qt has no SGI writer, so this goes through the temporary PNG and the
    // external converter -- the fallback that had never run
    if (!haveImageMagick()) GTEST_SKIP() << "no ImageMagick to convert with";
    const QString p = path("shot.sgi");
    SaveTo answer(p);
    QImage img = picture();
    exportImage(nullptr, &img, "Snapshot");
    QApplication::processEvents();

    ASSERT_TRUE(QFile::exists(p)) << "the conversion produced nothing: " << answer.all().toStdString();
    EXPECT_GT(QFileInfo(p).size(), 0) << "the converted file is empty";

    // read it back through the converter, since Qt cannot open it either
    QProcess back;
    back.start("convert", {p, path("back.png")});
    if (back.waitForFinished(20000) && back.exitCode() == 0)
        EXPECT_TRUE(looksLikeThePicture(path("back.png")))
            << "the converted file is not the image that was exported";
}

TEST_F(Export, AnExtensionQtDoesNotKnowStillProducesARealImage)
{
    // ImageMagick is asked to write whatever name the user typed.  For an
    // extension it does not recognise it falls back to its own default format
    // rather than refusing, so what must never happen is a stub file at that
    // name: the user would open it later expecting their snapshot.
    if (!haveImageMagick()) GTEST_SKIP() << "no ImageMagick to convert with";
    const QString p = path("shot.notaformat");
    SaveTo answer(p);
    QImage img = picture();
    exportImage(nullptr, &img, "Snapshot");
    QApplication::processEvents();

    ASSERT_TRUE(QFile::exists(p)) << "nothing was written: " << answer.all().toStdString();
    EXPECT_GT(QFileInfo(p).size(), 0) << "a zero-length file was left at the user's chosen name";

    QProcess back;
    back.start("convert", {p, path("back2.png")});
    if (back.waitForFinished(20000) && back.exitCode() == 0)
        EXPECT_TRUE(looksLikeThePicture(path("back2.png")))
            << "the file at that name is not the exported image";
}

TEST_F(Export, SavingWhereItCannotWriteSaysSoRatherThanFailingSilently)
{
    SaveTo answer("/proc/definitely/not/writable/shot.png");
    QImage img = picture();
    exportImage(nullptr, &img, "Snapshot");
    QApplication::processEvents();

    EXPECT_FALSE(answer.messages.isEmpty())
        << "an unwritable destination was accepted without a word";
    EXPECT_TRUE(answer.said("Could not save") || answer.said("Error"))
        << answer.all().toStdString();
}

TEST_F(Export, TheErrorIsTitledWithWhatWasBeingSaved)
{
    // the same function serves the snapshot viewer, the slide show and the
    // charts; the title is how the user knows which one failed
    SaveTo answer("/proc/definitely/not/writable/shot.png");
    QImage img = picture();
    exportImage(nullptr, &img, "Slide Show");
    QApplication::processEvents();
    EXPECT_TRUE(answer.said("Slide Show")) << answer.all().toStdString();
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_exportimage.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
