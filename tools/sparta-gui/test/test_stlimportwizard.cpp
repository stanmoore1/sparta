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

// The surface import wizard: six tab pages that turn an STL file into the
// SPARTA commands that read it, and write the .surf file those commands need.
//
// test_stlimport.cpp covers the parsers and the command builders underneath.
// This covers the wizard on top of them -- the part that decides which
// commands to build from what the user set, writes the surface file, and
// refuses a source it could not read.  It needs no simulator: the wizard checks
// its SpartaWrapper before every use, and with none it falls back to the
// preflight watertightness heuristic and its own mesh renderer.

#include "stlimportwizard.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QSpinBox>
#include <QTabWidget>
#include <QTemporaryDir>

using namespace StlImport;

namespace {

// A closed tetrahedron in ASCII STL: four facets, every edge shared by exactly
// two of them, so the watertightness preflight has something to say yes to.
QString tetrahedron()
{
    struct Facet {
        double n[3], v[3][3];
    };
    const Facet facets[4] = {
        {{0, 0, -1}, {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}}},
        {{0, -1, 0}, {{0, 0, 0}, {1, 0, 0}, {0, 0, 1}}},
        {{-1, 0, 0}, {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}}},
        {{1, 1, 1}, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
    };
    QString out = "solid tetra\n";
    for (const auto &f : facets) {
        out += QString("  facet normal %1 %2 %3\n").arg(f.n[0]).arg(f.n[1]).arg(f.n[2]);
        out += "    outer loop\n";
        for (const auto &v : f.v)
            out += QString("      vertex %1 %2 %3\n").arg(v[0]).arg(v[1]).arg(v[2]);
        out += "    endloop\n  endfacet\n";
    }
    return out + "endsolid tetra\n";
}

// The same tetrahedron with one facet removed: an open surface, which is what
// a leak looks like.
QString openMesh()
{
    QString out = tetrahedron();
    const int last = out.lastIndexOf("  facet normal");
    return out.left(last) + "endsolid tetra\n";
}

class Wizard : public ::testing::Test {
protected:
    QString writeStl(const QString &name, const QString &text) const
    {
        const QString p = dir.filePath(name);
        QFile f(p);
        EXPECT_TRUE(f.open(QIODevice::WriteOnly | QIODevice::Text));
        f.write(text.toUtf8());
        f.close();
        return p;
    }

    // A missing control has to abort the case rather than hand back a null the
    // caller then dereferences: a typo in a name should fail, not segfault.
    template <class W> static W *ctl(const QDialog &d, const char *name)
    {
        auto *w = d.findChild<W *>(QLatin1String(name));
        if (!w) ADD_FAILURE() << "no control named " << name;
        EXPECT_NE(w, nullptr);
        return w ? w : new W;
    }

    /// the generated commands as the Output page shows them
    static QString output(const StlImportWizard &w)
    {
        auto *p = w.findChild<QPlainTextEdit *>("output");
        return p ? p->toPlainText() : QString();
    }

    QTemporaryDir dir;
};

} // namespace

// ---------------------------------------------------------------- loading

TEST_F(Wizard, LoadsAnStlAndOffersEveryPage)
{
    StlImportWizard w(nullptr, nullptr, writeStl("tetra.stl", tetrahedron()));
    EXPECT_TRUE(w.loaded());

    auto *tabs = w.findChild<QTabWidget *>();
    ASSERT_NE(tabs, nullptr);
    EXPECT_EQ(tabs->count(), 6) << "a wizard page was added or lost";
    for (int i = 0; i < tabs->count(); ++i)
        EXPECT_TRUE(tabs->isTabEnabled(i)) << tabs->tabText(i).toStdString() << " is disabled";
}

TEST_F(Wizard, ASourceItCannotReadDisablesEverythingPastTheFirstPage)
{
    // what a user gets from a truncated download or the wrong file entirely:
    // the wizard says so by refusing to go on, rather than generating commands
    // for a mesh it does not have
    StlImportWizard w(nullptr, nullptr, writeStl("broken.stl", "solid nothing\nendsolid\n"));
    EXPECT_FALSE(w.loaded());

    auto *tabs = w.findChild<QTabWidget *>();
    ASSERT_NE(tabs, nullptr);
    EXPECT_TRUE(tabs->isTabEnabled(0)) << "even the source page was disabled";
    for (int i = 1; i < tabs->count(); ++i)
        EXPECT_FALSE(tabs->isTabEnabled(i)) << tabs->tabText(i).toStdString() << " is still usable";

    auto *box = w.findChild<QDialogButtonBox *>();
    ASSERT_NE(box, nullptr);
    EXPECT_FALSE(box->button(QDialogButtonBox::Ok)->isEnabled())
        << "the wizard would insert commands for a mesh it could not read";
}

TEST_F(Wizard, AFileThatIsNotThereIsRefusedTheSameWay)
{
    StlImportWizard w(nullptr, nullptr, dir.filePath("never-written.stl"));
    EXPECT_FALSE(w.loaded());
}

// ---------------------------------------------------------------- the explicit path

TEST_F(Wizard, TheDefaultOutputIsAReadSurfCommand)
{
    StlImportWizard w(nullptr, nullptr, writeStl("tetra.stl", tetrahedron()));
    const QString cmds = output(w);
    EXPECT_TRUE(cmds.startsWith("read_surf")) << cmds.toStdString();
    EXPECT_TRUE(cmds.contains("tetra.surf")) << "the command does not name the file it writes: "
                                             << cmds.toStdString();
}

TEST_F(Wizard, TheTransformControlsReachTheCommand)
{
    StlImportWizard w(nullptr, nullptr, writeStl("tetra.stl", tetrahedron()));

    ctl<QCheckBox>(w, "scaleOn")->setChecked(true);
    ctl<QDoubleSpinBox>(w, "scale0")->setValue(2.0);
    ctl<QDoubleSpinBox>(w, "scale1")->setValue(3.0);
    ctl<QDoubleSpinBox>(w, "scale2")->setValue(4.0);

    const QString cmds = output(w);
    EXPECT_TRUE(cmds.contains("scale")) << cmds.toStdString();
    EXPECT_TRUE(cmds.contains("2") && cmds.contains("3") && cmds.contains("4"))
        << cmds.toStdString();
}

TEST_F(Wizard, TurningAScaleOffTakesItBackOutOfTheCommand)
{
    StlImportWizard w(nullptr, nullptr, writeStl("tetra.stl", tetrahedron()));
    auto *on = ctl<QCheckBox>(w, "scaleOn");
    on->setChecked(true);
    ctl<QDoubleSpinBox>(w, "scale0")->setValue(7.0);
    ASSERT_TRUE(output(w).contains("scale"));

    on->setChecked(false);
    EXPECT_FALSE(output(w).contains("scale"))
        << "a disabled scale is still in the command: " << output(w).toStdString();
}

TEST_F(Wizard, TheOptionCheckboxesReachTheCommand)
{
    StlImportWizard w(nullptr, nullptr, writeStl("tetra.stl", tetrahedron()));
    ctl<QCheckBox>(w, "invert")->setChecked(true);
    ctl<QCheckBox>(w, "clip")->setChecked(true);
    ctl<QCheckBox>(w, "transparent")->setChecked(true);

    const QString cmds = output(w);
    EXPECT_TRUE(cmds.contains("invert")) << cmds.toStdString();
    EXPECT_TRUE(cmds.contains("clip")) << cmds.toStdString();
    EXPECT_TRUE(cmds.contains("transparent")) << cmds.toStdString();

    const auto s = w.settings();
    EXPECT_TRUE(s.invert);
    EXPECT_TRUE(s.useClip);
    EXPECT_TRUE(s.transparent);
}

TEST_F(Wizard, TheGroupNameReachesTheCommandAndIsTrimmed)
{
    StlImportWizard w(nullptr, nullptr, writeStl("tetra.stl", tetrahedron()));
    ctl<QLineEdit>(w, "group")->setText("  nose  ");
    EXPECT_EQ(w.settings().group, "nose") << "the group name was not trimmed";
    EXPECT_TRUE(output(w).contains("nose")) << output(w).toStdString();
}

TEST_F(Wizard, EveryTranslationKindIsOffered)
{
    StlImportWizard w(nullptr, nullptr, writeStl("tetra.stl", tetrahedron()));
    auto *kind = ctl<QComboBox>(w, "transKind");
    EXPECT_EQ(kind->count(), 4) << "a translation kind was added or lost";

    // each one has to produce a command the builder recognises
    for (int i = 0; i < kind->count(); ++i) {
        kind->setCurrentIndex(i);
        ctl<QDoubleSpinBox>(w, "trans0")->setValue(1.5);
        EXPECT_TRUE(output(w).startsWith("read_surf"))
            << "kind " << kind->itemText(i).toStdString() << ": " << output(w).toStdString();
    }
}

// ---------------------------------------------------------------- the implicit path

TEST_F(Wizard, SwitchingToImplicitProducesTheAblationCommandsInstead)
{
    StlImportWizard w(nullptr, nullptr, writeStl("tetra.stl", tetrahedron()));
    ASSERT_TRUE(output(w).startsWith("read_surf"));

    ctl<QRadioButton>(w, "modeImplicit")->setChecked(true);
    const QString cmds = output(w);
    EXPECT_TRUE(cmds.contains("create_isurf")) << cmds.toStdString();
    EXPECT_TRUE(cmds.contains("fix")) << "no ablation fix was generated: " << cmds.toStdString();
    EXPECT_EQ(w.settings().mode, StlImportSettings::Mode::Implicit);
}

TEST_F(Wizard, TheAblationControlsReachTheGeneratedCommands)
{
    StlImportWizard w(nullptr, nullptr, writeStl("tetra.stl", tetrahedron()));
    ctl<QRadioButton>(w, "modeImplicit")->setChecked(true);
    ctl<QLineEdit>(w, "isurfGroup")->setText("inner");
    ctl<QLineEdit>(w, "ablateId")->setText("myablate");
    ctl<QDoubleSpinBox>(w, "thresh")->setValue(120.0);

    const QString cmds = output(w);
    EXPECT_TRUE(cmds.contains("inner")) << cmds.toStdString();
    EXPECT_TRUE(cmds.contains("myablate")) << cmds.toStdString();
    EXPECT_TRUE(cmds.contains("120")) << "the threshold did not reach the commands: "
                                      << cmds.toStdString();

    const auto s = w.settings();
    EXPECT_EQ(s.isurfGroup, "inner");
    EXPECT_EQ(s.ablateId, "myablate");
    EXPECT_DOUBLE_EQ(s.thresh, 120.0);
}

TEST_F(Wizard, TheGridResolutionIsRecordedWithoutEnteringTheCommands)
{
    // The implicit-surface grid drives the preview render and the create_grid
    // the preview runs -- not the commands the wizard inserts, which take the
    // grid from the deck the user already has.  Emitting it would silently
    // override that.
    StlImportWizard w(nullptr, nullptr, writeStl("tetra.stl", tetrahedron()));
    ctl<QRadioButton>(w, "modeImplicit")->setChecked(true);
    ctl<QSpinBox>(w, "grid0")->setValue(12);
    ctl<QSpinBox>(w, "grid1")->setValue(13);
    ctl<QSpinBox>(w, "grid2")->setValue(14);

    const auto s = w.settings();
    EXPECT_EQ(s.gridNx, 12);
    EXPECT_EQ(s.gridNy, 13);
    EXPECT_EQ(s.gridNz, 14);
    EXPECT_FALSE(output(w).contains("create_grid"))
        << "the wizard emitted a grid, overriding the deck's own: " << output(w).toStdString();
}

TEST_F(Wizard, SwitchingBackToExplicitRestoresTheReadSurfCommand)
{
    StlImportWizard w(nullptr, nullptr, writeStl("tetra.stl", tetrahedron()));
    ctl<QRadioButton>(w, "modeImplicit")->setChecked(true);
    ASSERT_TRUE(output(w).contains("create_isurf"));

    ctl<QRadioButton>(w, "modeExplicit")->setChecked(true);
    EXPECT_TRUE(output(w).startsWith("read_surf")) << output(w).toStdString();
    EXPECT_FALSE(output(w).contains("create_isurf"))
        << "the ablation commands survived the switch back";
    EXPECT_EQ(w.settings().mode, StlImportSettings::Mode::Explicit);
}

// ---------------------------------------------------------------- accepting

TEST_F(Wizard, AcceptingWritesTheSurfaceFileTheCommandNames)
{
    const QString stl = writeStl("tetra.stl", tetrahedron());
    StlImportWizard w(nullptr, nullptr, stl);
    ASSERT_TRUE(w.loaded());

    static_cast<QDialog &>(w).accept();
    const QString surf = w.writtenSurfPath();
    ASSERT_FALSE(surf.isEmpty()) << "no surface file was written";
    EXPECT_EQ(QFileInfo(surf).fileName(), "tetra.surf")
        << "the file written is not the one the command reads";

    QFile f(surf);
    ASSERT_TRUE(f.open(QIODevice::ReadOnly | QIODevice::Text));
    const QString text = QString::fromUtf8(f.readAll());
    EXPECT_TRUE(text.contains("points")) << text.left(200).toStdString();
    EXPECT_TRUE(text.contains("triangles")) << text.left(200).toStdString();
    EXPECT_TRUE(text.contains("tetra.stl")) << "the surface file does not say where it came from";
}

TEST_F(Wizard, TheWrittenSurfaceIsWhatTheCommandAsksFor)
{
    const QString stl = writeStl("tetra.stl", tetrahedron());
    StlImportWizard w(nullptr, nullptr, stl);
    static_cast<QDialog &>(w).accept();
    EXPECT_TRUE(w.generatedText().contains(QFileInfo(w.writtenSurfPath()).fileName()))
        << w.generatedText().toStdString();
}

TEST_F(Wizard, AnExistingSurfFileIsUsedWhereItIsRatherThanRewritten)
{
    // the other source kind: a .surf the user already has needs no conversion
    const QString stl = writeStl("tetra.stl", tetrahedron());
    StlImportWizard maker(nullptr, nullptr, stl);
    static_cast<QDialog &>(maker).accept();
    const QString surf = maker.writtenSurfPath();
    ASSERT_FALSE(surf.isEmpty());

    StlImportWizard w(nullptr, nullptr, surf);
    ASSERT_TRUE(w.loaded()) << "the surface file this wizard just wrote could not be read back";
    static_cast<QDialog &>(w).accept();
    EXPECT_TRUE(w.writtenSurfPath().isEmpty())
        << "an existing .surf was rewritten instead of used as it is";
}

TEST_F(Wizard, RejectingWritesNothing)
{
    const QString stl = writeStl("tetra.stl", tetrahedron());
    StlImportWizard w(nullptr, nullptr, stl);
    static_cast<QDialog &>(w).reject();
    EXPECT_TRUE(w.writtenSurfPath().isEmpty());
    EXPECT_FALSE(QFile::exists(dir.filePath("tetra.surf")))
        << "cancelling still wrote a surface file";
}

// ---------------------------------------------------------------- watertightness

TEST_F(Wizard, AClosedMeshAndAnOpenOneAreReportedDifferently)
{
    StlImportWizard closed(nullptr, nullptr, writeStl("closed.stl", tetrahedron()));
    StlImportWizard open(nullptr, nullptr, writeStl("open.stl", openMesh()));
    ASSERT_TRUE(closed.loaded());
    ASSERT_TRUE(open.loaded());

    auto verdict = [](const StlImportWizard &w) {
        QString all;
        for (auto *l : w.findChildren<QLabel *>())
            all += l->text() + "\n";
        return all;
    };
    const QString a = verdict(closed);
    const QString b = verdict(open);
    EXPECT_NE(a, b) << "a closed mesh and one with a hole in it read the same:\n"
                    << a.toStdString();
}

TEST_F(Wizard, ItRendersItsPreviewWithoutASimulator)
{
    StlImportWizard w(nullptr, nullptr, writeStl("tetra.stl", tetrahedron()));
    w.resize(720, 640);
    EXPECT_FALSE(w.grab().isNull());
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_stlimportwizard.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
