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

// The dump-image settings dialog: the mapping between its ~120 controls and the
// DumpImageSettings struct.
//
// That mapping is the one untested link in the dump-image chain. The struct is
// plain data, the command builders downstream of it are pure and have 27 tests
// of their own, and the dialog upstream had none -- so a control wired to the
// wrong field would have rendered the wrong picture without anything noticing.
// It could not be tested before because the dialog was built on the stack
// inside a method that asked a live SPARTA instance for every fact as it went.
//
// Nothing here needs a simulator or a display: the dialog is handed a plain
// description of the simulation instead. Where it helps, the resulting struct
// is fed to buildDumpImageCommand() and the emitted command asserted, which
// turns a forty-field comparison into one readable string and leans on tests
// that already pass.

#include <gtest/gtest.h>

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QSignalSpy>
#include <QSlider>
#include <QSpinBox>
#include <QTabWidget>

#include <set>

#include "dumpimage.h"
#include "dumpimagesettingsdialog.h"

namespace {

// A 3d run with two species, two mixtures, a region, a group of each kind and
// one compute, one fix and one variable to colour by.
ImageSettingsEnv env3d()
{
    ImageSettingsEnv e;
    e.dimension  = 3;
    e.surfsExist = true;
    e.boxlo[0] = e.boxlo[1] = e.boxlo[2] = 0.0;
    e.boxhi[0] = e.boxhi[1] = e.boxhi[2] = 10.0;
    e.species    = {"N", "O"};
    e.mixtures   = {"all", "air"};
    // regions as SPARTA reports them; the "none" entry that means "do not clip"
    // is the dialog's own, not a region of the deck
    e.regions    = {"inner", "outer"};
    e.gridGroups = {"all", "coarse"};
    e.surfGroups = {"all", "wall"};
    e.gridSources     = {"proc", "c_temp", "f_ave", "v_scale"};
    e.surfSources     = {"one", "proc", "c_flux", "f_ave"};
    e.particleSources = {"c_temp", "f_ave", "v_scale"};
    return e;
}

// The same run in two dimensions and with no surfaces.
ImageSettingsEnv env2d()
{
    ImageSettingsEnv e = env3d();
    e.dimension        = 2;
    e.surfsExist       = false;
    return e;
}

DumpImageSettingsDialog::SpeciesColors twoSpecies()
{
    return {{"red", QColor(255, 0, 0)}, {"green", QColor(0, 255, 0)}};
}

// Settings with something non-default on every tab, so a round trip has to
// carry real values rather than the defaults it would produce anyway.
DumpImageSettings busy()
{
    DumpImageSettings s;
    s.mixture = "air";
    s.color   = "c_temp";
    s.region  = "inner";
    s.particle    = true;
    s.numericdiam = true;
    s.pdiamvalue  = 0.25;

    s.grid      = false;
    s.gridgroup = "coarse";
    s.gline     = true;
    s.glinediam = 0.004;
    s.glinecolor = "cyan";
    s.gcolors    = {{"*", "red/green"}};

    s.gridx = true;  s.gridxcoord = 2.5;  s.gridxcolor = "c_temp[2]";
    s.gridy = true;  s.gridycoord = 5.0;  s.gridycolor = "proc";
    s.gridz = true;  s.gridzcoord = 7.5;  s.gridzcolor = "v_scale";

    s.surf         = true;
    s.surfcolor    = "c_flux";
    s.surfcolorone = "orange";
    s.surfdiam     = 0.02;
    s.surfgroup    = "wall";
    s.scolors      = {{"1*4", "blue"}};
    s.sline        = true;
    s.slinediam    = 0.006;
    s.slinecolor   = "magenta";

    s.box = true;  s.boxdiam = 0.03;  s.boxcolor = "white";
    s.subbox = true;  s.subboxdiam = 0.01;  s.subboxcolor = "pink";
    s.axes = true;  s.axeslen = 0.7;  s.axesdiam = 0.04;

    s.theta = 45.0;  s.phi = 15.0;
    s.centerdynamic = true;
    s.cx = 0.25;  s.cy = 0.5;  s.cz = 0.75;
    s.upx = 1.0;  s.upy = 0.0;  s.upz = 0.0;
    s.zoom = 2.5;

    s.ssao = true;  s.ssaoint = 0.8;  s.fsaa = true;  s.shiny = 0.4;
    s.backcolor = "navy";  s.gradient = true;  s.backcolor2 = "skyblue";
    s.amblight = 0.2;  s.keylight = 0.7;  s.filllight = 0.3;  s.backlight = 0.5;

    s.cmap[DumpImageSettings::PARTICLE] = {true, "viridis", true, "0", "100", 'd', 'a', 2.0};
    s.cmap[DumpImageSettings::SURF]     = {true, "BWR", false, "min", "max", 's', 'f', 0.5};
    return s;
}

// ---- finding controls -------------------------------------------------------

template <class W> W *ctl(const DumpImageSettingsDialog &d, const char *name)
{
    return d.findChild<W *>(QString("ivs.") + name);
}

QString command(const DumpImageSettings &s)
{
    return buildDumpImageCommand(s);
}

} // namespace

class Settings : public ::testing::Test {
protected:
    ImageSettingsEnv env = env3d();
};

// ------------------------------------------------------------- construction

TEST_F(Settings, HasTheEightDocumentedTabs)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());
    auto *tabs = d.findChild<QTabWidget *>();
    ASSERT_NE(tabs, nullptr);
    ASSERT_EQ(tabs->count(), 8);

    const QStringList want = {"&Particles", "&Grid",     "Grid Pla&nes", "S&urfaces",
                              "Bo&x/Axes",  "&Camera",   "&Quality",     "Color &Maps"};
    for (int i = 0; i < tabs->count(); ++i)
        EXPECT_EQ(tabs->tabText(i), want.at(i)) << "tab " << i;
}

TEST_F(Settings, TheRequestedTabIsOpenedAndOutOfRangeIsClamped)
{
    for (int i = 0; i < 8; ++i) {
        DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies(), i);
        EXPECT_EQ(d.findChild<QTabWidget *>()->currentIndex(), i);
    }
    EXPECT_EQ(DumpImageSettingsDialog(DumpImageSettings{}, env, twoSpecies(), -1)
                  .findChild<QTabWidget *>()->currentIndex(), 0);
    EXPECT_EQ(DumpImageSettingsDialog(DumpImageSettings{}, env, twoSpecies(), 99)
                  .findChild<QTabWidget *>()->currentIndex(), 7);
}

TEST_F(Settings, AnUntouchedDefaultDialogEmitsTheDefaultCommand)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());
    // the same command test_dumpimage.cpp asserts for a default-constructed
    // struct: the dialog must not invent settings just by being built
    EXPECT_EQ(command(d.settings()), command(DumpImageSettings{}));
}

// -------------------------------------------------------------- round trip

// The single most valuable assertion here. Every control is populated from the
// struct and read straight back; a control wired to the wrong field cannot
// survive the trip, because the value would come back on the wrong member.
TEST_F(Settings, EverythingSurvivesTheRoundTrip)
{
    const DumpImageSettings in = busy();
    DumpImageSettingsDialog d(in, env, twoSpecies());
    const DumpImageSettings out = d.settings();

    EXPECT_EQ(command(out), command(in));
    EXPECT_EQ(buildDumpModifyCommands(out, "ID"), buildDumpModifyCommands(in, "ID"));

    // the fields the builders prune, which the command comparison cannot see
    EXPECT_EQ(out.mixture, in.mixture);
    EXPECT_EQ(out.region, in.region);
    EXPECT_EQ(out.numericdiam, in.numericdiam);
    EXPECT_DOUBLE_EQ(out.pdiamvalue, in.pdiamvalue);
    EXPECT_EQ(out.centerdynamic, in.centerdynamic);
    EXPECT_EQ(out.gridgroup, in.gridgroup);
    EXPECT_EQ(out.surfgroup, in.surfgroup);
}

TEST_F(Settings, ReadingTheSettingsTwiceGivesTheSameAnswer)
{
    DumpImageSettingsDialog d(busy(), env, twoSpecies());
    EXPECT_EQ(command(d.settings()), command(d.settings()))
        << "settings() is not repeatable, so it is changing the dialog as it reads it";
}

// Fields the dialog has no control for must be carried through, not reset:
// the image size, the SSAO seed, the tables the caller derives, and the movie
// settings all belong to somebody else.
TEST_F(Settings, FieldsTheDialogDoesNotOwnAreCarriedThrough)
{
    DumpImageSettings in;
    in.xsize = 1280;
    in.ysize = 720;
    in.ssaoseed = 12345;
    in.dimension = 2;
    in.customcolors = {{"mycol", "0.1 0.2 0.3"}};
    in.pcolors      = {{"1", "red"}};
    in.framerate    = 30;
    in.bitrate      = 2000;

    const DumpImageSettings out = DumpImageSettingsDialog(in, env, twoSpecies()).settings();
    EXPECT_EQ(out.xsize, 1280);
    EXPECT_EQ(out.ysize, 720);
    EXPECT_EQ(out.ssaoseed, 12345);
    EXPECT_EQ(out.dimension, 2);
    EXPECT_EQ(out.customcolors, in.customcolors);
    EXPECT_EQ(out.pcolors, in.pcolors);
    EXPECT_EQ(out.framerate, 30);
    EXPECT_EQ(out.bitrate, 2000);
}

// ------------------------------------------------------------- tab by tab

TEST_F(Settings, TheParticlesTabDrivesItsFields)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());

    ctl<QCheckBox>(d, "particle")->setChecked(false);
    ctl<QComboBox>(d, "mixture")->setCurrentText("air");
    ctl<QComboBox>(d, "color")->setCurrentText("c_temp");
    ctl<QComboBox>(d, "region")->setCurrentText("inner");
    ctl<QRadioButton>(d, "diameter.num")->setChecked(true);
    ctl<QLineEdit>(d, "pdiamvalue")->setText("0.75");

    const DumpImageSettings s = d.settings();
    EXPECT_FALSE(s.particle);
    EXPECT_EQ(s.mixture, "air");
    EXPECT_EQ(s.color, "c_temp");
    EXPECT_EQ(s.region, "inner");
    EXPECT_TRUE(s.numericdiam);
    EXPECT_DOUBLE_EQ(s.pdiamvalue, 0.75);
    EXPECT_TRUE(command(s).contains("pdiam 0.75")) << command(s).toStdString();
}

TEST_F(Settings, TheGridTabDrivesItsFields)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());

    ctl<QCheckBox>(d, "grid")->setChecked(true);
    ctl<QComboBox>(d, "gridcolor")->setCurrentText("c_temp");
    ctl<QSpinBox>(d, "gridcolor.col")->setValue(2);
    ctl<QLineEdit>(d, "gcolors")->setText("* red/green");
    ctl<QComboBox>(d, "gridgroup")->setCurrentText("coarse");
    ctl<QCheckBox>(d, "gline")->setChecked(true);
    ctl<QLineEdit>(d, "glinediam")->setText("0.004");
    ctl<QLineEdit>(d, "glinecolor")->setText("cyan");

    const DumpImageSettings s = d.settings();
    EXPECT_TRUE(s.grid);
    // the combo and its column spin box compose into one reference
    EXPECT_EQ(s.gridcolor, "c_temp[2]");
    EXPECT_EQ(s.gcolors, (QList<QPair<QString, QString>>{{"*", "red/green"}}));
    EXPECT_EQ(s.gridgroup, "coarse");
    EXPECT_TRUE(s.gline);
    EXPECT_DOUBLE_EQ(s.glinediam, 0.004);
    EXPECT_EQ(s.glinecolor, "cyan");
}

// A variable reference takes no array column, so the spin box must not
// subscript it -- "v_scale[2]" is not a thing SPARTA accepts.
TEST_F(Settings, AVariableSourceIsNotGivenAnArrayColumn)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());
    ctl<QCheckBox>(d, "grid")->setChecked(true);
    ctl<QComboBox>(d, "gridcolor")->setCurrentText("v_scale");
    ctl<QSpinBox>(d, "gridcolor.col")->setValue(3);
    EXPECT_EQ(d.settings().gridcolor, "v_scale");
}

TEST_F(Settings, TheGridPlanesTabDrivesItsFields)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());

    ctl<QCheckBox>(d, "gridx")->setChecked(true);
    ctl<QDoubleSpinBox>(d, "gridxcoord")->setValue(2.5);
    ctl<QComboBox>(d, "gridxcolor")->setCurrentText("proc");
    ctl<QCheckBox>(d, "gridz")->setChecked(true);
    ctl<QDoubleSpinBox>(d, "gridzcoord")->setValue(7.5);

    const DumpImageSettings s = d.settings();
    EXPECT_TRUE(s.gridx);
    EXPECT_DOUBLE_EQ(s.gridxcoord, 2.5);
    EXPECT_EQ(s.gridxcolor, "proc");
    EXPECT_TRUE(s.gridz);
    EXPECT_DOUBLE_EQ(s.gridzcoord, 7.5);
    EXPECT_FALSE(s.gridy);
}

// The cut planes are bounded by the simulation box, so a coordinate outside it
// cannot be entered at all.
TEST_F(Settings, ThePlaneCoordinatesAreBoundedByTheBox)
{
    env.boxlo[0] = -3.0;
    env.boxhi[0] = 4.0;
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());

    auto *coord = ctl<QDoubleSpinBox>(d, "gridxcoord");
    ASSERT_NE(coord, nullptr);
    EXPECT_DOUBLE_EQ(coord->minimum(), -3.0);
    EXPECT_DOUBLE_EQ(coord->maximum(), 4.0);
}

TEST_F(Settings, TheSurfacesTabDrivesItsFields)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());

    ctl<QCheckBox>(d, "surf")->setChecked(true);
    ctl<QComboBox>(d, "surfcolor")->setCurrentText("c_flux");
    ctl<QLineEdit>(d, "surfcolorone")->setText("orange");
    ctl<QLineEdit>(d, "surfdiam")->setText("0.02");
    ctl<QLineEdit>(d, "scolors")->setText("1*4 blue");
    ctl<QComboBox>(d, "surfgroup")->setCurrentText("wall");
    ctl<QCheckBox>(d, "sline")->setChecked(true);
    ctl<QLineEdit>(d, "slinediam")->setText("0.006");
    ctl<QLineEdit>(d, "slinecolor")->setText("magenta");

    const DumpImageSettings s = d.settings();
    EXPECT_TRUE(s.surf);
    EXPECT_EQ(s.surfcolor, "c_flux");
    EXPECT_EQ(s.surfcolorone, "orange");
    EXPECT_DOUBLE_EQ(s.surfdiam, 0.02);
    EXPECT_EQ(s.scolors, (QList<QPair<QString, QString>>{{"1*4", "blue"}}));
    EXPECT_EQ(s.surfgroup, "wall");
    EXPECT_TRUE(s.sline);
    EXPECT_DOUBLE_EQ(s.slinediam, 0.006);
    EXPECT_EQ(s.slinecolor, "magenta");
}

TEST_F(Settings, TheBoxAndAxesTabDrivesItsFields)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());

    ctl<QCheckBox>(d, "box")->setChecked(false);
    ctl<QLineEdit>(d, "boxdiam")->setText("0.03");
    ctl<QLineEdit>(d, "boxcolor")->setText("white");
    ctl<QCheckBox>(d, "subbox")->setChecked(true);
    ctl<QLineEdit>(d, "subboxdiam")->setText("0.01");
    ctl<QLineEdit>(d, "subboxcolor")->setText("pink");
    ctl<QCheckBox>(d, "axes")->setChecked(true);
    ctl<QLineEdit>(d, "axeslen")->setText("0.7");
    ctl<QLineEdit>(d, "axesdiam")->setText("0.04");

    const DumpImageSettings s = d.settings();
    EXPECT_FALSE(s.box);
    EXPECT_DOUBLE_EQ(s.boxdiam, 0.03);
    EXPECT_EQ(s.boxcolor, "white");
    EXPECT_TRUE(s.subbox);
    EXPECT_DOUBLE_EQ(s.subboxdiam, 0.01);
    EXPECT_EQ(s.subboxcolor, "pink");
    EXPECT_TRUE(s.axes);
    EXPECT_DOUBLE_EQ(s.axeslen, 0.7);
    EXPECT_DOUBLE_EQ(s.axesdiam, 0.04);
}

TEST_F(Settings, TheCameraTabDrivesItsFields)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());

    ctl<QDoubleSpinBox>(d, "theta")->setValue(45.0);
    ctl<QLineEdit>(d, "thetavar")->setText("tvar");
    ctl<QDoubleSpinBox>(d, "phi")->setValue(15.0);
    ctl<QRadioButton>(d, "centerdynamic")->setChecked(true);
    ctl<QDoubleSpinBox>(d, "center0")->setValue(0.25);
    ctl<QDoubleSpinBox>(d, "center2")->setValue(0.75);
    ctl<QLineEdit>(d, "up0")->setText("1.0");
    ctl<QLineEdit>(d, "up1")->setText("0.0");
    ctl<QLineEdit>(d, "up2")->setText("0.0");
    ctl<QLineEdit>(d, "zoom")->setText("2.5");
    ctl<QLineEdit>(d, "zoomvar")->setText("zvar");

    const DumpImageSettings s = d.settings();
    EXPECT_DOUBLE_EQ(s.theta, 45.0);
    EXPECT_EQ(s.thetavar, "tvar");
    EXPECT_DOUBLE_EQ(s.phi, 15.0);
    EXPECT_TRUE(s.centerdynamic);
    EXPECT_DOUBLE_EQ(s.cx, 0.25);
    EXPECT_DOUBLE_EQ(s.cz, 0.75);
    EXPECT_DOUBLE_EQ(s.upx, 1.0);
    EXPECT_DOUBLE_EQ(s.upz, 0.0);
    EXPECT_DOUBLE_EQ(s.zoom, 2.5);
    EXPECT_EQ(s.zoomvar, "zvar");
}

TEST_F(Settings, TheQualityTabDrivesItsFields)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());

    ctl<QCheckBox>(d, "ssao")->setChecked(true);
    ctl<QDoubleSpinBox>(d, "ssaoint")->setValue(0.8);
    ctl<QCheckBox>(d, "fsaa")->setChecked(true);
    ctl<QSlider>(d, "shiny")->setValue(40);
    ctl<QLineEdit>(d, "backcolor")->setText("navy");
    ctl<QCheckBox>(d, "gradient")->setChecked(true);
    ctl<QLineEdit>(d, "backcolor2")->setText("skyblue");
    ctl<QSlider>(d, "light0")->setValue(20);
    ctl<QSlider>(d, "light3")->setValue(50);

    const DumpImageSettings s = d.settings();
    EXPECT_TRUE(s.ssao);
    EXPECT_DOUBLE_EQ(s.ssaoint, 0.8);
    EXPECT_TRUE(s.fsaa);
    EXPECT_DOUBLE_EQ(s.shiny, 0.4);
    EXPECT_EQ(s.backcolor, "navy");
    EXPECT_TRUE(s.gradient);
    EXPECT_EQ(s.backcolor2, "skyblue");
    EXPECT_DOUBLE_EQ(s.amblight, 0.2);
    EXPECT_DOUBLE_EQ(s.backlight, 0.5);
    EXPECT_TRUE(command(s).contains("fsaa yes")) << command(s).toStdString();
}

// ------------------------------------------------------ environment gating

TEST_F(Settings, WithNoSurfacesTheSurfacesTabIsRefused)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env2d(), twoSpecies());

    auto *show = ctl<QCheckBox>(d, "surf");
    ASSERT_NE(show, nullptr);
    EXPECT_FALSE(show->isEnabled()) << "surfaces can be switched on in a run that has none";

    // and even forced on, the answer is still no -- the belt to that braces
    show->setChecked(true);
    EXPECT_FALSE(d.settings().surf);
}

TEST_F(Settings, InTwoDimensionsTheZPlaneAndTheUpVectorAreRefused)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env2d(), twoSpecies());

    EXPECT_FALSE(ctl<QCheckBox>(d, "gridz")->isEnabled()) << "a 2d run has no z cut plane";
    EXPECT_FALSE(ctl<QCheckBox>(d, "gridz")->isChecked());
    EXPECT_FALSE(ctl<QDoubleSpinBox>(d, "gridzcoord")->isEnabled());

    EXPECT_FALSE(ctl<QDoubleSpinBox>(d, "theta")->isEnabled()) << "a 2d run is viewed head on";
    EXPECT_FALSE(ctl<QDoubleSpinBox>(d, "phi")->isEnabled());
    for (const char *n : {"up0", "up1", "up2"})
        EXPECT_FALSE(ctl<QLineEdit>(d, n)->isEnabled()) << n;
}

TEST_F(Settings, TheCombosOfferWhatTheEnvironmentSays)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());

    auto items = [](QComboBox *b) {
        QStringList out;
        for (int i = 0; i < b->count(); ++i)
            out << b->itemText(i);
        return out;
    };
    EXPECT_EQ(items(ctl<QComboBox>(d, "mixture")), env.mixtures);
    // the region combo leads with its own "no clip" entry
    EXPECT_EQ(items(ctl<QComboBox>(d, "region")), QStringList{"none"} + env.regions);
    EXPECT_EQ(items(ctl<QComboBox>(d, "gridgroup")), env.gridGroups);
    EXPECT_EQ(items(ctl<QComboBox>(d, "surfgroup")), env.surfGroups);
    EXPECT_EQ(items(ctl<QComboBox>(d, "gridcolor")), env.gridSources);
    EXPECT_EQ(items(ctl<QComboBox>(d, "surfcolor")), env.surfSources);
}

// A grid group combo that came up empty would emit "gridgroup " with nothing
// after it, so it falls back to the group SPARTA always has.
TEST_F(Settings, AnEmptyGroupListFallsBackToAll)
{
    ImageSettingsEnv bare = env;
    bare.gridGroups.clear();
    bare.surfGroups.clear();
    DumpImageSettingsDialog d(DumpImageSettings{}, bare, twoSpecies());

    EXPECT_EQ(ctl<QComboBox>(d, "gridgroup")->currentText(), "all");
    EXPECT_EQ(ctl<QComboBox>(d, "surfgroup")->currentText(), "all");
}

// --------------------------------------------------------- guarded reads

// Every numeric and colour editor keeps its previous value when what is typed
// into it cannot be parsed. That is not politeness: the alternative is a field
// silently resetting to a SPARTA default the user never chose.
TEST_F(Settings, RubbishInAnEditorKeepsThePreviousValue)
{
    const DumpImageSettings in = busy();
    DumpImageSettingsDialog d(in, env, twoSpecies());

    for (const char *n : {"glinediam", "slinediam", "boxdiam", "subboxdiam", "axeslen",
                          "axesdiam", "surfdiam", "zoom", "pdiamvalue"})
        ctl<QLineEdit>(d, n)->setText("!!junk!!");

    const DumpImageSettings s = d.settings();
    EXPECT_DOUBLE_EQ(s.glinediam, in.glinediam);
    EXPECT_DOUBLE_EQ(s.slinediam, in.slinediam);
    EXPECT_DOUBLE_EQ(s.boxdiam, in.boxdiam);
    EXPECT_DOUBLE_EQ(s.subboxdiam, in.subboxdiam);
    EXPECT_DOUBLE_EQ(s.axeslen, in.axeslen);
    EXPECT_DOUBLE_EQ(s.axesdiam, in.axesdiam);
    EXPECT_DOUBLE_EQ(s.surfdiam, in.surfdiam);
    EXPECT_DOUBLE_EQ(s.zoom, in.zoom);
}

// The up vector is read as a unit: one unusable component, or a vector that
// points nowhere, leaves all three alone rather than half-applying.
TEST_F(Settings, AnUnusableUpVectorLeavesAllThreeComponentsAlone)
{
    DumpImageSettings in = busy();
    in.upx = 1.0; in.upy = 2.0; in.upz = 3.0;
    {
        DumpImageSettingsDialog d(in, env, twoSpecies());
        ctl<QLineEdit>(d, "up1")->setText("!!junk!!");
        const DumpImageSettings s = d.settings();
        EXPECT_DOUBLE_EQ(s.upx, 1.0);
        EXPECT_DOUBLE_EQ(s.upy, 2.0);
        EXPECT_DOUBLE_EQ(s.upz, 3.0) << "one bad component applied the other two";
    }
    {
        DumpImageSettingsDialog d(in, env, twoSpecies());
        ctl<QLineEdit>(d, "up0")->setText("0");
        ctl<QLineEdit>(d, "up1")->setText("0");
        ctl<QLineEdit>(d, "up2")->setText("0");
        const DumpImageSettings s = d.settings();
        EXPECT_DOUBLE_EQ(s.upy, 2.0) << "a zero up vector was accepted";
    }
    {
        DumpImageSettingsDialog d(in, env, twoSpecies());
        ctl<QLineEdit>(d, "up0")->setText("0");
        ctl<QLineEdit>(d, "up1")->setText("0");
        ctl<QLineEdit>(d, "up2")->setText("-1");
        const DumpImageSettings s = d.settings();
        EXPECT_DOUBLE_EQ(s.upz, -1.0) << "a valid up vector was refused";
    }
}

// ----------------------------------------------------------- exclusivity

// Volume rendering and cut planes are alternatives; SPARTA takes one or the
// other, and the dialog has to keep them apart while the user is editing.
TEST_F(Settings, TheGridVolumeAndTheCutPlanesExcludeEachOther)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());

    ctl<QCheckBox>(d, "gridx")->setChecked(true);
    ctl<QCheckBox>(d, "grid")->setChecked(true);
    EXPECT_FALSE(ctl<QCheckBox>(d, "gridx")->isChecked())
        << "switching to volume rendering left a cut plane on";

    ctl<QCheckBox>(d, "gridy")->setChecked(true);
    EXPECT_FALSE(ctl<QCheckBox>(d, "grid")->isChecked())
        << "switching a cut plane on left volume rendering on";

    const QString cmd = command(d.settings());
    EXPECT_FALSE(cmd.contains(" grid ") && cmd.contains(" gridy "))
        << "both were emitted: " << cmd.toStdString();
}

// --------------------------------------------------------- colour maps

// Six independent maps behind one set of controls. The mode combo has to store
// the map being left before it loads the one being entered, or edits are lost.
TEST_F(Settings, TheSixColourMapsAreIndependent)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());
    auto *mode = ctl<QComboBox>(d, "cmap.mode");
    ASSERT_NE(mode, nullptr);
    ASSERT_EQ(mode->count(), DumpImageSettings::NUM_CMAP_MODES);

    mode->setCurrentIndex(DumpImageSettings::PARTICLE);
    ctl<QCheckBox>(d, "cmap.active")->setChecked(true);
    ctl<QLineEdit>(d, "cmap.lo")->setText("0");

    mode->setCurrentIndex(DumpImageSettings::SURF);
    ctl<QCheckBox>(d, "cmap.active")->setChecked(true);
    ctl<QLineEdit>(d, "cmap.lo")->setText("5");

    mode->setCurrentIndex(DumpImageSettings::PARTICLE);
    EXPECT_EQ(ctl<QLineEdit>(d, "cmap.lo")->text(), "0")
        << "going back to a map did not restore what was typed into it";

    const DumpImageSettings s = d.settings();
    EXPECT_TRUE(s.cmap[DumpImageSettings::PARTICLE].active);
    EXPECT_EQ(s.cmap[DumpImageSettings::PARTICLE].lo, "0");
    EXPECT_TRUE(s.cmap[DumpImageSettings::SURF].active);
    EXPECT_EQ(s.cmap[DumpImageSettings::SURF].lo, "5");
    EXPECT_FALSE(s.cmap[DumpImageSettings::GRIDZ].active) << "a map nobody touched came on";
}

// The map on screen has not been stored into the working set yet -- that
// happens on a mode change -- so settings() has to flush it.
TEST_F(Settings, TheMapOnScreenIsFlushedWithoutSwitchingAway)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());
    ctl<QComboBox>(d, "cmap.mode")->setCurrentIndex(DumpImageSettings::GRID);
    ctl<QCheckBox>(d, "cmap.active")->setChecked(true);
    ctl<QLineEdit>(d, "cmap.hi")->setText("42");
    ctl<QRadioButton>(d, "cmap.style.d")->setChecked(true);

    const ColorMapSpec &m = d.settings().cmap[DumpImageSettings::GRID];
    EXPECT_TRUE(m.active) << "the map being edited was not read back";
    EXPECT_EQ(m.hi, "42");
    EXPECT_EQ(m.style, QChar('d'));
}

// ---------------------------------------------------------- species table

TEST_F(Settings, TheSpeciesTableReportsEditedColoursAndDiameters)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());

    ctl<QLineEdit>(d, "particle.colorName.1")->setText("blue");
    ctl<QLineEdit>(d, "particle.pdiam.2")->setText("0.5");

    const auto colors = d.speciesColors();
    ASSERT_EQ(colors.size(), 2);
    EXPECT_EQ(colors.at(0).first, "blue");
    EXPECT_EQ(colors.at(0).second, QColor("blue"));
    EXPECT_EQ(colors.at(1).first, "green") << "an untouched row changed";

    const DumpImageSettings s = d.settings();
    EXPECT_EQ(s.pdiams, (QList<QPair<QString, double>>{{"2", 0.5}}))
        << "the per-species diameter did not reach pdiams, or a default one did";
}

// A name Qt cannot parse keeps the row's RGB, which is how a colour picked from
// the swatch survives being given a name of its own.
TEST_F(Settings, AnUnparseableColourNameKeepsTheRowsRgb)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());
    ctl<QLineEdit>(d, "particle.colorName.1")->setText("guisp1");

    const auto colors = d.speciesColors();
    EXPECT_EQ(colors.at(0).first, "guisp1");
    EXPECT_EQ(colors.at(0).second, QColor(255, 0, 0)) << "the picked colour was thrown away";
}

// --------------------------------------------------------- degenerate envs

TEST_F(Settings, SurvivesAnEnvironmentWithNothingInIt)
{
    ImageSettingsEnv bare;   // no species, no mixtures, no sources, unit box
    DumpImageSettingsDialog d(DumpImageSettings{}, bare, {});
    EXPECT_EQ(d.findChild<QTabWidget *>()->count(), 8);
    EXPECT_TRUE(d.speciesColors().isEmpty());
    // whatever it answers must at least be a command the builders accept
    EXPECT_FALSE(command(d.settings()).isEmpty());
}

TEST_F(Settings, SurvivesAShortSpeciesColourList)
{
    // the caller tops the table up before constructing; if it did not, the
    // dialog must pad rather than index off the end
    DumpImageSettingsDialog d(DumpImageSettings{}, env, {});
    EXPECT_EQ(d.speciesColors().size(), env.species.size());
}

// ------------------------------------------------------------------- help

TEST_F(Settings, TheHelpButtonAsksForThePageRatherThanOpeningIt)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());
    QSignalSpy spy(&d, &DumpImageSettingsDialog::helpRequested);

    for (auto *b : d.findChildren<QPushButton *>())
        if (b->text().contains("Help")) b->click();

    ASSERT_EQ(spy.count(), 1) << "the Help button did nothing";
    EXPECT_EQ(spy.at(0).at(0).toString(), "dump_image.html");
}

// ------------------------------------------------------------------ naming

// The convention the tests above depend on. Without this, the first control
// added without a name is invisible to every test here and to the AT-SPI
// walker, and nobody finds out until someone goes looking for it.
TEST_F(Settings, EveryControlIsNamedAndTheNamesAreUnique)
{
    DumpImageSettingsDialog d(DumpImageSettings{}, env, twoSpecies());

    std::set<QString> seen;
    int named = 0, anonymous = 0;
    for (auto *w : d.findChildren<QWidget *>()) {
        const bool interactive = qobject_cast<QCheckBox *>(w) || qobject_cast<QComboBox *>(w) ||
                                 qobject_cast<QLineEdit *>(w) || qobject_cast<QRadioButton *>(w) ||
                                 qobject_cast<QSlider *>(w) || qobject_cast<QSpinBox *>(w) ||
                                 qobject_cast<QDoubleSpinBox *>(w);
        if (!interactive) continue;
        // combo boxes and spin boxes own an internal line edit; only the
        // outer control carries a name
        if (qobject_cast<QComboBox *>(w->parentWidget()) ||
            qobject_cast<QAbstractSpinBox *>(w->parentWidget()))
            continue;

        if (!w->objectName().startsWith("ivs.")) {
            ++anonymous;
            ADD_FAILURE() << "unnamed " << w->metaObject()->className() << " on tab \""
                          << (w->parentWidget() ? w->parentWidget()->objectName().toStdString()
                                                : std::string("?"))
                          << "\"";
            continue;
        }
        EXPECT_TRUE(seen.insert(w->objectName()).second)
            << w->objectName().toStdString() << " is used twice";
        EXPECT_FALSE(w->accessibleName().isEmpty())
            << w->objectName().toStdString() << " has no accessible name, so the AT-SPI "
                                                "walker and the screenshot sweep cannot find it";
        ++named;
    }
    EXPECT_EQ(anonymous, 0);
    EXPECT_GT(named, 80) << "only " << named << " named controls; the dialog looks truncated";
    RecordProperty("named_controls", named);
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
