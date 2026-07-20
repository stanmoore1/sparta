// Unit tests for the pure dump-image command builders (src/dumpimage.cpp).
//
// These tests exercise buildDumpImageCommand(), buildDumpModifyCommands(), and
// buildDumpSnippet() without a GUI or a live SPARTA instance: a
// DumpImageSettings struct is populated and the generated command strings are
// compared against the expected SPARTA syntax (verified against
// sparta/src/dump_image.cpp).  QString results are converted to std::string so
// failures print readable diffs.

#include "dumpimage.h"

#include <QString>
#include <QStringList>

#include "gtest/gtest.h"

#include <string>

namespace {

std::string args(const DumpImageSettings &s)
{
    return buildDumpImageCommand(s).toStdString();
}

std::string modify(const DumpImageSettings &s, bool movie = false)
{
    return buildDumpModifyCommands(s, "viz", movie).join('\n').toStdString();
}

// A default-constructed struct mirrors the SPARTA built-in defaults: only the
// positional arguments and the image size are emitted and no dump_modify
// command is needed.
TEST(DumpImage, MinimalDefaults)
{
    DumpImageSettings s;
    EXPECT_EQ(args(s), "type type size 600 600");
    EXPECT_TRUE(buildDumpModifyCommands(s, "viz").isEmpty());
}

// positional color and diameter attributes are passed through verbatim
TEST(DumpImage, PositionalColorAndDiameter)
{
    DumpImageSettings s;
    s.color    = "ke";
    s.diameter = "vx";
    EXPECT_EQ(args(s), "ke vx size 600 600");

    // custom per-particle and compute references are legal color sources
    s.color = "c_myc[2]";
    EXPECT_EQ(args(s), "c_myc[2] vx size 600 600");
}

// particle no and the numeric pdiam keyword
TEST(DumpImage, ParticleOffAndNumericDiameter)
{
    DumpImageSettings s;
    s.particle    = false;
    s.numericdiam = true;
    s.pdiamvalue  = 0.5;
    EXPECT_EQ(args(s), "type type particle no pdiam 0.5 size 600 600");
}

// grid volume rendering with a compute reference; gcolor rows are only
// emitted when the grid is colored by proc
TEST(DumpImage, GridVolume)
{
    DumpImageSettings s;
    s.grid      = true;
    s.gridcolor = "c_temp[2]";
    s.gcolors.emplaceBack("*", "red/green/blue");
    EXPECT_EQ(args(s), "type type grid c_temp[2] size 600 600");
    EXPECT_EQ(modify(s), ""); // no gcolor: not colored by proc

    s.gridcolor = "proc";
    EXPECT_EQ(args(s), "type type grid proc size 600 600");
    EXPECT_EQ(modify(s), "dump_modify viz gcolor * red/green/blue");
}

// the three grid cut planes with their coordinates and color sources
TEST(DumpImage, GridPlanes)
{
    DumpImageSettings s;
    s.gridx      = true;
    s.gridxcoord = 0.005;
    s.gridxcolor = "proc";
    s.gridy      = true;
    s.gridycoord = -1.5;
    s.gridycolor = "f_ave[1]";
    s.gridz      = true;
    s.gridzcoord = 2.0;
    s.gridzcolor = "v_gval";
    EXPECT_EQ(args(s), "type type gridx 0.005 proc gridy -1.5 f_ave[1] gridz 2 v_gval "
                       "size 600 600");
}

// grid volume rendering and grid planes are mutually exclusive; the planes win
TEST(DumpImage, GridVolumePlaneExclusivity)
{
    DumpImageSettings s;
    s.grid       = true;
    s.gridcolor  = "proc";
    s.gridx      = true;
    s.gridxcoord = 1.0;
    s.gridxcolor = "c_val";
    EXPECT_EQ(args(s), "type type gridx 1 c_val size 600 600");
}

// gridgroup applies to both volume and plane rendering
TEST(DumpImage, GridGroup)
{
    DumpImageSettings s;
    s.grid      = true;
    s.gridcolor = "c_temp";
    s.gridgroup = "inner";
    EXPECT_EQ(modify(s), "dump_modify viz gridgroup inner");

    // no grid displayed -> no gridgroup emitted
    s.grid = false;
    EXPECT_EQ(modify(s), "");
}

// surfaces: color modes one/proc/attribute, element diameter, group, outlines
TEST(DumpImage, Surfaces)
{
    DumpImageSettings s;
    s.surf      = true;
    s.surfcolor = "one";
    s.surfdiam  = 0.5;
    EXPECT_EQ(args(s), "type type surf one 0.5 size 600 600");
    EXPECT_EQ(modify(s), "");

    // non-default single color
    s.surfcolorone = "white";
    EXPECT_EQ(modify(s), "dump_modify viz scolor * white");

    // per-proc coloring with explicit proc color rows and a surf group
    s.surfcolor = "proc";
    s.scolors.emplaceBack("1*4", "red/blue");
    s.surfgroup = "walls";
    EXPECT_EQ(args(s), "type type surf proc 0.5 size 600 600");
    EXPECT_EQ(modify(s), "dump_modify viz scolor 1*4 red/blue\n"
                         "dump_modify viz surfgroup walls");

    // surf element outlines
    s.scolors.clear();
    s.surfgroup  = "all";
    s.sline      = true;
    s.slinediam  = 0.003;
    s.slinecolor = "orange";
    EXPECT_EQ(args(s), "type type surf proc 0.5 size 600 600 sline yes 0.003");
    EXPECT_EQ(modify(s), "dump_modify viz slinecolor orange");
}

// grid cell outlines with a non-default color
TEST(DumpImage, GridLines)
{
    DumpImageSettings s;
    s.gline      = true;
    s.glinediam  = 0.002;
    s.glinecolor = "black";
    EXPECT_EQ(args(s), "type type size 600 600 gline yes 0.002");
    EXPECT_EQ(modify(s), "dump_modify viz glinecolor black");
}

// camera settings with plain numbers
TEST(DumpImage, CameraNumeric)
{
    DumpImageSettings s;
    s.theta = 80.0;
    s.phi   = -30.0;
    s.cx    = 0.4;
    s.cz    = 0.6;
    s.upx   = 1.0;
    s.upz   = 0.0;
    s.zoom  = 1.5;
    EXPECT_EQ(args(s), "type type size 600 600 view 80 -30 center s 0.4 0.5 0.6 "
                       "up 1 0 0 zoom 1.5");
}

// v_name equal-style variable bindings for theta, phi, zoom, and the center
TEST(DumpImage, CameraVariables)
{
    DumpImageSettings s;
    s.thetavar = "th";
    s.cxvar    = "cx0";
    s.zoomvar  = "zm";
    EXPECT_EQ(args(s), "type type size 600 600 view v_th 30 center s v_cx0 0.5 0.5 "
                       "zoom v_zm");

    // dynamic center flag
    s.centerdynamic = true;
    EXPECT_EQ(args(s), "type type size 600 600 view v_th 30 center d v_cx0 0.5 0.5 "
                       "zoom v_zm");
}

// in 2d SPARTA forces view 0 0 and up 0 1 0, so neither keyword is emitted
TEST(DumpImage, TwoDimensional)
{
    DumpImageSettings s;
    s.dimension = 2;
    s.theta     = 80.0;
    s.phi       = -30.0;
    s.upx       = 1.0;
    EXPECT_EQ(args(s), "type type size 600 600");
}

// the new fsaa/subbox keywords, ssao with its seed, shiny, and box options
TEST(DumpImage, QualityBoxAndAxes)
{
    DumpImageSettings s;
    s.box        = true;
    s.boxdiam    = 0.03;
    s.subbox     = true;
    s.subboxdiam = 0.01;
    s.axes       = true;
    s.axeslen    = 0.4;
    s.axesdiam   = 0.01;
    s.shiny      = 0.5;
    s.ssao       = true;
    s.ssaoseed   = 453983;
    s.ssaoint    = 0.7;
    s.fsaa       = true;
    EXPECT_EQ(args(s), "type type size 600 600 box yes 0.03 subbox yes 0.01 "
                       "axes yes 0.4 0.01 shiny 0.5 ssao yes 453983 0.7 fsaa yes");

    s.box = false;
    EXPECT_EQ(args(s), "type type size 600 600 box no 0.03 subbox yes 0.01 "
                       "axes yes 0.4 0.01 shiny 0.5 ssao yes 453983 0.7 fsaa yes");
}

// backcolor/backcolor2 background gradient and the box/subbox colors
TEST(DumpImage, BackgroundAndBoxColors)
{
    DumpImageSettings s;
    s.backcolor = "white";
    EXPECT_EQ(modify(s), "dump_modify viz backcolor white");

    // with the gradient enabled both colors are always emitted
    s.backcolor   = "black";
    s.gradient    = true;
    s.backcolor2  = "gray";
    s.boxcolor    = "green";
    s.subbox      = true;
    s.subboxcolor = "red";
    EXPECT_EQ(modify(s), "dump_modify viz backcolor black\n"
                         "dump_modify viz backcolor2 gray\n"
                         "dump_modify viz boxcolor green\n"
                         "dump_modify viz subboxcolor red");
}

// lights are emitted only when changed from the SPARTA defaults
TEST(DumpImage, Lights)
{
    DumpImageSettings s;
    EXPECT_EQ(modify(s), "");

    s.amblight  = 0.1;
    s.keylight  = 0.8;
    s.filllight = 0.4;
    s.backlight = 0.7;
    EXPECT_EQ(modify(s), "dump_modify viz lights 0.1 0.8 0.4 0.7");
}

// per-type pcolor and pdiam rows, custom color definitions, and region clip
TEST(DumpImage, PerTypeRowsAndRegion)
{
    DumpImageSettings s;
    s.customcolors.emplaceBack("mycolor", "0.100 0.200 0.300");
    s.pcolors.emplaceBack("1", "mycolor");
    s.pcolors.emplaceBack("2*3", "blue/red");
    s.pdiams.emplaceBack("2", 0.5);
    s.region = "clip1";
    EXPECT_EQ(modify(s), "dump_modify viz color mycolor 0.100 0.200 0.300\n"
                         "dump_modify viz pcolor 1 mycolor\n"
                         "dump_modify viz pcolor 2*3 blue/red\n"
                         "dump_modify viz pdiam 2 0.5\n"
                         "dump_modify viz region clip1");

    // pcolor/pdiam only apply when coloring/sizing by type (or proc for pcolor)
    s.color    = "ke";
    s.diameter = "vx";
    EXPECT_EQ(modify(s), "dump_modify viz color mycolor 0.100 0.200 0.300\n"
                         "dump_modify viz region clip1");
}

// continuous color map for particles: the default BWR map defines its two RGB
// stops as custom colors on the same dump_modify line, entries run min -> max
TEST(DumpImage, CmapParticleContinuous)
{
    DumpImageSettings s;
    s.cmap[DumpImageSettings::PARTICLE].active = true;
    EXPECT_EQ(modify(s),
              "dump_modify viz color guimapp1 0.000 0.227 0.427 "
              "color guimapp2 0.459 0.055 0.075 "
              "cmap particle min max cf 0.0 3 min guimapp1 0.5 white max guimapp2");
}

// reversing a map flips the stop order and mirrors continuous positions
TEST(DumpImage, CmapReverse)
{
    DumpImageSettings s;
    auto &m   = s.cmap[DumpImageSettings::GRIDX];
    m.active  = true;
    m.reverse = true;
    EXPECT_EQ(modify(s),
              "dump_modify viz color guimapx1 0.459 0.055 0.075 "
              "color guimapx2 0.000 0.227 0.427 "
              "cmap gridx min max cf 0.0 3 min guimapx1 0.5 white max guimapx2");
}

// sequential style: named colors repeat in bins of the given width
TEST(DumpImage, CmapSequential)
{
    DumpImageSettings s;
    auto &m   = s.cmap[DumpImageSettings::GRID];
    m.active  = true;
    m.mapname = "Basic";
    m.style   = 's';
    m.delta   = 2.5;
    m.lo      = "0";
    m.hi      = "10";
    EXPECT_EQ(modify(s), "dump_modify viz cmap grid 0 10 sf 2.5 10 "
                         "red cyan green black magenta blue yellow purple white orange");
}

// discrete style with absolute range: equally wide bins mapped to [lo,hi]
TEST(DumpImage, CmapDiscreteAbsolute)
{
    DumpImageSettings s;
    auto &m   = s.cmap[DumpImageSettings::SURF];
    m.active  = true;
    m.mapname = "Grayscale";
    m.style   = 'd';
    m.range   = 'a';
    m.lo      = "0";
    m.hi      = "100";
    EXPECT_EQ(modify(s), "dump_modify viz cmap surf 0 100 da 0.0 2 "
                         "min 50 black min max white");

    // absolute needs numeric bounds; with min/max it falls back to fractional
    m.lo = "min";
    m.hi = "max";
    EXPECT_EQ(modify(s), "dump_modify viz cmap surf min max df 0.0 2 "
                         "min 0.5 black min max white");
}

// all six color map modes are independent and emitted in a fixed order
TEST(DumpImage, CmapAllSixModes)
{
    DumpImageSettings s;
    for (int mode = 0; mode < DumpImageSettings::NUM_CMAP_MODES; ++mode) {
        s.cmap[mode].active  = true;
        s.cmap[mode].mapname = "Grayscale"; // named stops only -> no color defs
    }
    const QStringList cmds = buildDumpModifyCommands(s, "viz");
    ASSERT_EQ(cmds.size(), 6);
    const char *modes[] = {"particle", "grid", "surf", "gridx", "gridy", "gridz"};
    for (int mode = 0; mode < 6; ++mode) {
        EXPECT_EQ(cmds[mode].toStdString(),
                  std::string("dump_modify viz cmap ") + modes[mode] +
                      " min max cf 0.0 2 min black max white");
    }
}

// movie-only settings and the dump movie snippet
TEST(DumpImage, MovieFramerateBitrate)
{
    DumpImageSettings s;
    s.framerate = 10.5;
    s.bitrate   = 2000;

    // not emitted for an image dump
    EXPECT_EQ(modify(s, false), "");
    EXPECT_EQ(modify(s, true), "dump_modify viz framerate 10.5\n"
                               "dump_modify viz bitrate 2000");

    const std::string snippet = buildDumpSnippet(s, true, "mymovie.mp4", 100).toStdString();
    EXPECT_EQ(snippet, "dump movie movie all 100 mymovie.mp4 type type size 600 600\n"
                       "dump_modify movie framerate 10.5\n"
                       "dump_modify movie bitrate 2000\n");
}

// the image snippet includes zero-padded frame numbers via dump_modify pad
TEST(DumpImage, ImageSnippet)
{
    DumpImageSettings s;
    s.mixture = "flow";
    const std::string snippet =
        buildDumpSnippet(s, false, "myimage-*.png", 250).toStdString();
    EXPECT_EQ(snippet, "dump viz image flow 250 myimage-*.png type type size 600 600\n"
                       "dump_modify viz pad 9\n");
}

// a representative full-featured render: every major keyword at once
TEST(DumpImage, FullFeaturedRender)
{
    DumpImageSettings s;
    s.mixture       = "flow";
    s.color         = "proc";
    s.diameter      = "type";
    s.gridx         = true;
    s.gridxcoord    = 0.005;
    s.gridxcolor    = "c_ave[1]";
    s.surf          = true;
    s.surfcolor     = "one";
    s.surfdiam      = 0.5;
    s.xsize         = 800;
    s.ysize         = 600;
    s.theta         = 80.0;
    s.phi           = -30.0;
    s.centerdynamic = true;
    s.cx            = 0.4;
    s.cz            = 0.6;
    s.zoom          = 1.5;
    s.boxdiam       = 0.03;
    s.subbox        = true;
    s.subboxdiam    = 0.01;
    s.gline         = true;
    s.glinediam     = 0.002;
    s.sline         = true;
    s.slinediam     = 0.003;
    s.axes          = true;
    s.axeslen       = 0.4;
    s.axesdiam      = 0.01;
    s.shiny         = 0.5;
    s.ssao          = true;
    s.ssaoint       = 0.7;
    s.fsaa          = true;
    s.gradient      = true;
    s.backcolor2    = "gray";
    s.boxcolor      = "green";
    s.subboxcolor   = "red";
    s.glinecolor    = "black";
    s.slinecolor    = "orange";
    s.surfcolorone  = "white";
    s.gridgroup     = "inner";
    s.surfgroup     = "walls";
    s.region        = "clip1";
    s.pcolors.emplaceBack("*", "blue/red");
    s.pdiams.emplaceBack("2", 0.5);
    s.amblight = 0.1;
    s.cmap[DumpImageSettings::GRIDX].active = true;

    EXPECT_EQ(args(s),
              "proc type gridx 0.005 c_ave[1] surf one 0.5 size 800 600 "
              "view 80 -30 center d 0.4 0.5 0.6 zoom 1.5 box yes 0.03 "
              "subbox yes 0.01 gline yes 0.002 sline yes 0.003 axes yes 0.4 0.01 "
              "shiny 0.5 ssao yes 453983 0.7 fsaa yes");

    EXPECT_EQ(modify(s),
              "dump_modify viz backcolor black\n"
              "dump_modify viz backcolor2 gray\n"
              "dump_modify viz boxcolor green\n"
              "dump_modify viz subboxcolor red\n"
              "dump_modify viz glinecolor black\n"
              "dump_modify viz gridgroup inner\n"
              "dump_modify viz pcolor * blue/red\n"
              "dump_modify viz pdiam 2 0.5\n"
              "dump_modify viz region clip1\n"
              "dump_modify viz scolor * white\n"
              "dump_modify viz slinecolor orange\n"
              "dump_modify viz surfgroup walls\n"
              "dump_modify viz lights 0.1 0.9 0.45 0.9\n"
              "dump_modify viz color guimapx1 0.000 0.227 0.427 "
              "color guimapx2 0.459 0.055 0.075 "
              "cmap gridx min max cf 0.0 3 min guimapx1 0.5 white max guimapx2");
}

TEST(DumpImageMath, WrapAzimuth)
{
    EXPECT_DOUBLE_EQ(wrapAzimuth(30.0), 30.0);
    EXPECT_DOUBLE_EQ(wrapAzimuth(190.0), -170.0);
    EXPECT_DOUBLE_EQ(wrapAzimuth(-190.0), 170.0);
    EXPECT_DOUBLE_EQ(wrapAzimuth(180.0), 180.0);   // upper bound kept
    EXPECT_DOUBLE_EQ(wrapAzimuth(540.0), 180.0);   // 540 -> 180
}

TEST(DumpImageMath, ClampPolar)
{
    EXPECT_DOUBLE_EQ(clampPolar(60.0), 60.0);
    EXPECT_DOUBLE_EQ(clampPolar(-5.0), 0.0);
    EXPECT_DOUBLE_EQ(clampPolar(200.0), 180.0);
}

TEST(DumpImageMath, ClampZoom)
{
    EXPECT_DOUBLE_EQ(clampZoom(1.0), 1.0);
    EXPECT_DOUBLE_EQ(clampZoom(0.01), 0.1);
    EXPECT_DOUBLE_EQ(clampZoom(99.0), 10.0);
}

} // namespace

// Local Variables:
// c-basic-offset: 4
// End:
