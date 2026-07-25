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

// What the renderer actually draws.
//
// Four dump image features were back-ported into the SPARTA core for the GUI --
// full-scene antialiasing, the sub-box outline, the background gradient, and
// the light intensities. Each has a control in the image viewer and a test that
// the right command string comes out, and neither of those runs a single line
// of the C++ in image.cpp and dump_image.cpp that has to make the picture
// different. A keyword the parser accepts and then ignores passes every check
// that existed.
//
// So each feature is rendered for real and compared against the same scene
// rendered without it, in the way that feature is supposed to differ. And the
// default render is compared against itself across the whole set, because a
// back-port that changes the picture nobody asked to change is the other way
// this goes wrong.
//
// PPM, not PNG: this build reports PNG and JPEG support as "no", so P6 is the
// only format the renderer can be asked for.

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QRgb>
#include <QSet>
#include <QStringList>
#include <QTemporaryDir>

#include <cmath>

#include <cstdio>
#include <string>
#include <vector>

#include "spartawrapper.h"

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

QString fixtures()
{
#if defined(SPARTA_RENDER_FIXTURES)
    return QString(SPARTA_RENDER_FIXTURES);
#else
    return QString();
#endif
}

struct Ppm {
    int w = 0, h = 0;
    std::vector<unsigned char> px;   // RGB triples
    bool ok() const { return w > 0 && h > 0 && px.size() == size_t(w) * h * 3; }
    const unsigned char *at(int x, int y) const { return &px[(size_t(y) * w + x) * 3]; }
};

// A P6 reader, deliberately minimal: the renderer writes a fixed header.
Ppm readPpm(const QString &path)
{
    Ppm img;
    FILE *f = fopen(path.toLocal8Bit().constData(), "rb");
    if (!f) return img;
    int maxval = 0;
    if (fscanf(f, "P6 %d %d %d", &img.w, &img.h, &maxval) != 3 || maxval != 255) {
        fclose(f);
        img.w = img.h = 0;
        return img;
    }
    fgetc(f); // the single whitespace byte before the raster
    img.px.resize(size_t(img.w) * img.h * 3);
    if (fread(img.px.data(), 1, img.px.size(), f) != img.px.size()) img.px.clear();
    fclose(f);
    return img;
}

// How many colours the picture uses. Antialiasing and lighting changes show up
// here even when the shapes are identical: both work by adding intermediate
// shades that were not there before.
size_t distinctColors(const Ppm &img)
{
    QSet<QRgb> seen;
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) {
            const unsigned char *p = img.at(x, y);
            seen.insert(qRgb(p[0], p[1], p[2]));
        }
    return size_t(seen.size());
}

size_t differingPixels(const Ppm &a, const Ppm &b)
{
    if (a.w != b.w || a.h != b.h) return size_t(-1);
    size_t n = 0;
    for (size_t i = 0; i < a.px.size(); i += 3)
        if (a.px[i] != b.px[i] || a.px[i + 1] != b.px[i + 1] || a.px[i + 2] != b.px[i + 2]) ++n;
    return n;
}

// Renders one frame of a fixed scene, with @p modify applied to the dump.
class Renderer : public ::testing::Test {
protected:
    void SetUp() override
    {
        if (!*testLibrary()) GTEST_SKIP() << "no shared libsparta to render with";
        if (fixtures().isEmpty()) GTEST_SKIP() << "no render fixtures configured";
        ASSERT_TRUE(dir.isValid());
        // The deck reads data.circle, air.species and air.vss by bare name, so
        // they have to be beside the process's working directory.
        for (const char *f : {"data.circle", "air.species", "air.vss"})
            ASSERT_TRUE(QFile::copy(fixtures() + "/" + f, dir.filePath(f)))
                << "missing fixture " << f;
    }

    // The same small 2d scene every time, rendered once, as a PPM.
    //
    // @p dumpArgs go on the dump image line, @p modify onto dump_modify lines.
    // Which of the two a keyword belongs to is not a detail: fsaa and subbox
    // are parsed by the dump image constructor, backcolor2 and lights by
    // modify_param, and putting one where the other goes is rejected outright.
    Ppm render(const QString &tag, const QStringList &modify,
               const QString &dumpArgs = QString())
    {
        SpartaWrapper sparta;
        if (!sparta.loadLib(testLibrary())) return {};
        char arg0[]  = "sparta";
        char *argv[] = {arg0, nullptr};
        sparta.open(1, argv);
        if (!sparta.isOpen()) return {};

        const QString prev = QDir::currentPath();
        QDir::setCurrent(dir.path());

        const QString out = tag + ".*.ppm";
        QStringList deck{
            "seed 12345",
            "dimension 2",
            "global gridcut 0.0 comm/sort yes",
            "boundary o r p",
            "create_box 0 10 0 10 -0.5 0.5",
            "create_grid 10 10 1",
            "balance_grid rcb cell",
            "global nrho 1.0 fnum 0.001",
            "species air.species N O",
            "mixture air N O vstream 100.0 0 0",
            "read_surf data.circle",
            "surf_collide 1 diffuse 300.0 0.0",
            "surf_modify all collide 1",
            "collide vss air air.vss",
            "timestep 0.0001",
            "dump 1 image all 1 " + out + " type type surf one 0.01 size 200 200 zoom 1.6" +
                (dumpArgs.isEmpty() ? QString() : " " + dumpArgs),
        };
        for (const QString &m : modify)
            deck << "dump_modify 1 " + m;
        deck << "run 0";

        sparta.commandsString(deck.join('\n'));
        QDir::setCurrent(prev);

        // The dump names the file after the timestep; run 0 makes that 0.
        return readPpm(dir.filePath(tag + ".0.ppm"));
    }

    QTemporaryDir dir;
};

} // namespace

TEST_F(Renderer, TheDefaultSceneRenders)
{
    const Ppm base = render("base", {});
    ASSERT_TRUE(base.ok()) << "the renderer produced no readable PPM at all";
    EXPECT_EQ(base.w, 200);
    EXPECT_EQ(base.h, 200);
    // Background plus a surface: a picture of one colour means nothing was drawn.
    EXPECT_GT(distinctColors(base), 2u) << "the render is a flat field; nothing reached it";
}

// Antialiasing renders at higher resolution and downsamples, so edges gain
// intermediate shades. A parser that accepts "fsaa yes" and does nothing gives
// back a byte-identical picture.
TEST_F(Renderer, AntialiasingChangesTheEdges)
{
    const Ppm plain = render("plain", {}, "fsaa no");
    const Ppm aa    = render("aa", {}, "fsaa yes");
    ASSERT_TRUE(plain.ok());
    ASSERT_TRUE(aa.ok());

    EXPECT_GT(differingPixels(plain, aa), 0u) << "fsaa yes rendered the identical picture";
    EXPECT_GT(distinctColors(aa), distinctColors(plain))
        << "antialiasing did not add intermediate shades, which is the whole of what it does";
}

// A vertical gradient: the top row and the bottom row of the background must
// differ. Without it both are the single background colour.
TEST_F(Renderer, TheBackgroundGradientPaintsTopAndBottomDifferently)
{
    const Ppm flat = render("flat", {"backcolor black", "backcolor2 none"});
    const Ppm grad = render("grad", {"backcolor black", "backcolor2 white"});
    ASSERT_TRUE(flat.ok());
    ASSERT_TRUE(grad.ok());

    // top-left and bottom-left are background in this scene, whatever is drawn
    const unsigned char *ftop = flat.at(0, 0), *fbot = flat.at(0, flat.h - 1);
    EXPECT_EQ(ftop[0], fbot[0]);
    EXPECT_EQ(ftop[1], fbot[1]);
    EXPECT_EQ(ftop[2], fbot[2]);

    const unsigned char *gtop = grad.at(0, 0), *gbot = grad.at(0, grad.h - 1);
    const int spread = std::abs(int(gtop[0]) - int(gbot[0])) +
                       std::abs(int(gtop[1]) - int(gbot[1])) +
                       std::abs(int(gtop[2]) - int(gbot[2]));
    EXPECT_GT(spread, 30) << "the background is the same at the top and the bottom, so "
                             "backcolor2 painted no gradient";
}

// The sub-box outline draws extra geometry, so it can only add pixels.
TEST_F(Renderer, TheSubBoxOutlineIsDrawn)
{
    const Ppm without = render("nosub", {}, "subbox no 0.02");
    const Ppm with    = render("sub", {}, "subbox yes 0.02");
    ASSERT_TRUE(without.ok());
    ASSERT_TRUE(with.ok());

    EXPECT_GT(differingPixels(without, with), 0u)
        << "subbox yes rendered the identical picture, so nothing was outlined";
}

// The lights are grayscale intensities applied to the shading. Turning them
// down has to darken what is lit; a scene that ignores them is unchanged.
TEST_F(Renderer, TheLightIntensitiesChangeTheShading)
{
    const Ppm bright = render("bright", {"lights 0.6 1.0 1.0 1.0"});
    const Ppm dim    = render("dim", {"lights 0.0 0.1 0.0 0.0"});
    ASSERT_TRUE(bright.ok());
    ASSERT_TRUE(dim.ok());

    EXPECT_GT(differingPixels(bright, dim), 0u)
        << "the two light settings rendered the identical picture";

    // and darker, not merely different
    auto meanLuma = [](const Ppm &img) {
        double sum = 0;
        for (size_t i = 0; i < img.px.size(); i += 3)
            sum += 0.299 * img.px[i] + 0.587 * img.px[i + 1] + 0.114 * img.px[i + 2];
        return sum / (img.px.size() / 3);
    };
    EXPECT_LT(meanLuma(dim), meanLuma(bright))
        << "turning every light down did not darken the picture";
}

// The other half: with none of the four asked for, the picture is what it
// always was. A back-port that changes the default render has changed every
// existing user's output without being asked.
TEST_F(Renderer, TheDefaultRenderIsReproducible)
{
    const Ppm a = render("rep1", {});
    const Ppm b = render("rep2", {});
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    EXPECT_EQ(differingPixels(a, b), 0u)
        << "two renders of the same scene with the same seed disagree";
}
