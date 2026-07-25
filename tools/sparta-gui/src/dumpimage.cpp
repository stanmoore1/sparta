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

// Pure command builders for SPARTA's `dump ID image mix-ID N file color diameter
// [keywords...]` command and its dump_modify options.  All knowledge about the
// SPARTA option syntax, defaults, and keyword order lives here (verified against
// sparta/src/dump_image.cpp); the GUI only fills a DumpImageSettings struct.

#include "dumpimage.h"

#include "colormaps.h"

#include <QStringList>
#include <algorithm>

namespace {

// SPARTA dump image built-in defaults (see src/dump_image.cpp and image.cpp).
// The builders emit an option only when it differs from these.
constexpr double DEF_BOXDIAM   = 0.02;
constexpr double DEF_THETA     = 60.0;
constexpr double DEF_PHI       = 30.0;
constexpr double DEF_SHINY     = 1.0;
constexpr double DEF_AMBIENT   = 0.0;
constexpr double DEF_KEYLIGHT  = 0.9;
constexpr double DEF_FILLLIGHT = 0.45;
constexpr double DEF_BACKLIGHT = 0.9;

// format a floating point number the way SPARTA reads it back (shortest form)
QString num(double v)
{
    return QString::number(v);
}

// emit "v_<name>" when a variable override is set, the number otherwise
QString numOrVar(double v, const QString &var)
{
    if (!var.isEmpty()) return "v_" + var;
    return num(v);
}

// prefix for the custom color names of the per-mode color-map stops, so maps
// of different modes never collide in the dump's global color namespace
const char *const cmapColorPrefix[DumpImageSettings::NUM_CMAP_MODES] = {
    "guimapp", "guimapg", "guimaps", "guimapx", "guimapy", "guimapz"};

// Build the argument text of one `cmap` dump_modify keyword for mode `mode`
// (including any needed leading `color <name> <R> <G> <B>` definitions) from
// the shared color-map table.  Returns an empty string when the map is not
// active. See ColorMapSpec for the meaning of style/range.
QString buildCmapArgs(const DumpImageSettings &s, int mode)
{
    const ColorMapSpec &m = s.cmap[mode];
    if (!m.active) return {};

    const ColorMapDef &def = colorMapDef(m.mapname);

    // work on a copy of the stops; reversal must not mutate the shared table
    QList<ColorMapStop> stops = def.stops;
    if (m.reverse) {
        std::reverse(stops.begin(), stops.end());
        if (def.continuous)
            for (auto &stop : stops)
                stop.pos = 1.0 - stop.pos;
    }
    const int n = stops.size();
    if (n < 1) return {};

    // 'a' (absolute) positions require numeric lo/hi bounds; fall back to 'f'
    bool lonum = false, hinum = false;
    const double loval = m.lo.toDouble(&lonum);
    const double hival = m.hi.toDouble(&hinum);
    const bool absolute = (m.range == 'a') && lonum && hinum && (hival > loval);
    const QChar range   = absolute ? QChar('a') : QChar('f');

    // map a 0..1 fraction to the emitted entry value
    auto mapval = [&](double frac) {
        return absolute ? (loval + frac * (hival - loval)) : frac;
    };

    // Resolve each stop to a color token: a SPARTA named color used verbatim,
    // or a custom color defined once via "color <prefix><k> R G B".  Distinct
    // RGB values share one custom color, numbered in order of first use.
    QString colordefs;
    QStringList colorref;
    QStringList rgbseen;
    const QString prefix = QString::fromLatin1(cmapColorPrefix[mode]);
    for (const auto &stop : stops) {
        if (!stop.name.isEmpty()) {
            colorref.append(stop.name);
            continue;
        }
        const QString rgb = QStringLiteral("%1 %2 %3")
                                .arg(stop.r, 0, 'f', 3)
                                .arg(stop.g, 0, 'f', 3)
                                .arg(stop.b, 0, 'f', 3);
        int idx = static_cast<int>(rgbseen.indexOf(rgb));
        if (idx < 0) {
            rgbseen.append(rgb);
            idx = rgbseen.size() - 1;
            colordefs += QString("color %1%2 %3 ").arg(prefix).arg(idx + 1).arg(rgb);
        }
        colorref.append(prefix + QString::number(idx + 1));
    }

    // fraction of stop i for maps whose table has no positions (sequences)
    auto stopfrac = [&](int i) {
        if (def.continuous) return stops[i].pos;
        return (n > 1) ? static_cast<double>(i) / (n - 1) : 0.0;
    };

    QString args = QString("cmap %1 %2 %3 ").arg(cmapModeName[mode], m.lo, m.hi);

    if (m.style == 's') {
        // sequential: colors repeat in bins of width delta
        args += QString("s%1 %2 %3").arg(range).arg(num(m.delta)).arg(n);
        for (const auto &c : colorref)
            args += " " + c;
    } else if (m.style == 'd') {
        // discrete: one equally wide value bin per stop. the first bin is
        // open at "min" and the last entry is the "min max" catch-all bin
        // that SPARTA requires as the final discrete map entry
        args += QString("d%1 0.0 %2").arg(range).arg(n);
        for (int i = 0; i < n - 1; ++i) {
            const QString lo =
                (i == 0) ? QStringLiteral("min") : num(mapval(static_cast<double>(i) / n));
            args += QString(" %1 %2 %3")
                        .arg(lo, num(mapval(static_cast<double>(i + 1) / n)), colorref[i]);
        }
        args += QString(" min max %1").arg(colorref[n - 1]);
    } else {
        // continuous: first entry at "min", last at "max", interior stops at
        // their (strictly ascending) table positions
        args += QString("c%1 0.0 %2").arg(range).arg(n);
        for (int i = 0; i < n; ++i) {
            QString pos;
            if (i == 0)
                pos = QStringLiteral("min");
            else if (i == n - 1)
                pos = QStringLiteral("max");
            else
                pos = num(mapval(stopfrac(i)));
            args += " " + pos + " " + colorref[i];
        }
    }
    return colordefs + args;
}

} // namespace

QString buildDumpImageCommand(const DumpImageSettings &s)
{
    QString cmd;

    // positional arguments: color and diameter source
    cmd += s.color + " " + s.diameter;

    // keyword order follows the SPARTA documentation / parse loop

    if (!s.particle) cmd += " particle no";
    if (s.numericdiam) cmd += " pdiam " + num(s.pdiamvalue);

    // grid volume rendering and grid cut planes are mutually exclusive in
    // SPARTA; the planes win when both are (erroneously) enabled
    const bool anyplane = s.gridx || s.gridy || s.gridz;
    if (s.grid && !anyplane && !s.gridcolor.isEmpty()) cmd += " grid " + s.gridcolor;
    if (s.gridx && !s.gridxcolor.isEmpty())
        cmd += " gridx " + num(s.gridxcoord) + " " + s.gridxcolor;
    if (s.gridy && !s.gridycolor.isEmpty())
        cmd += " gridy " + num(s.gridycoord) + " " + s.gridycolor;
    if (s.gridz && !s.gridzcolor.isEmpty())
        cmd += " gridz " + num(s.gridzcoord) + " " + s.gridzcolor;

    if (s.surf) cmd += " surf " + s.surfcolor + " " + num(s.surfdiam);

    cmd += QString(" size %1 %2").arg(s.xsize).arg(s.ysize);

    // camera settings; SPARTA forces view 0 0 and up 0 1 0 for 2d systems
    if (s.dimension == 3) {
        if ((s.theta != DEF_THETA) || (s.phi != DEF_PHI) || !s.thetavar.isEmpty() ||
            !s.phivar.isEmpty())
            cmd += " view " + numOrVar(s.theta, s.thetavar) + " " + numOrVar(s.phi, s.phivar);
    }
    if ((s.cx != 0.5) || (s.cy != 0.5) || (s.cz != 0.5) || s.centerdynamic ||
        !s.cxvar.isEmpty() || !s.cyvar.isEmpty() || !s.czvar.isEmpty()) {
        cmd += QString(" center %1 %2 %3 %4")
                   .arg(s.centerdynamic ? "d" : "s", numOrVar(s.cx, s.cxvar),
                        numOrVar(s.cy, s.cyvar), numOrVar(s.cz, s.czvar));
    }
    if (s.dimension == 3) {
        if ((s.upx != 0.0) || (s.upy != 0.0) || (s.upz != 1.0))
            cmd += QString(" up %1 %2 %3").arg(num(s.upx), num(s.upy), num(s.upz));
    }
    if ((s.zoom != 1.0) || !s.zoomvar.isEmpty()) cmd += " zoom " + numOrVar(s.zoom, s.zoomvar);

    // NOTE: no `persp` keyword: SPARTA errors out with "not yet supported"

    if (!s.box)
        cmd += " box no " + num(s.boxdiam);
    else if (s.boxdiam != DEF_BOXDIAM)
        cmd += " box yes " + num(s.boxdiam);
    if (s.subbox) cmd += " subbox yes " + num(s.subboxdiam);
    if (s.gline) cmd += " gline yes " + num(s.glinediam);
    if (s.sline) cmd += " sline yes " + num(s.slinediam);
    if (s.axes) cmd += QString(" axes yes %1 %2").arg(num(s.axeslen), num(s.axesdiam));
    if (s.shiny != DEF_SHINY) cmd += " shiny " + num(s.shiny);
    if (s.ssao) cmd += QString(" ssao yes %1 %2").arg(s.ssaoseed).arg(num(s.ssaoint));
    if (s.fsaa) cmd += " fsaa yes";

    return cmd;
}

QStringList buildDumpModifyCommands(const DumpImageSettings &s, const QString &id, bool movie)
{
    QStringList cmds;
    const QString head = "dump_modify " + id + " ";

    // background: a solid non-default color, or the bottom/top pair of the
    // vertical gradient (backcolor2 is only meaningful together with backcolor)
    if (s.gradient) {
        cmds << head + "backcolor " + s.backcolor;
        cmds << head + "backcolor2 " + s.backcolor2;
    } else if (s.backcolor != QLatin1String("black")) {
        cmds << head + "backcolor " + s.backcolor;
    }

    if (s.box && (s.boxcolor != QLatin1String("yellow"))) cmds << head + "boxcolor " + s.boxcolor;
    if (s.subbox && (s.subboxcolor != QLatin1String("yellow")))
        cmds << head + "subboxcolor " + s.subboxcolor;

    // user-defined custom colors (0..1 float RGB triples)
    for (const auto &cc : s.customcolors)
        cmds << head + "color " + cc.first + " " + cc.second;

    // grid: per-proc colors, outline color, grid group
    const bool anyplane = s.gridx || s.gridy || s.gridz;
    const bool anygrid  = (s.grid && !anyplane) || anyplane;
    if (s.grid && !anyplane && (s.gridcolor == QLatin1String("proc"))) {
        for (const auto &gc : s.gcolors)
            cmds << head + "gcolor " + gc.first + " " + gc.second;
    }
    if (s.gline && (s.glinecolor != QLatin1String("white")))
        cmds << head + "glinecolor " + s.glinecolor;
    if (anygrid && !s.gridgroup.isEmpty() && (s.gridgroup != QLatin1String("all")))
        cmds << head + "gridgroup " + s.gridgroup;

    // particles: per-type/per-proc colors and per-type diameters
    if (s.particle &&
        ((s.color == QLatin1String("type")) || (s.color == QLatin1String("proc")))) {
        for (const auto &pc : s.pcolors)
            cmds << head + "pcolor " + pc.first + " " + pc.second;
    }
    if (s.particle && !s.numericdiam && (s.diameter == QLatin1String("type"))) {
        for (const auto &pd : s.pdiams)
            cmds << head + "pdiam " + pd.first + " " + num(pd.second);
    }

    // region clip for particles (from dump particle)
    if (!s.region.isEmpty() && (s.region != QLatin1String("none")))
        cmds << head + "region " + s.region;

    // surfs: single color, per-proc colors, outline color, surf group
    if (s.surf) {
        if ((s.surfcolor == QLatin1String("one")) && (s.surfcolorone != QLatin1String("gray")))
            cmds << head + "scolor * " + s.surfcolorone;
        if (s.surfcolor == QLatin1String("proc")) {
            for (const auto &sc : s.scolors)
                cmds << head + "scolor " + sc.first + " " + sc.second;
        }
    }
    if (s.sline && (s.slinecolor != QLatin1String("white")))
        cmds << head + "slinecolor " + s.slinecolor;
    if (s.surf && !s.surfgroup.isEmpty() && (s.surfgroup != QLatin1String("all")))
        cmds << head + "surfgroup " + s.surfgroup;

    // light intensities (only when changed from the SPARTA defaults)
    if ((s.amblight != DEF_AMBIENT) || (s.keylight != DEF_KEYLIGHT) ||
        (s.filllight != DEF_FILLLIGHT) || (s.backlight != DEF_BACKLIGHT)) {
        cmds << head +
                QString("lights %1 %2 %3 %4")
                    .arg(num(s.amblight), num(s.keylight), num(s.filllight), num(s.backlight));
    }

    // the six independent color maps, in fixed mode order
    for (int mode = 0; mode < DumpImageSettings::NUM_CMAP_MODES; ++mode) {
        const QString args = buildCmapArgs(s, mode);
        if (!args.isEmpty()) cmds << head + args;
    }

    // movie-only settings
    if (movie) {
        if (s.framerate > 0.0) cmds << head + "framerate " + num(s.framerate);
        if (s.bitrate > 0) cmds << head + "bitrate " + QString::number(s.bitrate);
    }

    return cmds;
}

QString buildDumpSnippet(const DumpImageSettings &s, bool movie, const QString &file, int every)
{
    const QString id    = movie ? QStringLiteral("movie") : QStringLiteral("viz");
    const QString style = movie ? QStringLiteral("movie") : QStringLiteral("image");

    QString out = QString("dump %1 %2 %3 %4 %5 ").arg(id, style, s.mixture).arg(every).arg(file);
    out += buildDumpImageCommand(s);
    out += '\n';
    if (!movie) out += "dump_modify " + id + " pad 9\n";
    const QStringList modify = buildDumpModifyCommands(s, id, movie);
    for (const auto &cmd : modify)
        out += cmd + '\n';
    return out;
}

// Local Variables:
// c-basic-offset: 4
// End:
