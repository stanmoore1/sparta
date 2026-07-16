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

#ifndef DUMPIMAGE_H
#define DUMPIMAGE_H

#include <QList>
#include <QPair>
#include <QString>
#include <QStringList>

/**
 * @brief Per-mode color-map customization for a SPARTA dump image
 *
 * SPARTA's dump image maintains six independent color maps (particle, grid,
 * surf, gridx, gridy, gridz), each reset with a
 * `dump_modify ID cmap <mode> <lo> <hi> <style><range> <delta> <N> <entries...>`
 * command.  The GUI stores one of these per mode.  When @ref active is false
 * the SPARTA default map ("min max cf 0.0 2 min blue max red") is left
 * untouched and no cmap command is emitted for the mode.
 *
 * The color stops themselves come from the shared color-map table in
 * colormaps.cpp selected via @ref mapname, optionally mirrored via
 * @ref reverse.  @ref style selects how the stops are laid out:
 * - 'c' (continuous): first stop at `min`, last at `max`, interior stops at
 *   their table positions
 * - 'd' (discrete): the stops become N equally wide value bins
 * - 's' (sequential): the stops repeat in value bins of width @ref delta
 * @ref range 'f' interprets positions/bins as fractions of [lo,hi]; 'a' maps
 * them to absolute values, which requires both @ref lo and @ref hi to be
 * numeric (the builder falls back to 'f' otherwise).
 */
struct ColorMapSpec {
    bool active     = false;   ///< emit a cmap command for this mode
    QString mapname = "BWR";   ///< color-map name from colormaps.cpp
    bool reverse    = false;   ///< mirror the selected map
    QString lo      = "min";   ///< lower map bound: "min" or a number
    QString hi      = "max";   ///< upper map bound: "max" or a number
    QChar style     = 'c';     ///< 'c' continuous, 'd' discrete, 's' sequential
    QChar range     = 'f';     ///< 'a' absolute or 'f' fractional
    double delta    = 1.0;     ///< value bin width (sequential style only)
};

/**
 * @brief All state needed to assemble a SPARTA `dump ID image ...` command
 *
 * Every member corresponds to an option of SPARTA's dump image command (see
 * src/dump_image.cpp) or one of its dump_modify keywords.  Members are
 * initialized to the SPARTA built-in defaults, so a default-constructed
 * struct describes SPARTA's own default rendering; the builders emit only
 * settings that differ from those defaults (positional arguments and the
 * image size are always emitted).
 *
 * The struct is plain data: the builders below are pure functions of it and
 * never touch a GUI or a live SPARTA instance, which makes them directly
 * unit-testable (see test/test_dumpimage.cpp).
 */
struct DumpImageSettings {
    // ---- positional arguments --------------------------------------------
    QString mixture  = "all";  ///< mixture ID to render (positional arg 3)
    QString color    = "type"; ///< particle color: "type", "proc", or an attribute
    QString diameter = "type"; ///< particle diameter: "type" or a numeric attribute

    // ---- particles --------------------------------------------------------
    bool particle     = true;  ///< `particle yes/no` (default yes)
    bool numericdiam  = false; ///< use the numeric `pdiam <float>` keyword
    double pdiamvalue = 1.0;   ///< value for the `pdiam` keyword

    // ---- grid volume rendering (mutually exclusive with the planes) -------
    bool grid = false;      ///< `grid <color>` keyword present
    QString gridcolor;      ///< "proc" or a c_ID[N] / f_ID[N] / v_name reference

    // ---- grid cut planes ---------------------------------------------------
    bool gridx        = false; ///< `gridx <coord> <color>` keyword present
    double gridxcoord = 0.0;   ///< x coordinate of the yz cut plane
    QString gridxcolor;        ///< color source of the yz cut plane
    bool gridy        = false; ///< `gridy <coord> <color>` keyword present
    double gridycoord = 0.0;   ///< y coordinate of the xz cut plane
    QString gridycolor;        ///< color source of the xz cut plane
    bool gridz        = false; ///< `gridz <coord> <color>` keyword present
    double gridzcoord = 0.0;   ///< z coordinate of the xy cut plane
    QString gridzcolor;        ///< color source of the xy cut plane

    // ---- surface elements --------------------------------------------------
    bool surf         = false; ///< `surf <color> <diam>` keyword present
    QString surfcolor = "one"; ///< "one", "proc", or a c_/f_/v_ reference
    double surfdiam   = 0.01;  ///< surf element diameter (2d line width fraction)

    // ---- image size ---------------------------------------------------------
    int xsize = 600; ///< rendered image width in pixels
    int ysize = 600; ///< rendered image height in pixels

    // ---- camera --------------------------------------------------------------
    double theta = 60.0; ///< view angle from the +z axis in degrees (0..180)
    double phi   = 30.0; ///< azimuthal view angle in degrees
    QString thetavar;    ///< equal-style variable name overriding theta (no "v_")
    QString phivar;      ///< equal-style variable name overriding phi
    bool centerdynamic = false; ///< `center d ...` instead of `center s ...`
    double cx = 0.5, cy = 0.5, cz = 0.5; ///< view center as box fractions
    QString cxvar, cyvar, czvar;         ///< variable names overriding cx/cy/cz
    double upx = 0.0, upy = 0.0, upz = 1.0; ///< camera up vector
    double zoom = 1.0;   ///< zoom factor (> 0)
    QString zoomvar;     ///< equal-style variable name overriding zoom
    // NOTE: SPARTA errors out on the `persp` keyword ("not yet supported"),
    // so there is deliberately no perspective member.

    // ---- box / subbox / outlines / axes ---------------------------------------
    bool box          = true;  ///< draw the simulation box (default yes)
    double boxdiam    = 0.02;  ///< box edge diameter (fraction)
    bool subbox       = false; ///< draw the per-processor RCB sub-boxes
    double subboxdiam = 0.02;  ///< sub-box edge diameter (fraction)
    bool gline        = false; ///< draw grid cell outlines
    double glinediam  = 0.005; ///< grid outline diameter (fraction)
    bool sline        = false; ///< draw surf element outlines
    double slinediam  = 0.005; ///< surf outline diameter (fraction)
    bool axes         = false; ///< draw coordinate axes
    double axeslen    = 0.5;   ///< axes length (fraction of box)
    double axesdiam   = 0.02;  ///< axes diameter (fraction)

    // ---- quality ----------------------------------------------------------------
    double shiny   = 1.0;   ///< shininess 0..1 (SPARTA default 1.0)
    bool ssao      = false; ///< screen-space ambient occlusion
    int ssaoseed   = 453983; ///< RNG seed for SSAO (Cfg::SSAO_SEED)
    double ssaoint = 0.6;   ///< SSAO strength 0..1
    bool fsaa      = false; ///< full-scene anti-aliasing

    // ---- environment ---------------------------------------------------------------
    int dimension = 3; ///< 2 or 3; in 2d view/up are omitted (SPARTA forces them)

    // ---- dump_modify: colors ----------------------------------------------------
    QString backcolor  = "black";  ///< background color
    bool gradient      = false;    ///< vertical background gradient on/off
    QString backcolor2 = "white";  ///< upper background color of the gradient
    QString boxcolor   = "yellow"; ///< box color
    QString subboxcolor = "yellow"; ///< sub-box color
    QString glinecolor  = "white";  ///< grid outline color
    QString slinecolor  = "white";  ///< surf outline color
    QString surfcolorone = "gray";  ///< surf color for the "one" color mode

    /// custom color definitions: {name, "R G B"} with 0..1 floats
    QList<QPair<QString, QString>> customcolors;
    /// `gcolor <proc-range> <color[/color2/...]>` rows (grid colored by proc)
    QList<QPair<QString, QString>> gcolors;
    /// `pcolor <type-or-proc-range> <color[/...]>` rows
    QList<QPair<QString, QString>> pcolors;
    /// `pdiam <type-range> <diam>` rows (particle diameter mode "type")
    QList<QPair<QString, double>> pdiams;
    /// `scolor <proc-range> <color[/...]>` rows (surf colored by proc)
    QList<QPair<QString, QString>> scolors;

    // ---- dump_modify: groups / region clip -----------------------------------------
    QString gridgroup = "all";  ///< grid group (dump_modify gridgroup)
    QString surfgroup = "all";  ///< surf group (dump_modify surfgroup)
    QString region    = "none"; ///< region ID clipping particles (dump_modify region)

    // ---- dump_modify: lights ---------------------------------------------------------
    double amblight  = 0.0;  ///< ambient light intensity 0..1
    double keylight  = 0.9;  ///< key light intensity 0..1
    double filllight = 0.45; ///< fill light intensity 0..1
    double backlight = 0.9;  ///< back light intensity 0..1

    // ---- dump_modify: color maps (particle, grid, surf, gridx, gridy, gridz) ----------
    enum CmapMode { PARTICLE = 0, GRID, SURF, GRIDX, GRIDY, GRIDZ, NUM_CMAP_MODES };
    ColorMapSpec cmap[NUM_CMAP_MODES]; ///< one independent map per mode

    // ---- dump_modify: movie-only settings ----------------------------------------------
    double framerate = 0.0; ///< frames/s (0.1 - 24.0); emitted only when > 0
    int bitrate      = 0;   ///< bitrate in kbps; emitted only when > 0
};

/// the dump_modify cmap mode keywords, indexed by DumpImageSettings::CmapMode
inline const char *const cmapModeName[DumpImageSettings::NUM_CMAP_MODES] = {
    "particle", "grid", "surf", "gridx", "gridy", "gridz"};

/**
 * @brief Assemble the `dump ... image ...` arguments following the file name
 * @param s the populated settings struct
 * @return argument string "<color> <diameter> <keywords...>" (no leading space)
 *
 * Pure function of @p s.  Keywords are emitted in a fixed order (the order of
 * the SPARTA documentation / parse loop: particle, pdiam, grid, gridx, gridy,
 * gridz, surf, size, view, center, up, zoom, box, subbox, gline, sline, axes,
 * shiny, ssao, fsaa) so generated commands are deterministic and testable.
 * Options matching the SPARTA built-in defaults are omitted, except for the
 * positional color/diameter arguments and the image size.
 */
QString buildDumpImageCommand(const DumpImageSettings &s);

/**
 * @brief Assemble the `dump_modify` commands for all non-default modify settings
 * @param s  the populated settings struct
 * @param id the dump ID the commands refer to
 * @return ordered list of complete "dump_modify <id> ..." command strings
 *
 * Pure function of @p s and @p id.  Commands are emitted in a fixed order:
 * backcolor, backcolor2, boxcolor, subboxcolor, color, gcolor, glinecolor,
 * gridgroup, pcolor, pdiam, region, scolor, slinecolor, surfgroup, lights,
 * cmap (particle, grid, surf, gridx, gridy, gridz), framerate, bitrate.
 * Custom colors required by a color map are defined (via the `color` keyword)
 * on the same dump_modify line, directly before their cmap keyword.
 * Movie-only settings (framerate/bitrate) are emitted only when @p movie.
 */
QStringList buildDumpModifyCommands(const DumpImageSettings &s, const QString &id,
                                    bool movie = false);

/**
 * @brief Compose a complete, reusable dump image/movie snippet
 * @param s     the populated settings struct
 * @param movie true for a `dump ... movie` example, false for `dump ... image`
 * @param file  output file name (e.g. "myimage-*.png" or "mymovie.mp4")
 * @param every dump interval in timesteps
 * @return multi-line text with the dump command and its dump_modify commands
 *
 * Used by the "copy dump command" action of the image viewer so users can
 * paste the equivalent of the currently displayed rendering into their input.
 */
QString buildDumpSnippet(const DumpImageSettings &s, bool movie, const QString &file, int every);

#endif

// Local Variables:
// c-basic-offset: 4
// End:
