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

#ifndef IMAGEVIEWER_INTERNAL_H
#define IMAGEVIEWER_INTERNAL_H

// Implementation-detail symbols shared between imageviewer.cpp and
// imageviewersettings.cpp (the settings dialog builders). Not part of any
// public API.

#include <QColor>
#include <QIcon>
#include <QJsonArray>
#include <QJsonObject>
#include <QList>
#include <QPair>
#include <QPixmap>
#include <QString>
#include <QStringList>

class QComboBox;
class QWidget;

// ---- shared UI constants -------------------------------------------------
inline const QString blank(" ");
inline constexpr double SHINY_ON      = 1.0; // the SPARTA default
inline constexpr double SHINY_OFF     = 0.3;
inline constexpr double SHINY_CUT     = 0.65;
inline constexpr double ZOOM_MIN      = 0.1;
inline constexpr double ZOOM_MAX      = 10.0;
inline constexpr int TITLE_MARGIN     = 10;
inline constexpr int CONTENT_MARGIN   = 5;
inline constexpr int LAYOUT_SPACING   = 6;
/// bounds of the rendered image in pixels, shared by the size fields and by
/// "Fit Render to Panel" so the two can never disagree about what is allowed
inline constexpr int MIN_RENDER_SIZE  = 100;
inline constexpr int MAX_RENDER_SIZE  = 10000;
inline constexpr int MINIMUM_WIDTH    = 400;
inline constexpr int MINIMUM_HEIGHT   = 300;
inline constexpr int MAX_VALUE_COLS   = 99; // upper bound of the c_ID[N] column spinbox

// The default per-species color assignment of SPARTA's dump image: species i
// gets deftypecolors[(i - 1) % 6] (see sparta/src/dump_image.cpp). RGB values
// match SPARTA's image.cpp color database so the GUI preview is faithful and
// unchanged assignments can be omitted from the generated command.
inline const QList<QPair<QString, QColor>> defspeciescolors = {
    {{"red"}, {255, 0, 0}},    {{"green"}, {0, 255, 0}}, {{"blue"}, {0, 0, 255}},
    {{"yellow"}, {255, 255, 0}}, {{"aqua"}, {0, 255, 255}}, {{"purple"}, {128, 0, 128}}};

// particle attributes of SPARTA's dump particle usable for coloring or sizing
// particles in dump image (c_/f_/v_/p_/i_/d_ references may be typed in freely)
inline const QStringList particleAttributes = {"id", "type", "proc", "x",  "y",  "z",
                                               "xs", "ys",   "zs",   "vx", "vy", "vz",
                                               "ke", "erot", "evib"};

// ---- shared free helpers (defined in imageviewer.cpp) --------------------
//
// The colour swatches and selectComboItem() used to live here too; they moved
// to qaddon.h so that wanting a swatch does not mean linking the image viewer.
QJsonObject loadJsonColors(QWidget *parent);
void saveJsonColors(QWidget *parent, const QJsonArray &colors, const QJsonObject &lights);

#endif

// Local Variables:
// c-basic-offset: 4
// End:
