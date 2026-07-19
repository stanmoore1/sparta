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

#ifndef THEME_H
#define THEME_H

#include <QString>

/**
 * @brief Curated light/dark appearance applied on top of the Fusion style.
 *
 * SPARTA-GUI forces the Fusion QStyle so it looks identical and deliberate on
 * Windows, macOS, and Linux instead of inheriting each platform's native Qt
 * theming. On top of that this namespace installs a curated color palette
 * (light or dark) and a global stylesheet (:/style.qss) that only adjusts
 * spacing/padding/density -- never fonts or per-widget colors. Everything is
 * applied once, in main(), before the main window is constructed.
 */
namespace Theme {

/** @brief User-selectable appearance mode (persisted as Keys::THEME) */
enum class Mode { System, Light, Dark };

/** @brief Parse a persisted mode string ("system"/"light"/"dark"); default System */
Mode modeFromString(const QString &name);

/** @brief Serialize a mode to its persisted string */
QString modeToString(Mode mode);

/**
 * @brief Best-effort detection of the OS-level dark-mode preference.
 *
 * Uses QStyleHints::colorScheme() on Qt >= 6.5; on older Qt it inspects the
 * current application palette, so it must be called BEFORE the Fusion palette
 * is installed (i.e. before @ref apply) to read the platform default.
 */
bool osPrefersDark();

/**
 * @brief Install the curated palette + global stylesheet for @p mode.
 * @param mode   The desired appearance (System resolves via @ref osPrefersDark)
 * @param osDark The OS dark-mode preference captured before the style changed
 *               (used only to resolve Mode::System)
 *
 * Does not change the QStyle itself -- the caller sets Fusion (or a
 * command-line override) first; this only sets the palette and stylesheet.
 */
void apply(Mode mode, bool osDark);

} // namespace Theme

#endif // THEME_H

// Local Variables:
// c-basic-offset: 4
// End:
