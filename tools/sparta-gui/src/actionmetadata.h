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

#ifndef ACTIONMETADATA_H
#define ACTIONMETADATA_H

// One-line descriptions for every menu action, shown in the status bar while
// the entry is highlighted.  Kept as a table applied over the finished menu
// bar rather than as an extra argument to addMenuAction(): the ~70 call
// sites keep their upstream shape (so upstream patches still apply), and an
// action the table does not know simply gets no tip rather than breaking
// anything.  The gap that creates is closed by a test, not by trust:
// test_mainwindow asserts every menu action carries a non-empty status tip.

class QMenuBar;

/**
 * @brief Set the status tip (and tool tip) of every known action in the menus
 *
 * Looks each action up by its mnemonic-stripped text; dynamically generated
 * entries (recent files, the Open Example decks) are matched by their menu
 * path instead, since their texts are data.  Call once after the menus are
 * built, and again after a menu is rebuilt dynamically.
 */
void applyActionMetadata(const QMenuBar *bar);

#endif // ACTIONMETADATA_H

// Local Variables:
// c-basic-offset: 4
// End:
