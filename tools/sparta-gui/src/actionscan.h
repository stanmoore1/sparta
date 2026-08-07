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

#ifndef ACTIONSCAN_H
#define ACTIONSCAN_H

// The menu bar is the application's one complete list of what it can do, so
// features that need such a list -- the command palette, the generated
// keyboard-shortcut sheet, the status-tip table -- read it from the menus
// rather than keeping a registry of their own.  A registry would have to be
// fed by every place that creates an action and would drift the first time
// one forgot; the walk cannot, and it picks up actions created outside
// addMenuAction() (the dock panels' toggle-view actions, the dynamically
// built Open Example submenu) for free.

#include <QList>
#include <QPointer>
#include <QString>

class QAction;
class QMenu;
class QMenuBar;

/** @brief One triggerable menu entry, as found by scanMenuBar(). */
struct ActionInfo {
    QPointer<QAction> action; ///< the live action (guarded: menus rebuild)
    QString path;             ///< menu breadcrumb, e.g. "File > Open Example"
    QString text;             ///< action text with '&' mnemonics stripped
};

/**
 * @brief Collect every triggerable action reachable from a menu bar
 *
 * Walks all menus recursively.  Separators and submenu headers are skipped
 * (a submenu's own QAction opens the submenu, it does not do anything);
 * disabled actions are included -- what is disabled *now* is still part of
 * what the application can do, and the palette greys it rather than hiding
 * it.  Actions with empty text (unassigned recent-file slots) are skipped.
 */
QList<ActionInfo> scanMenuBar(const QMenuBar *bar);

/** @brief The same walk for a single menu (used by tests). */
void scanMenu(const QMenu *menu, const QString &prefix, QList<ActionInfo> &out);

/** @brief Action text with '&' mnemonics removed ("&Save && Run" -> "Save & Run"). */
QString strippedActionText(const QString &text);

#endif // ACTIONSCAN_H

// Local Variables:
// c-basic-offset: 4
// End:
