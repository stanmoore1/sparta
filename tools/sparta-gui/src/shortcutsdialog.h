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

#ifndef SHORTCUTSDIALOG_H
#define SHORTCUTSDIALOG_H

// The in-application help: a "Getting Started" page of task-shaped sections
// (what used to be a single unscrollable wall of text in a message box) and a
// "Keyboard Shortcuts" page generated from the menus themselves.  Generated,
// because a hand-written list of shortcuts is a list of what the shortcuts
// used to be: this one walks the live menu bar through scanMenuBar() every
// time it opens and therefore cannot go stale.

#include <QDialog>

class QMenuBar;
class QTabWidget;

class ShortcutsDialog : public QDialog {
    Q_OBJECT

public:
    enum Page { GettingStarted, Shortcuts };

    explicit ShortcutsDialog(QMenuBar *bar, QWidget *parent = nullptr);

    /** @brief Regenerate the shortcut table from the menus, open on @p page. */
    void popup(Page page);

private:
    /** @brief The generated shortcut table as HTML, grouped by menu. */
    [[nodiscard]] QString shortcutsHtml() const;

    QMenuBar *menubar;
    QTabWidget *tabs = nullptr;
};

#endif // SHORTCUTSDIALOG_H

// Local Variables:
// c-basic-offset: 4
// End:
