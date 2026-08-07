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

#ifndef COMMANDPALETTE_H
#define COMMANDPALETTE_H

// Type a few letters, see every matching action with its shortcut, hit Enter.
// One surface serves both audiences the GUI is for: someone new finds
// features without knowing which menu hides them, and someone experienced
// stops needing the menus at all -- while every match quietly teaches its
// keyboard shortcut by showing it.  The list is the menu bar itself, read
// through scanMenuBar() at each opening, so whatever the menus can do the
// palette can find, dynamically built submenus included.

#include <QDialog>

class QKeyEvent;
class QLineEdit;
class QMenuBar;
class QTreeWidget;

class CommandPalette : public QDialog {
    Q_OBJECT

public:
    /** @brief Build the palette over @p bar; scan happens on every show. */
    explicit CommandPalette(QMenuBar *bar, QWidget *parent = nullptr);

    /** @brief Rescan the menus, clear the filter, show and focus the input. */
    void popup();

protected:
    void keyPressEvent(QKeyEvent *event) override;
    bool eventFilter(QObject *watched, QEvent *event) override;
    void showEvent(QShowEvent *event) override;

private slots:
    void refilter(const QString &needle);
    void triggerCurrent();

private:
    void rebuild();

    QMenuBar *menubar;
    QLineEdit *input   = nullptr;
    QTreeWidget *list  = nullptr;
};

#endif // COMMANDPALETTE_H

// Local Variables:
// c-basic-offset: 4
// End:
