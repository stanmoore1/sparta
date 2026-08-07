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

#ifndef EMPTYSTATE_H
#define EMPTYSTATE_H

// A panel with nothing in it yet used to be a blank rectangle, which reads as
// broken -- especially to someone who just switched workspaces to look at it.
// This is the "nothing yet, and here is how to change that" card that stands
// in until real content arrives: a title stating what is absent and a hint
// naming the action (with its shortcut) that fills the panel.

#include <QWidget>

class QLabel;

class EmptyState : public QWidget {
    Q_OBJECT

public:
    EmptyState(const QString &title, const QString &hint, QWidget *parent = nullptr);

    /** @brief Marks placeholders so PanelManager never archives one as content. */
    static bool isPlaceholder(const QWidget *w);

private:
    QLabel *titleLabel = nullptr;
    QLabel *hintLabel  = nullptr;
};

#endif // EMPTYSTATE_H

// Local Variables:
// c-basic-offset: 4
// End:
