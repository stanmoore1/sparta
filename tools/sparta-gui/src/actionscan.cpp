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

#include "actionscan.h"

#include <QAction>
#include <QMenu>
#include <QMenuBar>

QString strippedActionText(const QString &text)
{
    // "&&" is a literal ampersand; a lone '&' marks the mnemonic
    QString out;
    out.reserve(text.size());
    for (int i = 0; i < text.size(); ++i) {
        if (text[i] == QLatin1Char('&')) {
            if (i + 1 < text.size() && text[i + 1] == QLatin1Char('&')) {
                out += QLatin1Char('&');
                ++i;
            }
            continue;
        }
        out += text[i];
    }
    return out;
}

void scanMenu(const QMenu *menu, const QString &prefix, QList<ActionInfo> &out)
{
    if (!menu) return;
    for (QAction *a : menu->actions()) {
        if (a->isSeparator()) continue;
        if (a->menu()) {
            scanMenu(a->menu(), prefix + QStringLiteral(" > ") + strippedActionText(a->text()),
                     out);
            continue;
        }
        const QString text = strippedActionText(a->text());
        if (text.isEmpty()) continue; // unassigned recent-file slots and the like
        out.append({a, prefix, text});
    }
}

QList<ActionInfo> scanMenuBar(const QMenuBar *bar)
{
    QList<ActionInfo> out;
    if (!bar) return out;
    for (QAction *top : bar->actions())
        if (top->menu()) scanMenu(top->menu(), strippedActionText(top->text()), out);
    return out;
}

// Local Variables:
// c-basic-offset: 4
// End:
