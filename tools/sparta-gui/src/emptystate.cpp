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

#include "emptystate.h"

#include <QLabel>
#include <QVBoxLayout>

EmptyState::EmptyState(const QString &title, const QString &hint, QWidget *parent) :
    QWidget(parent)
{
    setObjectName("emptystate");
    setProperty("spartaEmptyState", true);

    titleLabel = new QLabel(title, this);
    titleLabel->setObjectName("emptystateTitle");
    titleLabel->setAlignment(Qt::AlignCenter);
    QFont f = titleLabel->font();
    f.setBold(true);
    f.setPointSizeF(f.pointSizeF() * 1.15);
    titleLabel->setFont(f);

    hintLabel = new QLabel(hint, this);
    hintLabel->setObjectName("emptystateHint");
    hintLabel->setAlignment(Qt::AlignCenter);
    hintLabel->setWordWrap(true);
    // quieter than body text: this is scaffolding, not content
    hintLabel->setForegroundRole(QPalette::PlaceholderText);
    hintLabel->setEnabled(false);

    auto *layout = new QVBoxLayout(this);
    layout->addStretch(2);
    layout->addWidget(titleLabel);
    layout->addSpacing(6);
    layout->addWidget(hintLabel);
    layout->addStretch(3);
}

bool EmptyState::isPlaceholder(const QWidget *w)
{
    return w && w->property("spartaEmptyState").toBool();
}

// Local Variables:
// c-basic-offset: 4
// End:
