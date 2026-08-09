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

#include "viewersidebar.h"

#include "qaddon.h"

#include <QAbstractButton>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSizePolicy>
#include <QToolButton>
#include <QVBoxLayout>

/// spacing between the rows and their controls; matches the rest of the viewer
static constexpr int SIDEBAR_SPACING = 6;
/// width of the strip that a collapsed sidebar leaves behind
static constexpr int HANDLE_WIDTH = 16;

ViewerSidebar::ViewerSidebar(QWidget *parent) : QWidget(parent)
{
    setObjectName("viewersidebar");

    body = new QWidget;
    body->setObjectName("sidebarbody");
    bodyBox = new QVBoxLayout(body);
    bodyBox->setContentsMargins(0, 0, 0, 0);
    bodyBox->setSpacing(SIDEBAR_SPACING);

    rows = new QGridLayout;
    rows->setContentsMargins(0, 0, 0, 0);
    rows->setHorizontalSpacing(SIDEBAR_SPACING);
    rows->setVerticalSpacing(SIDEBAR_SPACING / 2);
    // Spare width goes to an empty fourth column rather than to the names, so
    // that a row's secondary toggles stay next to the name they belong to
    // instead of being flung to the far edge of a column stretched by whatever
    // the widest thing in the sidebar happens to be.
    rows->setColumnStretch(3, 1);

    // The collapse control sits with the rows rather than in the panel's menu
    // bar so that it is where the thing it hides is -- a user who finds the
    // sidebar in the way looks at the sidebar, not at a menu.
    auto *hide = new QToolButton;
    hide->setObjectName("sidebarhide");
    hide->setArrowType(Qt::RightArrow);
    hide->setAutoRaise(true);
    hide->setToolTip("Hide the settings sidebar");
    hide->setAccessibleName(hide->toolTip());
    connect(hide, &QToolButton::clicked, this, [this]() { setCollapsed(true); });

    auto *titleRow = new QHBoxLayout;
    titleRow->setContentsMargins(0, 0, 0, 0);
    titleRow->setSpacing(SIDEBAR_SPACING);
    auto *title = new QLabel("Settings");
    title->setObjectName("sidebartitle");
    titleRow->addWidget(title);
    titleRow->addStretch(1);
    titleRow->addWidget(hide);
    bodyBox->addLayout(titleRow);
    // Everything added later is inserted in front of this stretch, so the rows
    // stay packed at the top of a column that is taller than they are instead
    // of being spread out over its whole height.
    bodyBox->addStretch(1);

    handle = new QToolButton;
    handle->setObjectName("sidebarhandle");
    handle->setArrowType(Qt::LeftArrow);
    handle->setAutoRaise(true);
    handle->setToolTip("Show the settings sidebar");
    handle->setAccessibleName(handle->toolTip());
    handle->setFixedWidth(HANDLE_WIDTH);
    handle->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    handle->hide();
    connect(handle, &QToolButton::clicked, this, [this]() { setCollapsed(false); });

    auto *outer = new QHBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->setSpacing(0);
    outer->addWidget(body);
    outer->addWidget(handle);
}

/// index of the trailing stretch, which everything else is inserted before
int ViewerSidebar::tail() const
{
    return bodyBox->count() - 1;
}

void ViewerSidebar::addHeader(const QString &label, QWidget *widget)
{
    bodyBox->insertWidget(tail(), new QLabel(label));
    bodyBox->insertWidget(tail(), widget);
}

void ViewerSidebar::addRow(QAbstractButton *toggle, QAbstractButton *name,
                           const QList<QAbstractButton *> &extra)
{
    if (rows->parent() == nullptr) {
        bodyBox->insertWidget(tail(), new QHline);
        bodyBox->insertLayout(tail(), rows);
    }

    const int line = rows->rowCount();

    // A subject with nothing to switch (grid planes, camera, color maps) simply
    // leaves column 0 empty; the grid still sizes that column from the rows
    // that do have a toggle, so every name starts at the same x.
    if (toggle) rows->addWidget(toggle, line, 0);

    if (name) {
        name->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        rows->addWidget(name, line, 1);
    }

    if (!extra.isEmpty()) {
        auto *box = new QHBoxLayout;
        box->setContentsMargins(0, 0, 0, 0);
        box->setSpacing(SIDEBAR_SPACING / 2);
        for (auto *button : extra)
            box->addWidget(button);
        rows->addLayout(box, line, 2);
    }
}

void ViewerSidebar::addFooter(QWidget *widget)
{
    bodyBox->insertWidget(tail(), new QHline);
    bodyBox->insertWidget(tail(), widget);
}

void ViewerSidebar::setCollapsed(bool on)
{
    if (on == collapsed) return;
    collapsed = on;
    body->setVisible(!on);
    handle->setVisible(on);
    // The width the sidebar asks for is the body's while expanded and the
    // handle's while collapsed; without this the old, wide minimum sticks and
    // collapsing frees no space at all.
    updateGeometry();
    emit collapsedChanged(on);
}

// Local Variables:
// c-basic-offset: 4
// End:
