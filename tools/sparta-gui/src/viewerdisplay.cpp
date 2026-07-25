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

#include "viewerdisplay.h"

#include "helpers.h"

#include <QEvent>
#include <QLabel>
#include <QPalette>
#include <QPixmap>
#include <QScrollArea>
#include <QVBoxLayout>

////////////////////////////////////////////////////////////////////////////////
// ViewerDisplay                                                              //
////////////////////////////////////////////////////////////////////////////////

ViewerDisplay::ViewerDisplay(FitMode mode, QWidget *parent) :
    QWidget(parent), fit(mode), imageLabel(new QLabel), area(new QScrollArea)
{
    imageLabel->setBackgroundRole(QPalette::Base);
    imageLabel->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
    imageLabel->setScaledContents(false);

    area->setBackgroundRole(QPalette::Dark);
    area->setWidget(imageLabel);
    area->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

    // The fit is computed against the viewport, so it has to be redone whenever
    // that changes size.
    area->viewport()->installEventFilter(this);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(area);
}

ViewerDisplay::~ViewerDisplay() = default;

void ViewerDisplay::setImage(const QImage &image)
{
    raw = image;
    repaintPixmap();
}

void ViewerDisplay::setTransform(const DisplayTransform &t)
{
    xform = t;
    repaintPixmap();
}

void ViewerDisplay::refresh()
{
    repaintPixmap();
}

bool ViewerDisplay::eventFilter(QObject *watched, QEvent *event)
{
    if (watched == area->viewport() && event->type() == QEvent::Resize) repaintPixmap();
    return QWidget::eventFilter(watched, event);
}

void ViewerDisplay::repaintPixmap()
{
    if (raw.isNull()) return;

    // Resizing the label below can itself provoke a viewport resize, which
    // calls back in here; without this the two would chase each other.
    if (painting) return;
    painting = true;

    shown = applyDisplayTransform(raw, xform);
    QPixmap pix = QPixmap::fromImage(shown);

    if (fit == FitViewport) {
        // maximumViewportSize(), not viewport()->size(): the viewport's actual
        // size depends on whether a scroll bar is showing, which depends on the
        // size of the pixmap being computed here. Feeding that back in makes
        // the two chase each other -- fit to the full width, a vertical bar
        // appears, the narrower viewport wants a smaller fit, the bar goes away
        // -- and the re-entry guard above stops the oscillation wherever it
        // happens to be, which left the picture scaled to the panel's width and
        // cut off at the bottom. maximumViewportSize() is the room available
        // with no bars at all, so the result does not depend on the state it
        // produces.
        const QSize avail = area->maximumViewportSize();
        if (avail.isValid() && !avail.isEmpty() &&
            (pix.width() > avail.width() || pix.height() > avail.height())) {
            pix = pix.scaled(avail, Qt::KeepAspectRatio, Qt::SmoothTransformation);
            imageLabel->setToolTip(
                "Scaled to fit the panel; enlarge the panel to see it at full size");
        } else {
            imageLabel->setToolTip(QString());
        }
    }

    imageLabel->setPixmap(pix);
    // Size the label to what is actually painted, and never as a *minimum*: a
    // render can be several thousand pixels across, and a minimum that large
    // propagates out through the dock layout and forces the window wider than
    // the screen.
    imageLabel->resize(pix.size());
    painting = false;
}

void ViewerDisplay::fitHostWindow(QWidget *host, const QSize &content, const QSize &budget)
{
    lastFitSize = fitViewerWindow(host, area, content, budget, lastFitSize);
}

// Local Variables:
// c-basic-offset: 4
// End:
