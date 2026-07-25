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

#include "viewerwindow.h"

#include "constants.h"
#include "helpers.h"
#include "imageviewer.h"
#include "slideshow.h"
#include "viewersource.h"

#include <QIcon>

ViewerWindow::ViewerWindow(ViewerSource *source, const QString &titlePrefix, QWidget *parent) :
    QMainWindow(parent), view(source), prefix(titlePrefix)
{
    setCentralWidget(view);
    setWindowIcon(QIcon(Cfg::MAIN_ICON));
    setMinimumSize(Cfg::MINIMUM_WIDTH, Cfg::MINIMUM_HEIGHT);

    // The source no longer sets its own window title -- it cannot know whether
    // it is in a window or a dock tab -- so the host applies it.
    connect(view, &ViewerSource::titleChanged, this,
            [this](const QString &name) { setWindowTitle(prefix + name); });

    // Ctrl+W inside the source: a widget that is not a window cannot close
    // itself, so closing is the host's job.
    connect(view, &ViewerSource::closeRequested, this, &QWidget::close);

    applyWindowFlags(this);
}

ViewerWindow::~ViewerWindow() = default;

ViewerWindow *ViewerWindow::forSnapshot(const QString &file, SpartaWrapper *sparta,
                                        SpartaGui *spartagui, QWidget *parent)
{
    auto *viewer = new ImageViewer(file, sparta, spartagui);
    auto *win    = new ViewerWindow(viewer, QStringLiteral("SPARTA-GUI - Viewer - Image: "), parent);
    win->setWindowTitle(QStringLiteral("SPARTA-GUI - Viewer - Image: ") + file);
    return win;
}

ViewerWindow *ViewerWindow::forSequence(const QString &file, SpartaGui *spartagui, QWidget *parent)
{
    auto *show = new SlideShow(file, spartagui);
    // "Slide Show" stays in the title on purpose: it is the name users know
    // this window by, and the GUI tests find it by that name.
    auto *win =
        new ViewerWindow(show, QStringLiteral("SPARTA-GUI - Viewer - Slide Show: "), parent);
    win->setWindowTitle(QStringLiteral("SPARTA-GUI - Viewer - Slide Show: ") + file);
    return win;
}

SlideShow *ViewerWindow::sequence() const
{
    return qobject_cast<SlideShow *>(view);
}

ImageViewer *ViewerWindow::snapshot() const
{
    return qobject_cast<ImageViewer *>(view);
}

// Local Variables:
// c-basic-offset: 4
// End:
