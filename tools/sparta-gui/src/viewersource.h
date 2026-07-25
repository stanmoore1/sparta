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

#ifndef VIEWERSOURCE_H
#define VIEWERSOURCE_H

#include <QIcon>
#include <QImage>
#include <QString>
#include <QWidget>

class QMenu;

/**
 * @brief One of the things the viewer panel can show
 *
 * The application has three ways of putting a picture of a simulation on
 * screen: a snapshot rendered on demand through SPARTA, a sequence of frames a
 * run has already written, and an interactive 3D scene. They are genuinely
 * different -- different data, different controls, different cost per
 * interaction -- and this interface deliberately does not try to hide that.
 * It is only wide enough for a host to display one, label its tab, and know
 * whether it has anything to show.
 *
 * Keeping it this narrow is the point. The alternative, a common viewer
 * abstraction with the union of all three feature sets behind it, would have
 * exactly three implementations and every one of them would leave most of it
 * unimplemented.
 */
class ViewerSource : public QWidget {
    Q_OBJECT

public:
    explicit ViewerSource(QWidget *parent = nullptr) : QWidget(parent) {}
    ~ViewerSource() override = default;

    ViewerSource(const ViewerSource &)            = delete;
    ViewerSource(ViewerSource &&)                 = delete;
    ViewerSource &operator=(const ViewerSource &) = delete;
    ViewerSource &operator=(ViewerSource &&)      = delete;

    /// Short name for this source's tab, e.g. "Snapshot"
    [[nodiscard]] virtual QString sourceLabel() const = 0;

    /// Icon for the tab
    [[nodiscard]] virtual QIcon sourceIcon() const = 0;

    /// What the tab's tooltip says when the source is available
    [[nodiscard]] virtual QString sourceTip() const = 0;

    /// Why the tab is disabled, when hasContent() is false. Shown as the
    /// tooltip in that state: a source with nothing in it should say how to
    /// put something in it, rather than being an empty pane with no
    /// explanation -- which is what the old empty Slide Show panel was.
    [[nodiscard]] virtual QString emptyTip() const = 0;

    /// Is there anything to look at yet?
    [[nodiscard]] virtual bool hasContent() const = 0;

    /// The image the shared Save As and Copy act on. Null when the source has
    /// nothing, or has no single still image to give.
    [[nodiscard]] virtual QImage currentImage() const { return {}; }

    /// @name Chrome the host may hoist out of the source and share
    ///
    /// Declared now, returning nothing, so that consolidating the toolbars
    /// later does not have to change this interface or every implementation of
    /// it. Until then each source draws its own controls and the host only
    /// stacks them.
    /// @{
    virtual QWidget *topStrip() { return nullptr; }
    virtual QWidget *sideStrip() { return nullptr; }
    virtual QMenu *sourceMenu() { return nullptr; }
    /// @}

signals:
    /// The host applies this to the dock tab or the window title, so a source
    /// does not need to know which of the two it is living in.
    void titleChanged(const QString &title);

    /// Something appeared or went away; the host re-checks hasContent().
    void contentChanged();
};

#endif

// Local Variables:
// c-basic-offset: 4
// End:
