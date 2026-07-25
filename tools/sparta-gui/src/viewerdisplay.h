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

#ifndef VIEWERDISPLAY_H
#define VIEWERDISPLAY_H

#include "displaytransform.h"

#include <QImage>
#include <QSize>
#include <QWidget>

class QLabel;
class QScrollArea;

/**
 * @brief The surface a viewer paints its picture on.
 *
 * A scroll area holding a label holding a pixmap, plus the rules for how big
 * that pixmap should be. Both viewers had their own copy of this, along with
 * their own copy of the window-fitting dance around it, and the copies had
 * drifted apart: only the image viewer would scale a picture down to fit the
 * space it had, and only the slide show could rotate or mirror one.
 *
 * The difference that is real is @ref FitMode. Everything else was accident.
 */
class ViewerDisplay : public QWidget {
    Q_OBJECT

public:
    enum FitMode {
        /// Paint at the transform's own scale and let the window grow. What a
        /// slide show wants: the frame is the size it is.
        Natural,
        /// Shrink to the space available when the picture is larger. What a
        /// rendered snapshot wants: docked in a short panel it would otherwise
        /// show one corner of a 600x600 render, which reads as a broken panel
        /// rather than a cropped one.
        FitViewport,
    };

    explicit ViewerDisplay(FitMode mode, QWidget *parent = nullptr);
    ~ViewerDisplay() override;

    ViewerDisplay(const ViewerDisplay &)            = delete;
    ViewerDisplay &operator=(const ViewerDisplay &) = delete;

    /// The picture to show, before any displayed-image transform.
    void setImage(const QImage &image);
    [[nodiscard]] const QImage &sourceImage() const { return raw; }

    /// The picture as shown -- what Save As, Copy and the movie exporter act on.
    [[nodiscard]] const QImage &displayedImage() const { return shown; }

    [[nodiscard]] bool isEmpty() const { return raw.isNull(); }

    [[nodiscard]] DisplayTransform transform() const { return xform; }
    void setTransform(const DisplayTransform &t);

    /// Repaint after something outside changed the fit (a resize, a new mode).
    void refresh();

    /// The label and scroll area, for hosts that install event filters on them.
    [[nodiscard]] QLabel *label() const { return imageLabel; }
    [[nodiscard]] QScrollArea *scrollArea() const { return area; }

    /// @name Sizing the window this display is in
    ///
    /// Only meaningful when the display is in a window of its own; in a dock
    /// there is nothing to resize and fitViewerWindow() says so. The memo of
    /// the last fit lives here because both viewers kept one and both used it
    /// the same way.
    /// @{
    void fitHostWindow(QWidget *host, const QSize &content, const QSize &budget);
    void forgetHostFit() { lastFitSize = QSize(); }
    [[nodiscard]] bool needsHostFit() const { return !lastFitSize.isValid(); }
    /// @}

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    void repaintPixmap();

    FitMode fit;
    QLabel *imageLabel   = nullptr;
    QScrollArea *area    = nullptr;
    QImage raw;                 ///< as handed over
    QImage shown;               ///< after the displayed-image transform
    DisplayTransform xform;
    QSize lastFitSize;
    bool painting = false;      ///< guards repaintPixmap() against re-entry
};

#endif

// Local Variables:
// c-basic-offset: 4
// End:
