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

#include <QImage>
#include <QSize>
#include <QStringList>

/**
 * @brief How the image on screen is scaled, turned and mirrored
 *
 * This is the *displayed image* transform: it changes the picture the user is
 * looking at and nothing else. It is deliberately not the camera -- moving the
 * camera re-renders the scene through SPARTA (or VTK) and produces different
 * pixels, whereas everything here is a rearrangement of pixels that already
 * exist.
 *
 * Keeping it as plain data with free functions, rather than as state inside a
 * widget, is what lets the same transform drive the screen *and* the movie
 * exporter from one definition. Those were three separate implementations of
 * the same six lines of arithmetic, and the copies had already drifted: the
 * slide show's zoom out scaled by 0.9 against a zoom in of 1.1, so zooming in
 * and back out did not return to where it started.
 */
struct DisplayTransform {
    double scale = 1.0; ///< 1.0 is 1:1; never allowed below MIN_SCALE
    int rotation = 0;   ///< clockwise degrees, one of 0, 90, 180, 270
    bool flipH   = false;
    bool flipV   = false;

    /// Below this the image is too small to see and too easy to lose entirely
    static constexpr double MIN_SCALE = 0.1;
    /// One press of zoom in or zoom out. Zoom out divides rather than
    /// multiplying by 0.9, so that the two are exact inverses.
    static constexpr double STEP = 1.1;

    [[nodiscard]] bool isIdentity() const
    {
        return (scale == 1.0) && (rotation == 0) && !flipH && !flipV;
    }

    /// Does this transform exchange the width and the height?
    [[nodiscard]] bool isTransposed() const { return (rotation == 90) || (rotation == 270); }

    void zoomIn() { scale *= STEP; }
    void zoomOut() { scale = qMax(scale / STEP, MIN_SCALE); }
    void rotateCw() { rotation = (rotation + 90) % 360; }
    void rotateCcw() { rotation = (rotation + 270) % 360; }
    void mirrorH() { flipH = !flipH; }
    void mirrorV() { flipV = !flipV; }
    void reset() { *this = DisplayTransform(); }

    bool operator==(const DisplayTransform &o) const
    {
        return (scale == o.scale) && (rotation == o.rotation) && (flipH == o.flipH) &&
               (flipV == o.flipV);
    }
    bool operator!=(const DisplayTransform &o) const { return !(*this == o); }
};

/// Rotate, mirror and scale @p src, in that order.
QImage applyDisplayTransform(const QImage &src, const DisplayTransform &t);

/// Size @p raw ends up with under @p t, without doing the work of producing it.
QSize transformedSize(const QSize &raw, const DisplayTransform &t);

/// FFmpeg arguments reproducing @p t, or an empty list for an identity
/// transform. Returned as ready-to-append arguments ("-vf", "transpose=1,hflip").
QStringList ffmpegFilterArgs(const DisplayTransform &t);

/// The same for ImageMagick ("-rotate", "90", "-flop").
QStringList magickTransformArgs(const DisplayTransform &t);

#endif

// Local Variables:
// c-basic-offset: 4
// End:
