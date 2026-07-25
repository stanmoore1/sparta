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

#include <QTransform>

QImage applyDisplayTransform(const QImage &src, const DisplayTransform &t)
{
    if (src.isNull() || t.isIdentity()) return src;

    QImage img = src;

    if (t.rotation != 0) {
        QTransform rot;
        rot.rotate(t.rotation);
        img = img.transformed(rot, Qt::SmoothTransformation);
    }

    if (t.flipH) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 9, 0)
        img = img.flipped(Qt::Horizontal);
#else
        img = img.mirrored(true, false);
#endif
    }

    if (t.flipV) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 9, 0)
        img = img.flipped(Qt::Vertical);
#else
        img = img.mirrored(false, true);
#endif
    }

    if (t.scale != 1.0) {
        // IgnoreAspectRatio is safe because both extents carry the same factor;
        // it avoids the rounding KeepAspectRatio applies to the second one.
        img = img.scaled(static_cast<int>(img.width() * t.scale),
                         static_cast<int>(img.height() * t.scale), Qt::IgnoreAspectRatio,
                         Qt::SmoothTransformation);
    }

    return img;
}

QSize transformedSize(const QSize &raw, const DisplayTransform &t)
{
    QSize size = raw;
    if (t.isTransposed()) size.transpose();
    return {static_cast<int>(size.width() * t.scale), static_cast<int>(size.height() * t.scale)};
}

QStringList ffmpegFilterArgs(const DisplayTransform &t)
{
    QStringList filters;

    if (t.scale != 1.0) filters << QString("scale=iw*%1:-1").arg(t.scale);

    // ffmpeg has no arbitrary-angle filter in this pipeline, so a half turn is
    // two quarter turns. transpose=1 is clockwise, transpose=2 counter.
    switch (t.rotation) {
        case 90: filters << "transpose=1"; break;
        case 180: filters << "transpose=1" << "transpose=1"; break;
        case 270: filters << "transpose=2"; break;
        default: break;
    }

    if (t.flipH) filters << "hflip";
    if (t.flipV) filters << "vflip";

    if (filters.isEmpty()) return {};
    return {"-vf", filters.join(',')};
}

QStringList magickTransformArgs(const DisplayTransform &t)
{
    QStringList args;
    // The percent sign is appended rather than written into the format string:
    // "%%" is not an escape for QString::arg (that is printf, not Qt), so the
    // obvious QString("%1%%").arg(...) yields "50%%" and ImageMagick rejects it.
    if (t.scale != 1.0) args << "-resize" << QString::number(100.0 * t.scale) + QLatin1Char('%');
    if (t.rotation != 0) args << "-rotate" << QString::number(t.rotation);
    if (t.flipH) args << "-flop";
    if (t.flipV) args << "-flip";
    return args;
}

// Local Variables:
// c-basic-offset: 4
// End:
