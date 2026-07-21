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

#ifndef CASEMODEL_H
#define CASEMODEL_H

// Pure, GUI-free semantic model of the geometry and boundary-condition commands
// in a SPARTA input deck, plus surgical text edits back onto the deck.  It is the
// engine behind the visual case-setup canvas: parse() turns deck text into a
// structured Model the canvas can render (simulation box, boundary conditions,
// regions, imported surfaces, mixtures, face emitters), and the edit helpers
// (setBoundary/insertEmitFace/setBoxExtents) rewrite ONLY the affected command
// lines, preserving every comment and unrecognized command verbatim so a
// hand-written deck is never destroyed.  Depends on QtCore only (no Widgets, no
// SPARTA library), so it is fully unit-testable.

#include <QString>
#include <QStringList>
#include <QVector>

namespace CaseModel {

/// @brief Canonical SPARTA box-face order used throughout the model.
/// Index 0..5 == xlo, xhi, ylo, yhi, zlo, zhi.
extern const char *const FACE_NAMES[6];

/// @brief One physical line of the deck, preserved verbatim in order.
struct Line {
    QString text;     ///< the raw line (no trailing newline)
    QString command;  ///< first whitespace token with comments stripped ("" if blank/comment)
};

/// @brief The simulation box from `create_box` (+ `dimension`).
struct Box {
    bool present = false;
    int dimension = 3;               ///< from a `dimension` command; SPARTA default is 3 (2 for 2d)
    double lo[3] = {0.0, 0.0, 0.0};  ///< xlo, ylo, zlo
    double hi[3] = {0.0, 0.0, 0.0};  ///< xhi, yhi, zhi
    int sourceLine = -1;             ///< 0-based index of the create_box line, -1 if none
};

/// @brief The `boundary` command: one spec per axis (each 1-2 chars: o/p/r/s).
struct Boundary {
    bool present = false;
    QString spec[3];      ///< x, y, z as written, e.g. "o", "p", "rr"
    int sourceLine = -1;  ///< 0-based index of the boundary line, -1 if none
};

/// @brief A `region ID style args...` primitive.
struct Region {
    QString id;
    QString style;        ///< block, sphere, cylinder, ...
    QStringList args;     ///< remaining tokens after id+style
    int sourceLine = -1;
};

/// @brief A `read_surf file ...` import.
struct SurfImport {
    QString file;         ///< the surface file (first arg after read_surf)
    QStringList args;     ///< remaining tokens (group, transforms, ...)
    int sourceLine = -1;
};

/// @brief A `mixture ID ...` definition (may span multiple lines with the same ID).
struct Mixture {
    QString id;
    QStringList species;        ///< species names listed before any keyword
    QString nrho;               ///< value after the `nrho` keyword (empty if unset)
    QString temp;               ///< value after the `temp` keyword
    QStringList vstream;        ///< the three values after `vstream`
    QVector<int> sourceLines;   ///< every line that contributes to this mixture
};

/// @brief A `fix ID emit/face mixture faces...` boundary emitter.
struct EmitFace {
    QString id;
    QString mixture;
    QStringList faces;    ///< face tokens (xlo, xhi, ...) up to the first keyword
    int sourceLine = -1;
};

/// @brief The parsed case: all lines preserved, recognized elements extracted.
struct Model {
    QVector<Line> lines;          ///< every physical line, in order
    int dimension = 3;            ///< effective dimension (mirrors Box::dimension)
    Box box;
    Boundary boundary;
    QVector<Region> regions;
    QVector<SurfImport> surfaces;
    QVector<Mixture> mixtures;
    QVector<EmitFace> emits;

    /// @brief IDs of every defined mixture (for BC popovers).
    QStringList mixtureIds() const;
};

/// @brief Parse a SPARTA deck into a Model (never throws; unknown lines preserved).
Model parse(const QString &deckText);

/// @brief Split one deck line into whitespace tokens with a trailing `#` comment removed.
QStringList tokenize(const QString &line);

/// @brief True if @p token is a SPARTA box-face name (xlo/xhi/ylo/yhi/zlo/zhi).
bool isFaceName(const QString &token);

// --- surgical text edits: return a NEW deck string with only the relevant
//     command line(s) replaced/inserted; everything else is byte-preserved. ---

/// @brief Set the three-axis `boundary` command.  Replaces the existing boundary
/// line if present, else inserts one right after `create_box` (or at the top).
QString setBoundary(const QString &deckText, const QString &x, const QString &y,
                    const QString &z);

/// @brief Set the `create_box` extents in place (Phase 2 drag-to-resize).  No-op
/// (returns the input unchanged) if the deck has no create_box line.
QString setBoxExtents(const QString &deckText, const double lo[3], const double hi[3]);

/// @brief Insert a `fix <id> emit/face <mixture> <faces...>` line after the last
/// of (boundary, create_box) so it is defined once the box exists.
QString insertEmitFace(const QString &deckText, const QString &id,
                       const QString &mixture, const QStringList &faces);

} // namespace CaseModel

#endif // CASEMODEL_H

// Local Variables:
// c-basic-offset: 4
// End:
