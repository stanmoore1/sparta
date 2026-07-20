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

#ifndef STLIMPORT_H
#define STLIMPORT_H

// Pure, GUI-free core of the "Import Surface (STL / SPARTA)" wizard: parse an
// STL file (ASCII or binary) or an existing SPARTA surface file into a common
// mesh, run SPARTA's watertightness pre-check (reporting *where* it fails), and
// build the read_surf / create_isurf / fix ablate commands. None of this touches
// Qt widgets or a live SPARTA instance, so it is unit-testable on its own
// (mirrors the DumpImageSettings + builder pattern in dumpimage.h).

#include <QSet>
#include <QString>
#include <QStringList>
#include <QVector>

#include <array>

namespace StlImport {

/** @brief A triangulated (3d) or line (2d) surface mesh with dedup'd points. */
struct SurfMesh {
    QVector<std::array<double, 3>> points; ///< unique vertex coordinates
    QVector<std::array<int, 3>> elems;     ///< 0-based point indices; 2d lines use [a,b,-1]
    QVector<int> types;                    ///< per-element type, or empty if none present
    bool is2d = false;                     ///< true = Lines (2d), false = Triangles (3d)
    double lo[3] = {0, 0, 0};              ///< bounding-box minimum
    double hi[3] = {0, 0, 0};              ///< bounding-box maximum
    int nelements() const { return static_cast<int>(elems.size()); }
    int npoints() const { return static_cast<int>(points.size()); }
};

/** @brief What kind of file a path is, decided by extension then content. */
enum class SourceKind { Stl, Surf, Unknown };
SourceKind detectSource(const QString &path);

/** @brief Parse an STL file (auto-detects ASCII vs binary). Always yields a 3d mesh. */
bool parseStl(const QString &path, SurfMesh &out, QString &err);

/** @brief Parse an existing SPARTA surface file (Points + Lines/Triangles, optional type). */
bool parseSurf(const QString &path, SurfMesh &out, QString &err);

/**
 * @brief Result of the watertightness pre-check, including the failing locations.
 *
 * Reproduces the check SPARTA performs at read_surf time (surf.cpp), computed
 * here before SPARTA is ever touched so the wizard can both report counts and
 * highlight the offending elements in the preview render.
 */
struct WatertightReport {
    int duplicateEdges = 0;                       ///< same-direction edge seen >1x (non-manifold)
    int unmatchedEdges = 0;                        ///< directed edge whose reverse is absent (hole)
    QVector<std::array<int, 2>> unmatchedEdgeList; ///< point-index pairs of hole edges
    QVector<std::array<int, 2>> duplicateEdgeList; ///< point-index pairs of non-manifold edges
    QSet<int> leakingElems;                        ///< element indices touching any failing edge
    bool watertight() const { return duplicateEdges == 0 && unmatchedEdges == 0; }
};
WatertightReport checkWatertightPreflight(const SurfMesh &m);

/**
 * @brief Write the mesh as a SPARTA surface file (byte-compatible with read_surf).
 * @param badElems if non-empty, a per-element type column is written (2 = in the
 *        set / "leaking", 1 = ok) so a `read_surf ... type` + `dump_modify scolor
 *        2 red` render highlights those elements.
 */
QString buildSurfFile(const SurfMesh &m, const QString &sourceName, const QSet<int> &badElems = {});

/** @brief GUI-free settings the builders turn into SPARTA commands. */
struct StlImportSettings {
    enum class Mode { Explicit, Implicit };
    enum class TransKind { None, Trans, ATrans, FTrans };

    Mode mode = Mode::Explicit;

    // read_surf transform keywords (only emitted when their "use" flag is set)
    bool useOrigin = false;
    double origin[3] = {0, 0, 0};
    TransKind transKind = TransKind::None;
    double trans[3] = {0, 0, 0};
    bool useScale = false;
    double scale[3] = {1, 1, 1};
    bool useRotate = false;
    double rotate[4] = {0, 1, 0, 0}; // theta, Rx, Ry, Rz
    bool invert = false;
    bool transparent = false;
    bool useClip = false;
    bool clipHasFraction = false;
    double clipFraction = 0.0;
    QString group;         // read_surf group-ID (empty = omit)
    QString typeadd;       // read_surf typeadd offset (empty = omit)

    // implicit-surface / ablation (create_isurf + fix ablate)
    QString isurfGroup = "all"; // grid-cell group for create_isurf
    QString ablateId = "fablate";
    double thresh = 39.5;        // strictly non-integer in (0,255)
    QString isurfMode = "voxel"; // inout | voxel | ave | multi
    int nevery = 0;
    double ablateScale = 0.2;
    QString ablateSource = "random";
    int maxrandom = 0;
    int gridNx = 50, gridNy = 50, gridNz = 50; // preview / create_grid resolution
};

/** @brief `read_surf <path>` plus the non-default transform keywords in a fixed order. */
QString buildReadSurfCommand(const StlImportSettings &s, const QString &surfPath);

/**
 * @brief The implicit/ablation command block, in SPARTA's required order:
 * `global surfs explicit`, the read_surf line, `fix ... ablate ...`, then
 * `create_isurf ...` (fix ablate must precede create_isurf).
 */
QStringList buildAblationCommands(const StlImportSettings &s, const QString &surfPath);

/** @brief True if thresh is a valid create_isurf threshold: non-integer, 0<thresh<255. */
bool validThreshold(double thresh);

} // namespace StlImport

#endif // STLIMPORT_H

// Local Variables:
// c-basic-offset: 4
// End:
