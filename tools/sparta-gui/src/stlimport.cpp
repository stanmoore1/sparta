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

#include "stlimport.h"

#include <QDataStream>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QRegularExpression>
#include <QTextStream>

#include <cmath>

namespace StlImport {

namespace {

// dedup key for a vertex: identical coordinates (same STL text or same binary
// bytes for a shared vertex) format to the same key, matching stl2surf.py's
// text-based merging of shared triangle corners.
QString vkey(double x, double y, double z)
{
    return QStringLiteral("%1 %2 %3")
        .arg(x, 0, 'g', 17)
        .arg(y, 0, 'g', 17)
        .arg(z, 0, 'g', 17);
}

// coordinate formatting for the emitted surf file (read with atof by SPARTA)
QString coord(double v) { return QString::number(v, 'g', 12); }

void resetExtents(SurfMesh &m)
{
    for (int k = 0; k < 3; ++k) {
        m.lo[k] = 0.0;
        m.hi[k] = 0.0;
    }
}

void accumExtents(SurfMesh &m, bool &first, double x, double y, double z)
{
    const double v[3] = {x, y, z};
    if (first) {
        for (int k = 0; k < 3; ++k) m.lo[k] = m.hi[k] = v[k];
        first = false;
    } else {
        for (int k = 0; k < 3; ++k) {
            if (v[k] < m.lo[k]) m.lo[k] = v[k];
            if (v[k] > m.hi[k]) m.hi[k] = v[k];
        }
    }
}

// add a vertex, returning its unique index (dedup via the hash)
int addVertex(SurfMesh &m, QHash<QString, int> &uniq, bool &first, double x, double y, double z)
{
    const QString k = vkey(x, y, z);
    auto it = uniq.constFind(k);
    if (it != uniq.constEnd()) return it.value();
    const int idx = m.points.size();
    m.points.append({x, y, z});
    uniq.insert(k, idx);
    accumExtents(m, first, x, y, z);
    return idx;
}

} // namespace

SourceKind detectSource(const QString &path)
{
    const QString suffix = QFileInfo(path).suffix().toLower();
    if (suffix == "stl") return SourceKind::Stl;
    if (suffix == "surf") return SourceKind::Surf;
    // sniff: STL ASCII begins with "solid"; a SPARTA surf file has a "N points"
    // line and a "Points" section. Fall back to Surf for SPARTA data files.
    QFile f(path);
    if (f.open(QIODevice::ReadOnly)) {
        const QByteArray head = f.read(512);
        if (head.trimmed().startsWith("solid")) return SourceKind::Stl;
        if (head.contains("points") || head.contains("Points")) return SourceKind::Surf;
    }
    return SourceKind::Unknown;
}

// ---------------------------------------------------------------------------
// STL parsing (ASCII + binary)
// ---------------------------------------------------------------------------

static bool parseBinaryStl(QFile &f, quint32 ntri, SurfMesh &out, QString &err)
{
    QDataStream ds(&f);
    ds.setByteOrder(QDataStream::LittleEndian);
    ds.setFloatingPointPrecision(QDataStream::SinglePrecision);
    f.seek(84); // 80-byte header + 4-byte count already consumed by the caller

    QHash<QString, int> uniq;
    bool first = true;
    out.elems.reserve(static_cast<int>(ntri));
    for (quint32 t = 0; t < ntri; ++t) {
        float nx, ny, nz;
        ds >> nx >> ny >> nz; // normal, ignored
        int v[3];
        for (int c = 0; c < 3; ++c) {
            float x, y, z;
            ds >> x >> y >> z;
            v[c] = addVertex(out, uniq, first, x, y, z);
        }
        quint16 attr;
        ds >> attr;
        if (ds.status() != QDataStream::Ok) {
            err = QStringLiteral("Truncated binary STL file (read %1 of %2 triangles)")
                      .arg(t)
                      .arg(ntri);
            return false;
        }
        out.elems.append({v[0], v[1], v[2]});
    }
    return true;
}

static bool parseAsciiStl(const QByteArray &bytes, SurfMesh &out, QString &err)
{
    const QString text = QString::fromLatin1(bytes);
    static const QRegularExpression vre(
        QStringLiteral("vertex\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)"));
    QHash<QString, int> uniq;
    bool first = true;
    int nvert = 0;
    int tri[3];
    auto it = vre.globalMatch(text);
    while (it.hasNext()) {
        const auto m = it.next();
        bool okx, oky, okz;
        const double x = m.captured(1).toDouble(&okx);
        const double y = m.captured(2).toDouble(&oky);
        const double z = m.captured(3).toDouble(&okz);
        if (!okx || !oky || !okz) {
            err = QStringLiteral("Malformed vertex in ASCII STL: %1").arg(m.captured(0));
            return false;
        }
        tri[nvert % 3] = addVertex(out, uniq, first, x, y, z);
        if (++nvert % 3 == 0) out.elems.append({tri[0], tri[1], tri[2]});
    }
    if (nvert == 0) {
        err = QStringLiteral("No vertices found in ASCII STL file");
        return false;
    }
    if (nvert % 3 != 0) {
        err = QStringLiteral("ASCII STL vertex count (%1) is not a multiple of 3").arg(nvert);
        return false;
    }
    return true;
}

bool parseStl(const QString &path, SurfMesh &out, QString &err)
{
    out = SurfMesh();
    out.is2d = false;
    resetExtents(out);

    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        err = QStringLiteral("Cannot open STL file: %1").arg(path);
        return false;
    }
    const qint64 filesize = f.size();

    // binary STL: 80-byte header, uint32 count, 50 bytes per triangle. Use the
    // exact size arithmetic to disambiguate binary files that (illegally) begin
    // with the ASCII token "solid".
    bool isBinary = false;
    quint32 ntri = 0;
    if (filesize >= 84) {
        f.seek(80);
        QDataStream ds(&f);
        ds.setByteOrder(QDataStream::LittleEndian);
        ds >> ntri;
        if (static_cast<qint64>(84) + static_cast<qint64>(ntri) * 50 == filesize) isBinary = true;
    }

    bool ok;
    if (isBinary) {
        ok = parseBinaryStl(f, ntri, out, err);
    } else {
        f.seek(0);
        const QByteArray bytes = f.readAll();
        if (!bytes.trimmed().startsWith("solid")) {
            err = QStringLiteral("Not a recognized STL file (no 'solid' header, and the size "
                                 "does not match a binary STL)");
            return false;
        }
        ok = parseAsciiStl(bytes, out, err);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// SPARTA surface-file parsing
// ---------------------------------------------------------------------------

bool parseSurf(const QString &path, SurfMesh &out, QString &err)
{
    out = SurfMesh();
    resetExtents(out);

    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        err = QStringLiteral("Cannot open surface file: %1").arg(path);
        return false;
    }
    QTextStream in(&f);

    enum Section { None, Points, Lines, Triangles } sect = None;
    QHash<int, int> id2idx; // surf point id (1-based) -> our index
    QHash<QString, int> uniq;
    bool first = true;
    bool haveTri = false, haveLine = false;

    while (!in.atEnd()) {
        QString raw = in.readLine();
        const int hash = raw.indexOf('#');
        if (hash >= 0) raw = raw.left(hash);
        const QString line = raw.trimmed();
        if (line.isEmpty()) continue;

        if (line.compare("Points", Qt::CaseInsensitive) == 0) { sect = Points; continue; }
        if (line.compare("Lines", Qt::CaseInsensitive) == 0) { sect = Lines; haveLine = true; continue; }
        if (line.compare("Triangles", Qt::CaseInsensitive) == 0) { sect = Triangles; haveTri = true; continue; }
        // header count lines ("N points", "M triangles/lines") -> ignore
        if (line.endsWith("points", Qt::CaseInsensitive) ||
            line.endsWith("triangles", Qt::CaseInsensitive) ||
            line.endsWith("lines", Qt::CaseInsensitive))
            continue;

        const QStringList tok = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (sect == Points) {
            // "id x y [z]"
            if (tok.size() < 3) { err = "Malformed Points line: " + line; return false; }
            const int id = tok[0].toInt();
            const double x = tok[1].toDouble();
            const double y = tok[2].toDouble();
            const double z = (tok.size() >= 4) ? tok[3].toDouble() : 0.0;
            const int idx = addVertex(out, uniq, first, x, y, z);
            id2idx.insert(id, idx);
        } else if (sect == Lines) {
            out.is2d = true;
            // "id [type] v1 v2"
            int type = 0, v1, v2;
            if (tok.size() == 3) { v1 = tok[1].toInt(); v2 = tok[2].toInt(); }
            else if (tok.size() >= 4) { type = tok[1].toInt(); v1 = tok[2].toInt(); v2 = tok[3].toInt(); }
            else { err = "Malformed Lines line: " + line; return false; }
            out.elems.append({id2idx.value(v1, -1), id2idx.value(v2, -1), -1});
            if (tok.size() >= 4) out.types.append(type);
        } else if (sect == Triangles) {
            // "id [type] v1 v2 v3"
            int type = 0, v1, v2, v3;
            if (tok.size() == 4) { v1 = tok[1].toInt(); v2 = tok[2].toInt(); v3 = tok[3].toInt(); }
            else if (tok.size() >= 5) { type = tok[1].toInt(); v1 = tok[2].toInt(); v2 = tok[3].toInt(); v3 = tok[4].toInt(); }
            else { err = "Malformed Triangles line: " + line; return false; }
            out.elems.append({id2idx.value(v1, -1), id2idx.value(v2, -1), id2idx.value(v3, -1)});
            if (tok.size() >= 5) out.types.append(type);
        }
    }

    if (out.points.isEmpty()) { err = "Surface file has no Points section"; return false; }
    if (!haveTri && !haveLine) { err = "Surface file has neither Lines nor Triangles"; return false; }
    if (out.types.size() != out.nelements()) out.types.clear(); // partial/absent -> treat as none

    // validate indices resolved
    for (const auto &e : out.elems)
        if (e[0] < 0 || e[1] < 0 || (!out.is2d && e[2] < 0)) {
            err = "Surface file references an undefined point index";
            return false;
        }
    return true;
}

// ---------------------------------------------------------------------------
// Watertightness pre-check
// ---------------------------------------------------------------------------

WatertightReport checkWatertightPreflight(const SurfMesh &m)
{
    WatertightReport r;
    auto key = [](int a, int b) { return (static_cast<qint64>(a) << 32) | static_cast<quint32>(b); };

    // 2d surfaces are line segments, not triangles: a watertight, manifold loop
    // requires every point to be the *start* of exactly one line and the *end*
    // of exactly one line (in-degree == out-degree == 1).  The 3d directed-edge
    // reverse-matching test below is wrong for 2d -- a simple closed loop has no
    // reverse edges, so it would flag every segment as a hole (this is why a
    // known-good file such as examples/circle/data.circle failed).
    if (m.is2d) {
        QHash<int, int> inDeg, outDeg;   // point index -> incident line count
        QHash<int, QVector<int>> touch;  // point index -> incident element ids
        QHash<qint64, int> dirCount;     // directed line (a,b) -> multiplicity
        for (int e = 0; e < m.nelements(); ++e) {
            const int a = m.elems[e][0], b = m.elems[e][1];
            ++outDeg[a];
            ++inDeg[b];
            touch[a].append(e);
            touch[b].append(e);
            ++dirCount[key(a, b)];
        }
        // an exactly repeated directed line is non-manifold
        for (auto it = dirCount.constBegin(); it != dirCount.constEnd(); ++it) {
            if (it.value() <= 1) continue;
            const int a = static_cast<int>(it.key() >> 32);
            const int b = static_cast<int>(static_cast<quint32>(it.key() & 0xffffffff));
            r.duplicateEdges += it.value() - 1;
            r.duplicateEdgeList.append({a, b});
            for (int e : touch.value(a)) r.leakingElems.insert(e);
        }
        // a point whose in- and out-degree differ is an open end (a hole)
        QSet<int> pts;
        for (auto it = inDeg.constBegin(); it != inDeg.constEnd(); ++it) pts.insert(it.key());
        for (auto it = outDeg.constBegin(); it != outDeg.constEnd(); ++it) pts.insert(it.key());
        for (int p : std::as_const(pts)) {
            const int id = inDeg.value(p), od = outDeg.value(p);
            if (id == od) continue;
            r.unmatchedEdges += qAbs(id - od);
            r.unmatchedEdgeList.append({p, p}); // leak located at a point
            for (int e : touch.value(p)) r.leakingElems.insert(e);
        }
        return r;
    }

    // 3d triangles: directed edge (a,b) -> element indices owning it (same direction)
    QHash<qint64, QVector<int>> edge2elem;
    const int nseg = 3; // 3d tri = three directed edges
    for (int e = 0; e < m.nelements(); ++e) {
        const auto &el = m.elems[e];
        for (int s = 0; s < nseg; ++s) {
            int a = el[s], b = el[(s + 1) % 3];
            edge2elem[key(a, b)].append(e);
        }
    }

    QSet<qint64> seen;
    for (auto it = edge2elem.constBegin(); it != edge2elem.constEnd(); ++it) {
        const qint64 k = it.key();
        const int a = static_cast<int>(k >> 32);
        const int b = static_cast<int>(static_cast<quint32>(k & 0xffffffff));
        const QVector<int> &owners = it.value();
        // non-manifold: same directed edge used by >1 element
        if (owners.size() > 1) {
            r.duplicateEdges += owners.size() - 1;
            r.duplicateEdgeList.append({a, b});
            for (int e : owners) r.leakingElems.insert(e);
        }
        // hole: the reverse directed edge is absent
        if (!edge2elem.contains(key(b, a))) {
            r.unmatchedEdges += 1;
            r.unmatchedEdgeList.append({a, b});
            for (int e : owners) r.leakingElems.insert(e);
        }
        seen.insert(k);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Surface-file writer
// ---------------------------------------------------------------------------

QString buildSurfFile(const SurfMesh &m, const QString &sourceName, const QSet<int> &badElems)
{
    const bool tagged = !badElems.isEmpty();
    QString s;
    QTextStream out(&s);
    out << "# SPARTA surface file, from " << sourceName << " (SPARTA-GUI import)\n\n";
    out << m.npoints() << " points\n";
    out << m.nelements() << (m.is2d ? " lines\n" : " triangles\n");

    out << "\nPoints\n\n";
    for (int i = 0; i < m.points.size(); ++i) {
        const auto &p = m.points[i];
        out << (i + 1) << ' ' << coord(p[0]) << ' ' << coord(p[1]);
        if (!m.is2d) out << ' ' << coord(p[2]);
        out << '\n';
    }

    out << (m.is2d ? "\nLines\n\n" : "\nTriangles\n\n");
    for (int i = 0; i < m.elems.size(); ++i) {
        const auto &e = m.elems[i];
        out << (i + 1);
        if (tagged) out << ' ' << (badElems.contains(i) ? 2 : 1); // per-element type
        out << ' ' << (e[0] + 1) << ' ' << (e[1] + 1);
        if (!m.is2d) out << ' ' << (e[2] + 1);
        out << '\n';
    }
    return s;
}

// ---------------------------------------------------------------------------
// Command builders
// ---------------------------------------------------------------------------

bool validThreshold(double thresh)
{
    if (thresh <= 0.0 || thresh >= 255.0) return false;
    return std::floor(thresh) != thresh; // must be non-integer
}

QString buildReadSurfCommand(const StlImportSettings &s, const QString &surfPath)
{
    QStringList w;
    w << "read_surf" << surfPath;
    auto three = [](const double v[3]) {
        return QStringLiteral("%1 %2 %3")
            .arg(QString::number(v[0], 'g', 12))
            .arg(QString::number(v[1], 'g', 12))
            .arg(QString::number(v[2], 'g', 12));
    };
    // fixed, deterministic order (omit anything left at its default)
    if (s.useOrigin) w << "origin" << three(s.origin);
    switch (s.transKind) {
        case StlImportSettings::TransKind::Trans: w << "trans" << three(s.trans); break;
        case StlImportSettings::TransKind::ATrans: w << "atrans" << three(s.trans); break;
        case StlImportSettings::TransKind::FTrans: w << "ftrans" << three(s.trans); break;
        case StlImportSettings::TransKind::None: break;
    }
    if (s.useScale) w << "scale" << three(s.scale);
    if (s.useRotate)
        w << "rotate" << QString::number(s.rotate[0], 'g', 12) << QString::number(s.rotate[1], 'g', 12)
          << QString::number(s.rotate[2], 'g', 12) << QString::number(s.rotate[3], 'g', 12);
    if (s.invert) w << "invert";
    if (s.useClip) {
        w << "clip";
        if (s.clipHasFraction) w << QString::number(s.clipFraction, 'g', 12);
    }
    if (s.transparent) w << "transparent";
    if (!s.group.isEmpty()) w << "group" << s.group;
    if (!s.typeadd.isEmpty()) w << "typeadd" << s.typeadd;
    return w.join(' ');
}

QStringList buildAblationCommands(const StlImportSettings &s, const QString &surfPath)
{
    QStringList cmds;
    // create_isurf requires distributed explicit surfaces
    cmds << "global surfs explicit/distributed";
    cmds << buildReadSurfCommand(s, surfPath);
    // every element must carry a collision model before create_isurf consumes it
    // (verified vs examples/explicit2implicit/in.exp2imp.sphere.3d)
    cmds << "surf_collide 1 diffuse 300.0 0.0";
    cmds << "surf_modify all collide 1";
    // the fix ablate must be defined BEFORE create_isurf (verified vs
    // examples/explicit2implicit/in.exp2imp.sphere.3d)
    cmds << QStringLiteral("fix %1 ablate %2 %3 %4 %5 %6")
                .arg(s.ablateId, s.isurfGroup)
                .arg(s.nevery)
                .arg(QString::number(s.ablateScale, 'g', 12), s.ablateSource)
                .arg(s.maxrandom);
    cmds << QStringLiteral("create_isurf %1 %2 %3 %4")
                .arg(s.isurfGroup, s.ablateId)
                .arg(QString::number(s.thresh, 'g', 12), s.isurfMode);
    return cmds;
}

} // namespace StlImport

// Local Variables:
// c-basic-offset: 4
// End:
