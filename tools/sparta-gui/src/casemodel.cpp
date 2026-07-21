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

#include "casemodel.h"

#include <QRegularExpression>

namespace CaseModel {

const char *const FACE_NAMES[6] = {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"};

// keywords that terminate the leading species list of a `mixture` command
static const QStringList MIXTURE_KEYWORDS = {
    "nrho", "temp", "vstream", "frac", "group", "copy", "delete"};

QStringList tokenize(const QString &line)
{
    // strip a trailing comment: a '#' that starts a token runs to end of line
    static const QRegularExpression ws("\\s+");
    QStringList raw = line.split(ws, Qt::SkipEmptyParts);
    QStringList out;
    for (const QString &t : raw) {
        if (t.startsWith('#')) break;
        out << t;
    }
    return out;
}

bool isFaceName(const QString &token)
{
    for (const char *f : FACE_NAMES)
        if (token == QLatin1String(f)) return true;
    return false;
}

QStringList Model::mixtureIds() const
{
    QStringList ids;
    for (const Mixture &m : mixtures) ids << m.id;
    return ids;
}

Model parse(const QString &deckText)
{
    Model model;
    const QStringList rawLines = deckText.split('\n');

    for (int i = 0; i < rawLines.size(); ++i) {
        const QString &raw = rawLines.at(i);
        const QStringList tok = tokenize(raw);

        Line ln;
        ln.text = raw;
        ln.command = tok.isEmpty() ? QString() : tok.first();
        model.lines.push_back(ln);

        if (tok.isEmpty()) continue;
        const QString &cmd = tok.first();

        if (cmd == "dimension" && tok.size() >= 2) {
            model.dimension = tok.at(1).toInt();
            model.box.dimension = model.dimension;

        } else if (cmd == "create_box" && tok.size() >= 7) {
            model.box.present = true;
            model.box.sourceLine = i;
            model.box.lo[0] = tok.at(1).toDouble();
            model.box.hi[0] = tok.at(2).toDouble();
            model.box.lo[1] = tok.at(3).toDouble();
            model.box.hi[1] = tok.at(4).toDouble();
            model.box.lo[2] = tok.at(5).toDouble();
            model.box.hi[2] = tok.at(6).toDouble();

        } else if (cmd == "boundary" && tok.size() >= 4) {
            model.boundary.present = true;
            model.boundary.sourceLine = i;
            model.boundary.spec[0] = tok.at(1);
            model.boundary.spec[1] = tok.at(2);
            model.boundary.spec[2] = tok.at(3);

        } else if (cmd == "region" && tok.size() >= 3) {
            Region r;
            r.id = tok.at(1);
            r.style = tok.at(2);
            r.args = tok.mid(3);
            r.sourceLine = i;
            model.regions.push_back(r);

        } else if (cmd == "read_surf" && tok.size() >= 2) {
            SurfImport s;
            s.file = tok.at(1);
            s.args = tok.mid(2);
            s.sourceLine = i;
            model.surfaces.push_back(s);

        } else if (cmd == "mixture" && tok.size() >= 2) {
            const QString id = tok.at(1);
            // find or create the mixture (definitions accumulate across lines)
            int idx = -1;
            for (int m = 0; m < model.mixtures.size(); ++m)
                if (model.mixtures.at(m).id == id) { idx = m; break; }
            if (idx < 0) {
                Mixture mx;
                mx.id = id;
                model.mixtures.push_back(mx);
                idx = model.mixtures.size() - 1;
            }
            Mixture &mx = model.mixtures[idx];
            mx.sourceLines.push_back(i);
            // leading tokens before the first keyword are species names
            int j = 2;
            for (; j < tok.size(); ++j) {
                if (MIXTURE_KEYWORDS.contains(tok.at(j))) break;
                if (!mx.species.contains(tok.at(j))) mx.species << tok.at(j);
            }
            for (; j < tok.size(); ++j) {
                const QString &kw = tok.at(j);
                if (kw == "nrho" && j + 1 < tok.size()) mx.nrho = tok.at(++j);
                else if (kw == "temp" && j + 1 < tok.size()) mx.temp = tok.at(++j);
                else if (kw == "vstream" && j + 3 < tok.size()) {
                    mx.vstream = {tok.at(j + 1), tok.at(j + 2), tok.at(j + 3)};
                    j += 3;
                }
            }

        } else if (cmd == "fix" && tok.size() >= 4 && tok.at(2) == "emit/face") {
            EmitFace e;
            e.id = tok.at(1);
            e.mixture = tok.at(3);
            e.sourceLine = i;
            for (int j = 4; j < tok.size(); ++j) {
                if (!isFaceName(tok.at(j))) break;
                e.faces << tok.at(j);
            }
            model.emits.push_back(e);
        }
    }

    return model;
}

// ---------------------------------------------------------------------------
// surgical text edits
// ---------------------------------------------------------------------------

namespace {

// index of the first line whose first token equals cmd, or -1
int findCommand(const QStringList &lines, const QString &cmd)
{
    for (int i = 0; i < lines.size(); ++i) {
        const QStringList tok = tokenize(lines.at(i));
        if (!tok.isEmpty() && tok.first() == cmd) return i;
    }
    return -1;
}

// compact numeric formatting for rewritten commands
QString num(double v) { return QString::number(v, 'g', 10); }

} // namespace

QString setBoundary(const QString &deckText, const QString &x, const QString &y,
                    const QString &z)
{
    QStringList lines = deckText.split('\n');
    const QString newLine = QString("boundary %1 %2 %3").arg(x, y, z);

    const int b = findCommand(lines, "boundary");
    if (b >= 0) {
        lines[b] = newLine;
    } else {
        const int box = findCommand(lines, "create_box");
        // boundary must precede create_box in SPARTA; put it just before the box,
        // or at the top of the deck if there is no box yet
        lines.insert(box >= 0 ? box : 0, newLine);
    }
    return lines.join('\n');
}

QString setBoxExtents(const QString &deckText, const double lo[3], const double hi[3])
{
    QStringList lines = deckText.split('\n');
    const int box = findCommand(lines, "create_box");
    if (box < 0) return deckText;  // nothing to edit

    lines[box] = QString("create_box %1 %2 %3 %4 %5 %6")
                     .arg(num(lo[0]), num(hi[0]), num(lo[1]),
                          num(hi[1]), num(lo[2]), num(hi[2]));
    return lines.join('\n');
}

QString insertEmitFace(const QString &deckText, const QString &id,
                       const QString &mixture, const QStringList &faces)
{
    QStringList lines = deckText.split('\n');
    const QString newLine =
        QString("fix %1 emit/face %2 %3").arg(id, mixture, faces.join(' '));

    const int b = findCommand(lines, "boundary");
    const int box = findCommand(lines, "create_box");
    const int anchor = qMax(b, box);
    if (anchor >= 0) lines.insert(anchor + 1, newLine);
    else lines.push_back(newLine);
    return lines.join('\n');
}

} // namespace CaseModel
