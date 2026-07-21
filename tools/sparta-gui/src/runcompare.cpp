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

#include "runcompare.h"

#include <QFileInfo>
#include <QSet>

namespace RunCompare {

QVector<DiffLine> diffLines(const QStringList &a, const QStringList &b)
{
    const int n = a.size(), m = b.size();

    // LCS length table
    QVector<QVector<int>> lcs(n + 1, QVector<int>(m + 1, 0));
    for (int i = n - 1; i >= 0; --i)
        for (int j = m - 1; j >= 0; --j)
            lcs[i][j] = (a[i] == b[j]) ? lcs[i + 1][j + 1] + 1
                                       : qMax(lcs[i + 1][j], lcs[i][j + 1]);

    // backtrack into a unified diff
    QVector<DiffLine> out;
    int i = 0, j = 0;
    while (i < n && j < m) {
        if (a[i] == b[j]) {
            out.push_back({Op::Context, a[i]});
            ++i;
            ++j;
        } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
            out.push_back({Op::Removed, a[i]});
            ++i;
        } else {
            out.push_back({Op::Added, b[j]});
            ++j;
        }
    }
    while (i < n) out.push_back({Op::Removed, a[i++]});
    while (j < m) out.push_back({Op::Added, b[j++]});
    return out;
}

QVector<DiffLine> diffText(const QString &a, const QString &b)
{
    return diffLines(a.split('\n'), b.split('\n'));
}

bool decksDiffer(const QString &a, const QString &b)
{
    for (const DiffLine &d : diffText(a, b))
        if (d.op != Op::Context) return true;
    return false;
}

QVector<MetaDelta> diffMetadata(const QMap<QString, QString> &a,
                                const QMap<QString, QString> &b)
{
    QSet<QString> keys;
    for (auto it = a.constBegin(); it != a.constEnd(); ++it) keys.insert(it.key());
    for (auto it = b.constBegin(); it != b.constEnd(); ++it) keys.insert(it.key());

    QStringList sorted(keys.constBegin(), keys.constEnd());
    sorted.sort();

    QVector<MetaDelta> out;
    for (const QString &k : sorted)
        out.push_back({k, a.value(k), b.value(k)});
    return out;
}

// ---------------------------------------------------------------------------

namespace {

QString esc(const QString &s)
{
    QString o = s;
    o.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
    return o;
}

// inline the first available image of a record as a base64 data: URI
QString firstImageTag(const RunArchive::RunRecord &rec, const QMap<QString, QByteArray> &data)
{
    for (const QString &p : rec.imageFiles) {
        if (!data.contains(p)) continue;
        const QByteArray b64 = data.value(p).toBase64();
        return QString("<img alt=\"%1\" style=\"max-width:100%%;border:1px solid #ccc\" "
                       "src=\"data:image/png;base64,%2\">")
            .arg(esc(QFileInfo(p).fileName()), QString::fromLatin1(b64));
    }
    return QString("<em>(no image)</em>");
}

} // namespace

QString buildComparisonHtml(const RunArchive::RunRecord &a, const RunArchive::RunRecord &b,
                            const QMap<QString, QByteArray> &imagesA,
                            const QMap<QString, QByteArray> &imagesB)
{
    QString h;
    h += "<html><head><meta charset=\"utf-8\"><style>"
         "body{font-family:sans-serif;margin:1.5em;color:#222}"
         "h1{font-size:1.4em}h2{font-size:1.1em;border-bottom:1px solid #ddd;padding-bottom:2px}"
         "table{border-collapse:collapse;margin:0.5em 0}td,th{border:1px solid #ccc;padding:3px 8px;text-align:left;vertical-align:top}"
         "pre{font-family:monospace;font-size:0.9em;line-height:1.35;white-space:pre-wrap;margin:0}"
         ".add{background:#e6ffed}.del{background:#ffeef0}.ctx{color:#666}"
         ".diffkey{background:#fff8c5}.imgs td{width:50%}"
         "</style></head><body>";

    h += "<h1>Run comparison</h1>";
    h += QString("<table><tr><th></th><th>A</th><th>B</th></tr>"
                 "<tr><td>id</td><td>%1</td><td>%2</td></tr>"
                 "<tr><td>deck</td><td>%3</td><td>%4</td></tr>"
                 "<tr><td>time</td><td>%5</td><td>%6</td></tr>"
                 "<tr><td>status</td><td>%7</td><td>%8</td></tr></table>")
             .arg(esc(a.id), esc(b.id), esc(a.deckName), esc(b.deckName),
                  esc(a.timestamp), esc(b.timestamp), esc(a.status), esc(b.status));

    // metadata delta
    h += "<h2>Provenance metadata</h2><table><tr><th>key</th><th>A</th><th>B</th></tr>";
    for (const MetaDelta &d : diffMetadata(a.metadata, b.metadata)) {
        const QString cls = d.differs() ? " class=\"diffkey\"" : "";
        h += QString("<tr%1><td>%2</td><td>%3</td><td>%4</td></tr>")
                 .arg(cls, esc(d.key), esc(d.valueA), esc(d.valueB));
    }
    h += "</table>";

    // deck diff
    h += "<h2>Input deck diff</h2>";
    if (!decksDiffer(a.deckText, b.deckText)) {
        h += "<p><em>The input decks are identical.</em></p>";
    } else {
        h += "<pre>";
        for (const DiffLine &d : diffText(a.deckText, b.deckText)) {
            const char *cls = d.op == Op::Added ? "add" : d.op == Op::Removed ? "del" : "ctx";
            const char sign = d.op == Op::Added ? '+' : d.op == Op::Removed ? '-' : ' ';
            h += QString("<span class=\"%1\">%2 %3</span>\n")
                     .arg(QLatin1String(cls))
                     .arg(sign)
                     .arg(esc(d.text));
        }
        h += "</pre>";
    }

    // side-by-side images
    h += "<h2>Images</h2><table class=\"imgs\"><tr><th>A</th><th>B</th></tr><tr><td>";
    h += firstImageTag(a, imagesA);
    h += "</td><td>";
    h += firstImageTag(b, imagesB);
    h += "</td></tr></table>";

    h += "</body></html>";
    return h;
}

} // namespace RunCompare
