/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Run-provenance model + HTML report builder.  See runarchive.h.
------------------------------------------------------------------------- */

#include "runarchive.h"

#include <QBuffer>
#include <QFileInfo>
#include <QImage>
#include <QPair>
#include <QJsonArray>
#include <QJsonObject>

namespace {

QString esc(const QString &s)
{
    QString o = s;
    o.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;");
    return o;
}

/// The image as a browser can show it: formats every browser decodes go in as
/// they are; anything else (SPARTA's dump image writes PPM) is converted to PNG,
/// since labelling raw PPM bytes "image/png" only produced broken-image icons.
/// Returns an empty pair when the bytes cannot be decoded at all.
QPair<QString, QByteArray> browserImage(const QString &path, const QByteArray &bytes)
{
    const QString ext = QFileInfo(path).suffix().toLower();
    if (ext == "png") return {"image/png", bytes};
    if (ext == "jpg" || ext == "jpeg") return {"image/jpeg", bytes};
    if (ext == "gif") return {"image/gif", bytes};
    if (ext == "bmp") return {"image/bmp", bytes};
    if (ext == "webp") return {"image/webp", bytes};
    const QImage img = QImage::fromData(bytes);
    if (img.isNull()) return {};
    QByteArray png;
    QBuffer buf(&png);
    buf.open(QIODevice::WriteOnly);
    img.save(&buf, "PNG");
    return {"image/png", png};
}

} // namespace

namespace RunArchive {

QPair<QString, QByteArray> imageForBrowser(const QString &path, const QByteArray &bytes)
{
    return browserImage(path, bytes);
}

QString runArchiveDir(const QString &base, const QString &id)
{
    return base + "/" + id;
}

QJsonObject RunRecord::toJson() const
{
    QJsonObject o;
    o["id"] = id;
    o["timestamp"] = timestamp;
    o["deckName"] = deckName;
    o["deckText"] = deckText;
    o["logText"] = logText;
    o["thermoYaml"] = thermoYaml;
    o["imageFiles"] = QJsonArray::fromStringList(imageFiles);
    o["workDir"] = workDir;
    o["status"] = status;
    QJsonObject meta;
    for (auto it = metadata.constBegin(); it != metadata.constEnd(); ++it)
        meta[it.key()] = it.value();
    o["metadata"] = meta;
    return o;
}

RunRecord RunRecord::fromJson(const QJsonObject &o)
{
    RunRecord r;
    r.id = o["id"].toString();
    r.timestamp = o["timestamp"].toString();
    r.deckName = o["deckName"].toString();
    r.deckText = o["deckText"].toString();
    r.logText = o["logText"].toString();
    r.thermoYaml = o["thermoYaml"].toString();
    for (const auto &v : o["imageFiles"].toArray()) r.imageFiles << v.toString();
    r.workDir = o["workDir"].toString();
    r.status = o["status"].toString();
    const QJsonObject meta = o["metadata"].toObject();
    for (auto it = meta.constBegin(); it != meta.constEnd(); ++it)
        r.metadata.insert(it.key(), it.value().toString());
    return r;
}

QString buildRunReportHtml(const RunRecord &rec, const QMap<QString, QByteArray> &imageData)
{
    QString h;
    h += "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">\n";
    h += "<title>SPARTA-GUI run report: " + esc(rec.deckName) + "</title>\n";
    h += "<style>body{font-family:sans-serif;margin:2em;max-width:60em}"
         "h1,h2{border-bottom:1px solid #ccc;padding-bottom:.2em}"
         "table.meta{border-collapse:collapse}table.meta td{border:1px solid #ccc;"
         "padding:.2em .6em}pre{background:#f5f5f5;border:1px solid #ddd;padding:.6em;"
         "overflow:auto}img{max-width:100%;border:1px solid #ccc;margin:.4em 0}</style>\n";
    h += "</head><body>\n";
    h += "<h1>SPARTA-GUI Run Report</h1>\n";

    // metadata table
    h += "<table class=\"meta\">\n";
    auto row = [&](const QString &k, const QString &v) {
        h += "<tr><td><b>" + esc(k) + "</b></td><td>" + esc(v) + "</td></tr>\n";
    };
    row("Input deck", rec.deckName);
    row("Run finished", rec.timestamp);
    row("Working directory", rec.workDir);
    row("Status", rec.status);
    for (auto it = rec.metadata.constBegin(); it != rec.metadata.constEnd(); ++it)
        row(it.key(), it.value());
    h += "</table>\n";

    // images (inlined)
    QStringList shown;
    for (const QString &p : rec.imageFiles) {
        if (!imageData.contains(p)) continue;
        const auto img = browserImage(p, imageData.value(p));
        if (img.second.isEmpty()) continue;
        shown << "<img alt=\"" + esc(QFileInfo(p).fileName()) + "\" src=\"data:" + img.first +
                     ";base64," + QString::fromLatin1(img.second.toBase64()) + "\">";
    }
    if (!shown.isEmpty()) {
        h += "<h2>Images</h2>\n";
        h += shown.join("\n") + "\n";
    }

    if (!rec.deckText.isEmpty()) {
        h += "<h2>Input Deck</h2>\n<pre>" + esc(rec.deckText) + "</pre>\n";
    }
    if (!rec.thermoYaml.isEmpty()) {
        h += "<h2>Thermodynamic Output</h2>\n<pre>" + esc(rec.thermoYaml) + "</pre>\n";
    }
    if (!rec.logText.isEmpty()) {
        h += "<h2>Log</h2>\n<pre>" + esc(rec.logText) + "</pre>\n";
    }

    h += "</body></html>\n";
    return h;
}

} // namespace RunArchive

// Local Variables:
// c-basic-offset: 4
// End:
