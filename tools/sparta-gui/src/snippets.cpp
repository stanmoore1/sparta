/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Snippet library parsing.  See snippets.h.
------------------------------------------------------------------------- */

#include "snippets.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>

namespace Snippets {

QList<Snippet> parse(const QByteArray &json, QString *err)
{
    if (err) err->clear();
    QJsonParseError perr{};
    const QJsonDocument doc = QJsonDocument::fromJson(json, &perr);
    if (perr.error != QJsonParseError::NoError) {
        if (err) *err = perr.errorString();
        return {};
    }
    if (!doc.isArray()) {
        if (err) *err = "snippet document is not a JSON array";
        return {};
    }

    QList<Snippet> out;
    for (const auto &v : doc.array()) {
        const QJsonObject o = v.toObject();
        Snippet s;
        s.name = o["name"].toString();
        s.category = o["category"].toString("General");
        s.description = o["description"].toString();
        // body may be a single string or an array of lines
        const QJsonValue body = o["body"];
        if (body.isArray()) {
            QStringList lines;
            for (const auto &l : body.toArray()) lines << l.toString();
            s.body = lines.join('\n');
        } else {
            s.body = body.toString();
        }
        if (s.name.isEmpty() || s.body.isEmpty()) continue; // skip incomplete
        out << s;
    }
    return out;
}

QList<Snippet> builtin()
{
    QFile f(":/snippets.json");
    if (!f.open(QIODevice::ReadOnly)) return {};
    return parse(f.readAll());
}

} // namespace Snippets

// Local Variables:
// c-basic-offset: 4
// End:
