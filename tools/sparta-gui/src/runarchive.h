/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Pure, GUI-free run-provenance model: a RunRecord describing one archived
   run (deck, data files, log, images, thermo, metadata) plus a builder that
   renders it into a self-contained HTML report (images inlined as base64
   data: URIs so the file is portable).  No Qt widgets, no SPARTA, so it is
   unit-tested in isolation like the other pure cores.
------------------------------------------------------------------------- */

#ifndef RUNARCHIVE_H
#define RUNARCHIVE_H

#include <QMap>
#include <QString>
#include <QStringList>

class QJsonObject;

namespace RunArchive {

/** @brief One archived run's metadata + artifact paths. */
struct RunRecord {
    QString id;                 ///< stable archive id (also the subdir name)
    QString timestamp;          ///< ISO-8601 time the run finished
    QString deckName;           ///< input-deck file name
    QString deckText;           ///< full input-deck contents (embedded)
    QString logText;            ///< captured log (embedded)
    QString thermoYaml;         ///< finalized thermo as SPARTA YAML (embedded)
    QStringList imageFiles;     ///< absolute paths to run images (for the report)
    QString workDir;            ///< directory the run executed in
    QString status;             ///< "ok" / "failed" / ...
    QMap<QString, QString> metadata; ///< arbitrary key/value provenance

    QJsonObject toJson() const;
    static RunRecord fromJson(const QJsonObject &o);
};

/**
 * @brief Render a run into a self-contained HTML report.
 *
 * The report embeds the metadata table, the input deck, the log, and each
 * image inlined as a base64 `data:` URI (using @p imageData keyed by the same
 * paths as @p rec.imageFiles) so the single HTML file is portable.  Images
 * whose data is missing from @p imageData are skipped.
 */
QString buildRunReportHtml(const RunRecord &rec,
                           const QMap<QString, QByteArray> &imageData);

/** @brief Archive subdirectory path for a run under @p base. */
QString runArchiveDir(const QString &base, const QString &id);

} // namespace RunArchive

#endif // RUNARCHIVE_H
