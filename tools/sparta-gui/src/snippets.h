/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Editor snippet library: named, categorized multi-line SPARTA command
   templates the user can insert into the editor.  The parsing/validation of
   the snippet JSON is GUI-free and unit-tested; the bundled snippets live in
   the Qt resource :/snippets.json and optional user snippets in the app data
   dir.
------------------------------------------------------------------------- */

#ifndef SNIPPETS_H
#define SNIPPETS_H

#include <QList>
#include <QString>

namespace Snippets {

/** @brief One insertable command-block template. */
struct Snippet {
    QString name;        ///< short label
    QString category;    ///< grouping (e.g. "Setup", "Output", "Ablation")
    QString description; ///< one-line explanation
    QString body;        ///< the multi-line text inserted at the cursor
};

/**
 * @brief Parse a snippet JSON document into a list of snippets.
 *
 * The document is a JSON array of objects with "name", "category",
 * "description", and "body" (an array of lines or a single string).  Entries
 * missing a name or body are skipped.  On a malformed document @p err is set
 * (when non-null) and an empty list is returned.
 */
QList<Snippet> parse(const QByteArray &json, QString *err = nullptr);

/** @brief The bundled snippets from the :/snippets.json resource. */
QList<Snippet> builtin();

} // namespace Snippets

#endif // SNIPPETS_H
