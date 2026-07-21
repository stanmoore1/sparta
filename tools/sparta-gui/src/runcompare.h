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

#ifndef RUNCOMPARE_H
#define RUNCOMPARE_H

// Pure, GUI-free comparison of two archived runs: a longest-common-subsequence
// line diff of their input decks, a delta of their provenance metadata bags, and
// a self-contained HTML report combining both (plus optional side-by-side
// images).  No Qt widgets, no SPARTA, so the diff engine is unit-tested in
// isolation like the other pure cores.

#include "runarchive.h"

#include <QByteArray>
#include <QMap>
#include <QString>
#include <QStringList>
#include <QVector>

namespace RunCompare {

/** @brief One line of a unified diff. */
enum class Op { Context, Added, Removed };
struct DiffLine {
    Op op = Op::Context;
    QString text;
};

/** @brief LCS-based line diff of @p a vs @p b (added = in b only, removed = in a only). */
QVector<DiffLine> diffLines(const QStringList &a, const QStringList &b);

/** @brief Convenience: diff two multi-line strings (split on '\n'). */
QVector<DiffLine> diffText(const QString &a, const QString &b);

/** @brief True if the two decks differ in any line. */
bool decksDiffer(const QString &a, const QString &b);

/** @brief One metadata key's values in each run (empty string = key absent). */
struct MetaDelta {
    QString key;
    QString valueA;
    QString valueB;
    bool differs() const { return valueA != valueB; }
};

/** @brief Union of both metadata bags, sorted by key, with each run's value. */
QVector<MetaDelta> diffMetadata(const QMap<QString, QString> &a,
                                const QMap<QString, QString> &b);

/**
 * @brief Build a self-contained HTML comparison of two runs.
 * @param a,b        the two run records
 * @param imagesA,b  optional image bytes keyed by the records' image paths
 *                   (inlined as base64 so the report is portable)
 */
QString buildComparisonHtml(const RunArchive::RunRecord &a, const RunArchive::RunRecord &b,
                            const QMap<QString, QByteArray> &imagesA = {},
                            const QMap<QString, QByteArray> &imagesB = {});

} // namespace RunCompare

#endif // RUNCOMPARE_H

// Local Variables:
// c-basic-offset: 4
// End:
