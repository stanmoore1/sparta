// Unit tests for the dump-image colour-map table (src/colormaps.cpp).
//
// This table is the single source of truth shared by two things that have to
// agree: the command builder, which turns a map into "dump_modify ... cmap"
// arguments SPARTA renders from, and the settings dialog, which draws a swatch
// of the same map for the user to pick. A malformed entry does not fail
// loudly -- it produces a command SPARTA accepts and renders differently from
// the swatch that was clicked, which is only discoverable by looking at a
// finished picture and disbelieving it.
//
// So what is checked here is the shape every consumer relies on: stops in
// order, positions spanning the full range, colours that are either a name
// SPARTA knows or component values in range, and no two maps that are secretly
// the same map.

#include "colormaps.h"

#include <QColor>
#include <QSet>

#include "gtest/gtest.h"

namespace {

/// The colour a stop resolves to, by name or by components.
QColor resolve(const ColorMapStop &s)
{
    if (!s.name.isEmpty()) return QColor(s.name);
    return QColor::fromRgbF(s.r, s.g, s.b);
}

TEST(ColorMaps, ThereAreMapsToChooseFrom)
{
    EXPECT_FALSE(colorMapNames().isEmpty());
}

TEST(ColorMaps, NamesAreUniqueAndPresentable)
{
    QSet<QString> seen;
    for (const QString &n : colorMapNames()) {
        EXPECT_FALSE(n.isEmpty());
        EXPECT_FALSE(seen.contains(n)) << "'" << n.toStdString()
                                       << "' appears twice, so the combo has two entries that "
                                          "select the same thing";
        seen.insert(n);
    }
}

TEST(ColorMaps, EveryNamedMapHasADefinition)
{
    for (const QString &n : colorMapNames()) {
        const ColorMapDef &def = colorMapDef(n);
        EXPECT_GE(def.stops.size(), 2)
            << "'" << n.toStdString() << "' has fewer than two stops; a map has to go from "
            << "somewhere to somewhere";
    }
}

TEST(ColorMaps, StopsAreOrderedAndSpanTheWholeRange)
{
    // The first stop is the minimum of the data range and the last is the
    // maximum. Positions out of order, or a map that stops short of 1, leave
    // part of the data mapped to nothing in SPARTA and to the wrong end of
    // the swatch in the dialog.
    for (const QString &n : colorMapNames()) {
        const ColorMapDef &def = colorMapDef(n);
        if (!def.continuous) continue; // positions are not meaningful for a discrete sequence
        ASSERT_GE(def.stops.size(), 2) << n.toStdString();

        EXPECT_NEAR(def.stops.first().pos, 0.0, 1.0e-9) << n.toStdString();
        EXPECT_NEAR(def.stops.last().pos, 1.0, 1.0e-9) << n.toStdString();
        for (int i = 1; i < def.stops.size(); ++i)
            EXPECT_LT(def.stops[i - 1].pos, def.stops[i].pos)
                << "'" << n.toStdString() << "' stop " << i << " is not after the one before it";
    }
}

TEST(ColorMaps, EveryColorResolves)
{
    // A stop names a SPARTA colour or gives components. A name QColor cannot
    // resolve is a name SPARTA will not resolve either, and the swatch would
    // draw it black while the render refuses the command.
    for (const QString &n : colorMapNames()) {
        for (const ColorMapStop &s : colorMapDef(n).stops) {
            if (!s.name.isEmpty()) {
                EXPECT_TRUE(QColor::isValidColorName(s.name))
                    << "'" << n.toStdString() << "' uses the unknown colour name '"
                    << s.name.toStdString() << "'";
            } else {
                EXPECT_GE(s.r, 0.0) << n.toStdString();
                EXPECT_LE(s.r, 1.0) << n.toStdString();
                EXPECT_GE(s.g, 0.0) << n.toStdString();
                EXPECT_LE(s.g, 1.0) << n.toStdString();
                EXPECT_GE(s.b, 0.0) << n.toStdString();
                EXPECT_LE(s.b, 1.0) << n.toStdString();
            }
            EXPECT_TRUE(resolve(s).isValid()) << n.toStdString();
        }
    }
}

TEST(ColorMaps, MapsAreVisiblyDifferentFromOneAnother)
{
    // two entries with the same stops are one map offered twice
    QSet<QString> signatures;
    for (const QString &n : colorMapNames()) {
        QString sig;
        for (const ColorMapStop &s : colorMapDef(n).stops)
            sig += QString::number(s.pos, 'f', 4) + resolve(s).name() + ";";
        EXPECT_FALSE(signatures.contains(sig))
            << "'" << n.toStdString() << "' has the same stops as an earlier map";
        signatures.insert(sig);
    }
}

TEST(ColorMaps, EndsOfAContinuousMapDiffer)
{
    // a map whose first and last colour are the same shows no gradient at all
    for (const QString &n : colorMapNames()) {
        const ColorMapDef &def = colorMapDef(n);
        if (!def.continuous || def.stops.size() < 2) continue;
        const QColor lo = resolve(def.stops.first());
        const QColor hi = resolve(def.stops.last());
        EXPECT_NE(lo.name(), hi.name())
            << "'" << n.toStdString() << "' starts and ends on the same colour";
    }
}

TEST(ColorMaps, AnUnknownNameFallsBackInsteadOfReturningNothing)
{
    // the map name is stored in the settings; an old or hand-edited value must
    // not leave the dialog with an empty definition to draw
    const ColorMapDef &def = colorMapDef(QStringLiteral("no-such-map"));
    EXPECT_GE(def.stops.size(), 2);
    // documented to fall back to BWR
    const ColorMapDef &bwr = colorMapDef(QStringLiteral("BWR"));
    EXPECT_EQ(def.stops.size(), bwr.stops.size());
    EXPECT_EQ(def.continuous, bwr.continuous);
}

TEST(ColorMaps, LookupIsStableAcrossCalls)
{
    // both consumers hold on to the reference they are given
    const ColorMapDef &a = colorMapDef(colorMapNames().first());
    const ColorMapDef &b = colorMapDef(colorMapNames().first());
    EXPECT_EQ(&a, &b) << "each lookup builds a new definition, so a held reference dangles";
}

} // namespace
