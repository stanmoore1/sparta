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

#include "theme.h"

#include "constants.h"

#include <QApplication>
#include <QColor>
#include <QFile>
#include <QGuiApplication>
#include <QIODevice>
#include <QPalette>
#include <QStyleHints>

namespace Theme {

namespace {

// The full set of colors that define one appearance. The neutral, slightly
// desaturated tones mirror a modern IDE look and keep the two variants
// visually related (only the light/dark axis flips).
struct ColorSet {
    QColor window, windowText, base, alternateBase, text, button, buttonText, brightText;
    QColor highlight, highlightedText, tooltipBase, tooltipText, link, linkVisited;
    QColor light, midlight, mid, dark, shadow;
    QColor disabledText, disabledButtonText, disabledWindowText, disabledHighlight;
};

const ColorSet &lightColors()
{
    static const ColorSet c = {
        QColor("#eff0f1"), QColor("#232629"),           // window, windowText
        QColor("#fcfcfc"), QColor("#e6e7e9"),           // base, alternateBase
        QColor("#232629"), QColor("#eff0f1"),           // text, button
        QColor("#232629"), QColor("#ffffff"),           // buttonText, brightText
        QColor("#3daee9"), QColor("#ffffff"),           // highlight, highlightedText
        QColor("#f7f7f7"), QColor("#232629"),           // tooltipBase, tooltipText
        QColor("#2980b9"), QColor("#7f8c8d"),           // link, linkVisited
        QColor("#ffffff"), QColor("#f4f5f5"),           // light, midlight
        QColor("#c4c8cc"), QColor("#9ca0a4"),           // mid, dark
        QColor("#767a7e"),                              // shadow
        QColor("#9fa3a7"), QColor("#9fa3a7"),           // disabledText, disabledButtonText
        QColor("#9fa3a7"), QColor("#c8cbce"),           // disabledWindowText, disabledHighlight
    };
    return c;
}

const ColorSet &darkColors()
{
    static const ColorSet c = {
        QColor("#31363b"), QColor("#eff0f1"),           // window, windowText
        QColor("#232629"), QColor("#2a2e32"),           // base, alternateBase
        QColor("#eff0f1"), QColor("#31363b"),           // text, button
        QColor("#eff0f1"), QColor("#ffffff"),           // buttonText, brightText
        QColor("#3daee9"), QColor("#ffffff"),           // highlight, highlightedText
        QColor("#31363b"), QColor("#eff0f1"),           // tooltipBase, tooltipText
        QColor("#3daee9"), QColor("#8e44ad"),           // link, linkVisited
        QColor("#40454b"), QColor("#3b4046"),           // light, midlight
        QColor("#2c3034"), QColor("#212528"),           // mid, dark
        QColor("#16191c"),                              // shadow
        QColor("#6e7377"), QColor("#6e7377"),           // disabledText, disabledButtonText
        QColor("#6e7377"), QColor("#3b4046"),           // disabledWindowText, disabledHighlight
    };
    return c;
}

QPalette buildPalette(const ColorSet &c)
{
    QPalette p;
    // Active + Inactive groups share the same colors (Fusion dims inactive
    // highlights on its own where it matters).
    for (auto group : {QPalette::Active, QPalette::Inactive}) {
        p.setColor(group, QPalette::Window, c.window);
        p.setColor(group, QPalette::WindowText, c.windowText);
        p.setColor(group, QPalette::Base, c.base);
        p.setColor(group, QPalette::AlternateBase, c.alternateBase);
        p.setColor(group, QPalette::Text, c.text);
        p.setColor(group, QPalette::Button, c.button);
        p.setColor(group, QPalette::ButtonText, c.buttonText);
        p.setColor(group, QPalette::BrightText, c.brightText);
        p.setColor(group, QPalette::Highlight, c.highlight);
        p.setColor(group, QPalette::HighlightedText, c.highlightedText);
        p.setColor(group, QPalette::ToolTipBase, c.tooltipBase);
        p.setColor(group, QPalette::ToolTipText, c.tooltipText);
        p.setColor(group, QPalette::Link, c.link);
        p.setColor(group, QPalette::LinkVisited, c.linkVisited);
        p.setColor(group, QPalette::PlaceholderText, QColor(c.text.red(), c.text.green(),
                                                            c.text.blue(), 128));
        p.setColor(group, QPalette::Light, c.light);
        p.setColor(group, QPalette::Midlight, c.midlight);
        p.setColor(group, QPalette::Mid, c.mid);
        p.setColor(group, QPalette::Dark, c.dark);
        p.setColor(group, QPalette::Shadow, c.shadow);
    }
    // Disabled group: dimmed text so greyed-out controls stay legible but read
    // clearly as disabled.
    p.setColor(QPalette::Disabled, QPalette::WindowText, c.disabledWindowText);
    p.setColor(QPalette::Disabled, QPalette::Text, c.disabledText);
    p.setColor(QPalette::Disabled, QPalette::ButtonText, c.disabledButtonText);
    p.setColor(QPalette::Disabled, QPalette::Highlight, c.disabledHighlight);
    p.setColor(QPalette::Disabled, QPalette::HighlightedText, c.windowText);
    p.setColor(QPalette::Disabled, QPalette::Base, c.window);
    p.setColor(QPalette::Disabled, QPalette::Window, c.window);
    p.setColor(QPalette::Disabled, QPalette::Button, c.button);
    return p;
}

QString loadStyleSheet()
{
    QFile f(Cfg::STYLE_QSS);
    if (f.open(QIODevice::ReadOnly | QIODevice::Text)) return QString::fromUtf8(f.readAll());
    return QString();
}

} // namespace

Mode modeFromString(const QString &name)
{
    const QString n = name.trimmed().toLower();
    if (n == QLatin1String("light")) return Mode::Light;
    if (n == QLatin1String("dark")) return Mode::Dark;
    return Mode::System;
}

QString modeToString(Mode mode)
{
    switch (mode) {
        case Mode::Light: return QStringLiteral("light");
        case Mode::Dark: return QStringLiteral("dark");
        default: return QStringLiteral("system");
    }
}

bool osPrefersDark()
{
#if QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
    if (auto *hints = QGuiApplication::styleHints())
        return hints->colorScheme() == Qt::ColorScheme::Dark;
#endif
    // Fallback for Qt < 6.5: inspect the current palette. This is only correct
    // before the Fusion palette is installed, hence apply() takes the value
    // captured earlier in main() instead of calling this after setPalette().
    const QPalette p = QApplication::palette();
    return p.color(QPalette::Window).lightness() < p.color(QPalette::WindowText).lightness();
}

void apply(Mode mode, bool osDark)
{
    const bool dark = (mode == Mode::Dark) || (mode == Mode::System && osDark);
    qApp->setPalette(buildPalette(dark ? darkColors() : lightColors()));
    qApp->setStyleSheet(loadStyleSheet());
}

} // namespace Theme

// Local Variables:
// c-basic-offset: 4
// End:
