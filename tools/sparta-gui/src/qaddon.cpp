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

#include "qaddon.h"

#include <QColor>
#include <QComboBox>
#include <QLinearGradient>
#include <QPainter>
#include <QRect>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <algorithm>

namespace {
// clang-format off
const QStringList imagecolors = {
    "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black",
    "blanchedalmond", "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse",
    "chocolate", "coral", "cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
    "darkgoldenrod", "darkgray", "darkgreen", "darkkhaki", "darkmagenta", "darkolivegreen",
    "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue",
    "darkslategray", "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgray",
    "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro", "ghostwhite",
    "gold", "goldenrod", "gray", "green", "greenyellow", "honeydew", "hotpink", "indianred",
    "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon",
    "lightblue", "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgreen", "lightgrey",
    "lightpink", "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightsteelblue",
    "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine",
    "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue",
    "mediumspringgreen", "mediumturquoise", "mediumvioletred", "midnightblue", "mintcream",
    "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orange",
    "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred",
    "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple", "red", "rosybrown",
    "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna", "silver",
    "skyblue", "slateblue", "slategray", "snow", "springgreen", "steelblue", "tan", "teal",
    "thistle", "tomato", "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow",
    "yellowgreen"
};
// clang-format on
} // namespace

QHline::QHline(QWidget *parent) : QFrame(parent)
{
    setGeometry(QRect(0, 0, 100, 3));
    setFrameShape(QFrame::HLine);
    setFrameShadow(QFrame::Sunken);
}

QColorValidator::QColorValidator(QWidget *parent) : QValidator(parent) {}

void QColorValidator::fixup(QString &input) const
{
    // remove leading/trailing whitespace and make lowercase
    input = input.trimmed();
    input = input.toLower();
}

QValidator::State QColorValidator::validate(QString &input, int &) const
{
    QString match;

    // find if input string is contained in list of colors
    for (const auto &color : imagecolors) {
        if (color.startsWith(input)) {
            match = color;
            break;
        }
    }

    if (match == input) {
        return QValidator::Acceptable;
    } else if (match.size() > 0) {
        return QValidator::Intermediate;
    }
    return QValidator::Invalid;
}

// complete color inputs
QColorCompleter::QColorCompleter(QWidget *parent) : QCompleter(imagecolors, parent)
{
    setCompletionMode(QCompleter::InlineCompletion);
    setModelSorting(QCompleter::CaseInsensitivelySortedModel);
};

/* -------------------------------------------------------------------- */

VerticalLabel::VerticalLabel(const QString &text, QWidget *parent) : QWidget(parent), m_text(text)
{
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
}

void VerticalLabel::setText(const QString &text)
{
    m_text = text;
    updateGeometry();
    update();
}

QString VerticalLabel::text() const
{
    return m_text;
}

void VerticalLabel::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::TextAntialiasing);
    painter.setPen(palette().color(QPalette::WindowText));
    painter.setFont(font());
    painter.translate(0, height());
    painter.rotate(-90);
    QMargins m = contentsMargins();
    // In rotated coords (after translate+rotate-90): painter_x = widget_bottom-to-top,
    // painter_y = widget_left-to-right. So margins map: bottom→x_origin, left→y_origin.
    painter.drawText(QRect(m.bottom(), m.left(), height() - m.top() - m.bottom(),
                           width() - m.left() - m.right()),
                     Qt::AlignCenter, m_text);
}

QSize VerticalLabel::sizeHint() const
{
    QFontMetrics fm(font());
    QMargins m = contentsMargins();
    return QSize(fm.height() + 4 + m.left() + m.right(),
                 fm.horizontalAdvance(m_text) + 4 + m.top() + m.bottom());
}

QSize VerticalLabel::minimumSizeHint() const
{
    return sizeHint();
}

// Local Variables:
// c-basic-offset: 4
// End:

// ---------------------------------------------------------------------------
// Colour swatches and combo-box helpers.
//
// These live here rather than in imageviewer.cpp so that a translation unit
// needing a swatch does not have to link the image viewer, and through it the
// SPARTA wrapper and the main window.  The dump-image settings dialog is the
// case that matters: it wants swatches and nothing else from the viewer.
// ---------------------------------------------------------------------------

// 1) create a color gradient icon
QIcon gradient_icon(const QList<QPair<double, QColor>> &stops)
{
    if (stops.isEmpty()) return QIcon();

    // define pixmap and horizontal gradient
    QPixmap pixmap(ICON_SIZE, ICON_SIZE);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    QLinearGradient gradient(0, 0, ICON_SIZE, 0);

    // place each color at its stop position
    for (const auto &s : stops)
        gradient.setColorAt(std::clamp(s.first, 0.0, 1.0), s.second);

    painter.fillRect(pixmap.rect(), gradient);
    painter.end();

    return QIcon(pixmap);
}

// 2) create a color sequence icon
QIcon sequence_icon(const QList<QColor> &colors)
{
    // if no colors or too many colors return empty icon
    if (colors.isEmpty() || (colors.size() * 2 > ICON_SIZE)) return QIcon();

    // define pixmap
    QPixmap pixmap(ICON_SIZE, ICON_SIZE);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);

    // distribute colors across icon in evenly sized chunks
    const int chunk = ICON_SIZE / colors.size();
    for (int i = 0; i < colors.size(); ++i)
        painter.fillRect(QRect(i * chunk, 0, chunk, ICON_SIZE), colors[i]);

    painter.end();

    return QIcon(pixmap);
}

// 3) create a single color icon
QPixmap color_icon(const QColor &color)
{
    // define pixmap and fill with color
    QPixmap pixmap(ICON_SIZE / 2, ICON_SIZE / 2);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.fillRect(pixmap.rect(), color);
    painter.end();
    return pixmap;
}

// select the combo box entry matching the given text, if present (leave unchanged otherwise)
void selectComboItem(QComboBox *box, const QString &text)
{
    const int idx = box->findText(text);
    if (idx >= 0) box->setCurrentIndex(idx);
}
