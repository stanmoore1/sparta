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

// The chart window's four modal dialogs: chart style, postprocess analysis,
// the Birch-Murnaghan column setup and its result, and the reference lines.
//
// All four used to be built on the stack inside ChartWindow methods, reading a
// live ChartViewer as they went and writing back to it after exec() returned.
// That made them unreachable without a chart with data in it -- and left the
// only checked thing about them the fact that constructing them did not crash.
// Here each is a pure function of the plain struct it is handed.

#include "chartdialogs.h"

#include "constants.h"
#include "helpers.h"

#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPixmap>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QVBoxLayout>

#include <cmath>

namespace {

// A push button showing a colour and editing it in place.  The colour lives in
// the caller, because the picker is modal and the button has to survive it.
template <class Setter> QPushButton *colorButton(const QColor &initial, Setter &&apply)
{
    auto *btn      = new QPushButton;
    auto paint     = [btn](const QColor &c) {
        btn->setText(c.name());
        btn->setStyleSheet(QString("background-color: %1; color: %2;")
                               .arg(c.name(), (c.lightness() < 128) ? "white" : "black"));
    };
    paint(initial);
    QObject::connect(btn, &QPushButton::clicked, btn, [btn, paint, apply]() mutable {
        const QColor c = QColorDialog::getColor(QColor(btn->text()), btn, "Series Color");
        if (c.isValid()) {
            apply(c);
            paint(c);
        }
    });
    return btn;
}

// A display-mode selector preset to the given mode.
QComboBox *modeBox(ChartDisplayMode mode)
{
    auto *mb = new QComboBox;
    mb->addItem("Lines", static_cast<int>(ChartDisplayMode::Lines));
    mb->addItem("Points", static_cast<int>(ChartDisplayMode::Points));
    mb->addItem("Lines + Points", static_cast<int>(ChartDisplayMode::LinesAndPoints));
    mb->setCurrentIndex(static_cast<int>(mode));
    return mb;
}

// A line-width spin box preset to the given width.
QDoubleSpinBox *widthBox(qreal width)
{
    auto *w = new QDoubleSpinBox;
    w->setRange(0.5, 20.0);
    w->setSingleStep(0.5);
    w->setValue(width);
    return w;
}

// A point-diameter spin box preset to the given size.
QDoubleSpinBox *pointBox(qreal size)
{
    auto *w = new QDoubleSpinBox;
    w->setRange(1.0, 40.0);
    w->setSingleStep(1.0);
    w->setValue(size);
    return w;
}

// Ok/Cancel wired to the dialog, styled the way the rest of the app is.
QDialogButtonBox *okCancel(QDialog *dialog)
{
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    styleDialogButtons(buttons);
    QObject::connect(buttons, &QDialogButtonBox::accepted, dialog, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, dialog, &QDialog::reject);
    return buttons;
}

} // namespace

/* -------------------------------------------------------------------- */
/*  Chart style                                                          */
/* -------------------------------------------------------------------- */

ChartStyleDialog::ChartStyleDialog(const ChartStyle &initial, QWidget *parent) :
    QDialog(parent), rawColor(initial.rawColor), procColor(initial.procColor)
{
    setWindowTitle("Chart Style");
    auto *layout = new QVBoxLayout(this);

    // an unset series colour means "the theme default"; the dialog has to show
    // something, and these are what the chart draws in that case
    if (!rawColor.isValid()) rawColor = QColor(100, 150, 255);
    if (!procColor.isValid()) procColor = QColor(255, 125, 125);

    // raw data section
    rawMode  = modeBox(initial.rawMode);
    rawWidth = widthBox(initial.rawWidth);
    rawPoint = pointBox(initial.rawPointSize);
    auto *rawColorBtn =
        colorButton(rawColor, [this](const QColor &c) { rawColor = c; });
    auto *rawBox  = new QGroupBox("Raw data");
    auto *rawForm = new QFormLayout(rawBox);
    rawForm->addRow("Display:", rawMode);
    rawForm->addRow("Color:", rawColorBtn);
    rawForm->addRow("Line width:", rawWidth);
    rawForm->addRow("Point size:", rawPoint);
    layout->addWidget(rawBox);

    // processed data section
    procMode  = modeBox(initial.procMode);
    procWidth = widthBox(initial.procWidth);
    procPoint = pointBox(initial.procPointSize);
    auto *procColorBtn =
        colorButton(procColor, [this](const QColor &c) { procColor = c; });
    auto *procBox  = new QGroupBox("Processed data");
    auto *procForm = new QFormLayout(procBox);
    procForm->addRow("Display:", procMode);
    procForm->addRow("Color:", procColorBtn);
    procForm->addRow("Line width:", procWidth);
    procForm->addRow("Point size:", procPoint);
    layout->addWidget(procBox);

    // in-plot legend section
    legendBox = new QComboBox;
    legendBox->addItem("Off", static_cast<int>(LegendPos::Off));
    legendBox->addItem("Top left", static_cast<int>(LegendPos::TopLeft));
    legendBox->addItem("Top right", static_cast<int>(LegendPos::TopRight));
    legendBox->addItem("Bottom right", static_cast<int>(LegendPos::BottomRight));
    legendBox->addItem("Bottom left", static_cast<int>(LegendPos::BottomLeft));
    const int legendIdx = legendBox->findData(static_cast<int>(initial.legend));
    legendBox->setCurrentIndex(legendIdx < 0 ? 0 : legendIdx);
    auto *legendGroup = new QGroupBox("Legend");
    auto *legendForm  = new QFormLayout(legendGroup);
    legendForm->addRow("Placement:", legendBox);
    layout->addWidget(legendGroup);

    layout->addWidget(okCancel(this));
}

ChartStyle ChartStyleDialog::style() const
{
    ChartStyle s;
    s.rawMode       = static_cast<ChartDisplayMode>(rawMode->currentData().toInt());
    s.rawColor      = rawColor;
    s.rawWidth      = rawWidth->value();
    s.rawPointSize  = rawPoint->value();
    s.procMode      = static_cast<ChartDisplayMode>(procMode->currentData().toInt());
    s.procColor     = procColor;
    s.procWidth     = procWidth->value();
    s.procPointSize = procPoint->value();
    s.legend        = static_cast<LegendPos>(legendBox->currentData().toInt());
    return s;
}

/* -------------------------------------------------------------------- */
/*  Postprocess                                                          */
/* -------------------------------------------------------------------- */

PostProcessDialog::PostProcessDialog(int npoints, double dataXmin, double dataXmax,
                                     QWidget *parent) :
    QDialog(parent), m_npoints(npoints)
{
    setWindowTitle("Postprocess Chart Data");
    auto *form = new QFormLayout(this);

    analysisBox = new QComboBox;
    analysisBox->addItem("Autocorrelation");
    analysisBox->addItem("Polynomial fit");
    analysisBox->addItem("Birch-Murnaghan EOS fit");
    analysisBox->addItem("Custom function");
    analysisBox->addItem("Custom fit");
    analysisBox->addItem("Block-average uncertainty");
    analysisBox->addItem("Steady-state detection");
    form->addRow("Analysis:", analysisBox);

    paramLabel = new QLabel;
    paramSpin  = new QSpinBox;
    form->addRow(paramLabel, paramSpin);

    // expression field, shown for both the custom-function plot and fit
    exprLabel = new QLabel("f(x) =");
    exprEdit  = new QLineEdit;
    exprEdit->setPlaceholderText("e.g. 2*x^2 + 3*sin(x)");
    exprEdit->setMinimumWidth(Cfg::POSTPROCESS_EXPR_WIDTH);
    form->addRow(exprLabel, exprEdit);

    // parameter (initial-guess) and label fields, shown only for the custom fit
    paramsLabel = new QLabel("Parameters:");
    paramsEdit  = new QLineEdit;
    paramsEdit->setPlaceholderText("name=guess, e.g. a=1, b=0.5");
    paramsEdit->setMinimumWidth(Cfg::POSTPROCESS_EXPR_WIDTH);
    form->addRow(paramsLabel, paramsEdit);

    fitLabelLabel = new QLabel("Label:");
    fitLabelEdit  = new QLineEdit;
    fitLabelEdit->setPlaceholderText("optional name for the fitted curve");
    fitLabelEdit->setMinimumWidth(Cfg::POSTPROCESS_EXPR_WIDTH);
    form->addRow(fitLabelLabel, fitLabelEdit);

    // fit x-range (hidden for autocorrelation, shown for all fitting analyses)
    rangeLabel        = new QLabel("Fit x-range:");
    rangeWidget       = new QWidget;
    auto *fitRangeRow = new QHBoxLayout(rangeWidget);
    fitRangeRow->setContentsMargins(0, 0, 0, 0);
    fromSpin = new QDoubleSpinBox;
    fromSpin->setDecimals(6);
    fromSpin->setRange(-1e15, 1e15);
    fromSpin->setValue(dataXmin);
    toSpin = new QDoubleSpinBox;
    toSpin->setDecimals(6);
    toSpin->setRange(-1e15, 1e15);
    toSpin->setValue(dataXmax);
    fitRangeRow->addWidget(new QLabel("from"));
    fitRangeRow->addWidget(fromSpin, 1);
    fitRangeRow->addWidget(new QLabel("to"));
    fitRangeRow->addWidget(toSpin, 1);
    form->addRow(rangeLabel, rangeWidget);

    configure(0);
    connect(analysisBox, &QComboBox::currentIndexChanged, this, &PostProcessDialog::configure);

    form->addRow(okCancel(this));
}

// Show the controls this analysis needs and hide the rest.  The parameter
// limits follow the point count: a polynomial cannot have more coefficients
// than there are points to fit it with, and a lag cannot reach past the series.
void PostProcessDialog::configure(int idx)
{
    const bool plot      = (idx == PostProcessSpec::CustomFunction);
    const bool fit       = (idx == PostProcessSpec::CustomFit);
    const bool expr      = plot || fit;
    const bool eos       = (idx == PostProcessSpec::Eos);
    const bool block     = (idx == PostProcessSpec::BlockAverage);
    const bool steady    = (idx == PostProcessSpec::SteadyState);
    const bool showRange = (idx >= PostProcessSpec::Polynomial && idx <= PostProcessSpec::CustomFit);

    exprLabel->setVisible(expr);
    exprEdit->setVisible(expr);
    paramsLabel->setVisible(fit);
    paramsEdit->setVisible(fit);
    fitLabelLabel->setVisible(fit);
    fitLabelEdit->setVisible(fit);
    rangeLabel->setVisible(showRange);
    rangeWidget->setVisible(showRange);
    paramLabel->setVisible(idx == PostProcessSpec::Autocorrelation ||
                           idx == PostProcessSpec::Polynomial || block);

    if (idx == PostProcessSpec::Polynomial) { // polynomial degree
        paramLabel->setText("Degree:");
        paramSpin->setVisible(true);
        paramSpin->setRange(1, qMin(m_npoints - 1, 8));
        paramSpin->setValue(qMin(3, qMin(m_npoints - 1, 8)));
    } else if (block) { // number of blocks for batch-means averaging
        paramLabel->setText("Blocks:");
        paramSpin->setVisible(true);
        paramSpin->setRange(2, qMax(2, m_npoints / 2));
        paramSpin->setValue(
            qBound(2, int(std::sqrt(double(m_npoints))), qMax(2, m_npoints / 2)));
    } else if (eos || expr || steady) { // no scalar parameter needed
        paramSpin->setVisible(false);
    } else { // autocorrelation max lag
        paramLabel->setText("Max lag:");
        paramSpin->setVisible(true);
        paramSpin->setRange(1, m_npoints - 1);
        paramSpin->setValue(qMin(m_npoints - 1, m_npoints / 2));
    }
    adjustSize();
}

PostProcessSpec PostProcessDialog::spec() const
{
    PostProcessSpec s;
    s.analysis   = static_cast<PostProcessSpec::Analysis>(analysisBox->currentIndex());
    s.param      = paramSpin->value();
    s.expression = exprEdit->text().trimmed();
    s.parameters = paramsEdit->text();
    s.label      = fitLabelEdit->text().trimmed();
    s.fitFrom    = fromSpin->value();
    s.fitTo      = toSpin->value();
    return s;
}

/* -------------------------------------------------------------------- */
/*  Birch-Murnaghan column setup                                         */
/* -------------------------------------------------------------------- */

EosSetupDialog::EosSetupDialog(const QString &xLabel, const QString &yLabel, QWidget *parent) :
    QDialog(parent)
{
    setWindowTitle("Birch-Murnaghan EOS Fit — Column Setup");
    auto *eosLayout = new QVBoxLayout(this);
    eosLayout->addWidget(new QLabel(
        "The Birch-Murnaghan EOS fit expects volume on the x-axis and cohesive energy "
        "on the y-axis.\n\nThis chart has:"));
    auto *eosInfo = new QFormLayout;
    eosInfo->addRow("x-axis:", new QLabel("<b>" + xLabel + "</b>"));
    eosInfo->addRow("y-axis:", new QLabel("<b>" + yLabel + "</b>"));
    eosLayout->addLayout(eosInfo);
    eosLayout->addWidget(
        new QLabel("\nAtoms per unit cell N: the lattice constant is derived as\n"
                   "  a₀ = ∛(N × V₀)\n"
                   "Use the conventional unit cell (e.g. N=4 for FCC, N=2 for BCC/HCP).\n"
                   "Set N=1 only when the x-axis is already the conventional cell volume."));
    natSpin = new QSpinBox;
    natSpin->setRange(1, 1000);
    natSpin->setValue(1);
    natSpin->setToolTip("Number of atoms in the conventional unit cell\n"
                        "(e.g. 4 for FCC, 2 for BCC/HCP).\n"
                        "Use N=1 when x is already the conventional cell volume.");
    auto *natForm = new QFormLayout;
    natForm->addRow("Atoms per unit cell N:", natSpin);
    eosLayout->addLayout(natForm);
    eosLayout->addWidget(okCancel(this));
}

int EosSetupDialog::atomsPerCell() const
{
    return natSpin->value();
}

/* -------------------------------------------------------------------- */
/*  Birch-Murnaghan result                                               */
/* -------------------------------------------------------------------- */

double EosResultDialog::latticeConstant(double v0, int atomsPerCell)
{
    return std::cbrt(static_cast<double>(atomsPerCell) * v0);
}

EosResultDialog::EosResultDialog(const EosFit &fit, int atomsPerCell, QWidget *parent) :
    QDialog(parent)
{
    setWindowTitle("Birch-Murnaghan EOS Fit");
    auto *dlgLayout = new QVBoxLayout(this);

    auto *fmtLabel = new QLabel;
    fmtLabel->setPixmap(QPixmap(":/icons/birch-murnaghan-eos.png"));
    fmtLabel->setAlignment(Qt::AlignCenter);
    dlgLayout->addWidget(fmtLabel);

    auto *legend = new QLabel("where <i>V</i> is the unit cell volume "
                              "and <i>V</i><sub>0</sub> the equilibrium volume.");
    legend->setAlignment(Qt::AlignCenter);
    dlgLayout->addWidget(legend);

    auto *resultForm = new QFormLayout;
    auto makeVal     = [](double v, int prec, const QString &name) {
        auto *l = new QLabel(QString::number(v, 'g', prec));
        l->setTextInteractionFlags(Qt::TextSelectableByMouse);
        return l;
    };
    resultForm->addRow("<b>V<sub>0</sub></b> &mdash; Equilibrium volume (from fit):",
                       makeVal(fit.v0, 8, "v0"));
    resultForm->addRow(
        QString("<b>a<sub>0</sub></b> &mdash; Lattice constant ∛(%1 &times; V<sub>0</sub>):")
            .arg(atomsPerCell),
        makeVal(latticeConstant(fit.v0, atomsPerCell), 8, "a0"));
    resultForm->addRow("<b>E<sub>0</sub></b> &mdash; Cohesive energy at V<sub>0</sub>:",
                       makeVal(fit.e0, 8, "e0"));
    resultForm->addRow("<b>B<sub>0</sub></b> &mdash; Bulk modulus (&minus;V<sub>0</sub> dP/dV "
                       "at V<sub>0</sub>):",
                       makeVal(fit.b0, 8, "b0"));
    resultForm->addRow("<b>B<sub>0</sub>'</b> &mdash; Pressure derivative dB/dP at P=0:",
                       makeVal(fit.b0prime, 6, "b0prime"));
    resultForm->addRow("RMS residual:", makeVal(fit.rms, 6, "rms"));
    dlgLayout->addLayout(resultForm);

    auto *closeBtn = new QDialogButtonBox(QDialogButtonBox::Ok);
    styleDialogButtons(closeBtn);
    connect(closeBtn, &QDialogButtonBox::accepted, this, &QDialog::accept);
    dlgLayout->addWidget(closeBtn);
}

/* -------------------------------------------------------------------- */
/*  Reference lines                                                      */
/* -------------------------------------------------------------------- */

RefLinesDialog::RefLinesDialog(const QList<RefLine> &lines, const RefLineStyle &style,
                               QWidget *parent) :
    QDialog(parent)
{
    setWindowTitle("Reference Lines");
    setMinimumWidth(680); // room for the label field plus the anchor selector
    auto *layout = new QVBoxLayout(this);
    layout->addWidget(
        new QLabel("Reference lines (vertical at an x value or horizontal at a y value) are\n"
                   "applied to every chart. Labels are drawn next to the line."));

    // scrollable list of rows
    auto *listWidget = new QWidget;
    listLayout       = new QVBoxLayout(listWidget);
    listLayout->setContentsMargins(4, 4, 4, 4);

    auto *scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setWidget(listWidget);
    scroll->setMinimumHeight(100);
    // keep rows within the viewport width; only scroll vertically as lines are added
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    layout->addWidget(scroll, 1);

    for (const auto &rl : lines)
        appendRow(rl);

    auto *addBtn = new QPushButton("Add line");
    connect(addBtn, &QPushButton::clicked, this, [this]() { addLine(); });
    layout->addWidget(addBtn);

    // window-wide label style: font size, gap from the line, and a framed background
    auto *styleRow = new QHBoxLayout;
    fontSpin       = new QDoubleSpinBox;
    fontSpin->setRange(5.0, 30.0);
    fontSpin->setSingleStep(0.5);
    fontSpin->setValue(style.fontSize);
    distSpin = new QSpinBox;
    distSpin->setRange(0, 50);
    distSpin->setValue(static_cast<int>(style.gap));
    boxedCheck = new QCheckBox("Boxed labels");
    boxedCheck->setChecked(style.boxed);
    styleRow->addWidget(new QLabel("Label font:"));
    styleRow->addWidget(fontSpin);
    styleRow->addWidget(new QLabel("Gap:"));
    styleRow->addWidget(distSpin);
    styleRow->addWidget(boxedCheck);
    styleRow->addStretch(1);
    layout->addLayout(styleRow);

    layout->addWidget(okCancel(this));
}

// Build one row and append it.  The row's widgets go into a holder rather than
// straight into the list layout, so removing a row is one delete rather than a
// walk over a layout that has already been detached -- the old code deleted the
// layout's items by hand and left the row struct leaking on the Cancel path.
void RefLinesDialog::appendRow(const RefLine &line)
{
    auto *rd    = new Row;
    const int n = rows.size();

    rd->holder = new QWidget;
    auto *row  = new QHBoxLayout(rd->holder);
    row->setContentsMargins(0, 0, 0, 0);

    rd->orient = new QComboBox;
    rd->orient->addItems({"Vertical", "Horizontal"});
    rd->orient->setCurrentIndex(line.orient == RefOrient::Horizontal ? 1 : 0);

    rd->value = new QDoubleSpinBox;
    rd->value->setDecimals(6);
    rd->value->setRange(-1e15, 1e15);
    rd->value->setValue(line.value);
    // keep the value field compact so the label field has room
    rd->value->setMaximumWidth(110);

    rd->label = new QLineEdit(line.label);
    rd->label->setPlaceholderText("label");
    rd->color = line.color.isValid() ? line.color : QColor(80, 80, 80);

    // position label tracks the orientation: "x =" for vertical, "y =" for horizontal
    auto *posLabel = new QLabel;
    auto updatePos = [posLabel](int idx) { posLabel->setText(idx == 1 ? "y =" : "x ="); };
    updatePos(rd->orient->currentIndex());
    connect(rd->orient, &QComboBox::currentIndexChanged, this, updatePos);

    // label anchor along the line; the item texts track the orientation
    rd->anchor = new QComboBox;
    rd->anchor->addItem("Top", static_cast<int>(RefAnchor::Start));
    rd->anchor->addItem("Center", static_cast<int>(RefAnchor::Center));
    rd->anchor->addItem("Bottom", static_cast<int>(RefAnchor::End));
    rd->anchor->setCurrentIndex(static_cast<int>(line.anchor));
    auto *anchorCombo = rd->anchor;
    auto updateAnchor = [anchorCombo](int idx) {
        const bool horiz = (idx == 1);
        anchorCombo->setItemText(0, horiz ? "Left" : "Top");
        anchorCombo->setItemText(2, horiz ? "Right" : "Bottom");
    };
    updateAnchor(rd->orient->currentIndex());
    connect(rd->orient, &QComboBox::currentIndexChanged, this, updateAnchor);

    auto *colorBtn = colorButton(rd->color, [rd](const QColor &c) { rd->color = c; });

    auto *delBtn = new QPushButton("×");
    delBtn->setFixedWidth(24);
    connect(delBtn, &QPushButton::clicked, this, [this, rd]() { removeLine(rows.indexOf(rd)); });

    row->addWidget(rd->orient);
    row->addWidget(posLabel);
    row->addWidget(rd->value, 0);
    row->addWidget(new QLabel("Label:"));
    row->addWidget(rd->label, 1);
    row->addWidget(new QLabel("Pos:"));
    row->addWidget(rd->anchor);
    row->addWidget(new QLabel("Color:"));
    row->addWidget(colorBtn);
    row->addWidget(delBtn);

    listLayout->addWidget(rd->holder);
    rows.append(rd);
}

int RefLinesDialog::addLine()
{
    RefLine fresh;
    fresh.orient = RefOrient::Vertical;
    fresh.value  = 0.0;
    fresh.color  = QColor(80, 80, 80);
    fresh.anchor = RefAnchor::Start;
    appendRow(fresh);
    return rows.size();
}

bool RefLinesDialog::removeLine(int index)
{
    if ((index < 0) || (index >= rows.size())) return false;
    Row *rd = rows.takeAt(index);
    rd->holder->deleteLater();
    delete rd;
    return true;
}

QList<RefLine> RefLinesDialog::lines() const
{
    QList<RefLine> out;
    out.reserve(rows.size());
    for (const auto *rd : rows) {
        RefLine rl;
        rl.orient = rd->orient->currentIndex() == 1 ? RefOrient::Horizontal : RefOrient::Vertical;
        rl.value  = rd->value->value();
        rl.label  = rd->label->text().trimmed();
        rl.color  = rd->color;
        rl.anchor = static_cast<RefAnchor>(rd->anchor->currentData().toInt());
        out.append(rl);
    }
    return out;
}

RefLineStyle RefLinesDialog::labelStyle() const
{
    RefLineStyle s;
    s.fontSize = fontSpin->value();
    s.gap      = distSpin->value();
    s.boxed    = boxedCheck->isChecked();
    return s;
}

/* -------------------------------------------------------------------- */

QColor overlaySeriesColor(int index)
{
    // deliberately away from the raw (blue) and processed (pink) series colours
    static const QList<QColor> palette = {
        QColor(220, 80, 40),  // red-orange
        QColor(40, 160, 40),  // green
        QColor(160, 40, 220), // purple
        QColor(180, 140, 0),  // amber
        QColor(0, 160, 180),  // teal
    };
    const int n = palette.size();
    return palette.at(((index % n) + n) % n);
}

// Local Variables:
// c-basic-offset: 4
// End:
