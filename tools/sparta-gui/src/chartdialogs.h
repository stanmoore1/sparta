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

#ifndef CHARTDIALOGS_H
#define CHARTDIALOGS_H

#include <QDialog>

#include <QColor>
#include <QList>
#include <QString>

#include "chartviewer.h" // ChartDisplayMode, RefLine, RefOrient, RefAnchor
#include "fitting.h"     // EosFit
#include "plotwidget.h"  // LegendPos

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QSpinBox;
class QVBoxLayout;

/**
 * @brief How one chart's two series are drawn, plus the window's legend
 *
 * The dialog edits this and nothing else; binding it to a ChartViewer, storing
 * the legend placement and re-rendering all belong to ChartWindow.
 */
struct ChartStyle {
    ChartDisplayMode rawMode = ChartDisplayMode::Lines; ///< how the raw series is drawn
    QColor rawColor;              ///< raw series colour; invalid picks the dialog's default
    qreal rawWidth       = 3.0;   ///< raw line width
    qreal rawPointSize   = 8.0;   ///< raw marker diameter
    ChartDisplayMode procMode = ChartDisplayMode::Lines; ///< how the processed series is drawn
    QColor procColor;             ///< processed series colour; invalid picks the default
    qreal procWidth      = 3.0;   ///< processed line width
    qreal procPointSize  = 8.0;   ///< processed marker diameter
    LegendPos legend     = LegendPos::Off; ///< in-plot legend placement (window-wide)
};

/**
 * @brief The chart style editor
 *
 * A pure function of the style handed to it: raw series, processed series and
 * legend placement in, edited style out.  It owns no chart.
 */
class ChartStyleDialog : public QDialog {
    Q_OBJECT

public:
    explicit ChartStyleDialog(const ChartStyle &initial, QWidget *parent = nullptr);
    ~ChartStyleDialog() override = default;

    ChartStyleDialog()                                    = delete;
    ChartStyleDialog(const ChartStyleDialog &)            = delete;
    ChartStyleDialog(ChartStyleDialog &&)                 = delete;
    ChartStyleDialog &operator=(const ChartStyleDialog &) = delete;
    ChartStyleDialog &operator=(ChartStyleDialog &&)      = delete;

    /// the style as the controls currently express it; const and repeatable
    [[nodiscard]] ChartStyle style() const;

private:
    QComboBox *rawMode = nullptr, *procMode = nullptr, *legendBox = nullptr;
    QDoubleSpinBox *rawWidth = nullptr, *rawPoint = nullptr;
    QDoubleSpinBox *procWidth = nullptr, *procPoint = nullptr;
    QColor rawColor, procColor;
};

/**
 * @brief Which analysis to run over a chart's data, and with what
 *
 * The scalar @ref param means a different thing per analysis -- the maximum lag
 * of an autocorrelation, the degree of a polynomial, the block count of a
 * batch-means average -- which is why the dialog relabels one spin box rather
 * than showing three.
 */
struct PostProcessSpec {
    /// the analyses offered, in the order the dialog lists them
    enum Analysis {
        Autocorrelation = 0, ///< max lag in param
        Polynomial      = 1, ///< degree in param
        Eos             = 2, ///< Birch-Murnaghan; no scalar parameter
        CustomFunction  = 3, ///< plot f(x) from expression
        CustomFit       = 4, ///< nonlinear fit of expression with parameters
        BlockAverage    = 5, ///< block count in param
        SteadyState     = 6, ///< no scalar parameter
    };

    Analysis analysis = Autocorrelation;
    int param         = 1;   ///< max lag, degree or block count, per analysis
    QString expression;      ///< f(x), for the custom function and custom fit
    QString parameters;      ///< "a=1, b=0.5" initial guesses, for the custom fit
    QString label;           ///< optional name for the fitted curve
    double fitFrom = 0.0;    ///< first x of the fit range
    double fitTo   = 0.0;    ///< last x of the fit range

    /// true when this analysis is fitted over an x range rather than the whole series
    [[nodiscard]] bool usesFitRange() const
    {
        return analysis >= Polynomial && analysis <= CustomFit;
    }
};

/**
 * @brief The postprocess analysis picker
 *
 * Knows only how many points the chart holds and what x range they span; the
 * analysis itself, and everything it does to the chart afterwards, stays in
 * ChartWindow.  The parameter limits follow the point count -- a polynomial
 * cannot have more coefficients than there are points to fit -- which is the
 * part worth checking without a chart behind it.
 */
class PostProcessDialog : public QDialog {
    Q_OBJECT

public:
    /**
     * @param npoints  number of data points in the chart; bounds the parameters
     * @param dataXmin smallest x in the data, the fit range's lower default
     * @param dataXmax largest x in the data, the fit range's upper default
     */
    PostProcessDialog(int npoints, double dataXmin, double dataXmax, QWidget *parent = nullptr);
    ~PostProcessDialog() override = default;

    PostProcessDialog()                                     = delete;
    PostProcessDialog(const PostProcessDialog &)            = delete;
    PostProcessDialog(PostProcessDialog &&)                 = delete;
    PostProcessDialog &operator=(const PostProcessDialog &) = delete;
    PostProcessDialog &operator=(PostProcessDialog &&)      = delete;

    [[nodiscard]] PostProcessSpec spec() const;

private:
    void configure(int index); ///< show the controls this analysis needs, hide the rest

    const int m_npoints;
    QComboBox *analysisBox  = nullptr;
    QLabel *paramLabel      = nullptr;
    QSpinBox *paramSpin     = nullptr;
    QLabel *exprLabel       = nullptr;
    QLineEdit *exprEdit     = nullptr;
    QLabel *paramsLabel     = nullptr;
    QLineEdit *paramsEdit   = nullptr;
    QLabel *fitLabelLabel   = nullptr;
    QLineEdit *fitLabelEdit = nullptr;
    QLabel *rangeLabel      = nullptr;
    QWidget *rangeWidget    = nullptr;
    QDoubleSpinBox *fromSpin = nullptr, *toSpin = nullptr;
};

/**
 * @brief The column confirmation shown before a Birch-Murnaghan fit
 *
 * The fit expects volume against cohesive energy, and the lattice constant it
 * derives is a0 = cbrt(N * V0) -- so it has to ask for N, and show which
 * columns it is about to treat as volume and energy.
 */
class EosSetupDialog : public QDialog {
    Q_OBJECT

public:
    EosSetupDialog(const QString &xLabel, const QString &yLabel, QWidget *parent = nullptr);
    ~EosSetupDialog() override = default;

    EosSetupDialog()                                  = delete;
    EosSetupDialog(const EosSetupDialog &)            = delete;
    EosSetupDialog(EosSetupDialog &&)                 = delete;
    EosSetupDialog &operator=(const EosSetupDialog &) = delete;
    EosSetupDialog &operator=(EosSetupDialog &&)      = delete;

    /// atoms in the conventional unit cell; 1 when x is already the cell volume
    [[nodiscard]] int atomsPerCell() const;

private:
    QSpinBox *natSpin = nullptr;
};

/**
 * @brief The Birch-Murnaghan fit result, with the equation it fitted
 *
 * Read-only: it reports v0, the derived lattice constant, e0, b0, b0' and the
 * residual.  Split out with the rest so the numbers it formats can be checked
 * without running a fit through a chart.
 */
class EosResultDialog : public QDialog {
    Q_OBJECT

public:
    EosResultDialog(const EosFit &fit, int atomsPerCell, QWidget *parent = nullptr);
    ~EosResultDialog() override = default;

    EosResultDialog()                                   = delete;
    EosResultDialog(const EosResultDialog &)            = delete;
    EosResultDialog(EosResultDialog &&)                 = delete;
    EosResultDialog &operator=(const EosResultDialog &) = delete;
    EosResultDialog &operator=(EosResultDialog &&)      = delete;

    /// the lattice constant this dialog reports: cbrt(N * V0)
    [[nodiscard]] static double latticeConstant(double v0, int atomsPerCell);
};

/// window-wide styling of the reference-line labels
struct RefLineStyle {
    double fontSize = 9.0;   ///< label font size in points
    double gap      = 4.0;   ///< gap between the label and its line, in pixels
    bool boxed      = false; ///< draw the label on a framed opaque background
};

/**
 * @brief The reference-line editor
 *
 * One row per line: orientation, position, label, label anchor and colour, plus
 * a row of window-wide label style controls.  Rows can be added and removed
 * while the dialog is open, which is the part that goes wrong silently -- a
 * removed row that stays in the answer puts a line back on the chart.
 */
class RefLinesDialog : public QDialog {
    Q_OBJECT

public:
    RefLinesDialog(const QList<RefLine> &lines, const RefLineStyle &style,
                   QWidget *parent = nullptr);
    ~RefLinesDialog() override = default;

    RefLinesDialog()                                  = delete;
    RefLinesDialog(const RefLinesDialog &)            = delete;
    RefLinesDialog(RefLinesDialog &&)                 = delete;
    RefLinesDialog &operator=(const RefLinesDialog &) = delete;
    RefLinesDialog &operator=(RefLinesDialog &&)      = delete;

    /// the lines as the rows currently express them, in row order
    [[nodiscard]] QList<RefLine> lines() const;
    [[nodiscard]] RefLineStyle labelStyle() const;

    /// add an empty row, as the "Add line" button does; returns the new row count
    int addLine();
    /// remove row @p index, as its "x" button does; false when out of range
    bool removeLine(int index);
    [[nodiscard]] int lineCount() const { return rows.size(); }

private:
    /// one editable reference line
    struct Row {
        QComboBox *orient      = nullptr;
        QDoubleSpinBox *value  = nullptr;
        QLineEdit *label       = nullptr;
        QComboBox *anchor      = nullptr;
        QColor color;
        QWidget *holder = nullptr; ///< the row's widgets, so removal takes them all
    };

    void appendRow(const RefLine &line);

    QVBoxLayout *listLayout = nullptr;
    QList<Row *> rows;
    QDoubleSpinBox *fontSpin = nullptr;
    QSpinBox *distSpin       = nullptr;
    QCheckBox *boxedCheck    = nullptr;
};

/**
 * @brief The colour an overlay series takes, given how many precede it
 *
 * A fixed palette that deliberately avoids the raw and processed series
 * colours, so a file added on top of a chart does not look like part of it.
 * Wraps once the palette is exhausted.
 */
[[nodiscard]] QColor overlaySeriesColor(int index);

#endif

// Local Variables:
// c-basic-offset: 4
// End:
