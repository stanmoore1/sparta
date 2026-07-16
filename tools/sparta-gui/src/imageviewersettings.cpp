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

// The tabbed dump-image settings dialog of the image viewer, split out of
// imageviewer.cpp to keep that translation unit manageable.  One tab per
// option group: Particles, Grid, Grid Planes, Surfaces, Box/Axes, Camera,
// Quality, and Color Maps.  Together the tabs cover every option of SPARTA's
// dump image command and its image-related dump_modify keywords.

#include "imageviewer.h"

#include "imageviewer_internal.h"

#include "colormaps.h"
#include "constants.h"
#include "helpers.h"
#include "qaddon.h"
#include "spartawrapper.h"

#include <QButtonGroup>
#include <QCheckBox>
#include <QColor>
#include <QColorDialog>
#include <QComboBox>
#include <QDialog>
#include <QDoubleValidator>
#include <QFontMetrics>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QScrollArea>
#include <QSlider>
#include <QSpinBox>
#include <QString>
#include <QStringList>
#include <QTabWidget>
#include <QVBoxLayout>

#include <algorithm>

namespace {

// resolve a color-map stop to a QColor (named color or explicit RGB)
QColor stopColor(const ColorMapStop &s)
{
    return s.name.isEmpty() ? QColor::fromRgbF(s.r, s.g, s.b) : QColor(s.name);
}

// Populate a color-map combo box from the shared colormaps.cpp table, so the
// preview swatches match exactly what the dump-image cmap command renders.
void addColorMapItems(QComboBox *box)
{
    for (const QString &name : colorMapNames()) {
        const ColorMapDef &def = colorMapDef(name);
        if (def.continuous) {
            QList<QPair<double, QColor>> stops;
            for (const auto &s : def.stops)
                stops.append({s.pos, stopColor(s)});
            box->addItem(gradient_icon(stops), name);
        } else {
            QList<QColor> colors;
            for (const auto &s : def.stops)
                colors.append(stopColor(s));
            box->addItem(sequence_icon(colors), name);
        }
    }
}

// One "color source" selector: an editable combo with the offered value
// references plus a column spinbox for array references (c_ID[N] / f_ID[N]).
struct SourceRow {
    QComboBox *box = nullptr;
    QSpinBox *col  = nullptr;
};

// split a stored reference like "c_ID[3]" into its base and column number
void splitSource(const QString &ref, QString &base, int &col)
{
    static const QRegularExpression colref(QStringLiteral(R"(^(.*)\[(\d+)\]$)"));
    const auto match = colref.match(ref);
    if (match.hasMatch()) {
        base = match.captured(1);
        col  = match.captured(2).toInt();
    } else {
        base = ref;
        col  = 0;
    }
}

// compose the stored reference from the selector widgets
QString composeSource(const SourceRow &row)
{
    QString base = row.box->currentText().trimmed();
    const int n  = row.col->value();
    // only compute and fix references take a column subscript
    if ((n > 0) && (base.startsWith("c_") || base.startsWith("f_")))
        base += QString("[%1]").arg(n);
    return base;
}

// build a color source selector row into `layout` at `row`, starting at column
// `col`: "Source: [combo]  Column: [spin]"
SourceRow addSourceRow(QGridLayout *layout, int row, int col, const QStringList &sources,
                       const QString &current, QWidget *parent)
{
    QString base;
    int column = 0;
    splitSource(current, base, column);

    auto *box = new QComboBox(parent);
    box->setEditable(true);
    box->addItems(sources);
    if (!base.isEmpty()) {
        if (box->findText(base) < 0) box->addItem(base);
        selectComboItem(box, base);
    }
    box->setToolTip("Color source: \"proc\" or a per-grid/per-surf compute (c_ID), "
                    "fix (f_ID), or variable (v_name) reference");
    layout->addWidget(box, row, col);

    layout->addWidget(new QLabel("Column:"), row, col + 1, Qt::AlignRight);
    auto *colspin = new QSpinBox(parent);
    colspin->setRange(0, MAX_VALUE_COLS);
    colspin->setValue(column);
    colspin->setSpecialValueText("none");
    colspin->setToolTip("Column of an array-producing compute or fix (c_ID[N]); "
                        "\"none\" for a vector reference");
    layout->addWidget(colspin, row, col + 2);

    return {box, colspin};
}

// parse "range color[/color...]" rows separated by ';' into pair list
QList<QPair<QString, QString>> parseRangeColorRows(const QString &text)
{
    QList<QPair<QString, QString>> rows;
    static const QRegularExpression ws(QStringLiteral("\\s+"));
    const auto items = text.split(';', Qt::SkipEmptyParts);
    for (const auto &item : items) {
        const auto words = item.trimmed().split(ws, Qt::SkipEmptyParts);
        if (words.size() == 2) rows.append({words.at(0), words.at(1)});
    }
    return rows;
}

// inverse of parseRangeColorRows()
QString formatRangeColorRows(const QList<QPair<QString, QString>> &rows)
{
    QStringList items;
    for (const auto &row : rows)
        items << row.first + " " + row.second;
    return items.join("; ");
}

// standard dialog page: a vertical layout inside a scroll area
QWidget *makeTabPage(QGridLayout *&grid)
{
    auto *page   = new QWidget;
    auto *layout = new QVBoxLayout(page);
    grid         = new QGridLayout;
    grid->setSizeConstraint(QLayout::SetMinimumSize);
    layout->addLayout(grid);
    layout->addStretch(10);
    return page;
}

} // namespace

QStringList ImageViewer::valueSources(bool withproc, bool withone)
{
    QStringList list;
    if (withone) list << "one";
    if (withproc) list << "proc";
    int num = sparta->idCount("compute");
    for (int i = 0; i < num; ++i)
        list << "c_" + sparta->idName("compute", i);
    num = sparta->idCount("fix");
    for (int i = 0; i < num; ++i)
        list << "f_" + sparta->idName("fix", i);
    num = sparta->idCount("variable");
    for (int i = 0; i < num; ++i)
        list << "v_" + sparta->idName("variable", i);
    return list;
}

void ImageViewer::settingsDialog(int tab)
{
    QDialog dialog(this);
    dialog.setWindowTitle("SPARTA-GUI - Dump Image Settings");
    dialog.setWindowIcon(QIcon(Cfg::MAIN_ICON));
    dialog.setMinimumSize(MINIMUM_WIDTH + 200, MINIMUM_HEIGHT + 100);
    dialog.setContentsMargins(CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN);

    auto *mainLayout = new QVBoxLayout(&dialog);
    auto *tabs       = new QTabWidget(&dialog);
    mainLayout->addWidget(tabs, 10);

    auto *colorcompleter = new QColorCompleter(&dialog);
    auto *colorvalidator = new QColorValidator(&dialog);
    auto *fractvalidator = new QDoubleValidator(0.0, 10.0, 5, &dialog);
    auto *zoomvalidator  = new QDoubleValidator(ZOOM_MIN, ZOOM_MAX, 3, &dialog);
    auto *anyvalidator   = new QDoubleValidator(&dialog);
    QRegularExpression validminmax(
        QStringLiteral(R"(min|max|[+-]?(\d+\.?\d*|\d*\.?\d+)([eE][+-]?\d+)?)"));
    auto *minmaxvalidator = new QRegularExpressionValidator(validminmax, &dialog);
    QFontMetrics metrics(dialog.fontMetrics());
    const int fwidth = metrics.size(Qt::TextSingleLine, "0.00000000").width();

    const bool is3d     = sparta->extractSetting("dimension") == 3;
    const bool hassurfs = sparta->extractSetting("surf_exist") == 1;
    const int nspecies  = sparta->extractSetting("nspecies");

    // make sure the species color table covers all species
    const int ndef = defspeciescolors.size();
    for (int i = color_list.size(); i < nspecies; ++i)
        color_list.append(defspeciescolors[i % ndef]);

    // simulation box bounds for the cut-plane coordinate ranges
    double boxlo[3] = {0.0, 0.0, 0.0};
    double boxhi[3] = {1.0, 1.0, 1.0};
    if (const auto *lo = static_cast<const double *>(sparta->extractGlobal("boxlo")))
        std::copy(lo, lo + 3, boxlo);
    if (const auto *hi = static_cast<const double *>(sparta->extractGlobal("boxhi")))
        std::copy(hi, hi + 3, boxhi);

    const QStringList gridsources = valueSources(true, false);
    const QStringList surfsources = valueSources(true, true);

    // ---------------------------------------------------------------- Particles
    QGridLayout *grid = nullptr;
    auto *particlePage = makeTabPage(grid);
    int row = 0;

    auto *particleshow = new QCheckBox("Show particles");
    particleshow->setChecked(params.particle);
    grid->addWidget(particleshow, row, 0, 1, 2);

    grid->addWidget(new QLabel("Mixture:"), row, 2, Qt::AlignRight);
    auto *mixbox = new QComboBox;
    const int nmix = sparta->idCount("mixture");
    for (int i = 0; i < nmix; ++i)
        mixbox->addItem(sparta->idName("mixture", i));
    selectComboItem(mixbox, params.mixture);
    mixbox->setToolTip("Mixture of species rendered by the dump image command");
    grid->addWidget(mixbox, row++, 3);

    grid->addWidget(new QLabel("Color by:"), row, 0, Qt::AlignRight);
    auto *pcolorbox = new QComboBox;
    pcolorbox->setEditable(true);
    pcolorbox->addItems(QStringList() << "type" << "proc" << particleAttributes);
    // offer per-particle compute/fix/variable references, too
    pcolorbox->addItems(valueSources(false, false));
    if (pcolorbox->findText(params.color) < 0) pcolorbox->addItem(params.color);
    selectComboItem(pcolorbox, params.color);
    pcolorbox->setToolTip("Particle color: \"type\", \"proc\", a dump particle attribute, or "
                          "a custom (p_/i_/d_) or c_/f_/v_ reference");
    grid->addWidget(pcolorbox, row, 1);

    grid->addWidget(new QLabel("Region clip:"), row, 2, Qt::AlignRight);
    auto *regionbox = new QComboBox;
    regionbox->addItem("none");
    const int nregion = sparta->idCount("region");
    for (int i = 0; i < nregion; ++i)
        regionbox->addItem(sparta->idName("region", i));
    selectComboItem(regionbox, params.region);
    regionbox->setToolTip("Only render particles inside this region (dump_modify region)");
    grid->addWidget(regionbox, row++, 3);

    grid->addWidget(new QLabel("Diameter:"), row, 0, Qt::AlignRight);
    auto *diamgroup = new QButtonGroup(&dialog);
    auto *diamtype  = new QRadioButton("By type");
    auto *diamattr  = new QRadioButton("Attribute:");
    auto *diamnum   = new QRadioButton("Value:");
    diamgroup->addButton(diamtype);
    diamgroup->addButton(diamattr);
    diamgroup->addButton(diamnum);
    grid->addWidget(diamtype, row, 1);

    auto *pdiambox = new QComboBox;
    pdiambox->setEditable(true);
    pdiambox->addItems(particleAttributes);
    pdiambox->removeItem(pdiambox->findText("type"));
    grid->addWidget(diamattr, row, 2, Qt::AlignRight);
    grid->addWidget(pdiambox, row++, 3);

    auto *pdiamval = new QLineEdit(QString::number(params.pdiamvalue));
    pdiamval->setValidator(new QDoubleValidator(1.0e-30, 1.0e30, 10, &dialog));
    pdiamval->setMaximumWidth(fwidth);
    pdiamval->setToolTip("Fixed particle diameter in simulation length units (pdiam keyword)");
    grid->addWidget(diamnum, row, 2, Qt::AlignRight);
    grid->addWidget(pdiamval, row++, 3);

    if (params.numericdiam) {
        diamnum->setChecked(true);
    } else if (params.diameter == QLatin1String("type")) {
        diamtype->setChecked(true);
    } else {
        diamattr->setChecked(true);
        if (pdiambox->findText(params.diameter) < 0) pdiambox->addItem(params.diameter);
        selectComboItem(pdiambox, params.diameter);
    }

    // per-species color and diameter table (dump_modify pcolor/pdiam)
    grid->addWidget(new QHline, row++, 0, 1, 4);
    auto *tablehint = new QLabel("Per-species colors and diameters (used when coloring/sizing "
                                 "particles by type):");
    grid->addWidget(tablehint, row++, 0, 1, 4);

    auto *speciesarea   = new QScrollArea;
    auto *specieswidget = new QWidget;
    auto *speciesgrid   = new QGridLayout(specieswidget);
    speciesgrid->setSizeConstraint(QLayout::SetMinimumSize);
    int srow = 0;
    speciesgrid->addWidget(new QLabel("Type:"), srow, 0, Qt::AlignHCenter);
    speciesgrid->addWidget(new QLabel("Species:"), srow, 1, Qt::AlignHCenter);
    speciesgrid->addWidget(new QLabel(""), srow, 2);
    speciesgrid->addWidget(new QLabel("Color:"), srow, 3, Qt::AlignHCenter);
    speciesgrid->addWidget(new QLabel("Diameter:"), srow++, 4, Qt::AlignHCenter);

    // current per-species diameters from the pdiam rows (default 1.0)
    QList<double> curdiams;
    for (int i = 1; i <= nspecies; ++i) {
        double diam = 1.0;
        for (const auto &pd : params.pdiams)
            if (pd.first == QString::number(i)) diam = pd.second;
        curdiams.append(diam);
    }

    QList<QLineEdit *> colnames;
    QList<QPushButton *> colicons;
    QList<QLineEdit *> coldiams;
    for (int i = 1; i <= nspecies; ++i) {
        speciesgrid->addWidget(new QLabel(QString::number(i)), srow, 0, Qt::AlignRight);
        speciesgrid->addWidget(new QLabel(sparta->idName("species", i - 1)), srow, 1);

        const auto &cur = color_list[i - 1];
        auto *icon      = new QPushButton("");
        icon->setIcon(color_icon(cur.second));
        auto iconhint = icon->minimumSizeHint();
        iconhint.setWidth(iconhint.height());
        icon->setIconSize(QSize(iconhint.height() - 4, iconhint.height() - 4));
        icon->setMinimumSize(iconhint);
        icon->setMaximumSize(iconhint);
        speciesgrid->addWidget(icon, srow, 2);
        colicons.append(icon);

        auto *name = new QLineEdit(cur.first);
        name->setCompleter(colorcompleter);
        name->setValidator(colorvalidator);
        name->setFixedWidth(metrics.averageCharWidth() * 16);
        speciesgrid->addWidget(name, srow, 3);
        colnames.append(name);

        auto *diam = new QLineEdit(QString::number(curdiams[i - 1]));
        diam->setValidator(new QDoubleValidator(1.0e-30, 1.0e30, 10, &dialog));
        diam->setFixedWidth(metrics.averageCharWidth() * 10);
        speciesgrid->addWidget(diam, srow, 4);
        coldiams.append(diam);
        ++srow;

        connect(icon, &QPushButton::released, &dialog, [icon, name, &dialog]() {
            const QColor initial(name->text());
            const QColor selected =
                QColorDialog::getColor(initial, &dialog, "Select Species Color");
            if (selected.isValid()) {
                // adopt the SVG color name when Qt knows one for this RGB
                name->setText(selected.name());
                icon->setIcon(color_icon(selected));
            }
        });
        // update the icon when a valid color name is typed
        connect(name, &QLineEdit::editingFinished, &dialog, [icon, name]() {
            const QColor typed(name->text());
            if (typed.isValid()) icon->setIcon(color_icon(typed));
        });
    }
    speciesarea->setWidget(specieswidget);
    speciesarea->setWidgetResizable(true);
    speciesarea->setMinimumHeight(metrics.height() * 8);
    grid->addWidget(speciesarea, row++, 0, 1, 4);
    tabs->addTab(particlePage, "&Particles");

    // ---------------------------------------------------------------- Grid volume
    auto *gridPage = makeTabPage(grid);
    row = 0;

    auto *gridshow = new QCheckBox("Render grid cells (volume)");
    gridshow->setChecked(params.grid);
    grid->addWidget(gridshow, row++, 0, 1, 2);

    auto *gridhint =
        new QLabel("<i>Grid volume rendering and grid cut planes are mutually "
                   "exclusive: enabling one disables the other.</i>");
    gridhint->setWordWrap(true);
    grid->addWidget(gridhint, row++, 0, 1, 4);

    grid->addWidget(new QLabel("Color by:"), row, 0, Qt::AlignRight);
    const SourceRow gridsource =
        addSourceRow(grid, row++, 1, gridsources, params.grid ? params.gridcolor : "proc",
                     gridPage);

    grid->addWidget(new QLabel("Proc colors:"), row, 0, Qt::AlignRight);
    auto *gcolorrows = new QLineEdit(formatRangeColorRows(params.gcolors));
    gcolorrows->setToolTip("Per-processor grid colors when coloring by \"proc\": "
                           "semicolon-separated \"proc-range color[/color2/...]\" entries, "
                           "e.g. \"* red/green/blue\" (dump_modify gcolor)");
    grid->addWidget(gcolorrows, row++, 1, 1, 3);

    grid->addWidget(new QLabel("Grid group:"), row, 0, Qt::AlignRight);
    auto *gridgroupbox = new QComboBox;
    const int ngg      = sparta->idCount("group_grid");
    for (int i = 0; i < ngg; ++i)
        gridgroupbox->addItem(sparta->idName("group_grid", i));
    if (gridgroupbox->count() == 0) gridgroupbox->addItem("all");
    selectComboItem(gridgroupbox, params.gridgroup);
    gridgroupbox->setToolTip("Only render grid cells in this grid group (dump_modify gridgroup)");
    grid->addWidget(gridgroupbox, row++, 1);

    grid->addWidget(new QHline, row++, 0, 1, 4);

    auto *glineshow = new QCheckBox("Grid cell outlines (gline)");
    glineshow->setChecked(params.gline);
    grid->addWidget(glineshow, row, 0, 1, 2);
    grid->addWidget(new QLabel("Diameter:"), row, 2, Qt::AlignRight);
    auto *glinediam = new QLineEdit(QString::number(params.glinediam));
    glinediam->setValidator(fractvalidator);
    glinediam->setMaximumWidth(fwidth);
    grid->addWidget(glinediam, row++, 3);
    grid->addWidget(new QLabel("Outline color:"), row, 2, Qt::AlignRight);
    auto *glinecolor = new QLineEdit(params.glinecolor);
    glinecolor->setCompleter(colorcompleter);
    glinecolor->setValidator(colorvalidator);
    glinecolor->setMaximumWidth(fwidth * 2);
    grid->addWidget(glinecolor, row++, 3);
    tabs->addTab(gridPage, "&Grid");

    // ---------------------------------------------------------------- Grid planes
    auto *planePage = makeTabPage(grid);
    row = 0;

    auto *planehint =
        new QLabel("<i>Render colored cut planes through the grid. Mutually exclusive "
                   "with grid volume rendering on the Grid tab.</i>");
    planehint->setWordWrap(true);
    grid->addWidget(planehint, row++, 0, 1, 5);

    struct PlaneWidgets {
        QCheckBox *show;
        QDoubleSpinBox *coord;
        SourceRow source;
    } planes[3];
    const char *planelabels[3]  = {"X plane:", "Y plane:", "Z plane:"};
    const bool planeon[3]       = {params.gridx, params.gridy, params.gridz};
    const double planecoords[3] = {params.gridxcoord, params.gridycoord, params.gridzcoord};
    const QString planecolor[3] = {params.gridxcolor, params.gridycolor, params.gridzcolor};

    for (int dim = 0; dim < 3; ++dim) {
        auto *show = new QCheckBox(planelabels[dim]);
        show->setChecked(planeon[dim]);
        grid->addWidget(show, row, 0);

        auto *coord = new QDoubleSpinBox;
        coord->setRange(boxlo[dim], boxhi[dim]);
        coord->setDecimals(6);
        coord->setSingleStep((boxhi[dim] - boxlo[dim]) / 20.0);
        coord->setValue(planeon[dim] ? planecoords[dim] : 0.5 * (boxlo[dim] + boxhi[dim]));
        coord->setToolTip("Coordinate of the cut plane (within the simulation box)");
        grid->addWidget(coord, row, 1);

        const SourceRow source = addSourceRow(grid, row, 2, gridsources,
                                              planecolor[dim].isEmpty() ? QStringLiteral("proc")
                                                                        : planecolor[dim],
                                              planePage);
        planes[dim] = {show, coord, source};
        ++row;

        if ((dim == 2) && !is3d) {
            show->setChecked(false);
            show->setEnabled(false);
            coord->setEnabled(false);
            show->setToolTip("Not available for 2d simulations");
        }
    }
    tabs->addTab(planePage, "Grid Pla&nes");

    // grid volume and grid planes exclusivity, kept in sync while editing
    connect(gridshow, &QCheckBox::toggled, &dialog, [&planes](bool on) {
        if (on)
            for (auto &plane : planes)
                plane.show->setChecked(false);
    });
    for (auto &plane : planes) {
        connect(plane.show, &QCheckBox::toggled, gridshow, [gridshow](bool on) {
            if (on) gridshow->setChecked(false);
        });
    }

    // ---------------------------------------------------------------- Surfaces
    auto *surfPage = makeTabPage(grid);
    row = 0;

    auto *surfshow = new QCheckBox("Show surface elements");
    surfshow->setChecked(params.surf && hassurfs);
    grid->addWidget(surfshow, row++, 0, 1, 2);

    if (!hassurfs) {
        auto *nosurf = new QLabel("<i>No surfaces are defined in the current simulation "
                                  "(read_surf / read_isurf).</i>");
        nosurf->setWordWrap(true);
        grid->addWidget(nosurf, row++, 0, 1, 4);
    }

    grid->addWidget(new QLabel("Color by:"), row, 0, Qt::AlignRight);
    const SourceRow surfsource =
        addSourceRow(grid, row++, 1, surfsources,
                     params.surfcolor.isEmpty() ? QStringLiteral("one") : params.surfcolor,
                     surfPage);

    grid->addWidget(new QLabel("Color for \"one\":"), row, 0, Qt::AlignRight);
    auto *surfonecolor = new QLineEdit(params.surfcolorone);
    surfonecolor->setCompleter(colorcompleter);
    surfonecolor->setValidator(colorvalidator);
    surfonecolor->setMaximumWidth(fwidth * 2);
    surfonecolor->setToolTip("Single color of all surface elements (dump_modify scolor)");
    grid->addWidget(surfonecolor, row, 1);

    grid->addWidget(new QLabel("Element diameter:"), row, 2, Qt::AlignRight);
    auto *surfdiam = new QLineEdit(QString::number(params.surfdiam));
    surfdiam->setValidator(fractvalidator);
    surfdiam->setMaximumWidth(fwidth);
    surfdiam->setToolTip("Diameter of surface elements (2d: line width fraction)");
    grid->addWidget(surfdiam, row++, 3);

    grid->addWidget(new QLabel("Proc colors:"), row, 0, Qt::AlignRight);
    auto *scolorrows = new QLineEdit(formatRangeColorRows(params.scolors));
    scolorrows->setToolTip("Per-processor surface colors when coloring by \"proc\": "
                           "semicolon-separated \"proc-range color[/color2/...]\" entries "
                           "(dump_modify scolor)");
    grid->addWidget(scolorrows, row++, 1, 1, 3);

    grid->addWidget(new QLabel("Surface group:"), row, 0, Qt::AlignRight);
    auto *surfgroupbox = new QComboBox;
    const int nsg      = sparta->idCount("group_surf");
    for (int i = 0; i < nsg; ++i)
        surfgroupbox->addItem(sparta->idName("group_surf", i));
    if (surfgroupbox->count() == 0) surfgroupbox->addItem("all");
    selectComboItem(surfgroupbox, params.surfgroup);
    surfgroupbox->setToolTip(
        "Only render surface elements in this surf group (dump_modify surfgroup)");
    grid->addWidget(surfgroupbox, row++, 1);

    grid->addWidget(new QHline, row++, 0, 1, 4);

    auto *slineshow = new QCheckBox("Surface element outlines (sline)");
    slineshow->setChecked(params.sline);
    grid->addWidget(slineshow, row, 0, 1, 2);
    grid->addWidget(new QLabel("Diameter:"), row, 2, Qt::AlignRight);
    auto *slinediam = new QLineEdit(QString::number(params.slinediam));
    slinediam->setValidator(fractvalidator);
    slinediam->setMaximumWidth(fwidth);
    grid->addWidget(slinediam, row++, 3);
    grid->addWidget(new QLabel("Outline color:"), row, 2, Qt::AlignRight);
    auto *slinecolor = new QLineEdit(params.slinecolor);
    slinecolor->setCompleter(colorcompleter);
    slinecolor->setValidator(colorvalidator);
    slinecolor->setMaximumWidth(fwidth * 2);
    grid->addWidget(slinecolor, row++, 3);

    if (!hassurfs) {
        surfshow->setChecked(false);
        for (auto *w : std::initializer_list<QWidget *>{
                 surfshow, surfsource.box, surfsource.col, surfonecolor, surfdiam, scolorrows,
                 surfgroupbox, slineshow, slinediam, slinecolor})
            w->setEnabled(false);
    }
    tabs->addTab(surfPage, "S&urfaces");

    // ---------------------------------------------------------------- Box / Axes
    auto *boxPage = makeTabPage(grid);
    row = 0;

    auto *boxshow = new QCheckBox("Simulation box");
    boxshow->setChecked(params.box);
    grid->addWidget(boxshow, row, 0);
    grid->addWidget(new QLabel("Diameter:"), row, 1, Qt::AlignRight);
    auto *boxdiam = new QLineEdit(QString::number(params.boxdiam));
    boxdiam->setValidator(fractvalidator);
    boxdiam->setMaximumWidth(fwidth);
    grid->addWidget(boxdiam, row, 2);
    grid->addWidget(new QLabel("Color:"), row, 3, Qt::AlignRight);
    auto *boxcolor = new QLineEdit(params.boxcolor);
    boxcolor->setCompleter(colorcompleter);
    boxcolor->setValidator(colorvalidator);
    boxcolor->setMaximumWidth(fwidth * 2);
    grid->addWidget(boxcolor, row++, 4);

    auto *subboxshow = new QCheckBox("Processor sub-boxes");
    subboxshow->setChecked(params.subbox);
    subboxshow->setToolTip("Draw the RCB sub-box of each processor (subbox keyword)");
    grid->addWidget(subboxshow, row, 0);
    grid->addWidget(new QLabel("Diameter:"), row, 1, Qt::AlignRight);
    auto *subboxdiam = new QLineEdit(QString::number(params.subboxdiam));
    subboxdiam->setValidator(fractvalidator);
    subboxdiam->setMaximumWidth(fwidth);
    grid->addWidget(subboxdiam, row, 2);
    grid->addWidget(new QLabel("Color:"), row, 3, Qt::AlignRight);
    auto *subboxcolor = new QLineEdit(params.subboxcolor);
    subboxcolor->setCompleter(colorcompleter);
    subboxcolor->setValidator(colorvalidator);
    subboxcolor->setMaximumWidth(fwidth * 2);
    grid->addWidget(subboxcolor, row++, 4);

    auto *axesshow = new QCheckBox("Coordinate axes");
    axesshow->setChecked(params.axes);
    grid->addWidget(axesshow, row, 0);
    grid->addWidget(new QLabel("Length:"), row, 1, Qt::AlignRight);
    auto *axeslen = new QLineEdit(QString::number(params.axeslen));
    axeslen->setValidator(fractvalidator);
    axeslen->setMaximumWidth(fwidth);
    axeslen->setToolTip("Axes length as fraction of the box size");
    grid->addWidget(axeslen, row, 2);
    grid->addWidget(new QLabel("Diameter:"), row, 3, Qt::AlignRight);
    auto *axesdiam = new QLineEdit(QString::number(params.axesdiam));
    axesdiam->setValidator(fractvalidator);
    axesdiam->setMaximumWidth(fwidth);
    grid->addWidget(axesdiam, row++, 4);
    tabs->addTab(boxPage, "Bo&x/Axes");

    // ---------------------------------------------------------------- Camera
    auto *camPage = makeTabPage(grid);
    row = 0;

    grid->addWidget(new QLabel("View theta:"), row, 0, Qt::AlignRight);
    auto *thetaval = new QDoubleSpinBox;
    thetaval->setRange(0.0, 180.0);
    thetaval->setSingleStep(10.0);
    thetaval->setValue(params.theta);
    thetaval->setEnabled(is3d);
    thetaval->setToolTip("Viewing angle in degrees away from the +z axis");
    grid->addWidget(thetaval, row, 1);
    grid->addWidget(new QLabel("Variable:"), row, 2, Qt::AlignRight);
    auto *thetavar = new QLineEdit(params.thetavar);
    thetavar->setToolTip(
        "Optional equal-style variable name; when set, \"v_name\" is used instead of the number");
    thetavar->setEnabled(is3d);
    grid->addWidget(thetavar, row++, 3);

    grid->addWidget(new QLabel("View phi:"), row, 0, Qt::AlignRight);
    auto *phival = new QDoubleSpinBox;
    phival->setRange(-180.0, 180.0);
    phival->setSingleStep(10.0);
    phival->setValue(params.phi);
    phival->setEnabled(is3d);
    phival->setToolTip("Azimuthal viewing angle in degrees around the z axis");
    grid->addWidget(phival, row, 1);
    grid->addWidget(new QLabel("Variable:"), row, 2, Qt::AlignRight);
    auto *phivar = new QLineEdit(params.phivar);
    phivar->setEnabled(is3d);
    grid->addWidget(phivar, row++, 3);

    grid->addWidget(new QHline, row++, 0, 1, 4);

    grid->addWidget(new QLabel("Center:"), row, 0, Qt::AlignRight);
    auto *centerstatic  = new QRadioButton("Static");
    auto *centerdynamic = new QRadioButton("Dynamic");
    auto *centergroup   = new QButtonGroup(&dialog);
    centergroup->addButton(centerstatic);
    centergroup->addButton(centerdynamic);
    centerdynamic->setToolTip("Recompute the view center every frame (center d ...)");
    if (params.centerdynamic)
        centerdynamic->setChecked(true);
    else
        centerstatic->setChecked(true);
    grid->addWidget(centerstatic, row, 1);
    grid->addWidget(centerdynamic, row++, 2);

    QDoubleSpinBox *centerspin[3];
    QLineEdit *centervar[3];
    const double centerval[3]  = {params.cx, params.cy, params.cz};
    const QString centervars[3] = {params.cxvar, params.cyvar, params.czvar};
    const char *centerlbl[3]    = {"X fraction:", "Y fraction:", "Z fraction:"};
    for (int dim = 0; dim < 3; ++dim) {
        grid->addWidget(new QLabel(centerlbl[dim]), row, 0, Qt::AlignRight);
        centerspin[dim] = new QDoubleSpinBox;
        centerspin[dim]->setRange(-10.0, 10.0);
        centerspin[dim]->setDecimals(4);
        centerspin[dim]->setSingleStep(0.05);
        centerspin[dim]->setValue(centerval[dim]);
        centerspin[dim]->setToolTip("View center as fraction of the box dimension");
        grid->addWidget(centerspin[dim], row, 1);
        grid->addWidget(new QLabel("Variable:"), row, 2, Qt::AlignRight);
        centervar[dim] = new QLineEdit(centervars[dim]);
        grid->addWidget(centervar[dim], row++, 3);
        if ((dim == 2) && !is3d) {
            centerspin[dim]->setEnabled(false);
            centervar[dim]->setEnabled(false);
        }
    }

    grid->addWidget(new QHline, row++, 0, 1, 4);

    auto *uplabel = new QLabel("Camera up:");
    uplabel->setToolTip("Direction pointing up in the image; must not be the zero vector");
    grid->addWidget(uplabel, row, 0, Qt::AlignRight);
    QLineEdit *upval[3];
    const double upcur[3] = {params.upx, params.upy, params.upz};
    for (int dim = 0; dim < 3; ++dim) {
        upval[dim] = new QLineEdit(QString::number(upcur[dim]));
        upval[dim]->setValidator(anyvalidator);
        upval[dim]->setMaximumWidth(fwidth);
        upval[dim]->setEnabled(is3d);
        grid->addWidget(upval[dim], row, 1 + dim);
    }
    ++row;

    grid->addWidget(new QLabel("Zoom:"), row, 0, Qt::AlignRight);
    auto *zoomval = new QLineEdit(QString::number(params.zoom));
    zoomval->setValidator(zoomvalidator);
    zoomval->setMaximumWidth(fwidth);
    zoomval->setToolTip(
        QString("Zoom factor of the view (range: %1 -- %2)").arg(ZOOM_MIN).arg(ZOOM_MAX));
    grid->addWidget(zoomval, row, 1);
    grid->addWidget(new QLabel("Variable:"), row, 2, Qt::AlignRight);
    auto *zoomvar = new QLineEdit(params.zoomvar);
    grid->addWidget(zoomvar, row++, 3);

    grid->addWidget(new QLabel("Perspective:"), row, 0, Qt::AlignRight);
    auto *perspval = new QLineEdit("0.0");
    perspval->setEnabled(false);
    perspval->setMaximumWidth(fwidth);
    perspval->setToolTip("The persp keyword is not yet supported by SPARTA");
    grid->addWidget(perspval, row++, 1);
    tabs->addTab(camPage, "&Camera");

    // ---------------------------------------------------------------- Quality / Background
    auto *qualPage = makeTabPage(grid);
    row = 0;

    auto *ssaoshow = new QCheckBox("SSAO (screen-space ambient occlusion)");
    ssaoshow->setChecked(params.ssao);
    grid->addWidget(ssaoshow, row, 0, 1, 2);
    grid->addWidget(new QLabel("Strength:"), row, 2, Qt::AlignRight);
    auto *ssaoval = new QDoubleSpinBox;
    ssaoval->setRange(0.0, 1.0);
    ssaoval->setSingleStep(0.05);
    ssaoval->setValue(params.ssaoint);
    grid->addWidget(ssaoval, row++, 3);

    auto *fsaashow = new QCheckBox("FSAA (anti-aliasing)");
    fsaashow->setChecked(params.fsaa);
    grid->addWidget(fsaashow, row++, 0, 1, 2);

    grid->addWidget(new QLabel("Shininess:"), row, 0, Qt::AlignRight);
    auto *shinyslider = new QSlider(Qt::Horizontal);
    shinyslider->setRange(0, 100);
    shinyslider->setValue(static_cast<int>(params.shiny * 100.0));
    shinyslider->setToolTip("Shininess of particles and surfaces (0.0 - 1.0)");
    grid->addWidget(shinyslider, row++, 1, 1, 3);

    grid->addWidget(new QHline, row++, 0, 1, 4);

    grid->addWidget(new QLabel("Background:"), row, 0, Qt::AlignRight);
    auto *bgcolor = new QLineEdit(params.backcolor);
    bgcolor->setCompleter(colorcompleter);
    bgcolor->setValidator(colorvalidator);
    bgcolor->setMaximumWidth(fwidth * 2);
    grid->addWidget(bgcolor, row, 1);
    auto *gradientshow = new QCheckBox("Gradient to:");
    gradientshow->setChecked(params.gradient);
    gradientshow->setToolTip(
        "Blend the background vertically to a second color (dump_modify backcolor2)");
    grid->addWidget(gradientshow, row, 2, Qt::AlignRight);
    auto *bg2color = new QLineEdit(params.backcolor2);
    bg2color->setCompleter(colorcompleter);
    bg2color->setValidator(colorvalidator);
    bg2color->setMaximumWidth(fwidth * 2);
    bg2color->setEnabled(params.gradient);
    connect(gradientshow, &QCheckBox::toggled, bg2color, &QLineEdit::setEnabled);
    grid->addWidget(bg2color, row++, 3);

    grid->addWidget(new QHline, row++, 0, 1, 4);

    grid->addWidget(new QLabel("Lights:"), row++, 0);
    QSlider *lightslider[4];
    const char *lightlbl[4]  = {"Ambient:", "Key:", "Fill:", "Back:"};
    const double lightval[4] = {params.amblight, params.keylight, params.filllight,
                                params.backlight};
    for (int i = 0; i < 4; ++i) {
        grid->addWidget(new QLabel(lightlbl[i]), row, 0, Qt::AlignRight);
        lightslider[i] = new QSlider(Qt::Horizontal);
        lightslider[i]->setRange(0, 100);
        lightslider[i]->setValue(static_cast<int>(lightval[i] * 100.0));
        lightslider[i]->setToolTip("Light intensity 0.0 - 1.0 (defaults: ambient 0.0, "
                                   "key 0.9, fill 0.45, back 0.9)");
        grid->addWidget(lightslider[i], row++, 1, 1, 3);
    }
    tabs->addTab(qualPage, "&Quality");

    // ---------------------------------------------------------------- Color maps
    auto *mapPage = makeTabPage(grid);
    row = 0;

    // edit a local copy of the six per-mode specs; committed on Apply
    ColorMapSpec cmapspec[DumpImageSettings::NUM_CMAP_MODES];
    std::copy(params.cmap, params.cmap + DumpImageSettings::NUM_CMAP_MODES, cmapspec);

    grid->addWidget(new QLabel("Color map for:"), row, 0, Qt::AlignRight);
    auto *modebox = new QComboBox;
    for (int mode = 0; mode < DumpImageSettings::NUM_CMAP_MODES; ++mode)
        modebox->addItem(QString::fromLatin1(cmapModeName[mode]));
    grid->addWidget(modebox, row++, 1);

    auto *mapactive = new QCheckBox("Customize this color map");
    mapactive->setToolTip("When unchecked the SPARTA default map "
                          "(continuous, blue at min to red at max) is used");
    grid->addWidget(mapactive, row++, 0, 1, 2);

    grid->addWidget(new QLabel("Map:"), row, 0, Qt::AlignRight);
    auto *mapbox = new QComboBox;
    addColorMapItems(mapbox);
    grid->addWidget(mapbox, row, 1);
    auto *maprev = new QCheckBox("Reverse");
    grid->addWidget(maprev, row++, 2);

    grid->addWidget(new QLabel("Minimum:"), row, 0, Qt::AlignRight);
    auto *mapmin = new QLineEdit;
    mapmin->setValidator(minmaxvalidator);
    mapmin->setToolTip("Lower bound of the map: \"min\" (auto) or a number");
    grid->addWidget(mapmin, row, 1);
    grid->addWidget(new QLabel("Maximum:"), row, 2, Qt::AlignRight);
    auto *mapmax = new QLineEdit;
    mapmax->setValidator(minmaxvalidator);
    mapmax->setToolTip("Upper bound of the map: \"max\" (auto) or a number");
    grid->addWidget(mapmax, row++, 3);

    grid->addWidget(new QLabel("Style:"), row, 0, Qt::AlignRight);
    auto *stylec = new QRadioButton("Continuous");
    auto *styled = new QRadioButton("Discrete");
    auto *styles = new QRadioButton("Sequential");
    auto *stylegroup = new QButtonGroup(&dialog);
    stylegroup->addButton(stylec);
    stylegroup->addButton(styled);
    stylegroup->addButton(styles);
    grid->addWidget(stylec, row, 1);
    grid->addWidget(styled, row, 2);
    grid->addWidget(styles, row++, 3);

    grid->addWidget(new QLabel("Range:"), row, 0, Qt::AlignRight);
    auto *rangef = new QRadioButton("Fractional");
    auto *rangea = new QRadioButton("Absolute");
    auto *rangegroup = new QButtonGroup(&dialog);
    rangegroup->addButton(rangef);
    rangegroup->addButton(rangea);
    rangea->setToolTip("Absolute value positions require numeric minimum and maximum");
    grid->addWidget(rangef, row, 1);
    grid->addWidget(rangea, row++, 2);

    grid->addWidget(new QLabel("Bin size:"), row, 0, Qt::AlignRight);
    auto *mapdelta = new QLineEdit;
    mapdelta->setValidator(new QDoubleValidator(1.0e-30, 1.0e30, 10, &dialog));
    mapdelta->setMaximumWidth(fwidth);
    mapdelta->setToolTip("Value bin width for the sequential style");
    grid->addWidget(mapdelta, row++, 1);

    // transfer one spec into the widgets and back
    auto loadSpec = [&](const ColorMapSpec &spec) {
        mapactive->setChecked(spec.active);
        selectComboItem(mapbox, spec.mapname);
        maprev->setChecked(spec.reverse);
        mapmin->setText(spec.lo);
        mapmax->setText(spec.hi);
        stylec->setChecked(spec.style == 'c');
        styled->setChecked(spec.style == 'd');
        styles->setChecked(spec.style == 's');
        rangea->setChecked(spec.range == 'a');
        rangef->setChecked(spec.range != 'a');
        mapdelta->setText(QString::number(spec.delta));
    };
    auto storeSpec = [&](ColorMapSpec &spec) {
        spec.active  = mapactive->isChecked();
        spec.mapname = mapbox->currentText();
        spec.reverse = maprev->isChecked();
        if (!mapmin->text().isEmpty()) spec.lo = mapmin->text();
        if (!mapmax->text().isEmpty()) spec.hi = mapmax->text();
        spec.style = styles->isChecked() ? QChar('s') : styled->isChecked() ? QChar('d')
                                                                            : QChar('c');
        spec.range = rangea->isChecked() ? QChar('a') : QChar('f');
        if (!mapdelta->text().isEmpty()) spec.delta = mapdelta->text().toDouble();
    };

    int curmode = 0;
    loadSpec(cmapspec[curmode]);
    connect(modebox, QOverload<int>::of(&QComboBox::currentIndexChanged), &dialog,
            [&curmode, &cmapspec, loadSpec, storeSpec](int newmode) {
                storeSpec(cmapspec[curmode]);
                curmode = newmode;
                loadSpec(cmapspec[curmode]);
            });
    tabs->addTab(mapPage, "Color &Maps");

    // ---------------------------------------------------------------- buttons
    auto *bottomlayout = new QHBoxLayout;
    bottomlayout->setSpacing(LAYOUT_SPACING);
    auto *cancel = new QPushButton(QIcon(":/icons/dialog-cancel.svg"), "&Cancel");
    auto *apply  = new QPushButton(QIcon(":/icons/dialog-ok.svg"), "&Apply");
    auto *help   = new QPushButton(QIcon(":/icons/system-help.svg"), "&Help");
    cancel->setAutoDefault(false);
    help->setObjectName("dump_image.html");
    help->setAutoDefault(false);
    apply->setAutoDefault(true);
    apply->setDefault(true);
    apply->setFocus();

    connect(cancel, &QPushButton::released, &dialog, &QDialog::reject);
    connect(apply, &QPushButton::released, &dialog, &QDialog::accept);
    connect(help, &QPushButton::released, this, &ImageViewer::getHelp);

    bottomlayout->addWidget(cancel);
    bottomlayout->addWidget(apply);
    bottomlayout->addWidget(help);
    mainLayout->addLayout(bottomlayout);

    tabs->setCurrentIndex(std::clamp(tab, 0, tabs->count() - 1));

    const int rv = dialog.exec();

    // return immediately on cancel
    if (!rv) return;

    // ---- read back and apply all tabs ----

    // Particles
    params.particle = particleshow->isChecked();
    params.mixture  = mixbox->currentText();
    {
        // keep the main-window mixture selector in sync without re-rendering
        auto *mainmix = findChild<QComboBox *>("mixture");
        if (mainmix && (mainmix->currentText() != params.mixture)) {
            mainmix->blockSignals(true);
            selectComboItem(mainmix, params.mixture);
            mainmix->blockSignals(false);
        }
    }
    if (!pcolorbox->currentText().trimmed().isEmpty())
        params.color = pcolorbox->currentText().trimmed();
    params.region = regionbox->currentText();

    if (diamnum->isChecked()) {
        params.numericdiam = true;
        params.diameter    = "type";
        if (pdiamval->hasAcceptableInput()) params.pdiamvalue = pdiamval->text().toDouble();
    } else if (diamattr->isChecked() && !pdiambox->currentText().trimmed().isEmpty()) {
        params.numericdiam = false;
        params.diameter    = pdiambox->currentText().trimmed();
    } else {
        params.numericdiam = false;
        params.diameter    = "type";
    }

    params.pdiams.clear();
    for (int i = 1; i <= nspecies; ++i) {
        const QString cname = colnames[i - 1]->text().trimmed();
        if (!cname.isEmpty()) {
            QColor rgb(cname);
            // keep the RGB from the color picker when the name is unknown to Qt
            if (!rgb.isValid()) rgb = color_list[i - 1].second;
            color_list[i - 1] = {cname, rgb};
        }
        if (coldiams[i - 1]->hasAcceptableInput()) {
            const double diam = coldiams[i - 1]->text().toDouble();
            if (diam != 1.0) params.pdiams.append({QString::number(i), diam});
        }
    }

    // Grid
    params.grid      = gridshow->isChecked();
    params.gridcolor = composeSource(gridsource);
    params.gcolors   = parseRangeColorRows(gcolorrows->text());
    params.gridgroup = gridgroupbox->currentText();
    params.gline     = glineshow->isChecked();
    if (glinediam->hasAcceptableInput()) params.glinediam = glinediam->text().toDouble();
    if (glinecolor->hasAcceptableInput()) params.glinecolor = glinecolor->text();

    // Grid planes
    params.gridx      = planes[0].show->isChecked();
    params.gridxcoord = planes[0].coord->value();
    params.gridxcolor = composeSource(planes[0].source);
    params.gridy      = planes[1].show->isChecked();
    params.gridycoord = planes[1].coord->value();
    params.gridycolor = composeSource(planes[1].source);
    params.gridz      = planes[2].show->isChecked();
    params.gridzcoord = planes[2].coord->value();
    params.gridzcolor = composeSource(planes[2].source);

    // Surfaces
    params.surf      = surfshow->isChecked() && hassurfs;
    params.surfcolor = composeSource(surfsource);
    if (surfonecolor->hasAcceptableInput()) params.surfcolorone = surfonecolor->text();
    if (surfdiam->hasAcceptableInput()) params.surfdiam = surfdiam->text().toDouble();
    params.scolors   = parseRangeColorRows(scolorrows->text());
    params.surfgroup = surfgroupbox->currentText();
    params.sline     = slineshow->isChecked();
    if (slinediam->hasAcceptableInput()) params.slinediam = slinediam->text().toDouble();
    if (slinecolor->hasAcceptableInput()) params.slinecolor = slinecolor->text();

    // Box / Axes
    params.box = boxshow->isChecked();
    if (boxdiam->hasAcceptableInput()) params.boxdiam = boxdiam->text().toDouble();
    if (boxcolor->hasAcceptableInput()) params.boxcolor = boxcolor->text();
    params.subbox = subboxshow->isChecked();
    if (subboxdiam->hasAcceptableInput()) params.subboxdiam = subboxdiam->text().toDouble();
    if (subboxcolor->hasAcceptableInput()) params.subboxcolor = subboxcolor->text();
    params.axes = axesshow->isChecked();
    if (axeslen->hasAcceptableInput()) params.axeslen = axeslen->text().toDouble();
    if (axesdiam->hasAcceptableInput()) params.axesdiam = axesdiam->text().toDouble();

    // Camera
    params.theta    = thetaval->value();
    params.phi      = phival->value();
    params.thetavar = thetavar->text().trimmed();
    params.phivar   = phivar->text().trimmed();
    params.centerdynamic = centerdynamic->isChecked();
    params.cx       = centerspin[0]->value();
    params.cy       = centerspin[1]->value();
    params.cz       = centerspin[2]->value();
    params.cxvar    = centervar[0]->text().trimmed();
    params.cyvar    = centervar[1]->text().trimmed();
    params.czvar    = centervar[2]->text().trimmed();
    // SPARTA rejects a zero-length up vector, so keep the previous one in that case
    if (upval[0]->hasAcceptableInput() && upval[1]->hasAcceptableInput() &&
        upval[2]->hasAcceptableInput()) {
        const double ux = upval[0]->text().toDouble();
        const double uy = upval[1]->text().toDouble();
        const double uz = upval[2]->text().toDouble();
        if ((ux != 0.0) || (uy != 0.0) || (uz != 0.0)) {
            params.upx = ux;
            params.upy = uy;
            params.upz = uz;
        }
    }
    if (zoomval->hasAcceptableInput()) params.zoom = zoomval->text().toDouble();
    params.zoomvar = zoomvar->text().trimmed();

    // Quality / Background
    params.ssao    = ssaoshow->isChecked();
    params.ssaoint = ssaoval->value();
    params.fsaa    = fsaashow->isChecked();
    params.shiny   = shinyslider->value() / 100.0;
    if (bgcolor->hasAcceptableInput()) params.backcolor = bgcolor->text();
    params.gradient = gradientshow->isChecked();
    if (bg2color->hasAcceptableInput()) params.backcolor2 = bg2color->text();
    params.amblight  = lightslider[0]->value() / 100.0;
    params.keylight  = lightslider[1]->value() / 100.0;
    params.filllight = lightslider[2]->value() / 100.0;
    params.backlight = lightslider[3]->value() / 100.0;

    // Color maps
    storeSpec(cmapspec[curmode]);
    std::copy(cmapspec, cmapspec + DumpImageSettings::NUM_CMAP_MODES, params.cmap);

    // reflect the new state in the toolbar buttons and re-render
    syncButtons();
    createImage();
}

// Local Variables:
// c-basic-offset: 4
// End:
