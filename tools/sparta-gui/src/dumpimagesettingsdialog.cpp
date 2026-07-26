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

// The tabbed dump-image settings dialog.  One tab per option group: Particles,
// Grid, Grid Planes, Surfaces, Box/Axes, Camera, Quality, and Color Maps.
// Together the tabs cover every option of SPARTA's dump image command and its
// image-related dump_modify keywords.
//
// This was ImageViewer::settingsDialog(), an 881-line method that built its
// dialog on the stack, asked a live SpartaWrapper for the simulation it was
// describing, and read ~120 widgets back into a private member.  Nothing could
// construct it without a running simulator and a display, so nothing did -- and
// the widget-to-struct mapping, the one place a control wired to the wrong
// field silently renders the wrong picture, went unchecked.  Here the dialog is
// a pure function of (settings, environment): see test_dumpimagesettings.cpp.

#include "dumpimagesettingsdialog.h"

#include "colormaps.h"
#include "constants.h"
#include "imageviewer_internal.h"
#include "helpers.h"
#include "qaddon.h"

#include <QButtonGroup>
#include <QCheckBox>
#include <QColor>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
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
DumpImageSettingsDialog::DumpImageSettingsDialog(const DumpImageSettings &initial,
                                                 const ImageSettingsEnv &env,
                                                 const SpeciesColors &speciesColors, int tab,
                                                 QWidget *parent) :
    QDialog(parent), m_initial(initial), m_env(env), m_speciesColors(speciesColors)
{
    setWindowTitle("SPARTA-GUI - Dump Image Settings");
    setWindowIcon(QIcon(Cfg::MAIN_ICON));
    setMinimumSize(MINIMUM_WIDTH + 200, MINIMUM_HEIGHT + 100);
    setContentsMargins(CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN);

    auto *mainLayout = new QVBoxLayout(this);
    auto *tabs       = new QTabWidget(this);
    mainLayout->addWidget(tabs, 10);

    auto *colorcompleter = new QColorCompleter(this);
    auto *colorvalidator = new QColorValidator(this);
    auto *fractvalidator = new QDoubleValidator(0.0, 10.0, 5, this);
    auto *zoomvalidator  = new QDoubleValidator(ZOOM_MIN, ZOOM_MAX, 3, this);
    auto *anyvalidator   = new QDoubleValidator(this);
    QRegularExpression validminmax(
        QStringLiteral(R"(min|max|[+-]?(\d+\.?\d*|\d*\.?\d+)([eE][+-]?\d+)?)"));
    auto *minmaxvalidator = new QRegularExpressionValidator(validminmax, this);
    QFontMetrics metrics(fontMetrics());
    const int fwidth = metrics.size(Qt::TextSingleLine, "0.00000000").width();

    const bool is3d    = m_env.dimension == 3;
    const int nspecies = m_env.species.size();

    // the caller tops the colour table up to the species count before
    // constructing; pad defensively so a short list cannot index out of range
    while (m_speciesColors.size() < nspecies)
        m_speciesColors.append(defspeciescolors[m_speciesColors.size() % defspeciescolors.size()]);


    // ---------------------------------------------------------------- Particles
    QGridLayout *grid = nullptr;
    auto *particlePage = makeTabPage(grid);
    int row = 0;

    particleshow = new QCheckBox("Show particles");
    particleshow->setChecked(m_initial.particle);
    grid->addWidget(particleshow, row, 0, 1, 2);

    grid->addWidget(new QLabel("Mixture:"), row, 2, Qt::AlignRight);
    mixbox = new QComboBox();
    mixbox->addItems(m_env.mixtures);
    selectComboItem(mixbox, m_initial.mixture);
    mixbox->setToolTip("Mixture of species rendered by the dump image command");
    grid->addWidget(mixbox, row++, 3);

    grid->addWidget(new QLabel("Color by:"), row, 0, Qt::AlignRight);
    pcolorbox = new QComboBox();
    pcolorbox->setEditable(true);
    pcolorbox->addItems(QStringList() << "type" << "proc" << particleAttributes);
    // offer per-particle compute/fix/variable references, too
    pcolorbox->addItems(m_env.particleSources);
    if (pcolorbox->findText(m_initial.color) < 0) pcolorbox->addItem(m_initial.color);
    selectComboItem(pcolorbox, m_initial.color);
    pcolorbox->setToolTip("Particle color: \"type\", \"proc\", a dump particle attribute, or "
                          "a custom (p_/i_/d_) or c_/f_/v_ reference");
    grid->addWidget(pcolorbox, row, 1);

    grid->addWidget(new QLabel("Region clip:"), row, 2, Qt::AlignRight);
    regionbox = new QComboBox();
    regionbox->addItem("none");
    regionbox->addItems(m_env.regions);
    selectComboItem(regionbox, m_initial.region);
    regionbox->setToolTip("Only render particles inside this region (dump_modify region)");
    grid->addWidget(regionbox, row++, 3);

    grid->addWidget(new QLabel("Diameter:"), row, 0, Qt::AlignRight);
    auto *diamgroup = new QButtonGroup(this);
    auto *diamtype  = new QRadioButton("By type");
    diamattr  = new QRadioButton("Attribute:");
    diamnum   = new QRadioButton("Value:");
    diamgroup->addButton(diamtype);
    diamgroup->addButton(diamattr);
    diamgroup->addButton(diamnum);
    grid->addWidget(diamtype, row, 1);

    pdiambox = new QComboBox();
    pdiambox->setEditable(true);
    pdiambox->addItems(particleAttributes);
    pdiambox->removeItem(pdiambox->findText("type"));
    grid->addWidget(diamattr, row, 2, Qt::AlignRight);
    grid->addWidget(pdiambox, row++, 3);

    pdiamval = new QLineEdit(QString::number(m_initial.pdiamvalue));
    pdiamval->setValidator(new QDoubleValidator(1.0e-30, 1.0e30, 10, this));
    pdiamval->setMaximumWidth(fwidth);
    pdiamval->setToolTip("Fixed particle diameter in simulation length units (pdiam keyword)");
    grid->addWidget(diamnum, row, 2, Qt::AlignRight);
    grid->addWidget(pdiamval, row++, 3);

    if (m_initial.numericdiam) {
        diamnum->setChecked(true);
    } else if (m_initial.diameter == QLatin1String("type")) {
        diamtype->setChecked(true);
    } else {
        diamattr->setChecked(true);
        if (pdiambox->findText(m_initial.diameter) < 0) pdiambox->addItem(m_initial.diameter);
        selectComboItem(pdiambox, m_initial.diameter);
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
        for (const auto &pd : m_initial.pdiams)
            if (pd.first == QString::number(i)) diam = pd.second;
        curdiams.append(diam);
    }

    for (int i = 1; i <= nspecies; ++i) {
        speciesgrid->addWidget(new QLabel(QString::number(i)), srow, 0, Qt::AlignRight);
        speciesgrid->addWidget(new QLabel(m_env.species.at(i - 1)), srow, 1);

        const auto &cur = m_speciesColors[i - 1];
        auto *icon = new QPushButton("");
        icon->setObjectName("colorSwatch");
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
        diam->setValidator(new QDoubleValidator(1.0e-30, 1.0e30, 10, this));
        diam->setFixedWidth(metrics.averageCharWidth() * 10);
        speciesgrid->addWidget(diam, srow, 4);
        coldiams.append(diam);
        ++srow;

        connect(icon, &QPushButton::released, this, [icon, name, this]() {
            const QColor initial(name->text());
            const QColor selected =
                QColorDialog::getColor(initial, this, "Select Species Color");
            if (selected.isValid()) {
                // store the "#rrggbb" value; gatherSettings() defines it as a
                // custom SPARTA color, since SPARTA cannot parse a hex literal
                name->setText(selected.name());
                icon->setIcon(color_icon(selected));
            }
        });
        // update the icon when a valid color name is typed
        connect(name, &QLineEdit::editingFinished, this, [icon, name]() {
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

    gridshow = new QCheckBox("Render grid cells (volume)");
    gridshow->setChecked(m_initial.grid);
    grid->addWidget(gridshow, row++, 0, 1, 2);

    auto *gridhint =
        new QLabel("<i>Grid volume rendering and grid cut planes are mutually "
                   "exclusive: enabling one disables the other.</i>");
    gridhint->setWordWrap(true);
    grid->addWidget(gridhint, row++, 0, 1, 4);

    grid->addWidget(new QLabel("Color by:"), row, 0, Qt::AlignRight);
    gridsource =
        addSourceRow(grid, row++, 1, m_env.gridSources, m_initial.grid ? m_initial.gridcolor : "proc",
                     gridPage);

    grid->addWidget(new QLabel("Proc colors:"), row, 0, Qt::AlignRight);
    gcolorrows = new QLineEdit(formatRangeColorRows(m_initial.gcolors));
    gcolorrows->setToolTip("Per-processor grid colors when coloring by \"proc\": "
                           "semicolon-separated \"proc-range color[/color2/...]\" entries, "
                           "e.g. \"* red/green/blue\" (dump_modify gcolor)");
    grid->addWidget(gcolorrows, row++, 1, 1, 3);

    grid->addWidget(new QLabel("Grid group:"), row, 0, Qt::AlignRight);
    gridgroupbox = new QComboBox();
    gridgroupbox->addItems(m_env.gridGroups);
    if (gridgroupbox->count() == 0) gridgroupbox->addItem("all");
    selectComboItem(gridgroupbox, m_initial.gridgroup);
    gridgroupbox->setToolTip("Only render grid cells in this grid group (dump_modify gridgroup)");
    grid->addWidget(gridgroupbox, row++, 1);

    grid->addWidget(new QHline, row++, 0, 1, 4);

    glineshow = new QCheckBox("Grid cell outlines (gline)");
    glineshow->setChecked(m_initial.gline);
    grid->addWidget(glineshow, row, 0, 1, 2);
    grid->addWidget(new QLabel("Diameter:"), row, 2, Qt::AlignRight);
    glinediam = new QLineEdit(QString::number(m_initial.glinediam));
    glinediam->setValidator(fractvalidator);
    glinediam->setMaximumWidth(fwidth);
    grid->addWidget(glinediam, row++, 3);
    grid->addWidget(new QLabel("Outline color:"), row, 2, Qt::AlignRight);
    glinecolor = new QLineEdit(m_initial.glinecolor);
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

    const char *planelabels[3]  = {"X plane:", "Y plane:", "Z plane:"};
    const bool planeon[3]       = {m_initial.gridx, m_initial.gridy, m_initial.gridz};
    const double planecoords[3] = {m_initial.gridxcoord, m_initial.gridycoord, m_initial.gridzcoord};
    const QString planecolor[3] = {m_initial.gridxcolor, m_initial.gridycolor, m_initial.gridzcolor};

    for (int dim = 0; dim < 3; ++dim) {
        auto *show = new QCheckBox(planelabels[dim]);
        show->setChecked(planeon[dim]);
        grid->addWidget(show, row, 0);

        auto *coord = new QDoubleSpinBox;
        coord->setRange(m_env.boxlo[dim], m_env.boxhi[dim]);
        coord->setDecimals(6);
        coord->setSingleStep((m_env.boxhi[dim] - m_env.boxlo[dim]) / 20.0);
        coord->setValue(planeon[dim] ? planecoords[dim] : 0.5 * (m_env.boxlo[dim] + m_env.boxhi[dim]));
        coord->setToolTip("Coordinate of the cut plane (within the simulation box)");
        grid->addWidget(coord, row, 1);

        const SourceRow source =
            addSourceRow(grid, row, 2, m_env.gridSources,
                         planecolor[dim].isEmpty() ? QStringLiteral("proc") : planecolor[dim],
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
    connect(gridshow, &QCheckBox::toggled, this, [this](bool on) {
        if (on)
            for (auto &plane : planes)
                plane.show->setChecked(false);
    });
    for (auto &plane : planes) {
        connect(plane.show, &QCheckBox::toggled, gridshow, [this](bool on) {
            if (on) gridshow->setChecked(false);
        });
    }

    // ---------------------------------------------------------------- Surfaces
    auto *surfPage = makeTabPage(grid);
    row = 0;

    surfshow = new QCheckBox("Show surface elements");
    surfshow->setChecked(m_initial.surf && m_env.surfsExist);
    grid->addWidget(surfshow, row++, 0, 1, 2);

    if (!m_env.surfsExist) {
        auto *nosurf = new QLabel("<i>No surfaces are defined in the current simulation "
                                  "(read_surf / read_isurf).</i>");
        nosurf->setWordWrap(true);
        grid->addWidget(nosurf, row++, 0, 1, 4);
    }

    grid->addWidget(new QLabel("Color by:"), row, 0, Qt::AlignRight);
    surfsource =
        addSourceRow(grid, row++, 1, m_env.surfSources,
                     m_initial.surfcolor.isEmpty() ? QStringLiteral("one") : m_initial.surfcolor,
                     surfPage);

    grid->addWidget(new QLabel("Color for \"one\":"), row, 0, Qt::AlignRight);
    surfonecolor = new QLineEdit(m_initial.surfcolorone);
    surfonecolor->setCompleter(colorcompleter);
    surfonecolor->setValidator(colorvalidator);
    surfonecolor->setMaximumWidth(fwidth * 2);
    surfonecolor->setToolTip("Single color of all surface elements (dump_modify scolor)");
    grid->addWidget(surfonecolor, row, 1);

    grid->addWidget(new QLabel("Element diameter:"), row, 2, Qt::AlignRight);
    surfdiam = new QLineEdit(QString::number(m_initial.surfdiam));
    surfdiam->setValidator(fractvalidator);
    surfdiam->setMaximumWidth(fwidth);
    surfdiam->setToolTip("Diameter of surface elements (2d: line width fraction)");
    grid->addWidget(surfdiam, row++, 3);

    grid->addWidget(new QLabel("Proc colors:"), row, 0, Qt::AlignRight);
    scolorrows = new QLineEdit(formatRangeColorRows(m_initial.scolors));
    scolorrows->setToolTip("Per-processor surface colors when coloring by \"proc\": "
                           "semicolon-separated \"proc-range color[/color2/...]\" entries "
                           "(dump_modify scolor)");
    grid->addWidget(scolorrows, row++, 1, 1, 3);

    grid->addWidget(new QLabel("Surface group:"), row, 0, Qt::AlignRight);
    surfgroupbox = new QComboBox();
    surfgroupbox->addItems(m_env.surfGroups);
    if (surfgroupbox->count() == 0) surfgroupbox->addItem("all");
    selectComboItem(surfgroupbox, m_initial.surfgroup);
    surfgroupbox->setToolTip(
        "Only render surface elements in this surf group (dump_modify surfgroup)");
    grid->addWidget(surfgroupbox, row++, 1);

    grid->addWidget(new QHline, row++, 0, 1, 4);

    slineshow = new QCheckBox("Surface element outlines (sline)");
    slineshow->setChecked(m_initial.sline);
    grid->addWidget(slineshow, row, 0, 1, 2);
    grid->addWidget(new QLabel("Diameter:"), row, 2, Qt::AlignRight);
    slinediam = new QLineEdit(QString::number(m_initial.slinediam));
    slinediam->setValidator(fractvalidator);
    slinediam->setMaximumWidth(fwidth);
    grid->addWidget(slinediam, row++, 3);
    grid->addWidget(new QLabel("Outline color:"), row, 2, Qt::AlignRight);
    slinecolor = new QLineEdit(m_initial.slinecolor);
    slinecolor->setCompleter(colorcompleter);
    slinecolor->setValidator(colorvalidator);
    slinecolor->setMaximumWidth(fwidth * 2);
    grid->addWidget(slinecolor, row++, 3);

    if (!m_env.surfsExist) {
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

    boxshow = new QCheckBox("Simulation box");
    boxshow->setChecked(m_initial.box);
    grid->addWidget(boxshow, row, 0);
    grid->addWidget(new QLabel("Diameter:"), row, 1, Qt::AlignRight);
    boxdiam = new QLineEdit(QString::number(m_initial.boxdiam));
    boxdiam->setValidator(fractvalidator);
    boxdiam->setMaximumWidth(fwidth);
    grid->addWidget(boxdiam, row, 2);
    grid->addWidget(new QLabel("Color:"), row, 3, Qt::AlignRight);
    boxcolor = new QLineEdit(m_initial.boxcolor);
    boxcolor->setCompleter(colorcompleter);
    boxcolor->setValidator(colorvalidator);
    boxcolor->setMaximumWidth(fwidth * 2);
    grid->addWidget(boxcolor, row++, 4);

    subboxshow = new QCheckBox("Processor sub-boxes");
    subboxshow->setChecked(m_initial.subbox);
    subboxshow->setToolTip("Draw the RCB sub-box of each processor (subbox keyword)");
    grid->addWidget(subboxshow, row, 0);
    grid->addWidget(new QLabel("Diameter:"), row, 1, Qt::AlignRight);
    subboxdiam = new QLineEdit(QString::number(m_initial.subboxdiam));
    subboxdiam->setValidator(fractvalidator);
    subboxdiam->setMaximumWidth(fwidth);
    grid->addWidget(subboxdiam, row, 2);
    grid->addWidget(new QLabel("Color:"), row, 3, Qt::AlignRight);
    subboxcolor = new QLineEdit(m_initial.subboxcolor);
    subboxcolor->setCompleter(colorcompleter);
    subboxcolor->setValidator(colorvalidator);
    subboxcolor->setMaximumWidth(fwidth * 2);
    grid->addWidget(subboxcolor, row++, 4);

    axesshow = new QCheckBox("Coordinate axes");
    axesshow->setChecked(m_initial.axes);
    grid->addWidget(axesshow, row, 0);
    grid->addWidget(new QLabel("Length:"), row, 1, Qt::AlignRight);
    axeslen = new QLineEdit(QString::number(m_initial.axeslen));
    axeslen->setValidator(fractvalidator);
    axeslen->setMaximumWidth(fwidth);
    axeslen->setToolTip("Axes length as fraction of the box size");
    grid->addWidget(axeslen, row, 2);
    grid->addWidget(new QLabel("Diameter:"), row, 3, Qt::AlignRight);
    axesdiam = new QLineEdit(QString::number(m_initial.axesdiam));
    axesdiam->setValidator(fractvalidator);
    axesdiam->setMaximumWidth(fwidth);
    grid->addWidget(axesdiam, row++, 4);
    tabs->addTab(boxPage, "Bo&x/Axes");

    // ---------------------------------------------------------------- Camera
    auto *camPage = makeTabPage(grid);
    row = 0;

    grid->addWidget(new QLabel("View theta:"), row, 0, Qt::AlignRight);
    thetaval = new QDoubleSpinBox();
    thetaval->setRange(0.0, 180.0);
    thetaval->setSingleStep(10.0);
    thetaval->setValue(m_initial.theta);
    thetaval->setEnabled(is3d);
    thetaval->setToolTip("Viewing angle in degrees away from the +z axis");
    grid->addWidget(thetaval, row, 1);
    grid->addWidget(new QLabel("Variable:"), row, 2, Qt::AlignRight);
    thetavar = new QLineEdit(m_initial.thetavar);
    thetavar->setToolTip(
        "Optional equal-style variable name; when set, \"v_name\" is used instead of the number");
    thetavar->setEnabled(is3d);
    grid->addWidget(thetavar, row++, 3);

    grid->addWidget(new QLabel("View phi:"), row, 0, Qt::AlignRight);
    phival = new QDoubleSpinBox();
    phival->setRange(-180.0, 180.0);
    phival->setSingleStep(10.0);
    phival->setValue(m_initial.phi);
    phival->setEnabled(is3d);
    phival->setToolTip("Azimuthal viewing angle in degrees around the z axis");
    grid->addWidget(phival, row, 1);
    grid->addWidget(new QLabel("Variable:"), row, 2, Qt::AlignRight);
    phivar = new QLineEdit(m_initial.phivar);
    phivar->setEnabled(is3d);
    grid->addWidget(phivar, row++, 3);

    grid->addWidget(new QHline, row++, 0, 1, 4);

    grid->addWidget(new QLabel("Center:"), row, 0, Qt::AlignRight);
    auto *centerstatic  = new QRadioButton("Static");
    centerdynamic = new QRadioButton("Dynamic");
    auto *centergroup   = new QButtonGroup(this);
    centergroup->addButton(centerstatic);
    centergroup->addButton(centerdynamic);
    centerdynamic->setToolTip("Recompute the view center every frame (center d ...)");
    if (m_initial.centerdynamic)
        centerdynamic->setChecked(true);
    else
        centerstatic->setChecked(true);
    grid->addWidget(centerstatic, row, 1);
    grid->addWidget(centerdynamic, row++, 2);

    const double centerval[3]  = {m_initial.cx, m_initial.cy, m_initial.cz};
    const QString centervars[3] = {m_initial.cxvar, m_initial.cyvar, m_initial.czvar};
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
    const double upcur[3] = {m_initial.upx, m_initial.upy, m_initial.upz};
    for (int dim = 0; dim < 3; ++dim) {
        upval[dim] = new QLineEdit(QString::number(upcur[dim]));
        upval[dim]->setValidator(anyvalidator);
        upval[dim]->setMaximumWidth(fwidth);
        upval[dim]->setEnabled(is3d);
        grid->addWidget(upval[dim], row, 1 + dim);
    }
    ++row;

    grid->addWidget(new QLabel("Zoom:"), row, 0, Qt::AlignRight);
    zoomval = new QLineEdit(QString::number(m_initial.zoom));
    zoomval->setValidator(zoomvalidator);
    zoomval->setMaximumWidth(fwidth);
    zoomval->setToolTip(
        QString("Zoom factor of the view (range: %1 -- %2)").arg(ZOOM_MIN).arg(ZOOM_MAX));
    grid->addWidget(zoomval, row, 1);
    grid->addWidget(new QLabel("Variable:"), row, 2, Qt::AlignRight);
    zoomvar = new QLineEdit(m_initial.zoomvar);
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

    ssaoshow = new QCheckBox("SSAO (screen-space ambient occlusion)");
    ssaoshow->setChecked(m_initial.ssao);
    grid->addWidget(ssaoshow, row, 0, 1, 2);
    grid->addWidget(new QLabel("Strength:"), row, 2, Qt::AlignRight);
    ssaoval = new QDoubleSpinBox();
    ssaoval->setRange(0.0, 1.0);
    ssaoval->setSingleStep(0.05);
    ssaoval->setValue(m_initial.ssaoint);
    grid->addWidget(ssaoval, row++, 3);

    fsaashow = new QCheckBox("FSAA (anti-aliasing)");
    fsaashow->setChecked(m_initial.fsaa);
    grid->addWidget(fsaashow, row++, 0, 1, 2);

    grid->addWidget(new QLabel("Shininess:"), row, 0, Qt::AlignRight);
    shinyslider = new QSlider(Qt::Horizontal);
    shinyslider->setRange(0, 100);
    shinyslider->setValue(static_cast<int>(m_initial.shiny * 100.0));
    shinyslider->setToolTip("Shininess of particles and surfaces (0.0 - 1.0)");
    grid->addWidget(shinyslider, row++, 1, 1, 3);

    grid->addWidget(new QHline, row++, 0, 1, 4);

    grid->addWidget(new QLabel("Background:"), row, 0, Qt::AlignRight);
    bgcolor = new QLineEdit(m_initial.backcolor);
    bgcolor->setCompleter(colorcompleter);
    bgcolor->setValidator(colorvalidator);
    bgcolor->setMaximumWidth(fwidth * 2);
    grid->addWidget(bgcolor, row, 1);
    gradientshow = new QCheckBox("Gradient to:");
    gradientshow->setChecked(m_initial.gradient);
    gradientshow->setToolTip(
        "Blend the background vertically to a second color (dump_modify backcolor2)");
    grid->addWidget(gradientshow, row, 2, Qt::AlignRight);
    bg2color = new QLineEdit(m_initial.backcolor2);
    bg2color->setCompleter(colorcompleter);
    bg2color->setValidator(colorvalidator);
    bg2color->setMaximumWidth(fwidth * 2);
    bg2color->setEnabled(m_initial.gradient);
    connect(gradientshow, &QCheckBox::toggled, bg2color, &QLineEdit::setEnabled);
    grid->addWidget(bg2color, row++, 3);

    grid->addWidget(new QHline, row++, 0, 1, 4);

    grid->addWidget(new QLabel("Lights:"), row++, 0);
    const char *lightlbl[4]   = {"Ambient:", "Key:", "Fill:", "Back:"};
    const double lightval[4] = {m_initial.amblight, m_initial.keylight, m_initial.filllight,
                                m_initial.backlight};
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
    std::copy(m_initial.cmap, m_initial.cmap + DumpImageSettings::NUM_CMAP_MODES, cmapspec);

    grid->addWidget(new QLabel("Color map for:"), row, 0, Qt::AlignRight);
    modebox = new QComboBox();
    for (int mode = 0; mode < DumpImageSettings::NUM_CMAP_MODES; ++mode)
        modebox->addItem(QString::fromLatin1(cmapModeName[mode]));
    grid->addWidget(modebox, row++, 1);

    mapactive = new QCheckBox("Customize this color map");
    mapactive->setToolTip("When unchecked the SPARTA default map "
                          "(continuous, blue at min to red at max) is used");
    grid->addWidget(mapactive, row++, 0, 1, 2);

    grid->addWidget(new QLabel("Map:"), row, 0, Qt::AlignRight);
    mapbox = new QComboBox();
    addColorMapItems(mapbox);
    grid->addWidget(mapbox, row, 1);
    maprev = new QCheckBox("Reverse");
    grid->addWidget(maprev, row++, 2);

    grid->addWidget(new QLabel("Minimum:"), row, 0, Qt::AlignRight);
    mapmin = new QLineEdit();
    mapmin->setValidator(minmaxvalidator);
    mapmin->setToolTip("Lower bound of the map: \"min\" (auto) or a number");
    grid->addWidget(mapmin, row, 1);
    grid->addWidget(new QLabel("Maximum:"), row, 2, Qt::AlignRight);
    mapmax = new QLineEdit();
    mapmax->setValidator(minmaxvalidator);
    mapmax->setToolTip("Upper bound of the map: \"max\" (auto) or a number");
    grid->addWidget(mapmax, row++, 3);

    grid->addWidget(new QLabel("Style:"), row, 0, Qt::AlignRight);
    stylec = new QRadioButton("Continuous");
    styled = new QRadioButton("Discrete");
    styles = new QRadioButton("Sequential");
    auto *stylegroup = new QButtonGroup(this);
    stylegroup->addButton(stylec);
    stylegroup->addButton(styled);
    stylegroup->addButton(styles);
    grid->addWidget(stylec, row, 1);
    grid->addWidget(styled, row, 2);
    grid->addWidget(styles, row++, 3);

    grid->addWidget(new QLabel("Range:"), row, 0, Qt::AlignRight);
    rangef = new QRadioButton("Fractional");
    rangea = new QRadioButton("Absolute");
    auto *rangegroup = new QButtonGroup(this);
    rangegroup->addButton(rangef);
    rangegroup->addButton(rangea);
    rangea->setToolTip("Absolute value positions require numeric minimum and maximum");
    grid->addWidget(rangef, row, 1);
    grid->addWidget(rangea, row++, 2);

    grid->addWidget(new QLabel("Bin size:"), row, 0, Qt::AlignRight);
    mapdelta = new QLineEdit();
    mapdelta->setValidator(new QDoubleValidator(1.0e-30, 1.0e30, 10, this));
    mapdelta->setMaximumWidth(fwidth);
    mapdelta->setToolTip("Value bin width for the sequential style");
    grid->addWidget(mapdelta, row++, 1);


    loadColorMapSpec(cmapspec[curmode]);
    // Members, not captured locals: the handler outlives the constructor, and
    // capturing curmode and cmapspec by reference would leave it reading a
    // dead stack frame the moment the dialog was shown rather than run inline.
    connect(modebox, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int newmode) {
                storeColorMapSpec(cmapspec[curmode]);
                curmode = newmode;
                loadColorMapSpec(cmapspec[curmode]);
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

    connect(cancel, &QPushButton::released, this, &QDialog::reject);
    connect(apply, &QPushButton::released, this, &QDialog::accept);
    // the caller decides what opening the documentation means; the dialog
    // only says which page, so it needs no viewer to talk back to
    connect(help, &QPushButton::released, this,
            [this]() { emit helpRequested(QStringLiteral("dump_image.html")); });

    bottomlayout->addWidget(cancel);
    bottomlayout->addWidget(apply);
    bottomlayout->addWidget(help);
    mainLayout->addLayout(bottomlayout);

    tabs->setCurrentIndex(std::clamp(tab, 0, tabs->count() - 1));
}

DumpImageSettings DumpImageSettingsDialog::settings() const
{
    DumpImageSettings s = m_initial;

    // Particles
    s.particle = particleshow->isChecked();
    s.mixture  = mixbox->currentText();
    if (!pcolorbox->currentText().trimmed().isEmpty())
        s.color = pcolorbox->currentText().trimmed();
    s.region = regionbox->currentText();

    if (diamnum->isChecked()) {
        s.numericdiam = true;
        s.diameter    = "type";
        if (pdiamval->hasAcceptableInput()) s.pdiamvalue = pdiamval->text().toDouble();
    } else if (diamattr->isChecked() && !pdiambox->currentText().trimmed().isEmpty()) {
        s.numericdiam = false;
        s.diameter    = pdiambox->currentText().trimmed();
    } else {
        s.numericdiam = false;
        s.diameter    = "type";
    }

    s.pdiams.clear();
    for (int i = 1; i <= m_env.species.size(); ++i) {
        if (coldiams[i - 1]->hasAcceptableInput()) {
            const double diam = coldiams[i - 1]->text().toDouble();
            if (diam != 1.0) s.pdiams.append({QString::number(i), diam});
        }
    }

    // Grid
    s.grid      = gridshow->isChecked();
    s.gridcolor = composeSource(gridsource);
    s.gcolors   = parseRangeColorRows(gcolorrows->text());
    s.gridgroup = gridgroupbox->currentText();
    s.gline     = glineshow->isChecked();
    if (glinediam->hasAcceptableInput()) s.glinediam = glinediam->text().toDouble();
    if (glinecolor->hasAcceptableInput()) s.glinecolor = glinecolor->text();

    // Grid planes
    s.gridx      = planes[0].show->isChecked();
    s.gridxcoord = planes[0].coord->value();
    s.gridxcolor = composeSource(planes[0].source);
    s.gridy      = planes[1].show->isChecked();
    s.gridycoord = planes[1].coord->value();
    s.gridycolor = composeSource(planes[1].source);
    s.gridz      = planes[2].show->isChecked();
    s.gridzcoord = planes[2].coord->value();
    s.gridzcolor = composeSource(planes[2].source);

    // Surfaces
    s.surf      = surfshow->isChecked() && m_env.surfsExist;
    s.surfcolor = composeSource(surfsource);
    if (surfonecolor->hasAcceptableInput()) s.surfcolorone = surfonecolor->text();
    if (surfdiam->hasAcceptableInput()) s.surfdiam = surfdiam->text().toDouble();
    s.scolors   = parseRangeColorRows(scolorrows->text());
    s.surfgroup = surfgroupbox->currentText();
    s.sline     = slineshow->isChecked();
    if (slinediam->hasAcceptableInput()) s.slinediam = slinediam->text().toDouble();
    if (slinecolor->hasAcceptableInput()) s.slinecolor = slinecolor->text();

    // Box / Axes
    s.box = boxshow->isChecked();
    if (boxdiam->hasAcceptableInput()) s.boxdiam = boxdiam->text().toDouble();
    if (boxcolor->hasAcceptableInput()) s.boxcolor = boxcolor->text();
    s.subbox = subboxshow->isChecked();
    if (subboxdiam->hasAcceptableInput()) s.subboxdiam = subboxdiam->text().toDouble();
    if (subboxcolor->hasAcceptableInput()) s.subboxcolor = subboxcolor->text();
    s.axes = axesshow->isChecked();
    if (axeslen->hasAcceptableInput()) s.axeslen = axeslen->text().toDouble();
    if (axesdiam->hasAcceptableInput()) s.axesdiam = axesdiam->text().toDouble();

    // Camera
    s.theta    = thetaval->value();
    s.phi      = phival->value();
    s.thetavar = thetavar->text().trimmed();
    s.phivar   = phivar->text().trimmed();
    s.centerdynamic = centerdynamic->isChecked();
    s.cx       = centerspin[0]->value();
    s.cy       = centerspin[1]->value();
    s.cz       = centerspin[2]->value();
    s.cxvar    = centervar[0]->text().trimmed();
    s.cyvar    = centervar[1]->text().trimmed();
    s.czvar    = centervar[2]->text().trimmed();
    // SPARTA rejects a zero-length up vector, so keep the previous one in that case
    if (upval[0]->hasAcceptableInput() && upval[1]->hasAcceptableInput() &&
        upval[2]->hasAcceptableInput()) {
        const double ux = upval[0]->text().toDouble();
        const double uy = upval[1]->text().toDouble();
        const double uz = upval[2]->text().toDouble();
        if ((ux != 0.0) || (uy != 0.0) || (uz != 0.0)) {
            s.upx = ux;
            s.upy = uy;
            s.upz = uz;
        }
    }
    if (zoomval->hasAcceptableInput()) s.zoom = zoomval->text().toDouble();
    s.zoomvar = zoomvar->text().trimmed();

    // Quality / Background
    s.ssao    = ssaoshow->isChecked();
    s.ssaoint = ssaoval->value();
    s.fsaa    = fsaashow->isChecked();
    s.shiny   = shinyslider->value() / 100.0;
    if (bgcolor->hasAcceptableInput()) s.backcolor = bgcolor->text();
    s.gradient = gradientshow->isChecked();
    if (bg2color->hasAcceptableInput()) s.backcolor2 = bg2color->text();
    s.amblight  = lightslider[0]->value() / 100.0;
    s.keylight  = lightslider[1]->value() / 100.0;
    s.filllight = lightslider[2]->value() / 100.0;
    s.backlight = lightslider[3]->value() / 100.0;

    // Color maps.  The mode on screen has not been written back to the working
    // array yet -- that happens when the mode combo changes -- so flush it into
    // the copy rather than into the member, which settings() must not disturb.
    std::copy(cmapspec, cmapspec + DumpImageSettings::NUM_CMAP_MODES, s.cmap);
    storeColorMapSpec(s.cmap[curmode]);
    return s;
}

DumpImageSettingsDialog::SpeciesColors DumpImageSettingsDialog::speciesColors() const
{
    SpeciesColors out = m_speciesColors;
    for (int i = 1; i <= m_env.species.size(); ++i) {
        const QString cname = colnames[i - 1]->text().trimmed();
        if (cname.isEmpty()) continue;
        QColor rgb(cname);
        // keep the RGB from the color picker when the name is unknown to Qt
        if (!rgb.isValid()) rgb = out[i - 1].second;
        out[i - 1] = {cname, rgb};
    }
    return out;
}

void DumpImageSettingsDialog::loadColorMapSpec(const ColorMapSpec &spec)
{
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
}

void DumpImageSettingsDialog::storeColorMapSpec(ColorMapSpec &spec) const
{
    spec.active  = mapactive->isChecked();
    spec.mapname = mapbox->currentText();
    spec.reverse = maprev->isChecked();
    if (!mapmin->text().isEmpty()) spec.lo = mapmin->text();
    if (!mapmax->text().isEmpty()) spec.hi = mapmax->text();
    spec.style = styles->isChecked() ? QChar('s') : styled->isChecked() ? QChar('d')
                                                                        : QChar('c');
    spec.range = rangea->isChecked() ? QChar('a') : QChar('f');
    if (!mapdelta->text().isEmpty()) spec.delta = mapdelta->text().toDouble();
}
