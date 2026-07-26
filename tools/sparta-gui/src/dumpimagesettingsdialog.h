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

#ifndef DUMPIMAGESETTINGSDIALOG_H
#define DUMPIMAGESETTINGSDIALOG_H

#include <QDialog>

#include <QColor>
#include <QList>
#include <QPair>
#include <QString>
#include <QStringList>

#include "dumpimage.h"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QPushButton;
class QRadioButton;
class QSlider;
class QSpinBox;

/**
 * @brief A colour-source picker: the c_/f_/v_ reference and its array column
 *
 * The two are read together -- composeSource() turns "c_temp" plus column 2
 * into "c_temp[2]", and leaves a v_ reference unsubscripted -- so they travel
 * as a pair rather than as two loose pointers.
 */
struct SourceRow {
    QComboBox *box = nullptr;
    QSpinBox *col  = nullptr;
};

/**
 * @brief Everything the settings dialog needs to know about the simulation
 *
 * The dialog used to ask a live SpartaWrapper for each of these as it built
 * itself, which is what made it impossible to construct without a running
 * simulator.  Snapshotting them into plain data first means the dialog is a
 * pure function of (settings, environment) and can be tested on its own.
 *
 * A default-constructed environment describes a 3d run with no surfaces, no
 * mixtures and no species -- the emptiest thing the dialog has to survive.
 */
struct ImageSettingsEnv {
    int dimension   = 3;     ///< extractSetting("dimension"); 2 disables z and the up vector
    bool surfsExist = false; ///< extractSetting("surf_exist"); false disables the Surfaces tab
    double boxlo[3] = {0.0, 0.0, 0.0}; ///< extractGlobal("boxlo"), bounds the cut planes
    double boxhi[3] = {1.0, 1.0, 1.0}; ///< extractGlobal("boxhi")

    /// species names in index order; size() is the species count, so the two
    /// cannot disagree the way a separate nspecies field would let them
    QStringList species;

    QStringList mixtures;   ///< idName("mixture", i), the render subject choices
    QStringList regions;    ///< idName("region", i), for the dump_modify region clip
    QStringList gridGroups; ///< idName("group_grid", i)
    QStringList surfGroups; ///< idName("group_surf", i)

    /// colour sources offered per tab.  Three different lists because the
    /// dialog asks for three different mixes of the "one" and "proc" pseudo
    /// sources alongside the c_/f_/v_ references.
    QStringList gridSources;     ///< with proc, without one
    QStringList surfSources;     ///< with proc and with one
    QStringList particleSources; ///< with neither
};

/**
 * @brief The tabbed dump-image settings dialog
 *
 * One tab per option group: Particles, Grid, Grid Planes, Surfaces, Box/Axes,
 * Camera, Quality and Color Maps.  Together they cover every option of SPARTA's
 * dump image command and its image-related dump_modify keywords.
 *
 * The dialog owns no simulator and no viewer.  It is handed the settings to
 * edit and a snapshot of the environment, and answers with the edited settings;
 * rendering, the main window's mixture selector and the species colour table
 * belong to the caller.  That split is what makes the widget-to-struct mapping
 * -- the part where a control wired to the wrong field silently renders the
 * wrong picture -- testable without a display or a SPARTA library.
 */
class DumpImageSettingsDialog : public QDialog {
    Q_OBJECT

public:
    /// species name paired with the colour it renders in
    using SpeciesColors = QList<QPair<QString, QColor>>;

    /**
     * @brief Build the dialog for @p initial in the world described by @p env
     * @param initial       settings to edit; anything the dialog cannot express
     *                      is carried through to settings() untouched
     * @param env           snapshot of the simulation the settings apply to
     * @param speciesColors current per-species colours, one row per species
     * @param tab           tab to open on, clamped into range
     * @param parent        parent widget
     */
    DumpImageSettingsDialog(const DumpImageSettings &initial, const ImageSettingsEnv &env,
                            const SpeciesColors &speciesColors, int tab = 0,
                            QWidget *parent = nullptr);

    ~DumpImageSettingsDialog() override = default;

    DumpImageSettingsDialog()                                           = delete;
    DumpImageSettingsDialog(const DumpImageSettingsDialog &)            = delete;
    DumpImageSettingsDialog(DumpImageSettingsDialog &&)                 = delete;
    DumpImageSettingsDialog &operator=(const DumpImageSettingsDialog &) = delete;
    DumpImageSettingsDialog &operator=(DumpImageSettingsDialog &&)      = delete;

    /**
     * @brief The settings as the controls currently express them
     *
     * Starts from the settings the dialog was given and overlays what the
     * widgets say, which is not the same as building a fresh struct: a field
     * whose editor holds invalid text deliberately keeps its previous value,
     * and the fields this dialog has no control for (image size, the SSAO seed,
     * the per-species colour and diameter tables the caller derives, the movie
     * frame rate and bit rate) are carried through untouched.
     *
     * Const and repeatable -- calling it does not disturb the dialog.
     */
    [[nodiscard]] DumpImageSettings settings() const;

    /**
     * @brief The per-species colours as the table currently expresses them
     *
     * A name Qt cannot parse as a colour keeps the row's previous RGB, which is
     * how a name typed into the box survives alongside the picker's #rrggbb.
     */
    [[nodiscard]] SpeciesColors speciesColors() const;

signals:
    /// The Help button was pressed; @p page is the documentation page to open.
    void helpRequested(const QString &page);

private:
    /// one grid cut plane: enable, coordinate, and colour source
    struct PlaneRow {
        QCheckBox *show       = nullptr;
        QDoubleSpinBox *coord = nullptr;
        SourceRow source;
    };

    void loadColorMapSpec(const ColorMapSpec &spec);          ///< spec -> widgets
    void storeColorMapSpec(ColorMapSpec &spec) const;         ///< widgets -> spec

    const DumpImageSettings m_initial; ///< what settings() starts from
    const ImageSettingsEnv m_env;
    SpeciesColors m_speciesColors;     ///< the colours the table was built from

    // ---- Particles ---------------------------------------------------------
    QCheckBox *particleshow = nullptr;
    QComboBox *mixbox = nullptr, *pcolorbox = nullptr, *regionbox = nullptr, *pdiambox = nullptr;
    QRadioButton *diamattr = nullptr, *diamnum = nullptr;
    QLineEdit *pdiamval = nullptr;
    QList<QLineEdit *> colnames, coldiams;
    QList<QPushButton *> colicons;

    // ---- Grid --------------------------------------------------------------
    QCheckBox *gridshow = nullptr, *glineshow = nullptr;
    SourceRow gridsource;
    QLineEdit *gcolorrows = nullptr, *glinediam = nullptr, *glinecolor = nullptr;
    QComboBox *gridgroupbox = nullptr;

    // ---- Grid planes -------------------------------------------------------
    PlaneRow planes[3];

    // ---- Surfaces ----------------------------------------------------------
    QCheckBox *surfshow = nullptr, *slineshow = nullptr;
    SourceRow surfsource;
    QLineEdit *surfonecolor = nullptr, *surfdiam = nullptr, *scolorrows = nullptr;
    QLineEdit *slinediam = nullptr, *slinecolor = nullptr;
    QComboBox *surfgroupbox = nullptr;

    // ---- Box / axes --------------------------------------------------------
    QCheckBox *boxshow = nullptr, *subboxshow = nullptr, *axesshow = nullptr;
    QLineEdit *boxdiam = nullptr, *boxcolor = nullptr;
    QLineEdit *subboxdiam = nullptr, *subboxcolor = nullptr;
    QLineEdit *axeslen = nullptr, *axesdiam = nullptr;

    // ---- Camera ------------------------------------------------------------
    QDoubleSpinBox *thetaval = nullptr, *phival = nullptr;
    QLineEdit *thetavar = nullptr, *phivar = nullptr;
    QRadioButton *centerdynamic = nullptr;
    QDoubleSpinBox *centerspin[3] = {nullptr, nullptr, nullptr};
    QLineEdit *centervar[3]       = {nullptr, nullptr, nullptr};
    QLineEdit *upval[3]           = {nullptr, nullptr, nullptr};
    QLineEdit *zoomval = nullptr, *zoomvar = nullptr;

    // ---- Quality -----------------------------------------------------------
    QCheckBox *ssaoshow = nullptr, *fsaashow = nullptr, *gradientshow = nullptr;
    QDoubleSpinBox *ssaoval = nullptr;
    QSlider *shinyslider    = nullptr;
    QLineEdit *bgcolor = nullptr, *bg2color = nullptr;
    QSlider *lightslider[4] = {nullptr, nullptr, nullptr, nullptr};

    // ---- Colour maps -------------------------------------------------------
    //
    // The six specs and the selected mode are members rather than locals
    // because the mode combo's handler reads and writes them.  As locals of a
    // function that returned only after exec(), that was survivable; in a
    // dialog that outlives its constructor it would be a dangling reference.
    QComboBox *modebox = nullptr, *mapbox = nullptr;
    QCheckBox *mapactive = nullptr, *maprev = nullptr;
    QLineEdit *mapmin = nullptr, *mapmax = nullptr, *mapdelta = nullptr;
    QRadioButton *stylec = nullptr, *styled = nullptr, *styles = nullptr;
    QRadioButton *rangef = nullptr, *rangea = nullptr;
    ColorMapSpec cmapspec[DumpImageSettings::NUM_CMAP_MODES];
    int curmode = 0;
};

#endif

// Local Variables:
// c-basic-offset: 4
// End:
