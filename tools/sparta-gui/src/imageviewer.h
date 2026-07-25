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

#ifndef IMAGEVIEWER_H
#define IMAGEVIEWER_H

#include "dumpimage.h"
#include "viewersource.h"

#include <QColor>
#include <QImage>
#include <QList>
#include <QMap>
#include <QPair>
#include <QSize>
#include <QString>
#include <QStringList>

class QAction;
class QMenuBar;
class QComboBox;
class QEvent;
class QLabel;
class QObject;
class QScrollArea;
class ViewerDisplay;
class QShowEvent;
class SpartaWrapper;
class SpartaGui;

/**
 * @brief Dialog for viewing and manipulating SPARTA snapshot images
 *
 * This class provides an image viewer dialog for displaying SPARTA snapshots
 * created by the `dump image` command. It allows interactive manipulation of
 * all dump image visualization parameters -- particles, grid (volume and cut
 * planes), surfaces, box/axes, camera, render quality, and per-mode color
 * maps. Changes regenerate the image through the SPARTA library interface
 * (dump image + run 0 + undump).
 */
class ImageViewer : public ViewerSource {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param fileName Name of the input deck (used for the temp image files)
     * @param sparta Pointer to SpartaWrapper for regenerating images
     * @param spartagui Pointer to SpartaGui for sending signals
     * @param parent Parent widget
     */
    explicit ImageViewer(const QString &fileName, SpartaWrapper *sparta, SpartaGui *spartagui,
                         QWidget *parent = nullptr);

    /**
     * @brief Destructor
     */
    ~ImageViewer() override = default;

    ImageViewer()                               = delete;
    ImageViewer(const ImageViewer &)            = delete;
    ImageViewer(ImageViewer &&)                 = delete;
    ImageViewer &operator=(const ImageViewer &) = delete;
    ImageViewer &operator=(ImageViewer &&)      = delete;

private slots:
    void saveAs();             ///< Save image to file
    void copy();               ///< Copy image to clipboard
    void quit();               ///< Close application
    void getHelp();            ///< Open online help
    void editSize();           ///< Edit image dimensions
    void resetView();          ///< Reset view to defaults
    void resetWindowSize();    ///< Resize window to fit the configured image size
    void toggleSsao();         ///< Toggle screen-space ambient occlusion
    void toggleFsaa();         ///< Toggle full-scene anti-aliasing
    void toggleShiny();        ///< Toggle shiny/specular rendering
    void toggleParticles();    ///< Toggle particle display
    void toggleGrid();         ///< Toggle grid volume rendering
    void toggleSurf();         ///< Toggle surface element display
    void toggleBox();          ///< Toggle simulation box display
    void toggleAxes();         ///< Toggle coordinate axes display
    void doZoomIn();           ///< Zoom in view
    void doZoomOut();          ///< Zoom out view
    void doRotLeft();          ///< Rotate view left (decrease phi)
    void doRotRight();         ///< Rotate view right (increase phi)
    void doRotUp();            ///< Rotate view up (decrease theta)
    void doRotDown();          ///< Rotate view down (increase theta)
    void doRecenter();         ///< Reset the view center to the box center
    void cmdToClipboard();     ///< Copy dump image command to clipboard
    void movieToClipboard();   ///< Copy dump movie command to clipboard
    void resetColors();        ///< Restore the default species colors and lights
    void loadColors();         ///< Load species colors and lighting from JSON file
    void saveColors();         ///< Save species colors and lighting to JSON file
    void changeMixture(int);   ///< Change mixture selection
    void openSettings();       ///< Open the tabbed settings dialog (tab from sender)

public:
    /**
     * @brief Generate image using current settings
     *
     * Constructs and executes a SPARTA dump image command with the current
     * visualization parameters and updates the displayed image.  Shows a
     * message and returns when no simulation box or grid is defined yet.
     */
    void createImage();

    // --- ViewerSource ---
    [[nodiscard]] QString sourceLabel() const override { return QStringLiteral("Snapshot"); }
    [[nodiscard]] QIcon sourceIcon() const override;
    [[nodiscard]] QString sourceTip() const override
    {
        return QStringLiteral("The rendered SPARTA snapshot");
    }
    [[nodiscard]] QString emptyTip() const override
    {
        return QStringLiteral("No render yet: use Run > Create Image");
    }
    [[nodiscard]] bool hasContent() const override;
    [[nodiscard]] QImage currentImage() const override;

protected:

    bool eventFilter(QObject *watched, QEvent *event) override; ///< Intercept Alt-keystrokes
    void showEvent(QShowEvent *event) override; ///< Redo the initial window fit once shown

private:
    void createActions();     ///< Setup menu actions
    void updateActions();     ///< Update action states
    void adjustWindowSize();  ///< Auto-resize window to fit image
    void readImageSettings(); ///< Read snapshot settings from QSettings into params
    void syncButtons();       ///< Update toolbar button check states from params

    /// Refresh the params members derived from the SPARTA state (dimension,
    /// species colors, mixture) right before building the render command
    void gatherSettings();

    /// @name settings dialog (implemented in imageviewersettings.cpp)
    /// @{
    /// Show the tabbed dump-image settings dialog opened at tab @p tab;
    /// applies the widget state to params and re-renders on accept
    void settingsDialog(int tab);
    /// value sources ("proc" + c_/f_/v_ references) offered for grid/surf coloring
    QStringList valueSources(bool withproc, bool withone);
    /// @}

private:
    ViewerDisplay *display;  ///< Scroll area, label, and the fit-to-panel rule
    QMenuBar *menuBar;       ///< Menu bar
    QPoint dragLast;         ///< last mouse pos during an interactive view drag
    bool dragging = false;   ///< true while dragging to rotate/pan the render

    QAction *saveAsAct; ///< Save As action
    QAction *copyAct;   ///< Copy action
    QAction *cmdAct;    ///< Copy dump image command action
    QAction *movieAct;  ///< Copy dump movie command action

    QMap<QString, QString> fix_map;     ///< Fix style to help page mapping
    QMap<QString, QString> compute_map; ///< Compute style to help page mapping
    QStringList image_computes;         ///< compute styles with per-grid/per-surf data
    QStringList image_fixes;            ///< fix styles with per-grid/per-surf data
    SpartaWrapper *sparta;              ///< SPARTA interface for image generation
    SpartaGui *spartagui;               ///< Main widget pointer for receiving signals
    QString filename;                   ///< Input deck name (basis of temp image files)
    QString renderdumpid = "SPARTA_GUI_IMAGE"; ///< dump ID used for rendering

    DumpImageSettings params; ///< the complete dump image option state

    /// per-species display colors (name + RGB); translated into the pcolor
    /// and custom color rows of params by gatherSettings()
    QList<QPair<QString, QColor>> color_list;

    bool shutdown = false; ///< flag if class has entered the destructor
};
#endif

// Local Variables:
// c-basic-offset: 4
// End:
