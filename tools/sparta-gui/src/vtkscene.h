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

#ifndef VTKSCENE_H
#define VTKSCENE_H

// This widget is only compiled when SPARTA-GUI is built with an external VTK
// library (-D SPARTA_GUI_USE_VTK=on).  It renders the native VTK files written by
// the SPARTA "dump particle/vtk", "dump grid/vtk" and "dump surf/vtk" styles.
//
// To stay compatible with any system VTK -- including the common case where VTK's
// Qt integration was built against Qt5 and therefore cannot be linked into this
// Qt6 program -- the viewer does NOT use VTK's QVTKOpenGLNativeWidget.  Instead it
// renders the scene off-screen with VTK's OpenGL backend and blits the resulting
// frame into a plain Qt6 widget, translating Qt mouse/wheel events into camera
// moves.  It is a deliberately light-weight, interactive viewer (rotate/zoom/pan,
// color by a scalar field, pick a colormap) and leaves heavier analysis to
// ParaView (see the "Export to ParaView" dialog).

#include "viewersource.h"

#include <QImage>
#include <QMainWindow>
#include <QPoint>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <vtkSmartPointer.h>

#include <functional>

class QComboBox;
class QCheckBox;
class QLabel;
class QTimer;

class vtkActor;
class vtkDataSet;
class vtkDataSetMapper;
class vtkRenderWindow;
class vtkRenderer;
class vtkScalarBarActor;
class vtkScalarsToColors;

/**
 * @brief Off-screen VTK render surface embedded as a Qt widget.
 *
 * Owns a VTK render window (off-screen) and renderer.  On every change it renders
 * to the off-screen buffer, copies the pixels into a QImage and repaints.  Mouse
 * drag rotates (left) or pans (right/middle) the camera and the wheel zooms, so
 * the scene is fully interactive without linking VTK's Qt module.
 */
class VtkRenderArea : public QWidget {
public:
    explicit VtkRenderArea(QWidget *parent = nullptr);
    ~VtkRenderArea() override;

    vtkRenderer *renderer() const;
    vtkRenderWindow *window() const;

    /// @brief Re-render the scene off-screen and repaint the widget.
    void requestRender();
    /// @brief Frame all actors and re-render.
    void resetCamera();
    /// @brief Grab the current frame as a QImage (for screenshots).
    QImage grabFrame();

    /// @brief Install a click handler (for face/point picking).  When set, a
    /// left-button *click* (press+release without dragging) calls @p cb with the
    /// widget-space position instead of being treated as a camera move; drags
    /// still rotate/pan.  Pass a null function to disable picking.
    void setPickCallback(std::function<void(const QPoint &)> cb);

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    void pan(int fromX, int fromY, int toX, int toY); // display-space camera pan

    vtkSmartPointer<vtkRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderer> ren;
    QImage frame;
    QPoint lastPos;
    Qt::MouseButton dragButton = Qt::NoButton;
    std::function<void(const QPoint &)> pickCallback;
    bool dragMoved = false;
};

/**
 * @brief Interactive VTK 3D scene for SPARTA particle/grid/surf data.
 *
 * A toolbar (open file, color-by-field, colormap, edges, legend, camera reset,
 * screenshot) above a @ref VtkRenderArea, with a Filters menu and a status
 * line.  Datasets read from @c .vtu / @c .vtp / @c .vtk files become separate
 * layers.
 *
 * This is a plain widget rather than a window so that it can be shown either
 * inside the viewer panel or, via @ref SceneWindow, on its own.
 */
class VtkScene : public ViewerSource {
    Q_OBJECT

public:
    /// @brief The kind of SPARTA data a layer holds (drives default appearance).
    enum class Kind { Particles, Grid, Surface, Generic };

    explicit VtkScene(QWidget *parent = nullptr);
    ~VtkScene() override;

    /**
     * @brief Read a VTK dataset file and add it to the scene as a named layer.
     * @param path  path to a @c .vtu, @c .vtp or @c .vtk file
     * @param label human-readable layer name (shown in the status line)
     * @param kind  particle/grid/surface (selects default point size, color, ...)
     * @param err   optional error-message sink
     * @return true on success
     *
     * Named apart from the in-memory overload below on purpose: the two used to
     * be addDataset() and addDataSet(), one capital letter apart, which is a
     * typo waiting to compile.
     */
    bool addDatasetFile(const QString &path, const QString &label, Kind kind,
                        QString *err = nullptr);

    /// @brief Add an already-built VTK dataset as a layer (no file needed).
    void addDataset(vtkDataSet *data, const QString &label, Kind kind);

    /// @brief Programmatically color by a named field (e.g. "leak"); no-op if absent.
    void setColorField(const QString &name);

    /**
     * @brief Show or hide every layer of one kind.
     *
     * The three kinds SPARTA writes -- particles, grid cells, surface elements
     * -- occupy the same space, and which of them is worth looking at changes
     * from moment to moment: the grid hides the surface inside it, the
     * particles hide both. Toggling is therefore a property of the scene rather
     * than of a layer, so a category stays hidden as later frames arrive.
     */
    void setKindVisible(Kind kind, bool on);

    /// @brief Is this kind currently shown? (True for a kind with no layers.)
    [[nodiscard]] bool kindVisible(Kind kind) const;

    /// @brief How many layers of a kind the scene holds.
    [[nodiscard]] int layerCount(Kind kind) const;

    /// @brief Remove every layer from the scene.
    void clearScene();

    /// @brief Reset the camera to frame all layers.
    void resetView();

    // --- ViewerSource ---
    [[nodiscard]] QString sourceLabel() const override { return QStringLiteral("3D"); }
    [[nodiscard]] QIcon sourceIcon() const override;
    [[nodiscard]] QString sourceTip() const override
    {
        return QStringLiteral("The interactive 3D scene");
    }
    [[nodiscard]] QString emptyTitle() const override
    {
        return QStringLiteral("No 3D data yet");
    }
    [[nodiscard]] QString emptyTip() const override
    {
        return QStringLiteral(
            "Run \u25b8 3D Snapshot builds a scene from the simulation as it stands.\n\n"
            "For a scene that follows a run, add a VTK dump to your input deck and run it:\n"
            "    dump 1 grid/vtk all 100 grid.*.vtu\n\n"
            "Files written earlier can be opened from this panel's Open button.");
    }
    [[nodiscard]] bool hasContent() const override { return !layers.isEmpty(); }
    [[nodiscard]] QImage currentImage() const override;
    QMenu *sourceMenu() override { return filtersMenu; }

private slots:
    void openFileDialog();
    void onColorArrayChanged();
    void onColorMapChanged();
    void onEdgesToggled(bool on);
    void onScalarBarToggled(bool on);
    void saveScreenshot();

    // Feature 9: in-app field post-processing (heavier analysis stays in ParaView)
    void applyCutPlane();
    void applyIsoSurface();
    void applyFieldCalculator();
    void applyLineProbe();
    void togglePointProbe(bool on);

private:
    /// @brief One dataset in the scene: its data, mapper and actor plus metadata.
    struct Layer {
        vtkSmartPointer<vtkDataSet> data;
        vtkSmartPointer<vtkDataSetMapper> mapper;
        vtkSmartPointer<vtkActor> actor;
        QString label;
        Kind kind = Kind::Generic;
    };

    void buildUi();
    /// Put @p msg on the scene's own status line, optionally clearing it after
    /// @p ms milliseconds. Replaces QMainWindow::statusBar(), which a plain
    /// widget does not have.
    void showStatus(const QString &msg, int ms = 0);
    void addLayer(const vtkSmartPointer<vtkDataSet> &data, const QString &label, Kind kind);
    void refreshArrayCombo();
    void applyColoring();
    /// @brief The dataset most recently added (target of the filter actions), or null.
    vtkDataSet *currentData() const;
    /// @brief Names of the point-scalar arrays available across the layers.
    QStringList pointArrayNames() const;
    void onProbePick(const QPoint &pos); // point-probe click handler
    bool arrayRange(const QString &array, bool pointData, double range[2]) const;
    static vtkSmartPointer<vtkDataSet> readDataSet(const QString &path, QString *err);

    VtkRenderArea *renderArea = nullptr;
    vtkSmartPointer<vtkScalarBarActor> scalarBar;
    vtkSmartPointer<vtkScalarsToColors> colorMap;

    QList<Layer> layers;

    /// @brief Apply the per-kind show/hide state to every layer's actor.
    void applyKindVisibility();
    /// @brief Enable each kind's toggle only while the scene has layers of it.
    void syncKindBoxes();

    QComboBox *arrayCombo   = nullptr;
    QComboBox *cmapCombo    = nullptr;
    QCheckBox *edgesBox     = nullptr;
    QCheckBox *scalarBarBox = nullptr;

    /// Indexed by Kind. Generic has no toggle: it is whatever the user opened
    /// by hand from a file, and hiding that behind a category button would be
    /// a control with nothing predictable behind it.
    static constexpr int NKinds = 3;
    QCheckBox *kindBox[NKinds]  = {};
    bool kindShown[NKinds]      = {true, true, true};
    QLabel *infoLabel       = nullptr;
    QMenu *filtersMenu      = nullptr;
    QTimer *statusTimer     = nullptr;
    QString restingStatus;  ///< what the status line says when no message is up

    /// @brief Combo item-data bit marking a "point" (vs "cell") array entry.
    static constexpr int RolePointData = 0x1;
};

/**
 * @brief A @ref VtkScene on its own, as a top-level window.
 *
 * The STL import wizard needs this: it is itself a dialog running over the main
 * window, and its leak view is part of that workflow, so the result must not be
 * docked into the main window behind the dialog the user is still in.
 */
class SceneWindow : public QMainWindow {
    Q_OBJECT

public:
    using Kind = VtkScene::Kind;

    explicit SceneWindow(QWidget *parent = nullptr);
    ~SceneWindow() override;

    SceneWindow(const SceneWindow &)            = delete;
    SceneWindow &operator=(const SceneWindow &) = delete;

    [[nodiscard]] VtkScene *scene() const { return view; }

    bool addDatasetFile(const QString &path, const QString &label, Kind kind,
                        QString *err = nullptr);
    void addDataset(vtkDataSet *data, const QString &label, Kind kind);
    void setColorField(const QString &name);
    void clearScene();
    void resetView();

    /// @brief Show, raise and activate the window.
    void showViewer();

    [[nodiscard]] bool hasContent() const;

private:
    VtkScene *view = nullptr;
};

#endif // VTKSCENE_H

// Local Variables:
// c-basic-offset: 4
// End:
