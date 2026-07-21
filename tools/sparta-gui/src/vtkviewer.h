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

#ifndef VTKVIEWER_H
#define VTKVIEWER_H

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

#include <QImage>
#include <QMainWindow>
#include <QPoint>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <vtkSmartPointer.h>

class QComboBox;
class QCheckBox;
class QLabel;

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

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    void pan(int fromX, int fromY, int toX, int toY); // display-space camera pan

    vtkSmartPointer<vtkRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderer> ren;
    QImage frame;
    QPoint lastPos;
    Qt::MouseButton dragButton = Qt::NoButton;
};

/**
 * @brief Interactive VTK-based 3D viewer window for SPARTA particle/grid/surf data.
 *
 * A standalone top-level window: a toolbar (open file, color-by-field, colormap,
 * edges, legend, reset, screenshot) above a @ref VtkRenderArea.  Datasets loaded
 * from @c .vtu / @c .vtp / @c .vtk files are shown as separate layers.
 */
class VtkViewer : public QMainWindow {
    Q_OBJECT

public:
    /// @brief The kind of SPARTA data a layer holds (drives default appearance).
    enum class Kind { Particles, Grid, Surface, Generic };

    explicit VtkViewer(QWidget *parent = nullptr);
    ~VtkViewer() override;

    VtkViewer(const VtkViewer &)            = delete;
    VtkViewer &operator=(const VtkViewer &) = delete;

    /**
     * @brief Load a VTK dataset file and add it to the scene as a named layer.
     * @param path  path to a @c .vtu, @c .vtp or @c .vtk file
     * @param label human-readable layer name (shown in the status line)
     * @param kind  particle/grid/surface (selects default point size, color, ...)
     * @param err   optional error-message sink
     * @return true on success
     */
    bool addDataset(const QString &path, const QString &label, Kind kind, QString *err = nullptr);

    /// @brief Add an already-built VTK dataset as a layer (no file needed).
    void addDataSet(vtkDataSet *data, const QString &label, Kind kind);

    /// @brief Programmatically color by a named field (e.g. "leak"); no-op if absent.
    void setColorField(const QString &name);

    /// @brief Remove every layer from the scene.
    void clearScene();

    /// @brief Reset the camera to frame all layers.
    void resetView();

    /// @brief Show, raise and activate the viewer window.
    void showViewer();

    /// @brief Whether the scene currently holds any layer.
    bool hasContent() const { return !layers.isEmpty(); }

private slots:
    void openFileDialog();
    void onColorArrayChanged();
    void onColorMapChanged();
    void onEdgesToggled(bool on);
    void onScalarBarToggled(bool on);
    void saveScreenshot();

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
    void addLayer(const vtkSmartPointer<vtkDataSet> &data, const QString &label, Kind kind);
    void refreshArrayCombo();
    void applyColoring();
    bool arrayRange(const QString &array, bool pointData, double range[2]) const;
    static vtkSmartPointer<vtkDataSet> readDataSet(const QString &path, QString *err);

    VtkRenderArea *renderArea = nullptr;
    vtkSmartPointer<vtkScalarBarActor> scalarBar;
    vtkSmartPointer<vtkScalarsToColors> colorMap;

    QList<Layer> layers;

    QComboBox *arrayCombo   = nullptr;
    QComboBox *cmapCombo    = nullptr;
    QCheckBox *edgesBox     = nullptr;
    QCheckBox *scalarBarBox = nullptr;
    QLabel *infoLabel       = nullptr;

    /// @brief Combo item-data bit marking a "point" (vs "cell") array entry.
    static constexpr int RolePointData = 0x1;
};

#endif // VTKVIEWER_H

// Local Variables:
// c-basic-offset: 4
// End:
