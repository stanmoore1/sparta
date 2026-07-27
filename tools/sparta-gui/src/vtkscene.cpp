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

#include "vtkscene.h"

#include "chartviewer.h"
#include "constants.h"
#include "helpers.h"
#include "plotdata.h"
#include "vtkfilters.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QIcon>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QTimer>
#include <QToolBar>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <vtkPropPicker.h>

#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkCamera.h>
#include <vtkCellData.h>
#include <vtkColorTransferFunction.h>
#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkDataSetMapper.h>
#include <vtkDataSetReader.h>
#include <vtkLookupTable.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkScalarBarActor.h>
#include <vtkSmartPointer.h>
#include <vtkTextProperty.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkXMLUnstructuredGridReader.h>

#include <algorithm>
#include <cmath>

namespace {

// build a range-adapted colormap; both vtkLookupTable and vtkColorTransferFunction
// derive from vtkScalarsToColors, so the caller applies whichever we return the
// same way (mapper->SetLookupTable / scalarBar->SetLookupTable).
vtkSmartPointer<vtkScalarsToColors> makeColorMap(int idx, double lo, double hi)
{
    if (hi <= lo) hi = lo + 1.0; // avoid a degenerate range
    switch (idx) {
        case 1: { // cool-to-warm diverging (ParaView default flavor)
            auto ctf         = vtkSmartPointer<vtkColorTransferFunction>::New();
            const double mid = 0.5 * (lo + hi);
            ctf->AddRGBPoint(lo, 0.230, 0.299, 0.754);
            ctf->AddRGBPoint(mid, 0.865, 0.865, 0.865);
            ctf->AddRGBPoint(hi, 0.706, 0.016, 0.150);
            return ctf;
        }
        case 2: { // viridis-like perceptually uniform sequential
            auto ctf         = vtkSmartPointer<vtkColorTransferFunction>::New();
            const double t[] = {0.0, 0.25, 0.5, 0.75, 1.0};
            const double r[] = {0.267, 0.231, 0.128, 0.369, 0.993};
            const double g[] = {0.005, 0.318, 0.567, 0.789, 0.906};
            const double b[] = {0.329, 0.545, 0.551, 0.383, 0.144};
            for (int i = 0; i < 5; ++i) ctf->AddRGBPoint(lo + t[i] * (hi - lo), r[i], g[i], b[i]);
            return ctf;
        }
        case 3: { // grayscale
            auto lut = vtkSmartPointer<vtkLookupTable>::New();
            lut->SetHueRange(0.0, 0.0);
            lut->SetSaturationRange(0.0, 0.0);
            lut->SetValueRange(0.15, 1.0);
            lut->SetTableRange(lo, hi);
            lut->Build();
            return lut;
        }
        default: { // rainbow (blue -> red)
            auto lut = vtkSmartPointer<vtkLookupTable>::New();
            lut->SetHueRange(0.667, 0.0);
            lut->SetTableRange(lo, hi);
            lut->Build();
            return lut;
        }
    }
}

} // namespace

// ======================================================================== //
// VtkRenderArea -- off-screen render surface with Qt-driven camera control  //
// ======================================================================== //

VtkRenderArea::VtkRenderArea(QWidget *parent) : QWidget(parent)
{
    setMinimumSize(320, 240);
    setFocusPolicy(Qt::StrongFocus);
    setAttribute(Qt::WA_OpaquePaintEvent, true);

    renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetOffScreenRendering(1);
    renderWindow->SetMultiSamples(8); // anti-aliasing
    ren = vtkSmartPointer<vtkRenderer>::New();
    ren->SetBackground(0.15, 0.16, 0.18);
    ren->SetBackground2(0.30, 0.32, 0.36);
    ren->GradientBackgroundOn();
    renderWindow->AddRenderer(ren);
}

VtkRenderArea::~VtkRenderArea() = default;

vtkRenderer *VtkRenderArea::renderer() const
{
    return ren;
}

vtkRenderWindow *VtkRenderArea::window() const
{
    return renderWindow;
}

QImage VtkRenderArea::grabFrame()
{
    const int *size = renderWindow->GetSize();
    const int w = size[0], h = size[1];
    if (w <= 0 || h <= 0) return QImage();

    auto pixels = vtkSmartPointer<vtkUnsignedCharArray>::New();
    renderWindow->GetRGBACharPixelData(0, 0, w - 1, h - 1, /*front=*/0, pixels);

    QImage img(w, h, QImage::Format_RGBA8888);
    const unsigned char *src = pixels->GetPointer(0);
    // VTK's frame buffer is bottom-up; QImage is top-down, so flip rows.
    for (int row = 0; row < h; ++row)
        memcpy(img.scanLine(h - 1 - row), src + static_cast<size_t>(row) * w * 4, w * 4);
    img.setDevicePixelRatio(devicePixelRatioF());
    return img;
}

void VtkRenderArea::requestRender()
{
    if (renderWindow->GetSize()[0] <= 0) return;
    renderWindow->Render();
    frame = grabFrame();
    update();
}

void VtkRenderArea::resetCamera()
{
    // Point the camera back down -Z with +Y up before re-framing.
    // vtkRenderer::ResetCamera() on its own only moves the camera along the
    // direction it is already looking, so after a drag it re-frames the scene
    // at whatever angle the drag left it at. Dragging is the only way to turn
    // this view, so without this there is no way back to a known viewpoint --
    // a user who tumbles the scene and loses their bearings has to close the
    // window and start again.
    auto *cam = ren->GetActiveCamera();
    cam->SetPosition(0.0, 0.0, 1.0);
    cam->SetFocalPoint(0.0, 0.0, 0.0);
    cam->SetViewUp(0.0, 1.0, 0.0);
    ren->ResetCamera();
    requestRender();
}

void VtkRenderArea::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    if (frame.isNull()) {
        p.fillRect(rect(), QColor(38, 41, 46));
        return;
    }
    p.drawImage(0, 0, frame);
}

void VtkRenderArea::resizeEvent(QResizeEvent *event)
{
    const double dpr = devicePixelRatioF();
    const int w      = std::max(1, int(event->size().width() * dpr));
    const int h      = std::max(1, int(event->size().height() * dpr));
    renderWindow->SetSize(w, h);
    requestRender();
}

void VtkRenderArea::setPickCallback(std::function<void(const QPoint &)> cb)
{
    pickCallback = std::move(cb);
}

void VtkRenderArea::mousePressEvent(QMouseEvent *event)
{
    lastPos    = event->pos();
    dragButton = event->button();
    dragMoved  = false;
}

void VtkRenderArea::mouseReleaseEvent(QMouseEvent *event)
{
    // a left-button click that did not drag is a pick, when a handler is set
    if (pickCallback && event->button() == Qt::LeftButton && !dragMoved)
        pickCallback(event->pos());
    dragButton = Qt::NoButton;
}

void VtkRenderArea::mouseMoveEvent(QMouseEvent *event)
{
    if (dragButton == Qt::NoButton) return;
    const QPoint pos = event->pos();
    const int dx     = pos.x() - lastPos.x();
    const int dy     = pos.y() - lastPos.y();
    if (dx == 0 && dy == 0) return;
    // any real motion past a small threshold counts as a drag, not a click
    if (std::abs(dx) > 2 || std::abs(dy) > 2) dragMoved = true;

    auto *cam = ren->GetActiveCamera();
    if (dragButton == Qt::LeftButton) {
        // trackball rotate
        cam->Azimuth(-dx * 0.5);
        cam->Elevation(dy * 0.5);
        cam->OrthogonalizeViewUp();
        ren->ResetCameraClippingRange();
    } else {
        // pan (right/middle button): convert Qt logical coords to VTK display
        // pixels (bottom-up, device resolution)
        const double dpr = devicePixelRatioF();
        const int h      = renderWindow->GetSize()[1];
        pan(int(lastPos.x() * dpr), h - int(lastPos.y() * dpr), int(pos.x() * dpr),
            h - int(pos.y() * dpr));
    }
    lastPos = pos;
    requestRender();
}

void VtkRenderArea::wheelEvent(QWheelEvent *event)
{
    const double steps  = event->angleDelta().y() / 120.0;
    if (steps == 0.0) return;
    const double factor = std::pow(1.15, steps);
    ren->GetActiveCamera()->Dolly(factor);
    ren->ResetCameraClippingRange();
    requestRender();
}

void VtkRenderArea::pan(int fromX, int fromY, int toX, int toY)
{
    auto *cam = ren->GetActiveCamera();
    double viewFocus[4];
    cam->GetFocalPoint(viewFocus);

    // depth of the focal plane in display space
    ren->SetWorldPoint(viewFocus[0], viewFocus[1], viewFocus[2], 1.0);
    ren->WorldToDisplay();
    const double focalDepth = ren->GetDisplayPoint()[2];

    auto unproject = [&](int x, int y, double out[3]) {
        ren->SetDisplayPoint(double(x), double(y), focalDepth);
        ren->DisplayToWorld();
        const double *wp = ren->GetWorldPoint();
        const double winv = (wp[3] != 0.0) ? 1.0 / wp[3] : 1.0;
        out[0] = wp[0] * winv;
        out[1] = wp[1] * winv;
        out[2] = wp[2] * winv;
    };

    double newPick[3], oldPick[3];
    unproject(toX, toY, newPick);
    unproject(fromX, fromY, oldPick);

    const double motion[3] = {oldPick[0] - newPick[0], oldPick[1] - newPick[1],
                              oldPick[2] - newPick[2]};
    double fp[3], pos[3];
    cam->GetFocalPoint(fp);
    cam->GetPosition(pos);
    cam->SetFocalPoint(fp[0] + motion[0], fp[1] + motion[1], fp[2] + motion[2]);
    cam->SetPosition(pos[0] + motion[0], pos[1] + motion[1], pos[2] + motion[2]);
    ren->ResetCameraClippingRange();
}

// ======================================================================== //
// VtkScene -- toolbar + render area + coloring logic                        //
// ======================================================================== //

VtkScene::VtkScene(QWidget *parent) : ViewerSource(parent)
{
    buildUi();
}

VtkScene::~VtkScene() = default;

QIcon VtkScene::sourceIcon() const
{
    return QIcon(":/icons/x-office-drawing.svg");
}

QImage VtkScene::currentImage() const
{
    return renderArea ? renderArea->grabFrame() : QImage();
}

void VtkScene::showStatus(const QString &msg, int ms)
{
    if (!infoLabel) return;
    infoLabel->setText(msg);
    if (!statusTimer) {
        statusTimer = new QTimer(this);
        statusTimer->setSingleShot(true);
        connect(statusTimer, &QTimer::timeout, this,
                [this]() { infoLabel->setText(restingStatus); });
    }
    if (ms > 0)
        statusTimer->start(ms);
    else
        statusTimer->stop();
}

void VtkScene::buildUi()
{
    renderArea = new VtkRenderArea(this);
    renderArea->setObjectName("vtkRenderArea");

    // color legend, hidden until an array is chosen
    scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
    scalarBar->SetNumberOfLabels(5);
    scalarBar->SetBarRatio(0.15);
    scalarBar->GetTitleTextProperty()->SetFontSize(12);
    scalarBar->VisibilityOff();
    renderArea->renderer()->AddActor2D(scalarBar);

    auto *tb = new QToolBar("View", this);
    tb->setMovable(false);

    // Toolbar actions do not go through styleToolButtons(), so nothing here
    // picks up an accessible name from a tooltip the way the other viewers'
    // buttons do. Set the tooltips and name the generated buttons explicitly,
    // or every control in this window is anonymous to a screen reader -- and
    // unreachable by the GUI tests, which is why the 3D viewer had none.
    auto *openAct = tb->addAction(QIcon(":/icons/document-open.svg"), "Open VTK File...", this,
                                  &VtkScene::openFileDialog);
    openAct->setToolTip("Open a VTK data file (.vtu, .vtp or .vtk)");
    tb->addSeparator();

    tb->addWidget(new QLabel(" Color by: "));
    arrayCombo = new QComboBox(tb);
    arrayCombo->setObjectName("vtkArrayCombo");
    arrayCombo->setMinimumWidth(170);
    arrayCombo->setToolTip("Per-point or per-cell scalar field used to color the data");
    connect(arrayCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &VtkScene::onColorArrayChanged);
    tb->addWidget(arrayCombo);

    tb->addWidget(new QLabel("  Colormap: "));
    cmapCombo = new QComboBox(tb);
    cmapCombo->addItems({"Rainbow", "Cool to Warm", "Viridis", "Grayscale"});
    connect(cmapCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &VtkScene::onColorMapChanged);
    tb->addWidget(cmapCombo);

    tb->addSeparator();
    edgesBox = new QCheckBox("Edges", tb);
    edgesBox->setToolTip("Draw the outlines of grid cells / surface elements");
    connect(edgesBox, &QCheckBox::toggled, this, &VtkScene::onEdgesToggled);
    tb->addWidget(edgesBox);

    scalarBarBox = new QCheckBox("Legend", tb);
    scalarBarBox->setChecked(true);
    connect(scalarBarBox, &QCheckBox::toggled, this, &VtkScene::onScalarBarToggled);
    tb->addWidget(scalarBarBox);

    tb->addSeparator();
    auto *resetAct = tb->addAction(QIcon(":/icons/preferences-reset.svg"), "Reset View", this,
                                   &VtkScene::resetView);
    resetAct->setToolTip("Camera reset to the default view, framing all layers");
    auto *shotAct = tb->addAction(QIcon(":/icons/image-x-generic.svg"), "Save Screenshot...", this,
                                  &VtkScene::saveScreenshot);
    shotAct->setToolTip("Save the current 3D view to an image file");

    cmapCombo->setToolTip("Color map applied to the selected scalar field");
    scalarBarBox->setToolTip("Show the color scale for the selected field");

    for (auto *act : {openAct, resetAct, shotAct})
        nameFromToolTip(tb->widgetForAction(act));

    // Feature 9: in-app field post-processing (cut plane, iso-surface, probes,
    // field calculator).  Heavier analysis (streamlines, glyphs, volume
    // rendering) is left to the "Export to ParaView" dialog.
    auto *bar   = new QMenuBar(this);
    filtersMenu = bar->addMenu("&Filters");
    filtersMenu->addAction("Cut Plane...", this, &VtkScene::applyCutPlane);
    filtersMenu->addAction("Iso-surface...", this, &VtkScene::applyIsoSurface);
    filtersMenu->addAction("Field Calculator...", this, &VtkScene::applyFieldCalculator);
    filtersMenu->addAction("Line Probe...", this, &VtkScene::applyLineProbe);
    auto *ptProbe = filtersMenu->addAction("Point Probe (click points)");
    ptProbe->setCheckable(true);
    connect(ptProbe, &QAction::toggled, this, &VtkScene::togglePointProbe);
    filtersMenu->addSeparator();
    filtersMenu->addAction("Heavier analysis -> Export to ParaView", this,
                           []() {})->setEnabled(false);

    infoLabel = new QLabel("No data loaded.");
    infoLabel->setObjectName("status");
    infoLabel->setAccessibleName("Scene status and probe readout");
    restingStatus = infoLabel->text();

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(bar);
    layout->addWidget(tb);
    layout->addWidget(renderArea, 1);
    layout->addWidget(infoLabel);
}

vtkSmartPointer<vtkDataSet> VtkScene::readDataSet(const QString &path, QString *err)
{
    const QString ext  = QFileInfo(path).suffix().toLower();
    const QByteArray p = path.toLocal8Bit();
    // upcast the reader's concrete output to vtkDataSet* first, then wrap it, so
    // the smart-pointer constructor sees an exact vtkDataSet* (a derived*->base*
    // conversion inside the constructor call is ambiguous with some VTK builds).
    if (ext == "vtu") {
        auto rd = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
        rd->SetFileName(p.constData());
        rd->Update();
        if (vtkDataSet *out = rd->GetOutput()) return vtkSmartPointer<vtkDataSet>(out);
    } else if (ext == "vtp") {
        auto rd = vtkSmartPointer<vtkXMLPolyDataReader>::New();
        rd->SetFileName(p.constData());
        rd->Update();
        if (vtkDataSet *out = rd->GetOutput()) return vtkSmartPointer<vtkDataSet>(out);
    } else if (ext == "vtk") {
        auto rd = vtkSmartPointer<vtkDataSetReader>::New();
        rd->SetFileName(p.constData());
        rd->Update();
        if (vtkDataSet *out = rd->GetOutput()) return vtkSmartPointer<vtkDataSet>(out);
    } else if (err) {
        *err = QString("Unsupported file type '.%1' (expected .vtu, .vtp or .vtk)").arg(ext);
        return nullptr;
    }
    if (err) *err = QString("Could not read a VTK dataset from %1").arg(path);
    return nullptr;
}

void VtkScene::addLayer(const vtkSmartPointer<vtkDataSet> &data, const QString &label, Kind kind)
{
    Layer layer;
    layer.data   = data;
    layer.label  = label;
    layer.kind   = kind;
    layer.mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    layer.mapper->SetInputData(data);
    layer.mapper->ScalarVisibilityOff(); // solid color until an array is chosen
    layer.actor = vtkSmartPointer<vtkActor>::New();
    layer.actor->SetMapper(layer.mapper);

    auto *prop = layer.actor->GetProperty();
    switch (kind) {
        case Kind::Particles:
            prop->SetRepresentationToPoints();
            prop->SetPointSize(3.0);
            prop->SetColor(0.90, 0.90, 0.35); // pale yellow
            break;
        case Kind::Grid:
            prop->SetColor(0.55, 0.72, 0.90); // light blue
            prop->SetEdgeColor(0.20, 0.25, 0.30);
            break;
        case Kind::Surface: prop->SetColor(0.80, 0.80, 0.82); break; // light gray
        default: prop->SetColor(0.75, 0.75, 0.78); break;
    }
    prop->SetEdgeVisibility(edgesBox && edgesBox->isChecked());

    renderArea->renderer()->AddActor(layer.actor);
    layers.append(layer);

    refreshArrayCombo();
    applyColoring();
    resetView();

    vtkIdType np = 0, nc = 0;
    for (const auto &l : layers) {
        np += l.data->GetNumberOfPoints();
        nc += l.data->GetNumberOfCells();
    }
    restingStatus = QString("%1 layer(s), %2 points, %3 cells")
                        .arg(layers.size())
                        .arg(static_cast<qlonglong>(np))
                        .arg(static_cast<qlonglong>(nc));
    showStatus(restingStatus);
    emit contentChanged();
}

bool VtkScene::addDatasetFile(const QString &path, const QString &label, Kind kind, QString *err)
{
    auto data = readDataSet(path, err);
    if (!data || data->GetNumberOfPoints() == 0) {
        if (data && err) *err = QString("%1 contains no points").arg(path);
        return false;
    }
    addLayer(data, label, kind);
    return true;
}

void VtkScene::addDataset(vtkDataSet *data, const QString &label, Kind kind)
{
    if (data && data->GetNumberOfPoints() > 0)
        addLayer(vtkSmartPointer<vtkDataSet>(data), label, kind);
}

void VtkScene::setColorField(const QString &name)
{
    // pick the first combo entry whose base name matches (entries read "name  (point)")
    for (int i = 1; i < arrayCombo->count(); ++i) {
        QString t   = arrayCombo->itemText(i);
        const int c = t.lastIndexOf(QStringLiteral("  ("));
        if (c > 0) t = t.left(c);
        if (t == name) {
            arrayCombo->setCurrentIndex(i);
            return;
        }
    }
}

void VtkScene::clearScene()
{
    for (const auto &l : layers) renderArea->renderer()->RemoveActor(l.actor);
    layers.clear();
    scalarBar->VisibilityOff();
    refreshArrayCombo();
    restingStatus = "No data loaded.";
    showStatus(restingStatus);
    renderArea->requestRender();
    emit contentChanged();
}

void VtkScene::resetView()
{
    renderArea->resetCamera();
}


void VtkScene::refreshArrayCombo()
{
    const QString prev = arrayCombo->currentText();
    QSignalBlocker block(arrayCombo);
    arrayCombo->clear();
    arrayCombo->addItem("(solid color)");

    QStringList pointArrays, cellArrays;
    for (const auto &l : layers) {
        if (auto *pd = l.data->GetPointData())
            for (int i = 0; i < pd->GetNumberOfArrays(); ++i) {
                const char *n = pd->GetArrayName(i);
                if (n && *n && !pointArrays.contains(n)) pointArrays << n;
            }
        if (auto *cd = l.data->GetCellData())
            for (int i = 0; i < cd->GetNumberOfArrays(); ++i) {
                const char *n = cd->GetArrayName(i);
                if (n && *n && !cellArrays.contains(n)) cellArrays << n;
            }
    }
    pointArrays.sort();
    cellArrays.sort();
    for (const QString &n : pointArrays)
        arrayCombo->addItem(n + "  (point)", QVariant(RolePointData));
    for (const QString &n : cellArrays) arrayCombo->addItem(n + "  (cell)", QVariant(0));

    const int idx = arrayCombo->findText(prev);
    arrayCombo->setCurrentIndex(idx >= 0 ? idx : 0);
}

bool VtkScene::arrayRange(const QString &array, bool pointData, double range[2]) const
{
    bool found        = false;
    range[0] = range[1] = 0.0;
    const QByteArray a = array.toLocal8Bit();
    for (const auto &l : layers) {
        vtkDataArray *arr = nullptr;
        if (pointData && l.data->GetPointData())
            arr = l.data->GetPointData()->GetArray(a.constData());
        else if (!pointData && l.data->GetCellData())
            arr = l.data->GetCellData()->GetArray(a.constData());
        if (!arr) continue;
        double r[2];
        arr->GetRange(r, -1); // -1 = vector magnitude for multi-component arrays
        if (!found) {
            range[0] = r[0];
            range[1] = r[1];
            found    = true;
        } else {
            range[0] = std::min(range[0], r[0]);
            range[1] = std::max(range[1], r[1]);
        }
    }
    return found;
}

void VtkScene::applyColoring()
{
    const int sel = arrayCombo->currentIndex();
    if (sel <= 0) { // solid color
        for (const auto &l : layers) l.mapper->ScalarVisibilityOff();
        scalarBar->VisibilityOff();
        renderArea->requestRender();
        return;
    }

    const bool pointData = (arrayCombo->currentData().toInt() & RolePointData) != 0;
    QString name         = arrayCombo->currentText();
    const int cut        = name.lastIndexOf(QStringLiteral("  ("));
    if (cut > 0) name = name.left(cut);

    double range[2];
    if (!arrayRange(name, pointData, range)) {
        for (const auto &l : layers) l.mapper->ScalarVisibilityOff();
        scalarBar->VisibilityOff();
        renderArea->requestRender();
        return;
    }

    colorMap = makeColorMap(cmapCombo->currentIndex(), range[0], range[1]);

    const QByteArray cname = name.toLocal8Bit();
    for (const auto &l : layers) {
        const bool has =
            pointData ? (l.data->GetPointData() && l.data->GetPointData()->GetArray(cname.constData()))
                      : (l.data->GetCellData() && l.data->GetCellData()->GetArray(cname.constData()));
        if (!has) {
            l.mapper->ScalarVisibilityOff();
            continue;
        }
        if (pointData)
            l.mapper->SetScalarModeToUsePointFieldData();
        else
            l.mapper->SetScalarModeToUseCellFieldData();
        l.mapper->SelectColorArray(cname.constData());
        l.mapper->SetColorModeToMapScalars();
        l.mapper->SetScalarRange(range[0], range[1]);
        l.mapper->SetLookupTable(colorMap);
        l.mapper->ScalarVisibilityOn();
    }

    scalarBar->SetLookupTable(colorMap);
    scalarBar->SetTitle(cname.constData());
    scalarBar->SetVisibility(scalarBarBox && scalarBarBox->isChecked());
    renderArea->requestRender();
}

void VtkScene::onColorArrayChanged()
{
    applyColoring();
}

void VtkScene::onColorMapChanged()
{
    applyColoring();
}

void VtkScene::onEdgesToggled(bool on)
{
    for (const auto &l : layers)
        if (l.kind != Kind::Particles) l.actor->GetProperty()->SetEdgeVisibility(on);
    renderArea->requestRender();
}

void VtkScene::onScalarBarToggled(bool on)
{
    scalarBar->SetVisibility(on && arrayCombo->currentIndex() > 0);
    renderArea->requestRender();
}

void VtkScene::openFileDialog()
{
    const QString path = QFileDialog::getOpenFileName(
        this, "Open VTK Dataset", QString(), "VTK datasets (*.vtu *.vtp *.vtk);;All files (*)");
    if (path.isEmpty()) return;
    QString err;
    if (!addDatasetFile(path, QFileInfo(path).fileName(), Kind::Generic, &err))
        critical(this, "Open VTK Dataset", "Could not open the selected file:", err);
}

void VtkScene::saveScreenshot()
{
    const QString path = QFileDialog::getSaveFileName(this, "Save Screenshot", "sparta-view.png",
                                                      "PNG image (*.png)");
    if (path.isEmpty()) return;
    renderArea->requestRender();
    const QImage img = renderArea->grabFrame();
    if (img.isNull() || !img.save(path))
        critical(this, "Save Screenshot", "Could not write the screenshot to:", path);
    else
        showStatus(QString("Saved screenshot to %1").arg(path), 5000);
}

// ======================================================================== //
// Feature 9 -- in-app field post-processing (stock VTK filters)            //
// ======================================================================== //

vtkDataSet *VtkScene::currentData() const
{
    return layers.isEmpty() ? nullptr : layers.last().data.Get();
}

QStringList VtkScene::pointArrayNames() const
{
    QStringList names;
    for (const auto &l : layers) {
        auto *pd = l.data ? l.data->GetPointData() : nullptr;
        if (!pd) continue;
        for (int i = 0; i < pd->GetNumberOfArrays(); ++i) {
            auto *a = pd->GetArray(i);
            if (a && a->GetName() && !names.contains(a->GetName())) names << a->GetName();
        }
    }
    return names;
}

void VtkScene::applyCutPlane()
{
    vtkDataSet *data = currentData();
    if (!data) { QMessageBox::information(this, "Cut Plane", "Load a dataset first."); return; }

    QStringList axes = {"X", "Y", "Z"};
    bool ok = false;
    const QString axis = QInputDialog::getItem(this, "Cut Plane", "Slice normal to axis:",
                                               axes, 0, false, &ok);
    if (!ok) return;
    const int ai = axes.indexOf(axis);

    double b[6];
    data->GetBounds(b);
    const double lo = b[2 * ai], hi = b[2 * ai + 1], mid = 0.5 * (lo + hi);
    const double pos = QInputDialog::getDouble(this, "Cut Plane",
                                               QString("%1 position:").arg(axis), mid, lo, hi,
                                               4, &ok);
    if (!ok) return;

    double origin[3] = {0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])};
    origin[ai] = pos;
    double normal[3] = {0, 0, 0};
    normal[ai] = 1.0;

    auto cut = VtkFilters::cutPlane(data, origin, normal);
    if (!cut || cut->GetNumberOfPoints() == 0) {
        QMessageBox::information(this, "Cut Plane", "The plane did not intersect the data.");
        return;
    }
    addLayer(cut, QString("cut %1=%2").arg(axis).arg(pos, 0, 'g', 4), Kind::Generic);
}

void VtkScene::applyIsoSurface()
{
    vtkDataSet *data = currentData();
    if (!data) { QMessageBox::information(this, "Iso-surface", "Load a dataset first."); return; }
    const QStringList arrays = pointArrayNames();
    if (arrays.isEmpty()) {
        QMessageBox::information(this, "Iso-surface", "This dataset has no point-scalar field.");
        return;
    }
    bool ok = false;
    const QString arr = QInputDialog::getItem(this, "Iso-surface", "Scalar field:", arrays, 0,
                                              false, &ok);
    if (!ok) return;
    double range[2] = {0, 1};
    arrayRange(arr, /*pointData=*/true, range);
    const double val = QInputDialog::getDouble(this, "Iso-surface", "Iso value:",
                                               0.5 * (range[0] + range[1]), range[0], range[1],
                                               6, &ok);
    if (!ok) return;

    auto iso = VtkFilters::isoSurface(data, arr, val);
    if (!iso || iso->GetNumberOfPoints() == 0) {
        QMessageBox::information(this, "Iso-surface", "No surface at that value.");
        return;
    }
    addLayer(iso, QString("iso %1=%2").arg(arr).arg(val, 0, 'g', 4), Kind::Generic);
}

void VtkScene::applyFieldCalculator()
{
    vtkDataSet *data = currentData();
    if (!data) { QMessageBox::information(this, "Field Calculator", "Load a dataset first."); return; }
    bool ok = false;
    const QString name = QInputDialog::getText(this, "Field Calculator", "New field name:",
                                               QLineEdit::Normal, "derived", &ok);
    if (!ok || name.isEmpty()) return;
    const QString expr = QInputDialog::getText(
        this, "Field Calculator",
        "Expression using existing fields (e.g. mag(v), 2*rho):", QLineEdit::Normal, "", &ok);
    if (!ok || expr.isEmpty()) return;

    auto out = VtkFilters::calculate(data, name, expr);
    if (!out) {
        QMessageBox::warning(this, "Field Calculator",
                             "Could not evaluate the expression (check the field names).");
        return;
    }
    // update the last layer in place so the new field joins the color menu
    Layer &l = layers.last();
    l.data = out;
    l.mapper->SetInputData(out);
    refreshArrayCombo();
    // color by the freshly computed field
    for (int i = 0; i < arrayCombo->count(); ++i)
        if (arrayCombo->itemText(i).contains(name)) { arrayCombo->setCurrentIndex(i); break; }
    applyColoring();
    renderArea->requestRender();
}

void VtkScene::applyLineProbe()
{
    vtkDataSet *data = currentData();
    if (!data) { QMessageBox::information(this, "Line Probe", "Load a dataset first."); return; }
    const QStringList arrays = pointArrayNames();
    if (arrays.isEmpty()) {
        QMessageBox::information(this, "Line Probe", "This dataset has no point field to sample.");
        return;
    }
    bool ok = false;
    const QString arr = QInputDialog::getItem(this, "Line Probe", "Sample field:", arrays, 0,
                                              false, &ok);
    if (!ok) return;

    double b[6];
    data->GetBounds(b);
    // default line: along X through the domain centre
    double p1[3] = {b[0], 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])};
    double p2[3] = {b[1], 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])};
    const int nsamp = QInputDialog::getInt(this, "Line Probe", "Number of samples:", 100, 2,
                                           100000, 1, &ok);
    if (!ok) return;

    auto line = VtkFilters::probeLine(data, p1, p2, nsamp);
    if (!line || line->GetNumberOfPoints() == 0) {
        QMessageBox::information(this, "Line Probe", "The probe returned no samples.");
        return;
    }
    auto *field = line->GetPointData() ? line->GetPointData()->GetArray(arr.toUtf8().constData())
                                       : nullptr;
    if (!field) {
        QMessageBox::information(this, "Line Probe", "The field was not sampled along the line.");
        return;
    }

    // plot the sampled scalar vs. arc length in a chart window
    std::vector<double> dist, vals;
    double x0[3];
    line->GetPoint(0, x0);
    for (vtkIdType i = 0; i < line->GetNumberOfPoints(); ++i) {
        double x[3];
        line->GetPoint(i, x);
        const double d = std::sqrt((x[0] - x0[0]) * (x[0] - x0[0]) +
                                   (x[1] - x0[1]) * (x[1] - x0[1]) +
                                   (x[2] - x0[2]) * (x[2] - x0[2]));
        dist.push_back(d);
        vals.push_back(field->GetTuple1(i));
    }

    // addColumn() appends a column *and* its name, so setColumnNames() must not
    // be called as well: doing both leaves the first columns empty, rowCount()
    // reads zero from them and loadData() discards the whole table.
    PlotData pdata;
    pdata.addColumn("distance", dist);
    pdata.addColumn(arr, vals);
    auto *win = new ChartWindow(QString("Line probe: %1").arg(arr), nullptr);
    win->setAttribute(Qt::WA_DeleteOnClose);
    win->setWindowTitle(QString("Line probe: %1 - SPARTA-GUI").arg(arr));
    win->setWindowIcon(QIcon(Cfg::MAIN_ICON));
    win->loadData(pdata, 0, {1});
    win->show();
}

void VtkScene::togglePointProbe(bool on)
{
    if (on) {
        renderArea->setPickCallback([this](const QPoint &p) { onProbePick(p); });
        showStatus("Point probe: click a point to read its field values.");
    } else {
        renderArea->setPickCallback(nullptr);
        showStatus(restingStatus);
    }
}

void VtkScene::onProbePick(const QPoint &pos)
{
    vtkDataSet *data = currentData();
    if (!data) return;

    const double dpr = renderArea->devicePixelRatioF();
    const int h      = renderArea->window()->GetSize()[1];
    auto picker = vtkSmartPointer<vtkPropPicker>::New();
    if (!picker->Pick(pos.x() * dpr, h - pos.y() * dpr, 0.0, renderArea->renderer())) {
        showStatus("Point probe: no geometry under the cursor.", 3000);
        return;
    }
    double world[3];
    picker->GetPickPosition(world);

    auto probed = VtkFilters::probePoint(data, world);
    QString msg = QString("(%1, %2, %3): ")
                      .arg(world[0], 0, 'g', 4).arg(world[1], 0, 'g', 4).arg(world[2], 0, 'g', 4);
    auto *pd = probed ? probed->GetPointData() : nullptr;
    if (pd && pd->GetNumberOfArrays() > 0) {
        QStringList parts;
        for (int i = 0; i < pd->GetNumberOfArrays(); ++i) {
            auto *a = pd->GetArray(i);
            if (!a || !a->GetName()) continue;
            if (a->GetNumberOfComponents() == 1)
                parts << QString("%1=%2").arg(a->GetName()).arg(a->GetTuple1(0), 0, 'g', 5);
        }
        msg += parts.join("  ");
    } else {
        msg += "(no sampled fields)";
    }
    showStatus(msg, 15000);
}

////////////////////////////////////////////////////////////////////////////////
// SceneWindow -- the same scene on its own                                   //
////////////////////////////////////////////////////////////////////////////////

SceneWindow::SceneWindow(QWidget *parent) : QMainWindow(parent), view(new VtkScene)
{
    setWindowTitle("SPARTA 3D Viewer");
    setWindowIcon(QIcon(Cfg::MAIN_ICON));
    setMinimumSize(Cfg::MINIMUM_WIDTH, Cfg::MINIMUM_HEIGHT);
    setCentralWidget(view);
}

SceneWindow::~SceneWindow() = default;

bool SceneWindow::addDatasetFile(const QString &path, const QString &label, Kind kind, QString *err)
{
    return view->addDatasetFile(path, label, kind, err);
}

void SceneWindow::addDataset(vtkDataSet *data, const QString &label, Kind kind)
{
    view->addDataset(data, label, kind);
}

void SceneWindow::setColorField(const QString &name)
{
    view->setColorField(name);
}

void SceneWindow::clearScene()
{
    view->clearScene();
}

void SceneWindow::resetView()
{
    view->resetView();
}

bool SceneWindow::hasContent() const
{
    return view->hasContent();
}

void SceneWindow::showViewer()
{
    show();
    raise();
    activateWindow();
    view->resetView();
}

// Local Variables:
// c-basic-offset: 4
// End:
