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

#include "vtkviewer.h"

#include "constants.h"
#include "helpers.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QIcon>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QStatusBar>
#include <QToolBar>
#include <QWheelEvent>

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

void VtkRenderArea::mousePressEvent(QMouseEvent *event)
{
    lastPos    = event->pos();
    dragButton = event->button();
}

void VtkRenderArea::mouseMoveEvent(QMouseEvent *event)
{
    if (dragButton == Qt::NoButton) return;
    const QPoint pos = event->pos();
    const int dx     = pos.x() - lastPos.x();
    const int dy     = pos.y() - lastPos.y();
    if (dx == 0 && dy == 0) return;

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
// VtkViewer -- toolbar + render area + coloring logic                       //
// ======================================================================== //

VtkViewer::VtkViewer(QWidget *parent) : QMainWindow(parent)
{
    setWindowTitle("SPARTA 3D Viewer");
    setWindowIcon(QIcon(Cfg::MAIN_ICON));
    setMinimumSize(Cfg::MINIMUM_WIDTH, Cfg::MINIMUM_HEIGHT);
    buildUi();
}

VtkViewer::~VtkViewer() = default;

void VtkViewer::buildUi()
{
    renderArea = new VtkRenderArea(this);
    renderArea->setObjectName("vtkRenderArea");
    setCentralWidget(renderArea);

    // color legend, hidden until an array is chosen
    scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
    scalarBar->SetNumberOfLabels(5);
    scalarBar->SetBarRatio(0.15);
    scalarBar->GetTitleTextProperty()->SetFontSize(12);
    scalarBar->VisibilityOff();
    renderArea->renderer()->AddActor2D(scalarBar);

    auto *tb = addToolBar("View");
    tb->setMovable(false);

    tb->addAction(QIcon(":/icons/document-open.svg"), "Open VTK File...", this,
                  &VtkViewer::openFileDialog);
    tb->addSeparator();

    tb->addWidget(new QLabel(" Color by: "));
    arrayCombo = new QComboBox(tb);
    arrayCombo->setObjectName("vtkArrayCombo");
    arrayCombo->setMinimumWidth(170);
    arrayCombo->setToolTip("Per-point or per-cell scalar field used to color the data");
    connect(arrayCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &VtkViewer::onColorArrayChanged);
    tb->addWidget(arrayCombo);

    tb->addWidget(new QLabel("  Colormap: "));
    cmapCombo = new QComboBox(tb);
    cmapCombo->addItems({"Rainbow", "Cool to Warm", "Viridis", "Grayscale"});
    connect(cmapCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &VtkViewer::onColorMapChanged);
    tb->addWidget(cmapCombo);

    tb->addSeparator();
    edgesBox = new QCheckBox("Edges", tb);
    edgesBox->setToolTip("Draw the outlines of grid cells / surface elements");
    connect(edgesBox, &QCheckBox::toggled, this, &VtkViewer::onEdgesToggled);
    tb->addWidget(edgesBox);

    scalarBarBox = new QCheckBox("Legend", tb);
    scalarBarBox->setChecked(true);
    connect(scalarBarBox, &QCheckBox::toggled, this, &VtkViewer::onScalarBarToggled);
    tb->addWidget(scalarBarBox);

    tb->addSeparator();
    tb->addAction(QIcon(":/icons/preferences-reset.svg"), "Reset View", this,
                  &VtkViewer::resetView);
    tb->addAction(QIcon(":/icons/image-x-generic.svg"), "Save Screenshot...", this,
                  &VtkViewer::saveScreenshot);

    infoLabel = new QLabel("No data loaded.");
    statusBar()->addWidget(infoLabel);
}

vtkSmartPointer<vtkDataSet> VtkViewer::readDataSet(const QString &path, QString *err)
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

bool VtkViewer::addDataset(const QString &path, const QString &label, Kind kind, QString *err)
{
    auto data = readDataSet(path, err);
    if (!data || data->GetNumberOfPoints() == 0) {
        if (data && err) *err = QString("%1 contains no points").arg(path);
        return false;
    }

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
    infoLabel->setText(QString("%1 layer(s), %2 points, %3 cells")
                           .arg(layers.size())
                           .arg(static_cast<qlonglong>(np))
                           .arg(static_cast<qlonglong>(nc)));
    return true;
}

void VtkViewer::clearScene()
{
    for (const auto &l : layers) renderArea->renderer()->RemoveActor(l.actor);
    layers.clear();
    scalarBar->VisibilityOff();
    refreshArrayCombo();
    infoLabel->setText("No data loaded.");
    renderArea->requestRender();
}

void VtkViewer::resetView()
{
    renderArea->resetCamera();
}

void VtkViewer::showViewer()
{
    show();
    raise();
    activateWindow();
    renderArea->resetCamera();
}

void VtkViewer::refreshArrayCombo()
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

bool VtkViewer::arrayRange(const QString &array, bool pointData, double range[2]) const
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

void VtkViewer::applyColoring()
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

void VtkViewer::onColorArrayChanged()
{
    applyColoring();
}

void VtkViewer::onColorMapChanged()
{
    applyColoring();
}

void VtkViewer::onEdgesToggled(bool on)
{
    for (const auto &l : layers)
        if (l.kind != Kind::Particles) l.actor->GetProperty()->SetEdgeVisibility(on);
    renderArea->requestRender();
}

void VtkViewer::onScalarBarToggled(bool on)
{
    scalarBar->SetVisibility(on && arrayCombo->currentIndex() > 0);
    renderArea->requestRender();
}

void VtkViewer::openFileDialog()
{
    const QString path = QFileDialog::getOpenFileName(
        this, "Open VTK Dataset", QString(), "VTK datasets (*.vtu *.vtp *.vtk);;All files (*)");
    if (path.isEmpty()) return;
    QString err;
    if (!addDataset(path, QFileInfo(path).fileName(), Kind::Generic, &err))
        critical(this, "Open VTK Dataset", "Could not open the selected file:", err);
}

void VtkViewer::saveScreenshot()
{
    const QString path = QFileDialog::getSaveFileName(this, "Save Screenshot", "sparta-view.png",
                                                      "PNG image (*.png)");
    if (path.isEmpty()) return;
    renderArea->requestRender();
    const QImage img = renderArea->grabFrame();
    if (img.isNull() || !img.save(path))
        critical(this, "Save Screenshot", "Could not write the screenshot to:", path);
    else
        statusBar()->showMessage(QString("Saved screenshot to %1").arg(path), 5000);
}

// Local Variables:
// c-basic-offset: 4
// End:
