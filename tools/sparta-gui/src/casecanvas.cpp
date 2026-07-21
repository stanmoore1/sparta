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

#include "casecanvas.h"

#include "stlimport.h"
#include "vtkviewer.h"

#include <QDir>
#include <QFileInfo>
#include <QLabel>
#include <QMenu>
#include <QStatusBar>
#include <QToolBar>

#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkOutlineSource.h>
#include <vtkPlaneSource.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPropPicker.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>

namespace {

// SPARTA boundary character -> a face tint (RGB in 0..1) for visual feedback
void conditionColor(QChar c, double rgb[3])
{
    switch (c.toLatin1()) {
        case 'p': rgb[0] = 0.30; rgb[1] = 0.75; rgb[2] = 0.40; break; // periodic  green
        case 'r': rgb[0] = 0.90; rgb[1] = 0.55; rgb[2] = 0.20; break; // reflect   orange
        case 's': rgb[0] = 0.60; rgb[1] = 0.60; rgb[2] = 0.62; break; // surface   gray
        case 'o':
        default:  rgb[0] = 0.25; rgb[1] = 0.55; rgb[2] = 0.90; break; // outflow   blue
    }
}

// expand a one-or-two-char axis spec into its lo/hi characters
void axisChars(const QString &spec, QChar &lo, QChar &hi)
{
    if (spec.isEmpty()) { lo = hi = 'o'; return; }
    lo = spec.at(0);
    hi = spec.size() > 1 ? spec.at(1) : spec.at(0);
}

} // namespace

CaseCanvas::CaseCanvas(QWidget *parent) : QMainWindow(parent)
{
    picker = vtkSmartPointer<vtkPropPicker>::New();
    buildUi();
}

CaseCanvas::~CaseCanvas() = default;

void CaseCanvas::buildUi()
{
    setWindowTitle("Case Setup Canvas");
    resize(760, 620);

    renderArea = new VtkRenderArea(this);
    setCentralWidget(renderArea);
    renderArea->setPickCallback([this](const QPoint &p) { onPick(p); });

    auto *tb = addToolBar("Canvas");
    tb->setMovable(false);
    tb->addAction("Reset View", this, &CaseCanvas::resetView);

    info = new QLabel(this);
    info->setText("Click a box face to set its boundary condition or add an inflow.");
    statusBar()->addWidget(info);
}

void CaseCanvas::showCanvas()
{
    show();
    raise();
    activateWindow();
}

void CaseCanvas::setDeck(const QString &deckText, const QString &dir)
{
    deck    = deckText;
    baseDir = dir;
    model   = CaseModel::parse(deck);
    rebuildScene();
}

// ---------------------------------------------------------------------------
// scene construction
// ---------------------------------------------------------------------------

void CaseCanvas::rebuildScene()
{
    auto *ren = renderArea->renderer();
    for (const auto &a : actors) ren->RemoveActor(a);
    actors.clear();
    faceActors.clear();

    if (model.box.present) addBoxActors(model.box);
    for (const auto &s : model.surfaces) addSurfaceActors(s);
    for (const auto &r : model.regions) addRegionActors(r);

    if (model.box.present || !model.surfaces.isEmpty()) {
        renderArea->resetCamera();
    } else {
        info->setText("No create_box in the deck yet - add one to see the domain.");
        renderArea->requestRender();
    }
}

void CaseCanvas::addBoxActors(const CaseModel::Box &box)
{
    const double *lo = box.lo;
    const double *hi = box.hi;

    // domain outline (white wireframe)
    auto outline = vtkSmartPointer<vtkOutlineSource>::New();
    outline->SetBounds(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
    outline->Update();
    auto omap = vtkSmartPointer<vtkPolyDataMapper>::New();
    omap->SetInputConnection(outline->GetOutputPort());
    auto oact = vtkSmartPointer<vtkActor>::New();
    oact->SetMapper(omap);
    oact->GetProperty()->SetColor(0.9, 0.9, 0.9);
    oact->GetProperty()->SetLineWidth(1.5);
    oact->PickableOff();
    renderArea->renderer()->AddActor(oact);
    actors.push_back(oact);

    // six pickable, semi-transparent face quads (z faces only in 3d)
    // origin, point1, point2 for each face (a parallelogram)
    const double corners[6][3][3] = {
        {{lo[0], lo[1], lo[2]}, {lo[0], hi[1], lo[2]}, {lo[0], lo[1], hi[2]}}, // xlo
        {{hi[0], lo[1], lo[2]}, {hi[0], hi[1], lo[2]}, {hi[0], lo[1], hi[2]}}, // xhi
        {{lo[0], lo[1], lo[2]}, {hi[0], lo[1], lo[2]}, {lo[0], lo[1], hi[2]}}, // ylo
        {{lo[0], hi[1], lo[2]}, {hi[0], hi[1], lo[2]}, {lo[0], hi[1], hi[2]}}, // yhi
        {{lo[0], lo[1], lo[2]}, {hi[0], lo[1], lo[2]}, {lo[0], hi[1], lo[2]}}, // zlo
        {{lo[0], lo[1], hi[2]}, {hi[0], lo[1], hi[2]}, {lo[0], hi[1], hi[2]}}, // zhi
    };

    const int nfaces = (box.dimension >= 3) ? 6 : 4;
    for (int f = 0; f < nfaces; ++f) {
        auto plane = vtkSmartPointer<vtkPlaneSource>::New();
        plane->SetOrigin(corners[f][0][0], corners[f][0][1], corners[f][0][2]);
        plane->SetPoint1(corners[f][1][0], corners[f][1][1], corners[f][1][2]);
        plane->SetPoint2(corners[f][2][0], corners[f][2][1], corners[f][2][2]);
        plane->Update();
        auto pmap = vtkSmartPointer<vtkPolyDataMapper>::New();
        pmap->SetInputConnection(plane->GetOutputPort());
        auto pact = vtkSmartPointer<vtkActor>::New();
        pact->SetMapper(pmap);

        // tint by the current boundary condition for that face
        const int axis = f / 2;
        const bool isHi = (f % 2) != 0;
        QChar clo, chi;
        axisChars(model.boundary.present ? model.boundary.spec[axis] : QString(), clo, chi);
        double rgb[3];
        conditionColor(isHi ? chi : clo, rgb);
        pact->GetProperty()->SetColor(rgb[0], rgb[1], rgb[2]);
        pact->GetProperty()->SetOpacity(0.28);

        renderArea->renderer()->AddActor(pact);
        actors.push_back(pact);
        faceActors.insert(pact.Get(), f);
    }
}

void CaseCanvas::addSurfaceActors(const CaseModel::SurfImport &surf)
{
    // resolve the surface file relative to the deck directory
    QString path = surf.file;
    QFileInfo fi(path);
    if (fi.isRelative() && !baseDir.isEmpty())
        path = QDir(baseDir).absoluteFilePath(path);
    if (!QFileInfo::exists(path)) return;

    StlImport::SurfMesh mesh;
    QString err;
    const auto kind = StlImport::detectSource(path);
    bool ok = (kind == StlImport::SourceKind::Stl) ? StlImport::parseStl(path, mesh, err)
                                                   : StlImport::parseSurf(path, mesh, err);
    if (!ok || mesh.npoints() == 0) return;

    auto points = vtkSmartPointer<vtkPoints>::New();
    for (const auto &p : mesh.points) points->InsertNextPoint(p[0], p[1], p[2]);

    auto cells = vtkSmartPointer<vtkCellArray>::New();
    for (const auto &e : mesh.elems) {
        if (mesh.is2d || e[2] < 0) {
            cells->InsertNextCell(2);
            cells->InsertCellPoint(e[0]);
            cells->InsertCellPoint(e[1]);
        } else {
            cells->InsertNextCell(3);
            cells->InsertCellPoint(e[0]);
            cells->InsertCellPoint(e[1]);
            cells->InsertCellPoint(e[2]);
        }
    }

    auto poly = vtkSmartPointer<vtkPolyData>::New();
    poly->SetPoints(points);
    if (mesh.is2d) poly->SetLines(cells);
    else           poly->SetPolys(cells);

    auto map = vtkSmartPointer<vtkPolyDataMapper>::New();
    map->SetInputData(poly);
    auto act = vtkSmartPointer<vtkActor>::New();
    act->SetMapper(map);
    act->GetProperty()->SetColor(0.85, 0.72, 0.35); // surface: warm tan
    act->GetProperty()->SetLineWidth(2.0);
    act->PickableOff();
    renderArea->renderer()->AddActor(act);
    actors.push_back(act);
}

void CaseCanvas::addRegionActors(const CaseModel::Region &region)
{
    // Phase 1 renders only block regions (an axis-aligned wireframe box); other
    // region styles (sphere/cylinder/...) are recognized by the model but not
    // yet drawn -- they fall through here and are left to a later phase.
    if (region.style != "block" || region.args.size() < 6) return;

    double b[6];
    for (int i = 0; i < 6; ++i) b[i] = region.args.at(i).toDouble();

    auto outline = vtkSmartPointer<vtkOutlineSource>::New();
    outline->SetBounds(b[0], b[1], b[2], b[3], b[4], b[5]);
    outline->Update();
    auto map = vtkSmartPointer<vtkPolyDataMapper>::New();
    map->SetInputConnection(outline->GetOutputPort());
    auto act = vtkSmartPointer<vtkActor>::New();
    act->SetMapper(map);
    act->GetProperty()->SetColor(0.55, 0.75, 0.95); // region: light blue
    act->PickableOff();
    renderArea->renderer()->AddActor(act);
    actors.push_back(act);
}

// ---------------------------------------------------------------------------
// interaction
// ---------------------------------------------------------------------------

void CaseCanvas::resetView()
{
    renderArea->resetCamera();
}

void CaseCanvas::onPick(const QPoint &pos)
{
    if (faceActors.isEmpty()) return;

    const double dpr = renderArea->devicePixelRatioF();
    const int h      = renderArea->window()->GetSize()[1];
    const double dx  = pos.x() * dpr;
    const double dy  = h - pos.y() * dpr; // VTK display is bottom-up

    if (!picker->Pick(dx, dy, 0.0, renderArea->renderer())) return;
    vtkActor *hit = picker->GetActor();
    if (!hit) return;
    auto it = faceActors.constFind(hit);
    if (it == faceActors.constEnd()) return;

    showFaceMenu(it.value(), renderArea->mapToGlobal(pos));
}

void CaseCanvas::showFaceMenu(int face, const QPoint &globalPos)
{
    const QString faceName = CaseModel::FACE_NAMES[face];
    QMenu menu;
    menu.addAction(QString("Face %1").arg(faceName))->setEnabled(false);
    menu.addSeparator();
    QAction *aInflow  = menu.addAction("Inflow (open + emitter)");
    QAction *aOutflow = menu.addAction("Outflow (open)");
    QAction *aSpecular = menu.addAction("Specular wall");
    QAction *aPeriodic = menu.addAction("Periodic (axis)");

    QAction *chosen = menu.exec(globalPos);
    if (!chosen) return;

    if (chosen == aInflow)        applyBoundary(face, "o", /*addInflow=*/true);
    else if (chosen == aOutflow)  applyBoundary(face, "o", false);
    else if (chosen == aSpecular) applyBoundary(face, "r", false);
    else if (chosen == aPeriodic) applyBoundary(face, "p", false);
}

void CaseCanvas::applyBoundary(int face, const QString &condition, bool addInflow)
{
    const int axis  = face / 2;
    const bool isHi = (face % 2) != 0;

    // rebuild all three axis specs from the current model, changing only `axis`
    QString specs[3];
    for (int a = 0; a < 3; ++a) {
        QChar lo, hi;
        axisChars(model.boundary.present ? model.boundary.spec[a] : QString("o"), lo, hi);
        if (a == axis) {
            const QChar c = condition.at(0);
            if (c == 'p') { lo = hi = 'p'; }        // periodic must apply to both faces
            else if (isHi) hi = c; else lo = c;
        }
        specs[a] = (lo == hi) ? QString(lo) : (QString(lo) + hi);
    }

    QString newDeck = CaseModel::setBoundary(deck, specs[0], specs[1], specs[2]);

    if (addInflow) {
        // choose a mixture: first user mixture, else SPARTA's built-in "all"
        const QStringList mixes = model.mixtureIds();
        const QString mix = mixes.isEmpty() ? QString("all") : mixes.first();
        // a unique fix id for this face
        QString id = "in_" + QString(CaseModel::FACE_NAMES[face]);
        int n = 1;
        const CaseModel::Model probe = CaseModel::parse(newDeck);
        auto idTaken = [&](const QString &cand) {
            for (const auto &e : probe.emits)
                if (e.id == cand) return true;
            return false;
        };
        QString cand = id;
        while (idTaken(cand)) cand = id + QString::number(++n);
        newDeck = CaseModel::insertEmitFace(newDeck, cand, mix,
                                            {CaseModel::FACE_NAMES[face]});
    }

    // apply locally (re-render) and tell the main window to update the editor
    setDeck(newDeck, baseDir);
    info->setText(QString("Set %1 boundary.").arg(CaseModel::FACE_NAMES[face]));
    emit deckEdited(newDeck);
}
