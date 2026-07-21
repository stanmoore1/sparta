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

#ifndef CASECANVAS_H
#define CASECANVAS_H

// Visual, bidirectional case-setup canvas (Feature 8, "the big one").  Phase 1
// (this file): render the simulation box, imported surfaces and block regions
// parsed from the current deck (one-way text -> scene), and let the user click a
// box face to declare its boundary condition or add an inflow -- writing the
// corresponding SPARTA command(s) straight back into the editor.  The pure
// parse/edit engine lives in casemodel.{h,cpp}; this class is only the VTK/Qt
// shell over it.  Compiled only when SPARTA-GUI is built with VTK
// (-D SPARTA_GUI_USE_VTK=on); otherwise the menu entry is disabled and the text
// editor remains the only path.

#include "casemodel.h"

#include <QHash>
#include <QMainWindow>
#include <QString>

#include <vtkSmartPointer.h>

#include <vector>

class QLabel;
class VtkRenderArea;

class vtkActor;
class vtkPropPicker;

/**
 * @brief Interactive 3D case-setup canvas driven by CaseModel.
 *
 * setDeck() parses the deck and (re)builds the scene; a left click on a box
 * face opens a small menu to set that face's boundary condition or add an
 * inflow.  Any such action rewrites the affected command line(s) and emits
 * @ref deckEdited with the new deck text, which the main window applies to the
 * editor.  The window listens for follow-up setDeck() calls to refresh when the
 * user edits the text directly.
 */
class CaseCanvas : public QMainWindow {
    Q_OBJECT

public:
    explicit CaseCanvas(QWidget *parent = nullptr);
    ~CaseCanvas() override;

    CaseCanvas(const CaseCanvas &)            = delete;
    CaseCanvas &operator=(const CaseCanvas &) = delete;

    /// @brief Parse @p deckText and rebuild the scene; @p baseDir resolves
    /// relative surface-file paths (read_surf).  Safe to call repeatedly.
    void setDeck(const QString &deckText, const QString &baseDir);

    /// @brief Show, raise and activate the canvas window.
    void showCanvas();

signals:
    /// @brief A pick edited the deck; carries the full new deck text.
    void deckEdited(const QString &newDeckText);

private slots:
    void resetView();

private:
    void buildUi();
    void rebuildScene();
    void onPick(const QPoint &pos);       ///< face-pick handler installed on the render area
    void showFaceMenu(int face, const QPoint &globalPos);
    void applyBoundary(int face, const QString &condition, bool addInflow);

    // scene builders
    void addBoxActors(const CaseModel::Box &box);
    void addSurfaceActors(const CaseModel::SurfImport &surf);
    void addRegionActors(const CaseModel::Region &region);

    VtkRenderArea *renderArea = nullptr;
    QLabel *info             = nullptr;

    QString deck;
    QString baseDir;
    CaseModel::Model model;

    vtkSmartPointer<vtkPropPicker> picker;
    // actors we keep alive; the face quads map back to a face index 0..5
    std::vector<vtkSmartPointer<vtkActor>> actors;
    QHash<vtkActor *, int> faceActors;
};

#endif // CASECANVAS_H

// Local Variables:
// c-basic-offset: 4
// End:
