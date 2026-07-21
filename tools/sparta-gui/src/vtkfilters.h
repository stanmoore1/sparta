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

#ifndef VTKFILTERS_H
#define VTKFILTERS_H

// Small, window-free wrappers around a handful of stock VTK filters that bring
// routine field post-processing in-house (Feature 9): an arbitrary cut plane, an
// iso-surface, a line probe, a point probe, and a field calculator for derived
// quantities.  Heavier analysis (streamlines, glyphs, volume rendering) stays in
// ParaView via the export dialog.  These are pure data->data transforms (no
// render window), so they are unit-testable against a synthetic dataset.  Only
// compiled when SPARTA-GUI is built with VTK (-D SPARTA_GUI_USE_VTK=on).

#include <vtkSmartPointer.h>

#include <QString>

class vtkDataSet;
class vtkPolyData;

namespace VtkFilters {

/// @brief Slice @p input with the plane through @p origin with @p normal.
vtkSmartPointer<vtkPolyData> cutPlane(vtkDataSet *input, const double origin[3],
                                      const double normal[3]);

/// @brief Iso-surface of point-scalar @p array in @p input at @p value.
vtkSmartPointer<vtkPolyData> isoSurface(vtkDataSet *input, const QString &array, double value);

/// @brief Sample @p input at @p nsamples points along the segment p1->p2.
vtkSmartPointer<vtkPolyData> probeLine(vtkDataSet *input, const double p1[3],
                                       const double p2[3], int nsamples);

/// @brief Sample @p input at a single @p point (a 1-point probe dataset).
vtkSmartPointer<vtkPolyData> probePoint(vtkDataSet *input, const double point[3]);

/// @brief Add a derived point-data array @p name = @p expression to a copy of
/// @p input.  Expression variables are existing scalar/vector array names, e.g.
/// "mag(v)" or "0.5*rho*mag(v)^2".  Returns null on an invalid expression.
vtkSmartPointer<vtkDataSet> calculate(vtkDataSet *input, const QString &name,
                                      const QString &expression);

} // namespace VtkFilters

#endif // VTKFILTERS_H

// Local Variables:
// c-basic-offset: 4
// End:
