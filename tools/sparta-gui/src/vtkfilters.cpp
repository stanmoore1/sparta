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

#include "vtkfilters.h"

#include <vtkArrayCalculator.h>
#include <vtkContourFilter.h>
#include <vtkCutter.h>
#include <vtkDataArray.h>
#include <vtkDataObject.h>
#include <vtkDataSet.h>
#include <vtkLineSource.h>
#include <vtkPlane.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkProbeFilter.h>

namespace VtkFilters {

vtkSmartPointer<vtkPolyData> cutPlane(vtkDataSet *input, const double origin[3],
                                      const double normal[3])
{
    if (!input) return nullptr;
    auto plane = vtkSmartPointer<vtkPlane>::New();
    plane->SetOrigin(origin[0], origin[1], origin[2]);
    plane->SetNormal(normal[0], normal[1], normal[2]);

    auto cutter = vtkSmartPointer<vtkCutter>::New();
    cutter->SetCutFunction(plane);
    cutter->SetInputData(input);
    cutter->Update();

    auto out = vtkSmartPointer<vtkPolyData>::New();
    out->ShallowCopy(cutter->GetOutput());
    return out;
}

vtkSmartPointer<vtkPolyData> isoSurface(vtkDataSet *input, const QString &array, double value)
{
    if (!input) return nullptr;
    auto contour = vtkSmartPointer<vtkContourFilter>::New();
    contour->SetInputData(input);
    if (!array.isEmpty())
        contour->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                        array.toUtf8().constData());
    contour->SetValue(0, value);
    contour->Update();

    auto out = vtkSmartPointer<vtkPolyData>::New();
    out->ShallowCopy(contour->GetOutput());
    return out;
}

vtkSmartPointer<vtkPolyData> probeLine(vtkDataSet *input, const double p1[3],
                                       const double p2[3], int nsamples)
{
    if (!input) return nullptr;
    auto line = vtkSmartPointer<vtkLineSource>::New();
    line->SetPoint1(p1[0], p1[1], p1[2]);
    line->SetPoint2(p2[0], p2[1], p2[2]);
    line->SetResolution(qMax(1, nsamples - 1));
    line->Update();

    auto probe = vtkSmartPointer<vtkProbeFilter>::New();
    probe->SetInputConnection(line->GetOutputPort());
    probe->SetSourceData(input);
    probe->Update();

    auto out = vtkSmartPointer<vtkPolyData>::New();
    out->ShallowCopy(vtkPolyData::SafeDownCast(probe->GetOutput()));
    return out;
}

vtkSmartPointer<vtkPolyData> probePoint(vtkDataSet *input, const double point[3])
{
    if (!input) return nullptr;
    auto pts = vtkSmartPointer<vtkPoints>::New();
    pts->InsertNextPoint(point[0], point[1], point[2]);
    auto poly = vtkSmartPointer<vtkPolyData>::New();
    poly->SetPoints(pts);

    auto probe = vtkSmartPointer<vtkProbeFilter>::New();
    probe->SetInputData(poly);
    probe->SetSourceData(input);
    probe->Update();

    auto out = vtkSmartPointer<vtkPolyData>::New();
    out->ShallowCopy(vtkPolyData::SafeDownCast(probe->GetOutput()));
    return out;
}

vtkSmartPointer<vtkDataSet> calculate(vtkDataSet *input, const QString &name,
                                      const QString &expression)
{
    if (!input || name.isEmpty() || expression.isEmpty()) return nullptr;

    auto calc = vtkSmartPointer<vtkArrayCalculator>::New();
    calc->SetInputData(input);
    calc->SetAttributeTypeToPointData();

    // expose every existing point array as a variable (scalars and vectors)
    if (auto *pd = input->GetPointData()) {
        for (int i = 0; i < pd->GetNumberOfArrays(); ++i) {
            auto *arr = pd->GetArray(i);
            if (!arr || !arr->GetName()) continue;
            const char *an = arr->GetName();
            if (arr->GetNumberOfComponents() == 1) calc->AddScalarVariable(an, an);
            else if (arr->GetNumberOfComponents() == 3) calc->AddVectorVariable(an, an);
        }
    }
    calc->SetResultArrayName(name.toUtf8().constData());
    calc->SetFunction(expression.toUtf8().constData());
    calc->Update();

    auto *result = vtkDataSet::SafeDownCast(calc->GetOutput());
    if (!result) return nullptr;
    // verify the expression actually produced the array (invalid functions
    // leave it absent)
    if (!result->GetPointData() ||
        !result->GetPointData()->GetArray(name.toUtf8().constData()))
        return nullptr;

    auto out = vtkSmartPointer<vtkDataSet>(result->NewInstance());
    out->ShallowCopy(result);
    return out;
}

} // namespace VtkFilters
