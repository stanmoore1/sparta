// Unit tests for the in-app field post-processing filters (src/vtkfilters.cpp).
// Built only when a suitable VTK is available; they are pure data->data
// transforms so no render window / display is needed.

#include "vtkfilters.h"

#include "gtest/gtest.h"

#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>

#include <cmath>

namespace {

// 11^3 grid on the unit cube with a point scalar f == x-coordinate
vtkSmartPointer<vtkImageData> rampField()
{
    auto img = vtkSmartPointer<vtkImageData>::New();
    img->SetDimensions(11, 11, 11);
    img->SetSpacing(0.1, 0.1, 0.1);
    img->SetOrigin(0, 0, 0);
    auto f = vtkSmartPointer<vtkDoubleArray>::New();
    f->SetName("f");
    f->SetNumberOfTuples(11 * 11 * 11);
    int idx = 0;
    for (int k = 0; k < 11; ++k)
        for (int j = 0; j < 11; ++j)
            for (int i = 0; i < 11; ++i) f->SetValue(idx++, i * 0.1);
    img->GetPointData()->AddArray(f);
    img->GetPointData()->SetScalars(f);
    return img;
}

TEST(VtkFilters, CutPlaneSamplesConstant)
{
    auto img = rampField();
    double o[3] = {0.5, 0, 0}, n[3] = {1, 0, 0};
    auto cut = VtkFilters::cutPlane(img, o, n);
    ASSERT_TRUE(cut && cut->GetNumberOfPoints() > 0);
    auto *a = cut->GetPointData()->GetArray("f");
    ASSERT_TRUE(a);
    for (vtkIdType i = 0; i < a->GetNumberOfTuples(); ++i)
        EXPECT_NEAR(a->GetTuple1(i), 0.5, 1e-6); // the whole slice is at x=0.5
}

TEST(VtkFilters, IsoSurfaceNonEmpty)
{
    auto img = rampField();
    auto iso = VtkFilters::isoSurface(img, "f", 0.5);
    ASSERT_TRUE(iso);
    EXPECT_GT(iso->GetNumberOfPoints(), 0);
}

TEST(VtkFilters, LineProbeRamp)
{
    auto img = rampField();
    double p1[3] = {0, 0.5, 0.5}, p2[3] = {1, 0.5, 0.5};
    auto ln = VtkFilters::probeLine(img, p1, p2, 11);
    ASSERT_TRUE(ln && ln->GetNumberOfPoints() == 11);
    auto *a = ln->GetPointData()->GetArray("f");
    ASSERT_TRUE(a);
    EXPECT_NEAR(a->GetTuple1(0), 0.0, 1e-6);
    EXPECT_NEAR(a->GetTuple1(10), 1.0, 1e-6);
}

TEST(VtkFilters, PointProbe)
{
    auto img = rampField();
    double p[3] = {0.7, 0.5, 0.5};
    auto pt = VtkFilters::probePoint(img, p);
    ASSERT_TRUE(pt && pt->GetNumberOfPoints() == 1);
    EXPECT_NEAR(pt->GetPointData()->GetArray("f")->GetTuple1(0), 0.7, 1e-6);
}

TEST(VtkFilters, FieldCalculator)
{
    auto img = rampField();
    auto out = VtkFilters::calculate(img, "g", "2*f");
    ASSERT_TRUE(out);
    auto *g = out->GetPointData()->GetArray("g");
    ASSERT_TRUE(g);
    EXPECT_NEAR(g->GetTuple1(g->GetNumberOfTuples() - 1), 2.0, 1e-6); // x=1 -> g=2

    // an invalid expression yields null
    EXPECT_EQ(VtkFilters::calculate(img, "bad", "nonexistent_var*2"), nullptr);
}

} // namespace
