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

// 11^3 grid with a 3-component vector field v = (x, 2y, 0), so |v| has a
// closed form and the field calculator can be checked against it.
vtkSmartPointer<vtkImageData> vectorField()
{
    auto img = vtkSmartPointer<vtkImageData>::New();
    img->SetDimensions(11, 11, 11);
    img->SetSpacing(0.1, 0.1, 0.1);
    img->SetOrigin(0, 0, 0);
    auto v = vtkSmartPointer<vtkDoubleArray>::New();
    v->SetName("v");
    v->SetNumberOfComponents(3);
    v->SetNumberOfTuples(11 * 11 * 11);
    int idx = 0;
    for (int k = 0; k < 11; ++k)
        for (int j = 0; j < 11; ++j)
            for (int i = 0; i < 11; ++i) v->SetTuple3(idx++, i * 0.1, 2.0 * j * 0.1, 0.0);
    img->GetPointData()->AddArray(v);
    return img;
}

// --- cut plane -------------------------------------------------------------

TEST(VtkFilters, CutPlaneOnAnObliqueNormal)
{
    // the plane x + y = 1 through the middle of the unit cube. Every sampled
    // point lies on it, so f (== x) plus the y it was sampled at must sum to 1.
    auto img = rampField();
    double o[3] = {0.5, 0.5, 0.5};
    double n[3] = {1, 1, 0};
    auto cut = VtkFilters::cutPlane(img, o, n);
    ASSERT_TRUE(cut && cut->GetNumberOfPoints() > 0);
    auto *a = cut->GetPointData()->GetArray("f");
    ASSERT_TRUE(a);
    for (vtkIdType i = 0; i < cut->GetNumberOfPoints(); ++i) {
        double p[3];
        cut->GetPoint(i, p);
        EXPECT_NEAR(p[0] + p[1], 1.0, 1e-6);
        EXPECT_NEAR(a->GetTuple1(i), p[0], 1e-6);
    }
}

TEST(VtkFilters, CutPlaneOutsideTheDataIsEmptyRatherThanNull)
{
    // a plane that misses the dataset entirely: an empty result the caller can
    // report, not a null they have to guess the meaning of
    auto img = rampField();
    double o[3] = {5.0, 0, 0}, n[3] = {1, 0, 0};
    auto cut = VtkFilters::cutPlane(img, o, n);
    ASSERT_TRUE(cut);
    EXPECT_EQ(cut->GetNumberOfPoints(), 0);
}

// --- iso-surface -----------------------------------------------------------

TEST(VtkFilters, IsoSurfaceSitsAtTheRequestedValue)
{
    // f == x, so the 0.3 iso-surface is the plane x = 0.3
    auto img = rampField();
    auto iso = VtkFilters::isoSurface(img, "f", 0.3);
    ASSERT_TRUE(iso);
    ASSERT_GT(iso->GetNumberOfPoints(), 0);
    for (vtkIdType i = 0; i < iso->GetNumberOfPoints(); ++i) {
        double p[3];
        iso->GetPoint(i, p);
        EXPECT_NEAR(p[0], 0.3, 1e-5);
    }
}

TEST(VtkFilters, IsoSurfaceOutsideTheRangeIsEmpty)
{
    auto img = rampField();
    auto iso = VtkFilters::isoSurface(img, "f", 7.5); // f only spans 0..1
    ASSERT_TRUE(iso);
    EXPECT_EQ(iso->GetNumberOfPoints(), 0);
}

TEST(VtkFilters, IsoSurfaceOfAnAbsentArrayProducesNothing)
{
    auto img = rampField();
    auto iso = VtkFilters::isoSurface(img, "no-such-array", 0.5);
    ASSERT_TRUE(iso);
    // whatever it does, it must not invent a surface out of a field that is
    // not there -- the dialog offers only names it read from the data, so
    // this is the "the layer changed under me" case
    EXPECT_EQ(iso->GetNumberOfPoints(), 0);
}

// --- probes ----------------------------------------------------------------

TEST(VtkFilters, LineProbeHonorsTheSampleCount)
{
    auto img = rampField();
    double p1[3] = {0, 0.5, 0.5}, p2[3] = {1, 0.5, 0.5};
    EXPECT_EQ(VtkFilters::probeLine(img, p1, p2, 2)->GetNumberOfPoints(), 2);
    EXPECT_EQ(VtkFilters::probeLine(img, p1, p2, 51)->GetNumberOfPoints(), 51);
    // a nonsensical count still yields a usable line rather than a crash
    EXPECT_GE(VtkFilters::probeLine(img, p1, p2, 0)->GetNumberOfPoints(), 2);
    EXPECT_GE(VtkFilters::probeLine(img, p1, p2, -5)->GetNumberOfPoints(), 2);
}

TEST(VtkFilters, LineProbeIsLinearAlongTheRamp)
{
    auto img = rampField();
    double p1[3] = {0, 0.5, 0.5}, p2[3] = {1, 0.5, 0.5};
    auto ln = VtkFilters::probeLine(img, p1, p2, 21);
    ASSERT_TRUE(ln);
    auto *a = ln->GetPointData()->GetArray("f");
    ASSERT_TRUE(a);
    for (int i = 0; i < 21; ++i) EXPECT_NEAR(a->GetTuple1(i), i / 20.0, 1e-6);
}

TEST(VtkFilters, ProbingOutsideTheDataIsMarkedInvalid)
{
    // a point outside the dataset has no value; the probe must say so rather
    // than hand back a zero that reads as a real measurement
    auto img = rampField();
    double p[3] = {5.0, 5.0, 5.0};
    auto pt = VtkFilters::probePoint(img, p);
    ASSERT_TRUE(pt && pt->GetNumberOfPoints() == 1);
    auto *valid = pt->GetPointData()->GetArray("vtkValidPointMask");
    ASSERT_TRUE(valid) << "no validity mask, so a miss is indistinguishable from a zero";
    EXPECT_EQ(valid->GetTuple1(0), 0.0);
}

TEST(VtkFilters, ProbingInsideTheDataIsMarkedValid)
{
    auto img = rampField();
    double p[3] = {0.25, 0.5, 0.5};
    auto pt = VtkFilters::probePoint(img, p);
    ASSERT_TRUE(pt && pt->GetNumberOfPoints() == 1);
    auto *valid = pt->GetPointData()->GetArray("vtkValidPointMask");
    ASSERT_TRUE(valid);
    EXPECT_EQ(valid->GetTuple1(0), 1.0);
    EXPECT_NEAR(pt->GetPointData()->GetArray("f")->GetTuple1(0), 0.25, 1e-6);
}

// --- field calculator ------------------------------------------------------

TEST(VtkFilters, CalculatorHandlesVectorMagnitude)
{
    // v = (x, 2y, 0) so |v| = sqrt(x^2 + 4y^2); at the far corner (1,1,*)
    // that is sqrt(5)
    auto img = vectorField();
    auto out = VtkFilters::calculate(img, "speed", "mag(v)");
    ASSERT_TRUE(out);
    auto *s = out->GetPointData()->GetArray("speed");
    ASSERT_TRUE(s);
    for (vtkIdType i = 0; i < out->GetNumberOfPoints(); ++i) {
        double p[3];
        out->GetPoint(i, p);
        EXPECT_NEAR(s->GetTuple1(i), std::sqrt(p[0] * p[0] + 4.0 * p[1] * p[1]), 1e-6);
    }
}

TEST(VtkFilters, CalculatorLeavesTheInputAlone)
{
    // the result is a copy: a derived quantity must not quietly modify the
    // layer it was computed from, which is still on screen
    auto img = rampField();
    const int before = img->GetPointData()->GetNumberOfArrays();
    auto out = VtkFilters::calculate(img, "g", "2*f");
    ASSERT_TRUE(out);
    EXPECT_EQ(img->GetPointData()->GetNumberOfArrays(), before);
    EXPECT_EQ(out->GetPointData()->GetNumberOfArrays(), before + 1);
}

TEST(VtkFilters, CalculatorRejectsEmptyArguments)
{
    auto img = rampField();
    EXPECT_EQ(VtkFilters::calculate(img, "", "2*f"), nullptr);
    EXPECT_EQ(VtkFilters::calculate(img, "g", ""), nullptr);
}

TEST(VtkFilters, CalculatorAcceptsConstantsAndArithmetic)
{
    auto img = rampField();
    auto out = VtkFilters::calculate(img, "h", "0.5*f^2 + 3");
    ASSERT_TRUE(out);
    auto *h = out->GetPointData()->GetArray("h");
    ASSERT_TRUE(h);
    auto *f = out->GetPointData()->GetArray("f");
    ASSERT_TRUE(f);
    for (vtkIdType i = 0; i < out->GetNumberOfPoints(); i += 37) {
        const double fv = f->GetTuple1(i);
        EXPECT_NEAR(h->GetTuple1(i), 0.5 * fv * fv + 3.0, 1e-6);
    }
}

// --- no input --------------------------------------------------------------

TEST(VtkFilters, EveryFilterRefusesANullDataset)
{
    // the scene calls these with currentData(), which is null when every layer
    // has been cleared -- a menu entry left enabled is all it takes
    double a[3] = {0, 0, 0}, b[3] = {1, 1, 1};
    EXPECT_EQ(VtkFilters::cutPlane(nullptr, a, b), nullptr);
    EXPECT_EQ(VtkFilters::isoSurface(nullptr, "f", 0.5), nullptr);
    EXPECT_EQ(VtkFilters::probeLine(nullptr, a, b, 10), nullptr);
    EXPECT_EQ(VtkFilters::probePoint(nullptr, a), nullptr);
    EXPECT_EQ(VtkFilters::calculate(nullptr, "g", "2*f"), nullptr);
}

} // namespace
