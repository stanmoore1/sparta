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

#include "stlimportwizard.h"

#include "helpers.h"
#include "spartawrapper.h"
#include "stdcapture.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImageReader>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPlainTextEdit>
#include <QPolygonF>
#include <QPushButton>
#include <QRadioButton>
#include <QSpinBox>
#include <QTabWidget>
#include <QVBoxLayout>

#include <array>
#include <cmath>

using namespace StlImport;

namespace {

constexpr int PREVIEW_W = 460;
constexpr int PREVIEW_H = 380;
const char *const RENDER_ID = "sguiwiz";

using Vec3 = std::array<double, 3>;

Vec3 sub(const Vec3 &a, const Vec3 &b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 cross(const Vec3 &a, const Vec3 &b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double dot(const Vec3 &a, const Vec3 &b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double norm(const Vec3 &a) { return std::sqrt(dot(a, a)); }
Vec3 normalize(const Vec3 &a)
{
    const double n = norm(a);
    return (n > 1e-30) ? Vec3{a[0] / n, a[1] / n, a[2] / n} : Vec3{0, 0, 1};
}

// 3x3 matrix (row-major) times vector
Vec3 matvec(const double m[9], const Vec3 &v)
{
    return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

// Rodrigues rotation matrix for angle theta (deg) about a unit axis
void rotationMatrix(double thetaDeg, Vec3 axis, double m[9])
{
    axis = normalize(axis);
    const double t = thetaDeg * M_PI / 180.0;
    const double c = std::cos(t), s = std::sin(t), C = 1 - c;
    const double x = axis[0], y = axis[1], z = axis[2];
    m[0] = c + x * x * C;     m[1] = x * y * C - z * s; m[2] = x * z * C + y * s;
    m[3] = y * x * C + z * s; m[4] = c + y * y * C;     m[5] = y * z * C - x * s;
    m[6] = z * x * C - y * s; m[7] = z * y * C + x * s; m[8] = c + z * z * C;
}

QColor scaleColor(const QColor &c, double b)
{
    b = qBound(0.0, b, 1.0);
    return QColor(int(c.red() * b), int(c.green() * b), int(c.blue() * b));
}

} // namespace

StlImportWizard::StlImportWizard(QWidget *parent, SpartaWrapper *sparta, const QString &path) :
    QDialog(parent), sparta_(sparta), sourcePath_(path)
{
    setWindowTitle("SPARTA-GUI - Import Surface (STL / SPARTA)");
    setWindowIcon(QIcon(":/icons/sparta-gui-icon-128x128.png"));
    resize(720, 640);

    // parse the source into the common mesh
    QString err;
    kind_ = detectSource(sourcePath_);
    bool ok = false;
    if (kind_ == SourceKind::Stl)
        ok = parseStl(sourcePath_, mesh_, err);
    else
        ok = parseSurf(sourcePath_, mesh_, err); // Surf or Unknown -> try surf
    loaded_ = ok;
    if (ok) wt_ = checkWatertightPreflight(mesh_);

    auto *outer = new QVBoxLayout(this);
    tabs_ = new QTabWidget;
    outer->addWidget(tabs_, 1);
    tabs_->addTab(buildSourcePage(), "&Source");
    tabs_->addTab(buildTransformPage(), "&Transform");
    tabs_->addTab(buildPreviewPage(), "&Preview");
    tabs_->addTab(buildAblationPage(), "&Ablation");
    tabs_->addTab(buildDiagnosticsPage(), "&Diagnostics");
    tabs_->addTab(buildOutputPage(), "&Output");

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    buttons->button(QDialogButtonBox::Ok)->setText("&Insert into editor");
    connect(buttons, &QDialogButtonBox::accepted, this, &StlImportWizard::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &StlImportWizard::reject);
    outer->addWidget(buttons);

    if (!ok) {
        for (int i = 1; i < tabs_->count(); ++i) tabs_->setTabEnabled(i, false);
        buttons->button(QDialogButtonBox::Ok)->setEnabled(false);
        return;
    }

    syncControls();
    showWatertightDiagnostics();
    renderPreview();
}

// ---------------------------------------------------------------------------
// pages
// ---------------------------------------------------------------------------

QWidget *StlImportWizard::buildSourcePage()
{
    auto *w = new QWidget;
    auto *lay = new QFormLayout(w);

    lay->addRow("File:", new QLabel(QFileInfo(sourcePath_).fileName()));

    if (!loaded_) {
        auto *bad = new QLabel("Could not parse this file as an STL or SPARTA surface file.");
        bad->setWordWrap(true);
        lay->addRow(bad);
        return w;
    }

    const QString kindStr =
        (kind_ == SourceKind::Stl) ? "STL" : "SPARTA surface file";
    lay->addRow("Type:", new QLabel(kindStr));
    lay->addRow(mesh_.is2d ? "Lines:" : "Triangles:", new QLabel(QString::number(mesh_.nelements())));
    lay->addRow("Points:", new QLabel(QString::number(mesh_.npoints())));
    lay->addRow("Extent X:", new QLabel(QString("%1 .. %2").arg(mesh_.lo[0]).arg(mesh_.hi[0])));
    lay->addRow("Extent Y:", new QLabel(QString("%1 .. %2").arg(mesh_.lo[1]).arg(mesh_.hi[1])));
    if (!mesh_.is2d)
        lay->addRow("Extent Z:", new QLabel(QString("%1 .. %2").arg(mesh_.lo[2]).arg(mesh_.hi[2])));

    auto *wtl = new QLabel;
    wtl->setWordWrap(true);
    if (wt_.watertight())
        wtl->setText("<b style='color:#3aa563'>Watertight:</b> the surface passes the "
                     "closed-manifold check and can be used directly.");
    else
        wtl->setText(QString("<b style='color:#d9534f'>Not watertight:</b> %1 unmatched (hole) "
                             "and %2 duplicate (non-manifold) edges. See the Preview tab (leaks in "
                             "red) and Diagnostics.")
                         .arg(wt_.unmatchedEdges)
                         .arg(wt_.duplicateEdges));
    lay->addRow(wtl);
    return w;
}

QWidget *StlImportWizard::buildTransformPage()
{
    auto *w = new QWidget;
    auto *lay = new QVBoxLayout(w);

    auto mkspin = [this](double lo, double hi, double val, double step, int dec) {
        auto *s = new QDoubleSpinBox;
        s->setRange(lo, hi);
        s->setDecimals(dec);
        s->setSingleStep(step);
        s->setValue(val);
        connect(s, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                &StlImportWizard::syncControls);
        return s;
    };

    // scale
    auto *scaleBox = new QGroupBox("Scale (about origin)");
    auto *sl = new QHBoxLayout(scaleBox);
    scaleOn_ = new QCheckBox("enable");
    connect(scaleOn_, &QCheckBox::toggled, this, &StlImportWizard::syncControls);
    sl->addWidget(scaleOn_);
    for (int k = 0; k < 3; ++k) {
        sl->addWidget(new QLabel(QString(QChar('X' + k)) + ":"));
        scale_[k] = mkspin(1e-6, 1e6, 1.0, 0.1, 6);
        sl->addWidget(scale_[k]);
    }
    lay->addWidget(scaleBox);

    // translate
    auto *transBox = new QGroupBox("Translate");
    auto *tl = new QHBoxLayout(transBox);
    transKind_ = new QComboBox;
    transKind_->addItems({"none", "trans (by)", "atrans (to)", "ftrans (fractional)"});
    connect(transKind_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &StlImportWizard::syncControls);
    tl->addWidget(transKind_);
    for (int k = 0; k < 3; ++k) {
        tl->addWidget(new QLabel(QString(QChar('X' + k)) + ":"));
        trans_[k] = mkspin(-1e9, 1e9, 0.0, 0.1, 6);
        tl->addWidget(trans_[k]);
    }
    lay->addWidget(transBox);

    // rotate
    auto *rotBox = new QGroupBox("Rotate");
    auto *rl = new QHBoxLayout(rotBox);
    rotOn_ = new QCheckBox("enable");
    connect(rotOn_, &QCheckBox::toggled, this, &StlImportWizard::syncControls);
    rl->addWidget(rotOn_);
    rl->addWidget(new QLabel("angle:"));
    rot_[0] = mkspin(-360, 360, 0.0, 5.0, 3);
    rl->addWidget(rot_[0]);
    const char *axl[3] = {"axis X:", "Y:", "Z:"};
    double axdef[3] = {0, 0, 1};
    for (int k = 0; k < 3; ++k) {
        rl->addWidget(new QLabel(axl[k]));
        rot_[k + 1] = mkspin(-1, 1, axdef[k], 0.1, 3);
        rl->addWidget(rot_[k + 1]);
    }
    lay->addWidget(rotBox);

    // flags + group
    auto *flags = new QHBoxLayout;
    invert_ = new QCheckBox("invert normals");
    transparent_ = new QCheckBox("transparent");
    clipOn_ = new QCheckBox("clip to box");
    for (auto *c : {invert_, transparent_, clipOn_}) {
        connect(c, &QCheckBox::toggled, this, &StlImportWizard::syncControls);
        flags->addWidget(c);
    }
    flags->addWidget(new QLabel("group:"));
    group_ = new QLineEdit;
    group_->setPlaceholderText("(optional)");
    connect(group_, &QLineEdit::textChanged, this, &StlImportWizard::syncControls);
    flags->addWidget(group_);
    lay->addLayout(flags);

    cmdPreview_ = new QLabel;
    cmdPreview_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    cmdPreview_->setWordWrap(true);
    cmdPreview_->setStyleSheet("font-family: monospace; padding:6px;");
    lay->addWidget(new QLabel("Generated read_surf command:"));
    lay->addWidget(cmdPreview_);
    lay->addStretch();
    return w;
}

QWidget *StlImportWizard::buildPreviewPage()
{
    auto *w = new QWidget;
    auto *lay = new QVBoxLayout(w);

    previewLabel_ = new QLabel;
    previewLabel_->setAlignment(Qt::AlignCenter);
    previewLabel_->setMinimumSize(PREVIEW_W, PREVIEW_H);
    previewLabel_->setStyleSheet("background:#1c1e22;");
    lay->addWidget(previewLabel_, 1);

    previewNote_ = new QLabel;
    previewNote_->setWordWrap(true);
    lay->addWidget(previewNote_);

    auto *row = new QHBoxLayout;
    auto *meshBtn = new QPushButton(QIcon(":/icons/image-x-generic.svg"), "Update mesh preview");
    connect(meshBtn, &QPushButton::clicked, this, &StlImportWizard::renderPreview);
    row->addWidget(meshBtn);
    spartaPreviewBtn_ = new QPushButton(QIcon(":/icons/image-viewer.svg"), "Render in SPARTA");
    spartaPreviewBtn_->setToolTip("Authoritative dump-image render (watertight surfaces only). "
                                  "Resets the in-memory simulation; your input script is untouched.");
    connect(spartaPreviewBtn_, &QPushButton::clicked, this, &StlImportWizard::renderSpartaPreview);
    row->addWidget(spartaPreviewBtn_);
    row->addStretch();
    lay->addLayout(row);
    return w;
}

QWidget *StlImportWizard::buildAblationPage()
{
    auto *w = new QWidget;
    auto *lay = new QVBoxLayout(w);

    auto *intro = new QLabel(
        "Convert the surface to an <b>implicit</b> surface (for ablation) and compare how "
        "faithfully each <tt>create_isurf</tt> mode reproduces it. Rendered with SPARTA "
        "<tt>dump image</tt>; requires a watertight surface. Grid resolution is the fidelity knob. "
        "<i>Rendering resets the in-memory simulation; your input script is untouched.</i>");
    intro->setWordWrap(true);
    lay->addWidget(intro);

    auto *form = new QFormLayout;
    ablMode_ = new QComboBox;
    ablMode_->addItems({"inout", "voxel", "ave", "multi"});
    ablMode_->setCurrentText("voxel");
    connect(ablMode_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &StlImportWizard::rebuildOutput);
    form->addRow("create_isurf mode:", ablMode_);

    auto *gl = new QHBoxLayout;
    for (int k = 0; k < 3; ++k) {
        gl->addWidget(new QLabel(QString("N%1:").arg(QChar('x' + k))));
        grid_[k] = new QSpinBox;
        grid_[k]->setRange(2, 400);
        grid_[k]->setValue(50);
        connect(grid_[k], QOverload<int>::of(&QSpinBox::valueChanged), this,
                &StlImportWizard::rebuildOutput);
        gl->addWidget(grid_[k]);
    }
    auto *gw = new QWidget;
    gw->setLayout(gl);
    form->addRow("grid resolution:", gw);

    thresh_ = new QDoubleSpinBox;
    thresh_->setRange(0.5, 254.5);
    thresh_->setDecimals(1);
    thresh_->setValue(39.5);
    connect(thresh_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &StlImportWizard::rebuildOutput);
    form->addRow("threshold (non-integer):", thresh_);

    isurfGroup_ = new QLineEdit("all");
    ablateId_ = new QLineEdit("fablate");
    connect(isurfGroup_, &QLineEdit::textChanged, this, &StlImportWizard::rebuildOutput);
    connect(ablateId_, &QLineEdit::textChanged, this, &StlImportWizard::rebuildOutput);
    form->addRow("grid group / ablate ID:", isurfGroup_);
    form->addRow("fix ablate ID:", ablateId_);
    lay->addLayout(form);

    ablationLabel_ = new QLabel;
    ablationLabel_->setAlignment(Qt::AlignCenter);
    ablationLabel_->setMinimumSize(PREVIEW_W, 260);
    ablationLabel_->setStyleSheet("background:#1c1e22;");
    lay->addWidget(ablationLabel_, 1);

    auto *row = new QHBoxLayout;
    auto *one = new QPushButton("Render implicit (selected mode)");
    connect(one, &QPushButton::clicked, this, &StlImportWizard::renderAblation);
    auto *all = new QPushButton("Compare all modes");
    connect(all, &QPushButton::clicked, this, &StlImportWizard::compareAblationModes);
    row->addWidget(one);
    row->addWidget(all);
    row->addStretch();
    lay->addLayout(row);
    return w;
}

QWidget *StlImportWizard::buildDiagnosticsPage()
{
    auto *w = new QWidget;
    auto *lay = new QVBoxLayout(w);
    diag_ = new QPlainTextEdit;
    diag_->setReadOnly(true);
    diag_->setStyleSheet("font-family: monospace;");
    lay->addWidget(diag_);
    return w;
}

QWidget *StlImportWizard::buildOutputPage()
{
    auto *w = new QWidget;
    auto *lay = new QVBoxLayout(w);

    auto *modeBox = new QGroupBox("Insert as");
    auto *ml = new QHBoxLayout(modeBox);
    modeExplicit_ = new QRadioButton("Explicit surface (read_surf)");
    modeImplicit_ = new QRadioButton("Implicit surface for ablation (create_isurf + fix ablate)");
    modeExplicit_->setChecked(true);
    connect(modeExplicit_, &QRadioButton::toggled, this, &StlImportWizard::rebuildOutput);
    ml->addWidget(modeExplicit_);
    ml->addWidget(modeImplicit_);
    lay->addWidget(modeBox);

    lay->addWidget(new QLabel("Text inserted at the editor cursor:"));
    outputPreview_ = new QPlainTextEdit;
    outputPreview_->setReadOnly(true);
    outputPreview_->setStyleSheet("font-family: monospace;");
    lay->addWidget(outputPreview_, 1);
    return w;
}

// ---------------------------------------------------------------------------
// control sync + command generation
// ---------------------------------------------------------------------------

void StlImportWizard::syncControls()
{
    if (!scaleOn_) return;
    settings_.useScale = scaleOn_->isChecked();
    for (int k = 0; k < 3; ++k) settings_.scale[k] = scale_[k]->value();

    static const StlImportSettings::TransKind tk[4] = {
        StlImportSettings::TransKind::None, StlImportSettings::TransKind::Trans,
        StlImportSettings::TransKind::ATrans, StlImportSettings::TransKind::FTrans};
    settings_.transKind = tk[qBound(0, transKind_->currentIndex(), 3)];
    for (int k = 0; k < 3; ++k) settings_.trans[k] = trans_[k]->value();

    settings_.useRotate = rotOn_->isChecked();
    for (int k = 0; k < 4; ++k) settings_.rotate[k] = rot_[k]->value();

    settings_.invert = invert_->isChecked();
    settings_.transparent = transparent_->isChecked();
    settings_.useClip = clipOn_->isChecked();
    settings_.group = group_->text().trimmed();

    if (cmdPreview_)
        cmdPreview_->setText(buildReadSurfCommand(settings_, QFileInfo(targetSurfPath()).fileName()));
    rebuildOutput();
}

void StlImportWizard::rebuildOutput()
{
    if (!modeExplicit_) return;
    const QString surf = QFileInfo(targetSurfPath()).fileName();
    if (modeImplicit_->isChecked()) {
        settings_.mode = StlImportSettings::Mode::Implicit;
        settings_.isurfGroup = isurfGroup_->text().trimmed();
        settings_.ablateId = ablateId_->text().trimmed();
        settings_.thresh = thresh_->value();
        settings_.isurfMode = ablMode_->currentText();
        settings_.gridNx = grid_[0]->value();
        settings_.gridNy = grid_[1]->value();
        settings_.gridNz = grid_[2]->value();
        generated_ = buildAblationCommands(settings_, surf).join('\n');
    } else {
        settings_.mode = StlImportSettings::Mode::Explicit;
        generated_ = buildReadSurfCommand(settings_, surf);
    }
    if (outputPreview_) outputPreview_->setPlainText(generated_);
}

QString StlImportWizard::targetSurfPath() const
{
    QFileInfo fi(sourcePath_);
    if (kind_ == SourceKind::Stl)
        return fi.absolutePath() + "/" + fi.completeBaseName() + ".surf";
    return sourcePath_; // existing surf file used as-is
}

// ---------------------------------------------------------------------------
// native mesh render (handles non-watertight surfaces, leaks in red)
// ---------------------------------------------------------------------------

QImage StlImportWizard::renderMesh(const QSet<int> &leak, QSize size, bool applyTransform) const
{
    QImage img(size, QImage::Format_RGB32);
    img.fill(QColor(28, 30, 34));
    if (mesh_.points.isEmpty()) return img;

    QVector<Vec3> P = mesh_.points;
    if (applyTransform) {
        if (settings_.useScale)
            for (auto &p : P)
                for (int k = 0; k < 3; ++k) p[k] *= settings_.scale[k];
        if (settings_.useRotate) {
            double rm[9];
            rotationMatrix(settings_.rotate[0],
                           {settings_.rotate[1], settings_.rotate[2], settings_.rotate[3]}, rm);
            for (auto &p : P) p = matvec(rm, p);
        }
    }

    Vec3 lo = P[0], hi = P[0];
    for (const auto &p : P)
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], p[k]);
            hi[k] = std::max(hi[k], p[k]);
        }
    const Vec3 c = {(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2};
    double rad = 0;
    for (int k = 0; k < 3; ++k) rad = std::max(rad, (hi[k] - lo[k]) / 2);
    if (rad <= 0) rad = 1;

    // fixed isometric view: R = Rx(22) * Ry(-35)
    double rx[9], ry[9];
    rotationMatrix(22.0, {1, 0, 0}, rx);
    rotationMatrix(-35.0, {0, 1, 0}, ry);
    double R[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += rx[i * 3 + k] * ry[k * 3 + j];
            R[i * 3 + j] = s;
        }

    const double sc = 0.42 * std::min(size.width(), size.height()) / rad;
    const double cx = size.width() / 2.0, cy = size.height() / 2.0;
    QVector<QPointF> scr(P.size());
    QVector<Vec3> vp(P.size());
    for (int i = 0; i < P.size(); ++i) {
        vp[i] = matvec(R, sub(P[i], c));
        scr[i] = QPointF(cx + sc * vp[i][0], cy - sc * vp[i][1]);
    }

    const Vec3 light = normalize({0.35, 0.4, 0.85});
    struct Face {
        int idx;
        double z;
    };
    QVector<Face> faces;
    faces.reserve(mesh_.nelements());
    for (int e = 0; e < mesh_.nelements(); ++e) {
        const auto &el = mesh_.elems[e];
        double z = (vp[el[0]][2] + vp[el[1]][2] + (mesh_.is2d ? 0 : vp[el[2]][2])) /
                   (mesh_.is2d ? 2 : 3);
        faces.append({e, z});
    }
    std::sort(faces.begin(), faces.end(), [](const Face &a, const Face &b) { return a.z < b.z; });

    QPainter painter(&img);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const QColor okCol(150, 155, 168), badCol(232, 66, 54);
    for (const auto &f : faces) {
        const auto &el = mesh_.elems[f.idx];
        const bool bad = leak.contains(f.idx);
        if (mesh_.is2d) {
            painter.setPen(QPen(bad ? badCol : okCol, bad ? 2.4 : 1.4));
            painter.drawLine(scr[el[0]], scr[el[1]]);
            continue;
        }
        const Vec3 n = normalize(cross(sub(vp[el[1]], vp[el[0]]), sub(vp[el[2]], vp[el[0]])));
        const double b = 0.30 + 0.70 * std::fabs(dot(n, light));
        QPolygonF poly;
        poly << scr[el[0]] << scr[el[1]] << scr[el[2]];
        painter.setPen(Qt::NoPen);
        painter.setBrush(scaleColor(bad ? badCol : okCol, b));
        painter.drawPolygon(poly);
    }
    return img;
}

void StlImportWizard::renderPreview()
{
    if (!loaded_ || !previewLabel_) return;
    const QImage img = renderMesh(wt_.leakingElems, QSize(PREVIEW_W, PREVIEW_H), true);
    previewLabel_->setPixmap(QPixmap::fromImage(img));
    if (wt_.watertight()) {
        previewNote_->setText("Native mesh preview (scale/rotate applied). This surface is "
                              "watertight — use \"Render in SPARTA\" for the authoritative view.");
        if (spartaPreviewBtn_) spartaPreviewBtn_->setEnabled(true);
    } else {
        previewNote_->setText("<span style='color:#d9534f'>Leaking elements shown in red.</span> "
                              "SPARTA cannot load or render a non-watertight surface, so the "
                              "authoritative render is unavailable until the mesh is repaired.");
        if (spartaPreviewBtn_) spartaPreviewBtn_->setEnabled(false);
    }
}

// ---------------------------------------------------------------------------
// SPARTA dump-image render (watertight surfaces + ablation)
// ---------------------------------------------------------------------------

QStringList StlImportWizard::boxGridCommands() const
{
    double lo[3], hi[3];
    for (int k = 0; k < 3; ++k) {
        lo[k] = mesh_.lo[k];
        hi[k] = mesh_.hi[k];
    }
    double maxext = 0;
    for (int k = 0; k < 3; ++k) maxext = std::max(maxext, hi[k] - lo[k]);
    if (maxext <= 0) maxext = 1;
    const double pad = 0.08 * maxext;
    double blo[3], bhi[3];
    for (int k = 0; k < 3; ++k) {
        blo[k] = lo[k] - pad;
        bhi[k] = hi[k] + pad;
    }
    const bool d2 = mesh_.is2d;
    if (d2) {
        blo[2] = -0.5;
        bhi[2] = 0.5;
    }
    QStringList c;
    c << "seed 12345";
    c << (d2 ? "dimension 2" : "dimension 3");
    // explicit/distributed is required by create_isurf and works for plain
    // read_surf rendering too
    c << "global surfs explicit/distributed";
    c << "boundary p p p";
    c << QString("create_box %1 %2 %3 %4 %5 %6")
             .arg(blo[0]).arg(bhi[0]).arg(blo[1]).arg(bhi[1]).arg(blo[2]).arg(bhi[2]);
    const int nx = grid_ ? grid_[0]->value() : 50;
    const int ny = grid_ ? grid_[1]->value() : 50;
    const int nz = d2 ? 1 : (grid_ ? grid_[2]->value() : 50);
    c << QString("create_grid %1 %2 %3").arg(nx).arg(ny).arg(nz);
    return c;
}

QImage StlImportWizard::renderViaSparta(const QStringList &setup, const QString &group,
                                        const QString &dumpargs, QString &captured, QString &err)
{
    err.clear();
    captured.clear();
    if (!sparta_ || !sparta_->isOpen()) {
        err = "The SPARTA library is not loaded (set the plugin path in Preferences).";
        return {};
    }
    if (sparta_->isRunning()) {
        err = "A simulation is currently running; stop it before rendering here.";
        return {};
    }
    const QString id = RENDER_ID;
    const QString base = "sguiwizsurf";
    QDir dumpdir(QDir::tempPath());
    const QString starfile = dumpdir.absoluteFilePath(base + ".*.ppm");
    auto cleanup = [&]() {
        for (const auto &f : dumpdir.entryList({base + ".*.ppm"}, QDir::Files))
            QFile::remove(dumpdir.absoluteFilePath(f));
    };

    StdCapture cap;
    cap.beginCapture();
    if (sparta_->hasId("dump", id.toLocal8Bit())) {
        sparta_->command("undump " + id);
        (void)sparta_->lastErrorMessage();
    }
    sparta_->command("clear");
    for (const auto &cmd : setup) sparta_->command(cmd);
    QString setupErr = sparta_->lastErrorMessage();
    if (!setupErr.isEmpty()) {
        cap.endCapture();
        captured = QString::fromStdString(cap.getCapture());
        err = setupErr;
        cleanup();
        return {};
    }
    sparta_->command("dump " + id + " image " + group + " 1 '" + starfile + "' " + dumpargs);
    sparta_->command("dump_modify " + id + " first yes pad 0");
    sparta_->command("dump_modify " + id + " scolor * gray");
    sparta_->command("run 0 pre yes post no");
    sparta_->command("undump " + id);
    cap.endCapture();
    captured = QString::fromStdString(cap.getCapture());
    QString runErr = sparta_->lastErrorMessage();

    const auto step = static_cast<long long>(sparta_->getThermo("step"));
    const QString imgpath = dumpdir.absoluteFilePath(QString("%1.%2.ppm").arg(base).arg(step));
    if (!runErr.isEmpty()) {
        cleanup();
        err = runErr;
        return {};
    }
    QImageReader reader(imgpath);
    reader.setAutoTransform(true);
    QImage im = reader.read();
    cleanup();
    if (im.isNull()) err = "SPARTA produced no image.";
    return im;
}

void StlImportWizard::renderSpartaPreview()
{
    QString surf = QDir::tempPath() + "/sguiwiz_preview.surf";
    QFile f(surf);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        f.write(buildSurfFile(mesh_, QFileInfo(sourcePath_).fileName()).toUtf8());
        f.close();
    }
    StlImportSettings s = settings_;
    QStringList setup = boxGridCommands();
    setup << buildReadSurfCommand(s, surf);
    // every surface element must be assigned a collision model or the run aborts
    setup << "surf_collide 1 diffuse 300.0 0.0";
    setup << "surf_modify all collide 1";
    const QString dumpargs =
        QString("type type surf one 0.02 particle no size %1 %2 zoom 1.5 box no 0.0")
            .arg(PREVIEW_W).arg(PREVIEW_H);
    QString captured, err;
    const QImage img = renderViaSparta(setup, "all", dumpargs, captured, err);
    appendDiagnostics("SPARTA preview render", captured + (err.isEmpty() ? "" : "\nERROR: " + err));
    if (!img.isNull()) {
        previewLabel_->setPixmap(QPixmap::fromImage(img));
        previewNote_->setText("Authoritative SPARTA dump-image render.");
    } else {
        warning(this, "SPARTA render failed",
                "Could not render this surface in SPARTA:", QString("<code>%1</code>").arg(err));
    }
}

// ---------------------------------------------------------------------------
// ablation implicit-reconstruction renders
// ---------------------------------------------------------------------------

void StlImportWizard::renderAblation()
{
    if (!wt_.watertight()) {
        warning(this, "Ablation preview",
                "create_isurf needs a watertight surface. Repair the mesh (see the red leaks in "
                "the Preview tab) before converting it to an implicit surface.");
        return;
    }
    rebuildOutput(); // pull ablation params into settings_
    QString surf = QDir::tempPath() + "/sguiwiz_preview.surf";
    QFile f(surf);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        f.write(buildSurfFile(mesh_, QFileInfo(sourcePath_).fileName()).toUtf8());
        f.close();
    }
    QStringList setup = boxGridCommands();
    setup << buildReadSurfCommand(settings_, surf);
    setup << "surf_collide 1 diffuse 300.0 0.0";
    setup << "surf_modify all collide 1";
    setup << QString("fix %1 ablate %2 0 %3 random 0")
                 .arg(settings_.ablateId, settings_.isurfGroup)
                 .arg(settings_.ablateScale);
    setup << QString("create_isurf %1 %2 %3 %4")
                 .arg(settings_.isurfGroup, settings_.ablateId)
                 .arg(settings_.thresh)
                 .arg(settings_.isurfMode);
    const QString dumpargs =
        QString("type type surf one 0.02 particle no size %1 %2 zoom 1.5 box no 0.0")
            .arg(PREVIEW_W).arg(240);
    QString captured, err;
    const QImage img = renderViaSparta(setup, "all", dumpargs, captured, err);
    appendDiagnostics(QString("Ablation render (mode %1)").arg(settings_.isurfMode),
                      captured + (err.isEmpty() ? "" : "\nERROR: " + err));
    if (!img.isNull())
        ablationLabel_->setPixmap(QPixmap::fromImage(img));
    else
        warning(this, "Ablation render failed", "SPARTA could not build the implicit surface:",
                QString("<code>%1</code>").arg(err));
}

void StlImportWizard::compareAblationModes()
{
    if (!wt_.watertight()) {
        warning(this, "Ablation preview", "create_isurf needs a watertight surface.");
        return;
    }
    rebuildOutput();
    QString surf = QDir::tempPath() + "/sguiwiz_preview.surf";
    QFile f(surf);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        f.write(buildSurfFile(mesh_, QFileInfo(sourcePath_).fileName()).toUtf8());
        f.close();
    }
    const QStringList modes = {"inout", "voxel", "ave", "multi"};
    const int cw = PREVIEW_W / 2, ch = 240;
    QImage grid(cw * 2, ch * 2, QImage::Format_RGB32);
    grid.fill(QColor(20, 22, 26));
    QPainter gp(&grid);
    for (int i = 0; i < modes.size(); ++i) {
        QStringList setup = boxGridCommands();
        setup << buildReadSurfCommand(settings_, surf);
        setup << "surf_collide 1 diffuse 300.0 0.0";
        setup << "surf_modify all collide 1";
        setup << QString("fix %1 ablate %2 0 %3 random 0")
                     .arg(settings_.ablateId, settings_.isurfGroup)
                     .arg(settings_.ablateScale);
        setup << QString("create_isurf %1 %2 %3 %4")
                     .arg(settings_.isurfGroup, settings_.ablateId)
                     .arg(settings_.thresh)
                     .arg(modes[i]);
        const QString dumpargs =
            QString("type type surf one 0.02 particle no size %1 %2 zoom 1.5 box no 0.0")
                .arg(cw).arg(ch);
        QString captured, err;
        QImage im = renderViaSparta(setup, "all", dumpargs, captured, err);
        const int x = (i % 2) * cw, y = (i / 2) * ch;
        if (!im.isNull()) gp.drawImage(x, y, im.scaled(cw, ch, Qt::KeepAspectRatio));
        gp.setPen(Qt::white);
        gp.drawText(x + 6, y + 18, modes[i] + (err.isEmpty() ? "" : " (failed)"));
        if (!err.isEmpty())
            appendDiagnostics(QString("Ablation compare mode %1").arg(modes[i]), err);
    }
    gp.end();
    ablationLabel_->setPixmap(QPixmap::fromImage(grid));
}

// ---------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------

void StlImportWizard::appendDiagnostics(const QString &title, const QString &text)
{
    if (!diag_ || text.trimmed().isEmpty()) return;
    diag_->appendPlainText("=== " + title + " ===");
    diag_->appendPlainText(text.trimmed());
    diag_->appendPlainText("");
}

void StlImportWizard::showWatertightDiagnostics()
{
    if (!diag_) return;
    if (wt_.watertight()) {
        appendDiagnostics("Watertightness", "PASS: the surface is a closed manifold.");
        return;
    }
    QString s;
    s += QString("NOT WATERTIGHT: %1 unmatched (hole) edges, %2 duplicate (non-manifold) edges.\n")
             .arg(wt_.unmatchedEdges)
             .arg(wt_.duplicateEdges);
    s += QString("%1 leaking element(s) highlighted in red on the Preview tab.\n")
             .arg(wt_.leakingElems.size());
    const int show = std::min(12, static_cast<int>(wt_.unmatchedEdgeList.size()));
    if (show > 0) s += "First hole edges (point coordinates):\n";
    for (int i = 0; i < show; ++i) {
        const auto &e = wt_.unmatchedEdgeList[i];
        const auto &a = mesh_.points[e[0]];
        const auto &b = mesh_.points[e[1]];
        s += QString("  (%1, %2, %3) -> (%4, %5, %6)\n")
                 .arg(a[0]).arg(a[1]).arg(a[2]).arg(b[0]).arg(b[1]).arg(b[2]);
    }
    appendDiagnostics("Watertightness", s);
}

// ---------------------------------------------------------------------------
// accept: write the .surf (STL case) and stage the generated command block
// ---------------------------------------------------------------------------

void StlImportWizard::accept()
{
    rebuildOutput();
    if (kind_ == SourceKind::Stl) {
        const QString path = targetSurfPath();
        QFile f(path);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            warning(this, "Import Surface", "Could not write the SPARTA surface file:",
                    QString("<code>%1</code>").arg(path));
            return;
        }
        f.write(buildSurfFile(mesh_, QFileInfo(sourcePath_).fileName()).toUtf8());
        f.close();
        writtenSurf_ = path;
    }
    QDialog::accept();
}

// Local Variables:
// c-basic-offset: 4
// End:
