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

#ifndef STLIMPORTWIZARD_H
#define STLIMPORTWIZARD_H

// Tabbed "Import Surface (STL / SPARTA)" dialog wrapping the pure converter and
// command builders in stlimport.h. It loads an STL (ASCII/binary) or an existing
// SPARTA surface file; exposes scale/translate/rotate controls; previews the
// geometry with a lightweight native mesh render that highlights *where* the
// watertightness check fails (in red) -- SPARTA itself cannot load or render a
// non-watertight surface, its surf/grid cut requires a closed manifold; renders
// SPARTA's implicit-surface reconstruction per create_isurf mode for an ablation
// fidelity comparison (SPARTA dump image, watertight input required); and emits
// the read_surf / create_isurf+fix ablate command block. Built as a tabbed
// QDialog (the app has no QWizard); results read back via settings()/
// generatedText() after exec()==Accepted.

#include "stlimport.h"

#include <QDialog>
#include <QImage>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;
class QRadioButton;
class QSpinBox;
class QTabWidget;

class SpartaWrapper;

class StlImportWizard : public QDialog {
    Q_OBJECT

public:
    StlImportWizard(QWidget *parent, SpartaWrapper *sparta, const QString &path);

    StlImport::StlImportSettings settings() const { return settings_; }
    /** @brief SPARTA command block to insert into the editor (valid after Accept). */
    QString generatedText() const { return generated_; }
    /** @brief The .surf file written next to the source on Accept (empty if none). */
    QString writtenSurfPath() const { return writtenSurf_; }
    /** @brief True if the source parsed successfully in the constructor. */
    bool loaded() const { return loaded_; }

private slots:
    void syncControls();        // transform widgets -> settings_, refresh labels
    void renderPreview();       // native mesh render (leaks in red)
    void renderSpartaPreview(); // authoritative dump-image render (watertight only)
    void renderAblation();      // SPARTA implicit render, selected mode
    void compareAblationModes();
    void rebuildOutput();
    void accept() override;

private:
    QWidget *buildSourcePage();
    QWidget *buildTransformPage();
    QWidget *buildPreviewPage();
    QWidget *buildAblationPage();
    QWidget *buildDiagnosticsPage();
    QWidget *buildOutputPage();

    // native flat-shaded orthographic render of the mesh; leaking elements red.
    // applyTransform bakes the user's scale/rotate into the view.
    QImage renderMesh(const QSet<int> &leak, QSize size, bool applyTransform) const;
    // run a self-contained SPARTA deck (isolated by clear) and read back the PPM
    QImage renderViaSparta(const QStringList &setup, const QString &group, const QString &dumpargs,
                           QString &captured, QString &err);
    QStringList boxGridCommands() const; // create_box/create_grid from padded extents
    QString targetSurfPath() const;      // where the surf file lives / will be written
    void appendDiagnostics(const QString &title, const QString &text);
    void showWatertightDiagnostics();

    SpartaWrapper *sparta_;
    QString sourcePath_;
    StlImport::SourceKind kind_ = StlImport::SourceKind::Unknown;
    StlImport::SurfMesh mesh_;
    StlImport::WatertightReport wt_;
    StlImport::StlImportSettings settings_;
    QString generated_;
    QString writtenSurf_;
    bool loaded_ = false;

    QTabWidget *tabs_ = nullptr;

    // transform controls
    QCheckBox *scaleOn_ = nullptr;
    QDoubleSpinBox *scale_[3] = {nullptr, nullptr, nullptr};
    QComboBox *transKind_ = nullptr;
    QDoubleSpinBox *trans_[3] = {nullptr, nullptr, nullptr};
    QCheckBox *rotOn_ = nullptr;
    QDoubleSpinBox *rot_[4] = {nullptr, nullptr, nullptr, nullptr};
    QCheckBox *invert_ = nullptr;
    QCheckBox *transparent_ = nullptr;
    QCheckBox *clipOn_ = nullptr;
    QLineEdit *group_ = nullptr;
    QLabel *cmdPreview_ = nullptr;

    // preview
    QLabel *previewLabel_ = nullptr;
    QPushButton *spartaPreviewBtn_ = nullptr;
    QLabel *previewNote_ = nullptr;

    // ablation
    QComboBox *ablMode_ = nullptr;
    QSpinBox *grid_[3] = {nullptr, nullptr, nullptr};
    QDoubleSpinBox *thresh_ = nullptr;
    QLineEdit *isurfGroup_ = nullptr;
    QLineEdit *ablateId_ = nullptr;
    QLabel *ablationLabel_ = nullptr;

    // diagnostics
    QPlainTextEdit *diag_ = nullptr;

    // output
    QRadioButton *modeExplicit_ = nullptr;
    QRadioButton *modeImplicit_ = nullptr;
    QPlainTextEdit *outputPreview_ = nullptr;
};

#endif // STLIMPORTWIZARD_H

// Local Variables:
// c-basic-offset: 4
// End:
