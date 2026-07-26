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

// What the image viewer has to do to show its settings dialog: describe the
// running simulation, hand that description over, and apply the answer.
//
// The dialog itself is in dumpimagesettingsdialog.cpp.  It used to live here,
// as one 881-line method that asked the SPARTA wrapper for each fact as it
// built the corresponding control -- which meant it could not be constructed
// without a running simulator, and so was never tested.  Everything that still
// needs the viewer stays here: reading the environment, keeping the main
// window's mixture selector in step, and re-rendering afterwards.

#include "imageviewer.h"

#include "imageviewer_internal.h"

#include "constants.h"
#include "dumpimagesettingsdialog.h"
#include "qaddon.h"
#include "spartawrapper.h"

#include <QComboBox>
#include <QDesktopServices>
#include <QString>
#include <QStringList>
#include <QUrl>

#include <algorithm>

QStringList ImageViewer::valueSources(bool withproc, bool withone)
{
    QStringList list;
    if (withone) list << "one";
    if (withproc) list << "proc";
    int num = sparta->idCount("compute");
    for (int i = 0; i < num; ++i)
        list << "c_" + sparta->idName("compute", i);
    num = sparta->idCount("fix");
    for (int i = 0; i < num; ++i)
        list << "f_" + sparta->idName("fix", i);
    num = sparta->idCount("variable");
    for (int i = 0; i < num; ++i)
        list << "v_" + sparta->idName("variable", i);
    return list;
}

// Everything the dialog needs to know about the simulation, read once.
//
// This is the only place the settings dialog touches SPARTA at all; gathering
// it here rather than control by control is what lets the dialog be built and
// driven with no simulator behind it.
ImageSettingsEnv ImageViewer::settingsEnv()
{
    ImageSettingsEnv env;

    env.dimension  = sparta->extractSetting("dimension");
    env.surfsExist = sparta->extractSetting("surf_exist") == 1;

    if (const auto *lo = static_cast<const double *>(sparta->extractGlobal("boxlo")))
        std::copy(lo, lo + 3, env.boxlo);
    if (const auto *hi = static_cast<const double *>(sparta->extractGlobal("boxhi")))
        std::copy(hi, hi + 3, env.boxhi);

    const int nspecies = sparta->extractSetting("nspecies");
    for (int i = 0; i < nspecies; ++i)
        env.species << sparta->idName("species", i);

    for (const char *cat : {"mixture", "region", "group_grid", "group_surf"}) {
        QStringList *dest = (qstrcmp(cat, "mixture") == 0)      ? &env.mixtures
                            : (qstrcmp(cat, "region") == 0)     ? &env.regions
                            : (qstrcmp(cat, "group_grid") == 0) ? &env.gridGroups
                                                                : &env.surfGroups;
        const int num = sparta->idCount(cat);
        for (int i = 0; i < num; ++i)
            *dest << sparta->idName(cat, i);
    }

    env.gridSources     = valueSources(true, false);
    env.surfSources     = valueSources(true, true);
    env.particleSources = valueSources(false, false);

    return env;
}

void ImageViewer::settingsDialog(int tab)
{
    // Top the species colour table up before the dialog is built, not after it
    // is accepted: this has always happened whether or not the user goes on to
    // press Cancel, and moving it would change that silently.
    const int nspecies = sparta->extractSetting("nspecies");
    const int ndef     = defspeciescolors.size();
    for (int i = color_list.size(); i < nspecies; ++i)
        color_list.append(defspeciescolors[i % ndef]);

    DumpImageSettingsDialog dialog(params, settingsEnv(), color_list, tab, this);
    // Same URL getHelp() builds for a SPARTA manual page. The dialog only says
    // which page it wants; it used to reach getHelp() through the Help button's
    // object name, which meant the dialog needed a viewer to talk back to.
    connect(&dialog, &DumpImageSettingsDialog::helpRequested, this, [](const QString &page) {
        QDesktopServices::openUrl(QUrl(QString("%1/doc/%2").arg(Cfg::DOCS_URL, page)));
    });

    // return immediately on cancel
    if (!dialog.exec()) return;

    params     = dialog.settings();
    color_list = dialog.speciesColors();

    // keep the main-window mixture selector in sync without re-rendering
    auto *mainmix = findChild<QComboBox *>("mixture");
    if (mainmix && (mainmix->currentText() != params.mixture)) {
        mainmix->blockSignals(true);
        selectComboItem(mainmix, params.mixture);
        mainmix->blockSignals(false);
    }

    // reflect the new state in the toolbar buttons and re-render
    syncButtons();
    createImage();
}

// Local Variables:
// c-basic-offset: 4
// End:
