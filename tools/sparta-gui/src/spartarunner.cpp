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

#include "spartarunner.h"
#include "spartawrapper.h"

#include <utility>

SpartaRunner::SpartaRunner(QObject *parent) : QThread(parent), sparta(nullptr) {}

void SpartaRunner::setupRun(SpartaWrapper *_sparta, std::string _input, std::string _file,
                            bool _clearfirst)
{
    sparta = _sparta;
    input  = std::move(_input);
    file   = std::move(_file);
    // Not when continuing: "clear" destroys the grid, the particles and every
    // definition, which is exactly what an extended run needs to keep.
    if (_clearfirst) sparta->command("clear");
}

void SpartaRunner::run()
{
    if (!input.empty()) {
        sparta->commandsString(input.c_str());
    } else if (!file.empty()) {
        sparta->file(file.c_str());
    }
    emit resultReady();
}

// Local Variables:
// c-basic-offset: 4
// End:
