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

#include "setupcard.h"

#include "helpers.h"

#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QPalette>
#include <QPixmap>
#include <QSizePolicy>
#include <QPushButton>
#include <QVBoxLayout>

SetupCard::SetupCard(QWidget *parent) :
    QFrame(parent), explanation(new QLabel), problem(new QLabel),
    downloadButton(nullptr)
{
    setObjectName("setupcard");
    setFrameShape(QFrame::StyledPanel);
    setAutoFillBackground(true);

    // A tint drawn from the palette rather than a fixed colour, so the strip
    // reads as "attention" in a light theme and in a dark one without either
    // being hard-coded.
    QPalette tint = palette();
    tint.setColor(QPalette::Window, palette().color(QPalette::AlternateBase));
    setPalette(tint);

    auto *icon = new QLabel;
    icon->setObjectName("setupicon");
    icon->setPixmap(QPixmap(":/icons/sparta-plugin.png")
                        .scaled(48, 48, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    icon->setAlignment(Qt::AlignTop);

    auto *title = new QLabel("<b>No SPARTA shared library yet</b>");
    title->setObjectName("setuptitle");

    explanation->setObjectName("setupexplain");
    explanation->setWordWrap(true);
    // The card lives above the editor, which in a three-panel workspace can be
    // a narrow column.  Let the wrapped prose give up width first: with the
    // default policy the layout squeezed the buttons instead and clipped their
    // labels, so the one part of the card that has to stay readable was the
    // part that stopped being.
    explanation->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    explanation->setText(
        "You can write, open and save input decks without one; running them needs "
        "the SPARTA shared library.");

    problem->setObjectName("setupproblem");
    problem->setWordWrap(true);
    problem->hide();
    QPalette bad = problem->palette();
    bad.setColor(QPalette::WindowText, QColor(0xb0, 0x30, 0x20));
    problem->setPalette(bad);

    auto *text = new QVBoxLayout;
    text->setContentsMargins(0, 0, 0, 0);
    text->setSpacing(2);
    text->addWidget(title);
    text->addWidget(explanation);
    text->addWidget(problem);

    auto *buttons = new QHBoxLayout;
    buttons->setContentsMargins(0, 0, 0, 0);
    buttons->setSpacing(6);

    // Offered only when this build can actually use the pre-compiled libraries
    // on the webserver; a button that always answers "not for you" is worse
    // than no button.
    if (!getSpartaDownloadUrl().isEmpty()) {
        downloadButton = new QPushButton(QIcon(":/icons/download-file.svg"), "&Download");
        downloadButton->setObjectName("setupdownload");
        downloadButton->setToolTip("Fetch the pre-compiled SPARTA library for this platform");
        downloadButton->setDefault(true);
        connect(downloadButton, &QPushButton::clicked, this, &SetupCard::downloadRequested);
        buttons->addWidget(downloadButton);
    }

    auto *browse = new QPushButton(QIcon(":/icons/document-open.svg"), "&Browse...");
    browse->setObjectName("setupbrowse");
    browse->setToolTip("Select a SPARTA shared library already on this computer");
    connect(browse, &QPushButton::clicked, this, &SetupCard::browseRequested);
    buttons->addWidget(browse);

    auto *what = new QPushButton(QIcon(":/icons/help-browser.svg"), "&What is this?");
    what->setObjectName("setuphelp");
    what->setToolTip("Read about the SPARTA shared library and how to build one");
    connect(what, &QPushButton::clicked, this, &SetupCard::helpRequested);
    buttons->addWidget(what);

    auto *row = new QHBoxLayout(this);
    row->setContentsMargins(10, 8, 10, 8);
    row->setSpacing(10);
    row->addWidget(icon);
    row->addLayout(text, 1);
    row->addLayout(buttons);
}

void SetupCard::setError(const QString &message)
{
    problem->setText(message);
    problem->setVisible(!message.isEmpty());
}

void SetupCard::clearError()
{
    setError(QString());
}

bool SetupCard::canDownload() const
{
    return downloadButton != nullptr;
}

// Local Variables:
// c-basic-offset: 4
// End:
