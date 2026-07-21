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

#include "welcomescreen.h"

#include "constants.h"

#include <QDir>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPixmap>
#include <QPushButton>
#include <QScrollArea>
#include <QVBoxLayout>

namespace {
constexpr int GALLERY_ICON_W = 132;
constexpr int GALLERY_ICON_H = 96;
constexpr int GALLERY_CELL_W = 168;
constexpr int GALLERY_CELL_H = 148;

// the file-name key an example maps to under :/examples, e.g. circle/in.circle
// -> "circle__in.circle" (matches the thumbnails committed as resources)
QString exampleKey(const QString &inpath)
{
    QFileInfo fi(inpath);
    return QFileInfo(fi.absolutePath()).fileName() + "__" + fi.fileName();
}
} // namespace

WelcomeScreen::WelcomeScreen(QWidget *parent) :
    QWidget(parent), recentList(nullptr), exampleList(nullptr), recentEmpty(nullptr)
{
    setObjectName("welcomeScreen");
    setAutoFillBackground(true);

    auto *outer = new QVBoxLayout(this);
    outer->setContentsMargins(24, 20, 24, 20);
    outer->setSpacing(16);

    // --- header banner ----------------------------------------------------
    auto *banner = new QLabel;
    QPixmap pix(":/icons/sparta-gui-banner.png");
    if (!pix.isNull())
        banner->setPixmap(pix.scaledToHeight(120, Qt::SmoothTransformation));
    banner->setAlignment(Qt::AlignCenter);
    outer->addWidget(banner);

    auto *subtitle = new QLabel("A graphical editor and runner for SPARTA DSMC simulations");
    subtitle->setAlignment(Qt::AlignCenter);
    subtitle->setEnabled(false);
    outer->addWidget(subtitle);

    // --- quick-start action buttons --------------------------------------
    auto *actions = new QHBoxLayout;
    actions->addStretch();
    auto *newbtn = new QPushButton(QIcon(":/icons/document-new.svg"), "  New Input File");
    auto *openbtn = new QPushButton(QIcon(":/icons/document-open.svg"), "  Open File...");
    for (auto *b : {newbtn, openbtn}) b->setMinimumHeight(34);
    connect(newbtn, &QPushButton::clicked, this, &WelcomeScreen::newFileRequested);
    connect(openbtn, &QPushButton::clicked, this, &WelcomeScreen::browseRequested);
    actions->addWidget(newbtn);
    actions->addWidget(openbtn);
    actions->addStretch();
    outer->addLayout(actions);

    auto *rule = new QFrame;
    rule->setFrameShape(QFrame::HLine);
    rule->setFrameShadow(QFrame::Sunken);
    outer->addWidget(rule);

    // --- two columns: recent files | examples gallery --------------------
    auto *columns = new QHBoxLayout;
    columns->setSpacing(20);

    // recent files (left)
    auto *recentCol = new QVBoxLayout;
    auto *recentHeading = new QLabel("<b>Recent Files</b>");
    recentCol->addWidget(recentHeading);
    recentList = new QListWidget;
    recentList->setObjectName("recentList");
    recentList->setIconSize(QSize(20, 20));
    recentList->setUniformItemSizes(false);
    // a single click opens the entry (in addition to Enter / double-click)
    connect(recentList, &QListWidget::itemClicked, this, [this](QListWidgetItem *item) {
        if (item) emit openFileRequested(item->data(Qt::UserRole).toString());
    });
    recentCol->addWidget(recentList, 1);
    recentEmpty = new QLabel("No recent files yet.\nOpen an example to get started.");
    recentEmpty->setEnabled(false);
    recentEmpty->setAlignment(Qt::AlignCenter);
    recentEmpty->setWordWrap(true);
    recentCol->addWidget(recentEmpty);
    recentEmpty->hide();
    columns->addLayout(recentCol, 2);

    // examples gallery (right)
    auto *exCol = new QVBoxLayout;
    exCol->addWidget(new QLabel("<b>Examples</b>"));
    exampleList = new QListWidget;
    exampleList->setObjectName("exampleList");
    exampleList->setViewMode(QListView::IconMode);
    exampleList->setResizeMode(QListView::Adjust);
    exampleList->setMovement(QListView::Static);
    exampleList->setWordWrap(true);
    exampleList->setSpacing(8);
    exampleList->setIconSize(QSize(GALLERY_ICON_W, GALLERY_ICON_H));
    exampleList->setGridSize(QSize(GALLERY_CELL_W, GALLERY_CELL_H));
    // a single click opens the example (in addition to Enter / double-click)
    connect(exampleList, &QListWidget::itemClicked, this, [this](QListWidgetItem *item) {
        if (item) emit openExampleRequested(item->data(Qt::UserRole).toString());
    });
    exCol->addWidget(exampleList, 1);
    columns->addLayout(exCol, 5);

    outer->addLayout(columns, 1);
}

QPixmap WelcomeScreen::thumbnailFor(const QString &inpath) const
{
    // an example is shown only when it has an official gallery thumbnail
    // shipped as :/examples/<subdir>__<in.file>.png -- decks without one are
    // skipped entirely (see rebuildExamples()).
    return QPixmap(":/examples/" + exampleKey(inpath) + ".png");
}

void WelcomeScreen::setRecentFiles(const QStringList &files)
{
    recentFiles = files;
    rebuildRecents();
}

void WelcomeScreen::setExamplesDir(const QString &dir)
{
    if (dir == examplesDir && exampleList->count() > 0) return;
    examplesDir = dir;
    rebuildExamples();
}

void WelcomeScreen::rebuildRecents()
{
    recentList->clear();
    int shown = 0;
    for (const auto &path : recentFiles) {
        QFileInfo fi(path);
        if (!fi.exists()) continue;
        auto *item = new QListWidgetItem(QIcon(":/icons/document-open-recent.svg"),
                                         fi.fileName(), recentList);
        item->setData(Qt::UserRole, fi.absoluteFilePath());
        item->setToolTip(fi.absoluteFilePath());
        ++shown;
    }
    const bool empty = (shown == 0);
    recentList->setVisible(!empty);
    recentEmpty->setVisible(empty);
}

void WelcomeScreen::rebuildExamples()
{
    exampleList->clear();
    if (examplesDir.isEmpty()) return;

    // enumerate exactly like SpartaGui::buildExampleMenu(): each in.* file in
    // each example subdirectory (skipping the benchmark directory) is one deck
    QDir exdir(examplesDir);
    const auto subdirs = exdir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
    for (const auto &sub : subdirs) {
        if (sub.fileName() == "bench") continue;
        const auto inputs = QDir(sub.absoluteFilePath())
                                .entryInfoList({QStringLiteral("in.*")}, QDir::Files, QDir::Name);
        for (const auto &input : inputs) {
            // only decks with an official gallery thumbnail are shown
            const QPixmap thumb = thumbnailFor(input.absoluteFilePath());
            if (thumb.isNull()) continue;
            // label with just the example name: drop the leading "in." from the
            // deck file so "ambi/in.ambi" reads as "ambi", "emit/in.emit.face"
            // as "emit.face", etc. (the full path stays in the tooltip)
            QString label = input.fileName();
            if (label.startsWith("in.")) label.remove(0, 3);
            auto *item = new QListWidgetItem(QIcon(thumb), label);
            item->setData(Qt::UserRole, input.absoluteFilePath());
            item->setToolTip(input.absoluteFilePath());
            item->setTextAlignment(Qt::AlignHCenter | Qt::AlignTop);
            item->setSizeHint(QSize(GALLERY_CELL_W, GALLERY_CELL_H));
            exampleList->addItem(item);
        }
    }
}

// Local Variables:
// c-basic-offset: 4
// End:
