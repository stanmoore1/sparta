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

#include "imageviewer.h"

#include "dumpimage.h"
#include "imageviewer_internal.h"

#include "constants.h"
#include "helpers.h"
#include "qaddon.h"
#include "spartagui.h"
#include "spartawrapper.h"
#include "viewerdisplay.h"
#include "viewersidebar.h"
#include "stdcapture.h"

#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QColor>
#include <QComboBox>
#include <QDesktopServices>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontMetrics>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QIcon>
#include <QImage>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <cmath>
#include <QKeySequence>
#include <QLabel>
#include <QLinearGradient>
#include <QMenu>
#include <QMenuBar>
#include <QPainter>
#include <QPalette>
#include <QPixmap>
#include <QPushButton>
#include <QRect>
#include <QRegularExpression>
#include <QScreen>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QShowEvent>
#include <QSizePolicy>
#include <QSpinBox>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>
#include <QVariant>

#include <algorithm>


// read JSON color and light data from file
QJsonObject loadJsonColors(QWidget *parent)
{
    QJsonObject obj;
    QString fileName = QFileDialog::getOpenFileName(parent, "Load Colors from JSON", "",
                                                    "JSON files (*.json);;All files (*)");
    if (fileName.isEmpty()) return obj;

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly)) {
        warning(parent, "Load Colors", "Could not open file '" + fileName + "' for reading.");
        return obj;
    }

    QJsonParseError err;
    auto doc = QJsonDocument::fromJson(file.readAll(), &err);
    if (doc.isNull() || !doc.isObject()) {
        warning(parent, "Load Colors",
                "Invalid JSON colors file '" + fileName + "': " + err.errorString());
        return obj;
    }
    obj      = doc.object();
    auto app = obj.value("application").toString();
    auto key = obj.value("format").toString();
    auto rev = obj.value("revision").toInt();
    if ((app != "SPARTA") || (key != "colors")) {
        warning(parent, "Load Colors",
                "JSON colors file '" + fileName + "' is not a SPARTA colors file.");
        return {};
    }
    if (rev != 1) {
        warning(parent, "Load Colors",
                QString("JSON colors file '%1' has incompatible revision %2 instead of 1")
                    .arg(fileName)
                    .arg(rev));
        return {};
    }

    auto arr = obj.value("colors").toArray();
    if (arr.isEmpty()) {
        warning(parent, "Load Colors",
                "JSON colors file '" + fileName + "' contains no colors entry");
        return obj;
    }

    if (obj.value("lights").toObject().isEmpty()) {
        warning(parent, "Load Colors",
                "JSON colors file '" + fileName + "' contains no lights entry");
        return obj;
    }
    return obj;
}

// save JSON color and light data to file
void saveJsonColors(QWidget *parent, const QJsonArray &colors, const QJsonObject &lights)
{
    QJsonObject root;
    root["application"] = QStringLiteral("SPARTA");
    root["format"]      = QStringLiteral("colors");
    root["revision"]    = 1;
    root["schema"]      = QStringLiteral("https://sparta.github.io/json/color-schema.json");
    root["colors"]      = colors;
    root["lights"]      = lights;

    QString fileName = QFileDialog::getSaveFileName(parent, "Save Colors to JSON", "",
                                                    "JSON files (*.json);;All files (*)");
    if (fileName.isEmpty()) return;

    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly)) {
        warning(parent, "Save Colors", "Could not open file '" + fileName + "' for writing.");
        return;
    }
    file.write(QJsonDocument(root).toJson());
}


ImageViewer::ImageViewer(const QString &fileName, SpartaWrapper *_sparta, SpartaGui *_spartagui,
                         QWidget *parent) :
    ViewerSource(parent), display(new ViewerDisplay(ViewerDisplay::FitViewport)),
    sidebar(new ViewerSidebar), menuBar(new QMenuBar),
    cmdAct(nullptr), movieAct(nullptr), sparta(_sparta),
    spartagui(_spartagui), filename(QFileInfo(fileName).fileName())
{
    // interactive view: drag to rotate/pan, wheel to zoom -- handled in
    // eventFilter(), which re-renders through the same createImage() path as
    // the toolbar buttons
    display->label()->installEventFilter(this);
    display->label()->setCursor(Qt::OpenHandCursor);
    display->setVisible(false);

    auto *imageLayout = new QHBoxLayout;
    auto *mainLayout  = new QVBoxLayout;

    // list of compute and fix styles producing per-grid or per-surf data
    // usable with dump image (regenerated table)
    QFile image_styles(":/image_style.table");
    if (image_styles.open(QIODevice::ReadOnly | QIODevice::Text)) {
        while (!image_styles.atEnd()) {
            auto line  = QString(image_styles.readLine());
            auto words = line.trimmed().split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
            if (words.size() == 2) {
                if (words.at(0) == "compute") {
                    image_computes << words.at(1);
                } else if (words.at(0) == "fix") {
                    image_fixes << words.at(1);
                } else {
                    fprintf(stderr, "unhandled image style: %s\n", qPrintable(line));
                }
            } else {
                fprintf(stderr, "unhandled image style: %s\n", qPrintable(line));
            }
        }
        image_styles.close();
    }

    // store help URL info for computes and fixes with dump image support
    QFile help_index(":/help_index.table");
    if (help_index.open(QIODevice::ReadOnly | QIODevice::Text)) {
        while (!help_index.atEnd()) {
            auto line  = QString(help_index.readLine());
            auto words = line.trimmed().split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
            if (words.size() == 3) {
                if (words.at(1) == "fix") {
                    if (image_fixes.contains(words.at(2))) fix_map[words.at(2)] = words.at(0);
                } else if (words.at(1) == "compute") {
                    if (image_computes.contains(words.at(2)))
                        compute_map[words.at(2)] = words.at(0);
                }
            }
        }
        help_index.close();
    }

    readImageSettings();

    auto fsize = QFontMetrics(QApplication::font()).size(Qt::TextSingleLine, "Height: 200");
#if defined(Q_OS_WIN32)
    fsize = fsize * 3 / 2;
#endif

    auto *renderstatus = new QLabel(QString());
    // The render-status icon shows a full-color "active" pixmap while an image is
    // rendering and a faded "idle" pixmap otherwise. Swap the two pixmaps
    // explicitly (stored as properties) rather than relying on the disabled-widget
    // visual, which does not refresh reliably on all platforms (e.g. macOS 12).
    auto pix                = QPixmap(":/icons/emblem-photos.png");
    const QPixmap activePix = pix.scaled(22, 22, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    const QPixmap idlePix   = grayscalePixmap(activePix);
    renderstatus->setProperty("activePix", activePix);
    renderstatus->setProperty("idlePix", idlePix);
    renderstatus->setPixmap(idlePix);
    renderstatus->setToolTip("Render status");
    renderstatus->setObjectName("renderstatus");

    auto *xval = new QSpinBox;
    xval->setRange(MIN_RENDER_SIZE, MAX_RENDER_SIZE);
    xval->setStepType(QAbstractSpinBox::AdaptiveDecimalStepType);
    xval->setValue(params.xsize);
    xval->setObjectName("xsize");
    xval->setToolTip("Set rendered image width");
    xval->setMinimumSize(fsize);
    auto *yval = new QSpinBox;
    yval->setRange(MIN_RENDER_SIZE, MAX_RENDER_SIZE);
    yval->setStepType(QAbstractSpinBox::AdaptiveDecimalStepType);
    yval->setValue(params.ysize);
    yval->setObjectName("ysize");
    yval->setToolTip("Set rendered image height");
    yval->setMinimumSize(fsize);

    auto *fitrender = new QPushButton(QIcon(":/icons/gtk-zoom-fit.svg"), "");
    fitrender->setToolTip("Render at the size of the space the picture is shown in");
    fitrender->setObjectName("fitrender");
    fitrender->setAccessibleName(fitrender->toolTip());

    connect(xval, &QAbstractSpinBox::editingFinished, this, &ImageViewer::editSize);
    connect(yval, &QAbstractSpinBox::editingFinished, this, &ImageViewer::editSize);
    connect(fitrender, &QPushButton::released, this, &ImageViewer::fitRenderToPanel);

    // workaround for incorrect highlight bug on macOS
    auto *dummy1 = new QPushButton(QIcon(), "");
    dummy1->hide();
    auto *dummy2 = new QPushButton(QIcon(), "");
    dummy2->hide();

    auto *dossao = new QPushButton(QIcon(":/icons/hd-img.svg"), "");
    dossao->setCheckable(true);
    dossao->setToolTip("Toggle SSAO rendering");
    dossao->setObjectName("ssao");
    const QSize buttonhint = toolButtonSize(dossao);
    auto *doanti           = new QPushButton(QIcon(":/icons/antialias.svg"), "");
    doanti->setCheckable(true);
    doanti->setToolTip("Toggle anti-aliasing (fsaa)");
    doanti->setObjectName("antialias");
    auto *doshiny = new QPushButton(QIcon(":/icons/image-shiny.svg"), "");
    doshiny->setCheckable(true);
    doshiny->setToolTip("Toggle shininess");
    doshiny->setObjectName("shiny");
    auto *dopart = new QPushButton(QIcon(":/icons/vdw-style.svg"), "");
    dopart->setCheckable(true);
    dopart->setToolTip("Toggle displaying particles");
    dopart->setObjectName("particles");
    auto *dogrid = new QPushButton(QIcon(":/icons/reference-lines.svg"), "");
    dogrid->setCheckable(true);
    dogrid->setToolTip("Toggle grid volume rendering (colored by owning processor)");
    dogrid->setObjectName("grid");
    auto *dosurf = new QPushButton(QIcon(":/icons/x-office-drawing.svg"), "");
    dosurf->setCheckable(true);
    dosurf->setToolTip("Toggle displaying surface elements");
    dosurf->setObjectName("surf");
    auto *dobox = new QPushButton(QIcon(":/icons/show-box.png"), "");
    dobox->setCheckable(true);
    dobox->setToolTip("Toggle displaying box");
    dobox->setObjectName("box");
    auto *doaxes = new QPushButton(QIcon(":/icons/show-axes.png"), "");
    doaxes->setCheckable(true);
    doaxes->setToolTip("Toggle displaying axes");
    doaxes->setObjectName("axes");
    auto *zoomin = new QPushButton(QIcon(":/icons/gtk-zoom-in.svg"), "");
    zoomin->setToolTip("Camera zoom in by 10 percent");
    zoomin->setObjectName("zoomin");
    auto *zoomout = new QPushButton(QIcon(":/icons/gtk-zoom-out.svg"), "");
    zoomout->setToolTip("Camera zoom out by 10 percent");
    zoomout->setObjectName("zoomout");
// the SVG versions do not render correctly with Qt before 6.7
#if QT_VERSION >= QT_VERSION_CHECK(6, 7, 0)
    auto *rotleft  = new QPushButton(QIcon(":/icons/rotate-left.svg"), "");
    auto *rotright = new QPushButton(QIcon(":/icons/rotate-right.svg"), "");
    auto *rotup    = new QPushButton(QIcon(":/icons/rotate-up.svg"), "");
    auto *rotdown  = new QPushButton(QIcon(":/icons/rotate-down.svg"), "");
#else
    auto *rotleft  = new QPushButton(QIcon(":/icons/rotate-left.png"), "");
    auto *rotright = new QPushButton(QIcon(":/icons/rotate-right.png"), "");
    auto *rotup    = new QPushButton(QIcon(":/icons/rotate-up.png"), "");
    auto *rotdown  = new QPushButton(QIcon(":/icons/rotate-down.png"), "");
#endif
    rotleft->setToolTip("Camera rotate left by 10 degrees");
    rotleft->setObjectName("rotleft");
    rotright->setToolTip("Camera rotate right by 10 degrees");
    rotright->setObjectName("rotright");
    rotup->setToolTip("Camera rotate up by 10 degrees");
    rotup->setObjectName("rotup");
    rotdown->setToolTip("Camera rotate down by 10 degrees");
    rotdown->setObjectName("rotdown");
    auto *recenter = new QPushButton(QIcon(":/icons/move-recenter.svg"), "");
    recenter->setToolTip("Camera recenter on the box center");
    recenter->setObjectName("recenter");
    auto *reset = new QPushButton(QIcon(":/icons/preferences-reset.svg"), "");
    reset->setToolTip("Camera reset to the default view");
    reset->setObjectName("resetview");
    auto *fitwin = new QPushButton(QIcon(":/icons/fit-window.svg"), "");
    fitwin->setToolTip("Resize window to fit the image size");
    fitwin->setObjectName("fitwindow");

    // rotations only apply to 3d systems
    if (sparta->extractSetting("dimension") == 2) {
        for (auto *b : {rotleft, rotright, rotup, rotdown}) {
            b->setEnabled(false);
            b->setToolTip("Not available for 2d simulations");
        }
    }

    // square toolbar buttons with a snug, uniform icon (shared policy)
    styleToolButtons(buttonhint,
                     {dossao, doanti, doshiny, dopart, dogrid, dosurf, dobox, doaxes, zoomin,
                      zoomout, rotleft, rotright, rotup, rotdown, recenter, reset, fitwin,
                      fitrender});

    // These carry an icon and no text, so without an accessible name they reach
    // the AT-SPI walker (and the screenshot sweep) as anonymous buttons. The
    // tooltip already says what each one does, so it is the name to use.
    for (auto *b : {dossao, doanti, doshiny, dopart, dogrid, dosurf, dobox, doaxes, zoomin,
                    zoomout, rotleft, rotright, rotup, rotdown, recenter, reset, fitwin})
        b->setAccessibleName(b->toolTip());

    // match the first-row controls (menu bar and size fields) to the toolbar
    // button height so both rows line up and the layout looks balanced
    menuBar->setFixedHeight(buttonhint.height());
    // This viewer is embedded as a dock panel of the main window, so its menu
    // bar must render inline in the panel and NOT be promoted to the native
    // (global) macOS menu bar -- otherwise focusing the Image panel would
    // replace the main window's menus with just this viewer's File menu, with
    // no way back.
    if (spartagui) menuBar->setNativeMenuBar(false);
    xval->setFixedHeight(buttonhint.height());
    yval->setFixedHeight(buttonhint.height());

    // one button per settings dialog tab
    auto *partviz = new QPushButton("&Particles...");
    partviz->setToolTip("Particle display settings: color, diameter, species colors, region clip");
    partviz->setProperty("tab", 0);
    partviz->setObjectName("particlesettings");
    auto *gridviz = new QPushButton("&Grid...");
    gridviz->setToolTip("Grid volume rendering settings");
    gridviz->setProperty("tab", 1);
    gridviz->setObjectName("gridsettings");
    auto *planeviz = new QPushButton("Grid Pla&nes...");
    planeviz->setToolTip("Grid cut plane rendering settings");
    planeviz->setProperty("tab", 2);
    planeviz->setObjectName("planes");
    auto *surfviz = new QPushButton("S&urfaces...");
    surfviz->setToolTip("Surface element display settings");
    surfviz->setProperty("tab", 3);
    surfviz->setObjectName("surfsettings");
    // NOT "Bo&x": Alt-X is the documented shortcut for the mixture chooser, and
    // a button mnemonic is matched by Qt's shortcut map before the panel's event
    // filter ever sees the key -- so the mixture shortcut silently opened this
    // dialog instead.  B is free among the settings buttons (P, G, N, U, C, Q, M).
    auto *boxviz = new QPushButton("&Box && Axes...");
    boxviz->setToolTip("Box, sub-box, and axes display settings");
    boxviz->setProperty("tab", 4);
    boxviz->setObjectName("boxsettings");
    auto *camviz = new QPushButton("&Camera...");
    camviz->setToolTip("View direction, center, up vector, and zoom");
    camviz->setProperty("tab", 5);
    camviz->setObjectName("camera");
    auto *qualviz = new QPushButton("&Quality...");
    qualviz->setToolTip("Render quality, background, and lights");
    qualviz->setProperty("tab", 6);
    qualviz->setObjectName("quality");
    auto *mapviz = new QPushButton("Color &Maps...");
    mapviz->setToolTip("Color maps for particles, grid, surfaces, and grid planes");
    mapviz->setProperty("tab", 7);
    mapviz->setObjectName("colormaps");
    auto *help = new QPushButton("Help");
    help->setToolTip("Open online help");
    help->setObjectName("visualization.html");

    auto *combo = new QComboBox;
    combo->setToolTip("Select mixture to display");
    combo->setObjectName("mixture");
    const int nmix = sparta->idCount("mixture");
    for (int i = 0; i < nmix; ++i)
        combo->addItem(sparta->idName("mixture", i));
    selectComboItem(combo, params.mixture);

    auto *menuLayout   = new QHBoxLayout;
    auto *buttonLayout = new QHBoxLayout;
    auto *topLayout    = new QVBoxLayout;
    topLayout->addLayout(menuLayout);
    topLayout->setSpacing(LAYOUT_SPACING);

    // a hidden dummy button as the first item works around a macOS bug where the
    // first widget in a toolbar row misbehaves (here: renderstatus not refreshing)
    menuLayout->addWidget(dummy1);
    menuLayout->addWidget(menuBar);
    menuLayout->insertStretch(2, 10);
    menuLayout->addWidget(renderstatus);
    menuLayout->addWidget(new QLabel(" <u>W</u>idth: "));
    menuLayout->addWidget(xval);
    menuLayout->addWidget(new QLabel(" <u>H</u>eight: "));
    menuLayout->addWidget(yval);
    menuLayout->addWidget(fitrender);
    menuLayout->insertStretch(-1, 50);
    menuLayout->setSpacing(LAYOUT_SPACING);

    // The top row is the camera and nothing else.  It used to carry the eight
    // render toggles as well, as unlabelled icons with no visible relation to
    // the settings buttons down the right-hand side that configure the very
    // same things; those have moved into the sidebar next to their subject.
    buttonLayout->addWidget(dummy2);
    auto *camlabel = new QLabel("Camera:");
    camlabel->setToolTip("These controls move the camera and re-render the scene");
    buttonLayout->addWidget(camlabel);
    buttonLayout->addWidget(zoomin);
    buttonLayout->addWidget(zoomout);
    buttonLayout->addWidget(rotleft);
    buttonLayout->addWidget(rotright);
    buttonLayout->addWidget(rotup);
    buttonLayout->addWidget(rotdown);
    buttonLayout->addWidget(recenter);
    buttonLayout->addWidget(reset);
    buttonLayout->addWidget(fitwin);
    buttonLayout->insertStretch(-1, 1);
    buttonLayout->setSizeConstraint(QLayout::SetMinimumSize);
    buttonLayout->setContentsMargins(0, 0, 0, 0);
    buttonLayout->setSpacing(LAYOUT_SPACING);
    // The sidebar puts each subject's on/off switch on the same line as the
    // button that configures it.  The buttons are the ones built above, handed
    // over unchanged: same object names, tooltips, mnemonics and connections,
    // so every slot below and syncButtons() are unaffected by the move.
    //
    //   old top-row toggle -> sidebar row
    //     particles  -> Particles       (primary)
    //     grid       -> Grid            (primary)
    //     surf       -> Surfaces        (primary)
    //     box        -> Box & Axes      (primary)
    //     axes       -> Box & Axes      (secondary)
    //     ssao       -> Quality         (secondary)
    //     antialias  -> Quality         (secondary)
    //     shiny      -> Quality         (secondary)
    //
    // Grid Planes, Camera and Color Maps have no toggle of their own and so
    // occupy only the name column.
    sidebar->addHeader("Mixture:", combo);
    sidebar->addRow(dopart, partviz);
    sidebar->addRow(dogrid, gridviz);
    sidebar->addRow(nullptr, planeviz);
    sidebar->addRow(dosurf, surfviz);
    sidebar->addRow(dobox, boxviz, {doaxes});
    sidebar->addRow(nullptr, camviz);
    sidebar->addRow(nullptr, qualviz, {dossao, doanti, doshiny});
    sidebar->addRow(nullptr, mapviz);
    sidebar->addFooter(help);

    connect(dossao, &QPushButton::released, this, &ImageViewer::toggleSsao);
    connect(doanti, &QPushButton::released, this, &ImageViewer::toggleFsaa);
    connect(doshiny, &QPushButton::released, this, &ImageViewer::toggleShiny);
    connect(dopart, &QPushButton::released, this, &ImageViewer::toggleParticles);
    connect(dogrid, &QPushButton::released, this, &ImageViewer::toggleGrid);
    connect(dosurf, &QPushButton::released, this, &ImageViewer::toggleSurf);
    connect(dobox, &QPushButton::released, this, &ImageViewer::toggleBox);
    connect(doaxes, &QPushButton::released, this, &ImageViewer::toggleAxes);
    connect(zoomin, &QPushButton::released, this, &ImageViewer::doZoomIn);
    connect(zoomout, &QPushButton::released, this, &ImageViewer::doZoomOut);
    connect(rotleft, &QPushButton::released, this, &ImageViewer::doRotLeft);
    connect(rotright, &QPushButton::released, this, &ImageViewer::doRotRight);
    connect(rotup, &QPushButton::released, this, &ImageViewer::doRotUp);
    connect(rotdown, &QPushButton::released, this, &ImageViewer::doRotDown);
    connect(recenter, &QPushButton::released, this, &ImageViewer::doRecenter);
    connect(reset, &QPushButton::released, this, &ImageViewer::resetView);
    connect(fitwin, &QPushButton::released, this, &ImageViewer::resetWindowSize);
    for (auto *button : {partviz, gridviz, planeviz, surfviz, boxviz, camviz, qualviz, mapviz})
        connect(button, &QPushButton::released, this, &ImageViewer::openSettings);
    connect(help, &QPushButton::released, this, &ImageViewer::getHelp);
    connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &ImageViewer::changeMixture);

    // Both rows of chrome are wrapped in scroll areas so that a panel too small
    // to show them scrolls instead of squashing them. Without this the toolbar
    // buttons and the settings column are compressed to a few pixels each --
    // present, but unreadable and barely clickable, which is worse than being
    // one scroll away.
    buttonLayout->addStretch(1);
    auto *buttonBar = new QWidget;
    buttonBar->setLayout(buttonLayout);
    auto *buttonScroll = new QScrollArea;
    buttonScroll->setWidget(buttonBar);
    buttonScroll->setWidgetResizable(true);
    buttonScroll->setFrameShape(QFrame::NoFrame);
    buttonScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    buttonScroll->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    buttonScroll->setFixedHeight(buttonBar->sizeHint().height());

    auto *settingsScroll = new QScrollArea;
    settingsScroll->setObjectName("settingsscroll");
    settingsScroll->setWidget(sidebar);
    settingsScroll->setWidgetResizable(true);
    settingsScroll->setFrameShape(QFrame::NoFrame);
    settingsScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    settingsScroll->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);

    // The column is as wide as it needs to be while it is expanded and as wide
    // as the handle once it is not, so the width has to follow the sidebar
    // rather than be measured once at construction time.
    auto fitSidebar = [settingsScroll, this]() {
        settingsScroll->setFixedWidth(sidebar->sizeHint().width() +
                                      settingsScroll->verticalScrollBar()->sizeHint().width());
    };
    fitSidebar();
    connect(sidebar, &ViewerSidebar::collapsedChanged, this, [fitSidebar](bool) { fitSidebar(); });

    topLayout->addWidget(buttonScroll);
    mainLayout->addLayout(topLayout);
    imageLayout->addWidget(display, 10);
    imageLayout->addWidget(settingsScroll, 0);
    imageLayout->setSpacing(LAYOUT_SPACING);
    mainLayout->addLayout(imageLayout);
    mainLayout->setSpacing(LAYOUT_SPACING);
    setWindowIcon(QIcon(Cfg::MAIN_ICON));
    emit titleChanged(QFileInfo(fileName).fileName());
    createActions();

    // surfaces can only be shown when surfs are defined
    dosurf->setEnabled(sparta->extractSetting("surf_exist") == 1);

    resetView();
    // layout has not yet been established, so we need to fix up the checkable
    // pushbutton properties directly since lookup in syncButtons() failed
    dossao->setChecked(params.ssao);
    doanti->setChecked(params.fsaa);
    doshiny->setChecked(params.shiny > SHINY_CUT);
    dopart->setChecked(params.particle);
    dogrid->setChecked(params.grid);
    dosurf->setChecked(params.surf);
    dobox->setChecked(params.box);
    doaxes->setChecked(params.axes);

    display->setVisible(true);
    updateActions();
    setLayout(mainLayout);
    // Deliberately NOT SetMinAndMaxSize. That made the widget's minimum track
    // the layout's, which suited a dialog that sized itself to its contents but
    // is wrong for a panel: the settings column is a stack of fourteen widgets,
    // so the minimum came out around 500px tall, the dock could not shrink the
    // viewer that far, and the surplus was clipped off the bottom of the
    // window. The render then fitted itself to a scroll area that was partly
    // off screen, so the bottom of every snapshot was simply not visible.
    mainLayout->setSizeConstraint(QLayout::SetDefaultConstraint);
    adjustWindowSize();
    menuBar->setFocus();

    // make Alt-H, Alt-W, and Alt-X hotkeys work for comboboxes and spinboxes
    xval->installEventFilter(this);
    yval->installEventFilter(this);
    combo->installEventFilter(this);
    for (auto &obj : combo->children())
        obj->installEventFilter(this);
    installEventFilter(this);

    applyWindowFlags(this);
}

void ImageViewer::readImageSettings()
{
    QSettings settings;
    settings.beginGroup(Keys::GROUP_SNAPSHOT);
    params        = DumpImageSettings(); // start from the SPARTA defaults
    params.xsize  = settings.value(Keys::XSIZE, "600").toInt();
    params.ysize  = settings.value(Keys::YSIZE, "600").toInt();
    params.zoom   = settings.value(Keys::ZOOM, 1.0).toDouble();
    params.theta  = settings.value(Keys::HROT, 60).toInt();
    params.phi    = settings.value(Keys::VROT, 30).toInt();
    params.shiny  = settings.value(Keys::SHINYSTYLE, true).toBool() ? SHINY_ON : SHINY_OFF;
    params.ssao   = settings.value(Keys::SSAO, false).toBool();
    params.fsaa   = settings.value(Keys::ANTIALIAS, false).toBool();
    params.box    = settings.value(Keys::BOX, true).toBool();
    params.boxdiam    = settings.value(Keys::BOXDIAM, 0.02).toDouble();
    params.boxcolor   = settings.value(Keys::BOXCOLOR, "yellow").toString();
    params.axes       = settings.value(Keys::AXES, false).toBool();
    params.axeslen    = settings.value(Keys::AXESLEN, 0.5).toDouble();
    params.axesdiam   = settings.value(Keys::AXESDIAM, 0.02).toDouble();
    params.backcolor  = settings.value(Keys::BACKCOLOR, "black").toString();
    params.backcolor2 = settings.value(Keys::BACKCOLOR2, "white").toString();
    params.gradient   = settings.value(Keys::USEGRADIENT, false).toBool();
    params.color      = settings.value(Keys::COLOR, "type").toString();
    params.diameter   = settings.value(Keys::DIAMETER, "type").toString();
    // SPARTA-specific settings without a preferences entry
    params.subboxdiam = settings.value("subboxdiam", 0.02).toDouble();
    params.glinediam  = settings.value("glinediam", 0.005).toDouble();
    params.slinediam  = settings.value("slinediam", 0.005).toDouble();
    // fraction of the shortest box length, like the box/gline diameters
    params.surfdiam   = settings.value("surfdiam", 0.01).toDouble();
    params.ssaoint    = settings.value("ssaoint", 0.6).toDouble();
    settings.endGroup();

    params.ssaoseed  = Cfg::SSAO_SEED;
    params.mixture   = "all";
    params.dimension = sparta ? sparta->extractSetting("dimension") : 3;

    // default particle diameter: SPARTA's per-type default of 1.0 length
    // units is usually far too large, so scale it to the simulation box
    // unless the user chose an explicit diameter
    if ((params.diameter == "type") && sparta) {
        auto *boxlo = static_cast<double *>(sparta->extractGlobal("boxlo"));
        auto *boxhi = static_cast<double *>(sparta->extractGlobal("boxhi"));
        if (boxlo && boxhi) {
            double minext = boxhi[0] - boxlo[0];
            minext        = qMin(minext, boxhi[1] - boxlo[1]);
            if (params.dimension == 3) minext = qMin(minext, boxhi[2] - boxlo[2]);
            if (minext > 0.0) {
                params.numericdiam = true;
                params.pdiamvalue  = 0.01 * minext;
            }
        }
    }
    // start with the grid volume rendering enabled when there are no
    // particles yet, so the first rendering is not just an empty box
    if (sparta && (sparta->extractSetting("nplocal") < 1)) {
        params.grid      = true;
        params.gridcolor = "proc";
    }
    // show surfaces when they exist
    if (sparta && (sparta->extractSetting("surf_exist") == 1)) {
        params.surf      = true;
        params.surfcolor = "one";
    }

    if (color_list.isEmpty()) resetColors(); // create list of default colors
}

void ImageViewer::resetView()
{
    readImageSettings();

    auto *field = findChild<QSpinBox *>("xsize");
    if (field) field->setValue(params.xsize);
    field = findChild<QSpinBox *>("ysize");
    if (field) field->setValue(params.ysize);

    auto *cb = findChild<QComboBox *>("mixture");
    if (cb && (cb->currentText() != "all")) {
        cb->setCurrentText("all"); // triggers changeMixture() -> createImage()
        syncButtons();
    } else {
        syncButtons();
        createImage();
    }
}

void ImageViewer::syncButtons()
{
    struct {
        const char *name;
        bool checked;
    } state[] = {{"ssao", params.ssao},         {"antialias", params.fsaa},
                 {"shiny", params.shiny > SHINY_CUT}, {"particles", params.particle},
                 {"grid", params.grid},         {"surf", params.surf},
                 {"box", params.box},           {"axes", params.axes}};
    for (const auto &s : state) {
        auto *button = findChild<QPushButton *>(s.name);
        if (button) button->setChecked(s.checked);
    }
}

void ImageViewer::editSize()
{
    // this arrives from editingFinished(), which Qt also emits when the field
    // merely loses focus -- including the focus change that hiding the widget
    // during destruction causes
    if (shutdown) return;
    auto *field = qobject_cast<QSpinBox *>(sender());
    if (!field) return;
    if (field->objectName() == "xsize") {
        params.xsize = field->value();
    } else if (field->objectName() == "ysize") {
        params.ysize = field->value();
    }
    createImage();
}

void ImageViewer::toggleSsao()
{
    auto *button = qobject_cast<QPushButton *>(sender());
    if (!button) return;
    params.ssao = !params.ssao;
    button->setChecked(params.ssao);
    createImage();
}

void ImageViewer::toggleFsaa()
{
    auto *button = qobject_cast<QPushButton *>(sender());
    if (!button) return;
    params.fsaa = !params.fsaa;
    button->setChecked(params.fsaa);
    createImage();
}

void ImageViewer::toggleShiny()
{
    auto *button = qobject_cast<QPushButton *>(sender());
    if (!button) return;
    if (params.shiny > SHINY_CUT)
        params.shiny = SHINY_OFF;
    else
        params.shiny = SHINY_ON;
    button->setChecked(params.shiny > SHINY_CUT);
    createImage();
}

void ImageViewer::toggleParticles()
{
    auto *button = qobject_cast<QPushButton *>(sender());
    if (!button) return;
    params.particle = !params.particle;
    button->setChecked(params.particle);
    createImage();
}

void ImageViewer::toggleGrid()
{
    auto *button = qobject_cast<QPushButton *>(sender());
    if (!button) return;
    params.grid = !params.grid;
    if (params.grid) {
        // grid volume rendering and grid cut planes are mutually exclusive
        params.gridx = params.gridy = params.gridz = false;
        if (params.gridcolor.isEmpty()) params.gridcolor = "proc";
    }
    button->setChecked(params.grid);
    createImage();
}

void ImageViewer::toggleSurf()
{
    auto *button = qobject_cast<QPushButton *>(sender());
    if (!button) return;
    params.surf = !params.surf;
    button->setChecked(params.surf);
    createImage();
}

void ImageViewer::toggleBox()
{
    auto *button = qobject_cast<QPushButton *>(sender());
    if (!button) return;
    params.box = !params.box;
    button->setChecked(params.box);
    createImage();
}

void ImageViewer::toggleAxes()
{
    auto *button = qobject_cast<QPushButton *>(sender());
    if (!button) return;
    params.axes = !params.axes;
    button->setChecked(params.axes);
    createImage();
}

void ImageViewer::doZoomIn()
{
    params.zoom = std::min(params.zoom * 1.1, ZOOM_MAX);
    createImage();
}

void ImageViewer::doZoomOut()
{
    params.zoom = std::max(params.zoom / 1.1, ZOOM_MIN);
    createImage();
}

// phi is the azimuthal angle around the z-axis (-180 to 180, wrapping)

void ImageViewer::doRotLeft()
{
    params.phi -= 10;
    if (params.phi < -180.0) params.phi += 360.0;
    createImage();
}

void ImageViewer::doRotRight()
{
    params.phi += 10;
    if (params.phi > 180.0) params.phi -= 360.0;
    createImage();
}

// theta is the viewing angle from the +z axis and must stay within 0-180

void ImageViewer::doRotUp()
{
    params.theta = std::max(0.0, params.theta - 10.0);
    createImage();
}

void ImageViewer::doRotDown()
{
    params.theta = std::min(180.0, params.theta + 10.0);
    createImage();
}

void ImageViewer::doRecenter()
{
    params.cx = params.cy = params.cz = 0.5;
    params.cxvar.clear();
    params.cyvar.clear();
    params.czvar.clear();
    params.centerdynamic = false;
    createImage();
}

void ImageViewer::cmdToClipboard()
{
    // compose a reusable dump image command with dump_modify lines for the
    // current settings. the file type prefers png/jpg when compiled in.
    QString imagefile = "myimage-*.ppm";
    if (sparta->configHasPngSupport())
        imagefile = "myimage-*.png";
    else if (sparta->configHasJpegSupport())
        imagefile = "myimage-*.jpg";

    const QString out = buildDumpSnippet(params, false, imagefile, 100);

#if QT_CONFIG(clipboard)
    auto *clip = QGuiApplication::clipboard();
    if (clip) {
        clip->setText(out, QClipboard::Clipboard);
        if (clip->supportsSelection()) clip->setText(out, QClipboard::Selection);
    } else
        fprintf(stderr, "# customized dump image command:\n%s", qPrintable(out));
#else
    fprintf(stderr, "# customized dump image command:\n%s", qPrintable(out));
#endif
}

void ImageViewer::movieToClipboard()
{
    // same settings as the image dump, but writing a movie via ffmpeg
    const QString out = buildDumpSnippet(params, true, "mymovie.mp4", 100);

#if QT_CONFIG(clipboard)
    auto *clip = QGuiApplication::clipboard();
    if (clip) {
        clip->setText(out, QClipboard::Clipboard);
        if (clip->supportsSelection()) clip->setText(out, QClipboard::Selection);
    } else
        fprintf(stderr, "# customized dump movie command:\n%s", qPrintable(out));
#else
    fprintf(stderr, "# customized dump movie command:\n%s", qPrintable(out));
#endif
}

void ImageViewer::resetColors()
{
    color_list.clear();
    const int nspecies = sparta ? sparta->extractSetting("nspecies") : 0;
    const int ndef     = defspeciescolors.size();
    for (int i = 0; i < std::max(nspecies, ndef); ++i)
        color_list.append(defspeciescolors[i % ndef]);
    params.pdiams.clear();
    params.pcolors.clear();
    params.customcolors.clear();
    params.amblight  = 0.0;
    params.keylight  = 0.9;
    params.filllight = 0.45;
    params.backlight = 0.9;
}

void ImageViewer::loadColors()
{
    auto root = loadJsonColors(this);
    if (root.isEmpty()) return;

    auto arr = root["colors"].toArray();
    if (arr.isEmpty()) return;

    color_list.clear();
    for (const auto &item : arr) {
        auto obj  = item.toObject();
        QString n = obj.value("name").toString();
        double r  = std::clamp(obj.value("red").toDouble(1.0), 0.0, 1.0);
        double g  = std::clamp(obj.value("green").toDouble(1.0), 0.0, 1.0);
        double b  = std::clamp(obj.value("blue").toDouble(1.0), 0.0, 1.0);
        color_list.append({n, QColor::fromRgbF(r, g, b)});
    }
    if (color_list.isEmpty()) resetColors();

    auto lights = root.value("lights").toObject();
    if (!lights.isEmpty()) {
        params.amblight  = lights.value("ambient").toDouble();
        params.keylight  = lights.value("key").toDouble();
        params.filllight = lights.value("fill").toDouble();
        params.backlight = lights.value("back").toDouble();
    }
    createImage();
}

void ImageViewer::saveColors()
{
    QJsonArray colors;
    for (const auto &c : color_list) {
        QJsonObject obj;
        obj["name"]  = c.first;
        obj["red"]   = c.second.redF();
        obj["green"] = c.second.greenF();
        obj["blue"]  = c.second.blueF();
        colors.append(obj);
    }
    QJsonObject lights;
    lights["ambient"] = params.amblight;
    lights["key"]     = params.keylight;
    lights["fill"]    = params.filllight;
    lights["back"]    = params.backlight;

    saveJsonColors(this, colors, lights);
}

void ImageViewer::changeMixture(int)
{
    auto *box      = findChild<QComboBox *>("mixture");
    params.mixture = box ? box->currentText() : "all";
    createImage();
}

void ImageViewer::fitRenderToPanel()
{
    // "Fit window" resizes the window to the picture.  This is the other
    // direction, and the one a docked panel needs: the render is a fixed
    // number of pixels, so a panel with room to spare -- after collapsing the
    // sidebar, or widening the dock -- shows the same small picture with more
    // grey around it.  Rendering at the panel's size is what actually spends
    // the space, and it is a deliberate act rather than something that fires
    // on every resize, because each one costs a full SPARTA render.
    if (sparta && sparta->isRunning()) return;
    auto *area = display->scrollArea();
    if (!area || !area->viewport()) return;

    const QSize vp = area->viewport()->size();
    if ((vp.width() < 1) || (vp.height() < 1)) return;

    auto *xfield = findChild<QSpinBox *>("xsize");
    auto *yfield = findChild<QSpinBox *>("ysize");
    // the same bounds the size fields carry, so the two can never disagree
    const int wide = std::clamp(vp.width(), MIN_RENDER_SIZE, MAX_RENDER_SIZE);
    const int high = std::clamp(vp.height(), MIN_RENDER_SIZE, MAX_RENDER_SIZE);
    params.xsize = wide;
    params.ysize = high;

    // set the fields without their editingFinished() re-entering createImage()
    if (xfield) {
        QSignalBlocker block(xfield);
        xfield->setValue(wide);
    }
    if (yfield) {
        QSignalBlocker block(yfield);
        yfield->setValue(high);
    }

    createImage();
}

void ImageViewer::openSettings()
{
    auto *src = sender();
    if (!src) return;
    settingsDialog(src->property("tab").toInt());
}

// intercept events
bool ImageViewer::eventFilter(QObject *watched, QEvent *event)
{
    // interactive view manipulation on the rendered image: left-drag rotates
    // (Shift+drag pans), wheel zooms. Each gesture re-renders through the same
    // createImage() path the toolbar buttons use, guarded against re-entrancy
    // while a run is active.
    // the panel can be resized at any time, and the fit above is computed
    // against the viewport, so it has to be recomputed when that changes
    if (watched == display->label() && !shutdown) {
        const bool is2d = sparta && (sparta->extractSetting("dimension") == 2);
        switch (event->type()) {
        case QEvent::MouseButtonPress: {
            auto *me = static_cast<QMouseEvent *>(event);
            if (me->button() == Qt::LeftButton) {
                dragLast = me->pos();
                dragging = true;
                display->label()->setCursor(Qt::ClosedHandCursor);
                return true;
            }
            break;
        }
        case QEvent::MouseMove: {
            if (!dragging) break;
            auto *me = static_cast<QMouseEvent *>(event);
            const QPoint d = me->pos() - dragLast;
            dragLast = me->pos();
            if (sparta && sparta->isRunning()) return true;
            if (me->modifiers().testFlag(Qt::ShiftModifier)) {
                // pan by moving the view center (box fractions)
                params.cx = std::clamp(params.cx - d.x() * 0.0025, 0.0, 1.0);
                params.cy = std::clamp(params.cy + d.y() * 0.0025, 0.0, 1.0);
                params.centerdynamic = false;
                params.cxvar.clear();
                params.cyvar.clear();
            } else {
                params.phi = wrapAzimuth(params.phi + d.x() * 0.5);
                if (!is2d) params.theta = clampPolar(params.theta + d.y() * 0.5);
            }
            createImage();
            return true;
        }
        case QEvent::MouseButtonRelease: {
            if (dragging) {
                dragging = false;
                display->label()->setCursor(Qt::OpenHandCursor);
                return true;
            }
            break;
        }
        case QEvent::Wheel: {
            auto *we = static_cast<QWheelEvent *>(event);
            if (sparta && sparta->isRunning()) return true;
            const double steps = we->angleDelta().y() / 120.0;
            if (steps != 0.0) {
                params.zoom = clampZoom(params.zoom * std::pow(1.1, steps));
                createImage();
            }
            return true;
        }
        default:
            break;
        }
    }

    // now that this window is a docked panel sharing the main window's
    // shortcut context, its own Ctrl+W/Q would otherwise be ambiguous with the
    // identically bound main-window menu shortcuts.  Ctrl+S and Ctrl+C are not
    // claimed here any more: the panel owns them for every tab, so claiming
    // them again from inside one tab would decide by focus which of two
    // identical actions ran.
    if (!shutdown &&
        dispatchCtrlShortcut(event, {{'W', [this]() { emit closeRequested(); }},
                                     {'Q', [this]() { quit(); }}}))
        return true;
    if (event->type() == QEvent::KeyPress) {
        // don't handle any more key press events after entering destructor
        if (shutdown) return false;
        QKeyEvent *kev = static_cast<QKeyEvent *>(event);
        if ((kev->key() == Qt::Key_X) && (kev->modifiers() == Qt::AltModifier)) {
            auto *box = findChild<QComboBox *>("mixture");
            if (box) {
                box->setFocus();
                box->showPopup();
                return true;
            } else
                return false;
        } else if ((kev->key() == Qt::Key_W) && (kev->modifiers() == Qt::AltModifier)) {
            auto *combo = findChild<QComboBox *>("mixture");
            if (combo) combo->hidePopup();

            auto *box = findChild<QSpinBox *>("xsize");
            if (box) {
                box->setFocus();
                box->selectAll();
                return true;
            } else
                return false;
        } else if ((kev->key() == Qt::Key_H) && (kev->modifiers() == Qt::AltModifier)) {
            auto *combo = findChild<QComboBox *>("mixture");
            if (combo) combo->hidePopup();

            auto *box = findChild<QSpinBox *>("ysize");
            if (box) {
                box->setFocus();
                box->selectAll();
                return true;
            } else
                return false;
        } else if (kev->modifiers() == Qt::AltModifier) {
            auto *combo = findChild<QComboBox *>("mixture");
            if (combo) combo->hidePopup();
            setFocus();
        }
    }
    return ViewerSource::eventFilter(watched, event);
}

// Refresh the SPARTA-derived members of the settings struct right before a
// render and translate the per-species color/diameter tables into the pcolor,
// pdiam, and custom color rows of the command builder.
void ImageViewer::gatherSettings()
{
    params.dimension = sparta->extractSetting("dimension");
    params.ssaoseed  = Cfg::SSAO_SEED;

    // surfaces may have been removed by a re-run of the input
    if (sparta->extractSetting("surf_exist") != 1) params.surf = false;

    // extend the per-species color table when new species appeared
    const int nspecies = sparta->extractSetting("nspecies");
    const int ndef     = defspeciescolors.size();
    for (int i = color_list.size(); i < nspecies; ++i)
        color_list.append(defspeciescolors[i % ndef]);

    // translate the per-species colors into pcolor rows and custom color
    // definitions; entries matching the SPARTA default assignment are omitted
    params.pcolors.clear();
    params.customcolors.clear();
    if (params.color == QLatin1String("type")) {
        for (int i = 1; i <= nspecies; ++i) {
            const auto &def = defspeciescolors[(i - 1) % ndef];
            const auto &cur = color_list[i - 1];
            if ((cur.first == def.first) && (cur.second == def.second)) continue;
            // SPARTA's color parser only understands its built-in SVG color
            // names and user-defined names, never a "#rrggbb" literal (as the
            // color picker produces). Reference such colors through a
            // SPARTA-safe synthetic name defined via "color <name> R G B".
            QString token = cur.first;
            if (cur.second.isValid() && token.startsWith('#'))
                token = QStringLiteral("guisp%1").arg(i);
            // define/redefine the color when the token is not a stock SPARTA name
            if (cur.second.isValid() && (token != cur.first || cur.second != QColor(cur.first))) {
                params.customcolors.append(
                    {token, QStringLiteral("%1 %2 %3")
                                .arg(cur.second.redF(), 0, 'f', 3)
                                .arg(cur.second.greenF(), 0, 'f', 3)
                                .arg(cur.second.blueF(), 0, 'f', 3)});
            }
            params.pcolors.append({QString::number(i), token});
        }
    }
}

// This function creates a visualization of the current system using the
// "dump image" command and reads and displays the rendered image.
// SPARTA has no write_dump command, so an explicit dump is created, output
// is forced with a "run 0", and the dump is removed again.
void ImageViewer::createImage()
{
    // no point in trying to update the image when triggered after the destructor started
    if (shutdown) return;

    // SPARTA is not re-entrant. The viewer is a separate, non-modal window, so
    // the user can change a setting here while a run started from the main
    // window is still in progress on the runner thread. Issuing dump/run/undump
    // commands on the same instance from this (GUI) thread would race with the
    // runner. Skip the re-render until the run has finished.
    if (sparta->isRunning()) return;

    // rendering requires a defined simulation box and grid
    if (!sparta->extractSetting("box_exist") || !sparta->extractSetting("grid_exist")) {
        warning(this, "Image Viewer",
                "Cannot render an image: the SPARTA dump image command requires "
                "a defined simulation box and grid (create_box / create_grid).");
        return;
    }

    auto *renderstatus = findChild<QLabel *>("renderstatus");
    if (renderstatus) renderstatus->setPixmap(renderstatus->property("activePix").value<QPixmap>());
    repaint();
#if defined(Q_OS_MACOS)
    // Workaround for macOS: while the main thread is busy with the synchronous
    // render below, the backing store is not flushed to the screen, so repaint()
    // alone leaves the render-status icon stuck on its idle (gray) pixmap. Pump
    // the event loop once -- excluding user input so this slot is not re-entered
    // -- to push the active (colored) pixmap to the display.
    if (renderstatus) renderstatus->repaint();
    QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
#endif

    // attempt to clean up if a previous render left our dump defined
    if (sparta->hasId("dump", renderdumpid.toLocal8Bit().constData())) {
        StdoutSilencer guard;
        sparta->command("undump " + renderdumpid);
        (void)sparta->lastErrorMessage(); // clear a possible pending error
    }

    // refresh SPARTA-derived state and assemble the dump/dump_modify commands
    gatherSettings();
    const QString dumpargs     = buildDumpImageCommand(params);
    const QStringList modcmds  = buildDumpModifyCommands(params, renderdumpid);

    // render into a PPM file in the temp directory: PPM support is always
    // compiled in, and QImageReader handles it. dump image needs a '*' in the
    // file name (it is replaced by the timestep); "first yes" forces the frame
    // to be written by the "run 0".
    QDir dumpdir(QDir::tempPath());
    const QString starfile = dumpdir.absoluteFilePath(filename + ".*.ppm");
    {
        StdoutSilencer guard;
        sparta->command("dump " + renderdumpid + " image " + params.mixture + " 1 '" + starfile +
                        "' " + dumpargs);
        sparta->command("dump_modify " + renderdumpid + " first yes pad 0");
        for (const auto &cmd : modcmds)
            sparta->command(cmd);
        sparta->command("run 0 pre yes post no");
        sparta->command("undump " + renderdumpid);
    }
    const QString errmsg = sparta->lastErrorMessage();

    const auto step = static_cast<long long>(sparta->getThermo("step"));
    const QString imagepath =
        dumpdir.absoluteFilePath(QString("%1.%2.ppm").arg(filename).arg(step));

    // reset the render-status icon on every exit path, otherwise a failed
    // render leaves the icon stuck on "active"
    auto restoreRenderState = [&]() {
        if (renderstatus)
            renderstatus->setPixmap(renderstatus->property("idlePix").value<QPixmap>());
    };

    // remove the per-step frame file(s) this render produced. dump image may
    // have flushed a frame even when the run reported an error, so this must run
    // on every exit path below, not only the success path.
    auto cleanupFrames = [&]() {
        for (const auto &f : dumpdir.entryList({filename + ".*.ppm"}, QDir::Files))
            QFile::remove(dumpdir.absoluteFilePath(f));
    };

    // display error message
    if (!errmsg.isEmpty()) {
        restoreRenderState();
        cleanupFrames();
        // ignore "Invalid SPARTA handle", but report other errors
        if (!errmsg.contains("Invalid SPARTA handle"))
            warning(this, "Image Viewer File Creation Error",
                    "SPARTA failed to create the image:", QString("<code>%1</code>").arg(errmsg));
        return;
    }

    QImageReader reader(imagepath);
    reader.setAutoTransform(true);
    const QImage newImage = reader.read();
    cleanupFrames();

    // read of new image failed. nothing left to do.
    if (newImage.isNull()) {
        restoreRenderState();
        return;
    }

    // show image
    display->setImage(newImage);
    // The fit above measures the viewport as it is right now, which during the
    // first render into a freshly-opened dock is not yet its final size. Fit
    // again once the layout has settled.
    QTimer::singleShot(0, this, [this]() { display->refresh(); });
    adjustWindowSize();
    restoreRenderState();
    repaint();
    updateActions();
}

void ImageViewer::saveAs()
{
    QImage shot = display->displayedImage();
    exportImage(this, &shot, "ImageViewer");
}

void ImageViewer::copy()
{
    copyImageToClipboard(display->displayedImage());
}

ImageViewer::~ImageViewer()
{
    // Before any member is gone: ~QWidget hides the window, the focus moves off
    // whichever size field held it, and the editingFinished() that follows
    // re-enters editSize() and createImage() on an object that is halfway
    // destroyed.  Focusing a size field and then closing the panel was enough
    // to crash the application.
    shutdown = true;

    // The flag alone is not enough, because the call itself is already invalid
    // by then: with ~ImageViewer finished and ~QWidget running, `this` is no
    // longer an ImageViewer, and a connection made to a member function pointer
    // is dispatched on the wrong type before the guard inside it can run.  A
    // debug build of Qt asserts on exactly that ("Called object is not of the
    // correct type (class destructor may have already run)"); a release build
    // dereferences whatever is there.
    //
    // So the connections into this object come down here, while it still is
    // one.  Per child rather than in one call: QObject::disconnect() takes the
    // sender as its first argument and that argument may never be null, so the
    // "any sender" form is not available and would silently do nothing.
    for (QObject *child : findChildren<QObject *>())
        QObject::disconnect(child, nullptr, this, nullptr);
}

void ImageViewer::quit()
{
    shutdown = true;
    if (spartagui) spartagui->quit();
}

void ImageViewer::getHelp()
{
    auto *src = sender();
    if (src) {
        QString page = src->objectName();

        if (page == "visualization.html") {
            // sparta-gui docs
            QDesktopServices::openUrl(
                QUrl(QString("https://sparta.github.io/sparta-gui/%1").arg(page)));
        } else {
            // SPARTA manual page
            QDesktopServices::openUrl(QUrl(QString("%1/doc/%2").arg(Cfg::DOCS_URL, page)));
        }
    }
}

void ImageViewer::createActions()
{
    QMenu *fileMenu = menuBar->addMenu("&File");

    // Saving the render and copying it are the viewer panel's shared controls
    // now, beside the tabs, with the same names and keys whichever tab is in
    // front.  They used to be here, spelled differently from the slide show's
    // pair of buttons doing the same thing one tab away.  What stays is what
    // only this viewer can do: hand over the commands that reproduce the
    // picture, and its colours.
    cmdAct = addMenuAction(fileMenu, "Copy &dump image command", ":/icons/file-clipboard.svg", this,
                           &ImageViewer::cmdToClipboard);
    cmdAct->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_D));
    cmdAct->setShortcutContext(Qt::WidgetWithChildrenShortcut);
    movieAct = addMenuAction(fileMenu, "Copy dump &movie command", ":/icons/export-movie.svg", this,
                             &ImageViewer::movieToClipboard);
    fileMenu->addSeparator();
    addMenuAction(fileMenu, "&Load Colors from JSON...", ":/icons/document-open.svg", this,
                  &ImageViewer::loadColors);
    addMenuAction(fileMenu, "S&ave Colors to JSON...", ":/icons/document-save.svg", this,
                  &ImageViewer::saveColors);
    addMenuAction(fileMenu, "&Reset Colors", ":/icons/system-restart.svg", this, [this]() {
        resetColors();
        createImage();
    });
    fileMenu->addSeparator();
    auto *closeAct =
        addMenuAction(fileMenu, "&Close Panel", ":/icons/window-close.svg", this,
                      &ImageViewer::closeRequested);
    closeAct->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_W));
    closeAct->setShortcutContext(Qt::WidgetWithChildrenShortcut);
    auto *quitAct =
        addMenuAction(fileMenu, "&Quit", ":/icons/application-exit.svg", this, &ImageViewer::quit);
    quitAct->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Q));
    quitAct->setShortcutContext(Qt::WidgetWithChildrenShortcut);

    // The sidebar can also be collapsed from its own header, but a menu entry
    // is what a user looks for when a panel has gone missing, and it is the
    // only way back if the handle is scrolled out of view in a very short panel.
    QMenu *viewMenu = menuBar->addMenu("&View");
    sidebarAct      = viewMenu->addAction("Settings &Sidebar");
    sidebarAct->setCheckable(true);
    sidebarAct->setChecked(!sidebar->isCollapsed());
    sidebarAct->setStatusTip("Show or hide the column of per-subject display settings");
    sidebarAct->setShortcut(QKeySequence(Qt::Key_F9));
    sidebarAct->setShortcutContext(Qt::WidgetWithChildrenShortcut);
    connect(sidebarAct, &QAction::toggled, sidebar,
            [this](bool on) { sidebar->setCollapsed(!on); });
    connect(sidebar, &ViewerSidebar::collapsedChanged, sidebarAct,
            [this](bool on) { sidebarAct->setChecked(!on); });

    viewMenu->addSeparator();
    auto *fitRenderAct = addMenuAction(viewMenu, "Fit &Render to Panel", ":/icons/gtk-zoom-fit.svg",
                                       this, &ImageViewer::fitRenderToPanel);
    fitRenderAct->setStatusTip("Render at the size of the space the picture is shown in");
    fitRenderAct->setShortcut(QKeySequence(Qt::Key_F8));
    fitRenderAct->setShortcutContext(Qt::WidgetWithChildrenShortcut);

    auto *fitWindowAct = addMenuAction(viewMenu, "Fit &Window to Render", ":/icons/fit-window.svg",
                                       this, &ImageViewer::resetWindowSize);
    fitWindowAct->setStatusTip("Resize the window so the picture is shown at its full size");
}

QIcon ImageViewer::sourceIcon() const
{
    return QIcon(":/icons/image-viewer.svg");
}

void ImageViewer::updateActions()
{
    // Save and Copy live on the panel now and follow currentImage(); telling
    // it the render arrived is this viewer's whole part in that.
    emit contentChanged();
}

bool ImageViewer::hasContent() const
{
    return !display->isEmpty();
}

QImage ImageViewer::currentImage() const
{
    return display->displayedImage();
}

void ImageViewer::adjustWindowSize()
{
    // the render size is set in the settings panel, so the size to fit is
    // known even before the first image has been rendered
    if ((params.xsize < 1) || (params.ysize < 1)) return;

    // make sure the scroll area is not resized beyond a certain fraction of the screen
    const QSize avail = screen()->availableSize();
    constexpr int EXTRA_WIDTH  = 150;
    constexpr int EXTRA_HEIGHT = 100;
    const QSize budget((avail.width() * 4 / 5) - EXTRA_WIDTH,
                       (avail.height() * 9 / 10) - EXTRA_HEIGHT);
    display->fitHostWindow(this, QSize(params.xsize, params.ysize), budget);
}

void ImageViewer::resetWindowSize()
{
    // discard both a manual window resize and the memoized fit
    display->forgetHostFit();
    adjustWindowSize();
}

void ImageViewer::showEvent(QShowEvent *event)
{
    ViewerSource::showEvent(event);
    // any fit computed while the window was hidden used unpolished style
    // metrics and was not memoized (see fitViewerWindow()); apply the fit
    // again as soon as the shown window has settled
    if (display->needsHostFit()) QTimer::singleShot(0, this, &ImageViewer::adjustWindowSize);
}

// Local Variables:
// c-basic-offset: 4
// End:
