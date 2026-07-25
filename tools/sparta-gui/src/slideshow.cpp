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

#include "slideshow.h"

#include "constants.h"
#include "helpers.h"
#include "spartagui.h"
#include "movieimport.h"
#include "qaddon.h"
#include "rangebandslider.h"
#include "viewerdisplay.h"

#include <QApplication>
#include <QClipboard>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QImage>
#include <QKeyEvent>
#include <QKeySequence>
#include <QLabel>
#include <QLocale>
#include <QMessageBox>
#include <QPalette>
#include <QPixmap>
#include <QProcess>
#include <QPushButton>
#include <QScreen>
#include <QScrollArea>
#include <QShortcut>
#include <QShowEvent>
#include <QSlider>
#include <QSpacerItem>
#include <QSpinBox>
#include <QTemporaryFile>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>

namespace {
constexpr int LAYOUT_SPACING = 6;
constexpr int EXTRA_HEIGHT   = 130;
} // namespace

SlideShow::SlideShow(const QString &fileName, SpartaGui *_spartagui, QWidget *parent) :
    ViewerSource(parent), spartagui(_spartagui),
    display(new ViewerDisplay(ViewerDisplay::Natural)), playtimer(nullptr),
    scrollBar(new RangeBandSlider),
    imageCounter(new QLabel("Image   0 /   0 :")), imageName(new QLabel("(none)")),
    startBox(new QSpinBox), stopBox(new QSpinBox), cacheButton(new QPushButton), current(0),
    maxwidth(0), maxheight(0), timerDelay(100), doLoop(true)
{
    display->setVisible(false);

    scrollBar->setMinimum(0);
    scrollBar->setMaximum(1);
    scrollBar->setOrientation(Qt::Horizontal);
    scrollBar->setToolTip("Select Image to display");
    connect(scrollBar, &QSlider::valueChanged, this, &SlideShow::loadImage);

    imageCounter->setFrameStyle(QFrame::Raised);
    imageCounter->setFrameShape(QFrame::Panel);
    imageCounter->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    imageCounter->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);

    imageName->setFrameStyle(QFrame::Raised);
    imageName->setFrameShape(QFrame::Panel);
    imageName->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    imageName->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    imageName->setTextInteractionFlags(Qt::TextSelectableByMouse);

    auto *stoprun = new QPushButton(QIcon(":/icons/process-stop.svg"), "");
    stoprun->setToolTip("Stop running simulation");
    // shared toolbar button policy (matches the image viewer); the spin boxes
    // and image-name/counter labels below are matched to the button height
    const QSize buttonhint = toolButtonSize(stoprun);
    connect(stoprun, &QPushButton::released, this, &SlideShow::stopRun);

    imageCounter->setMinimumHeight(buttonhint.height());
    imageCounter->setMaximumHeight(buttonhint.height());
    imageName->setMinimumHeight(buttonhint.height());
    imageName->setMaximumHeight(buttonhint.height());

    // WidgetWithChildrenShortcut (rather than the default WindowShortcut) so
    // these only compete with the identically-bound main-window menu
    // shortcuts while focus is actually inside this docked panel, instead of
    // unconditionally while its window (now shared with the main window) is
    // active; the eventFilter() override below resolves that in-focus case
    auto *shortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_W), this);
    shortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(shortcut, &QShortcut::activated, this, &SlideShow::closeRequested);
    shortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_Slash), this);
    shortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(shortcut, &QShortcut::activated, this, &SlideShow::stopRun);
    shortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_Q), this);
    shortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(shortcut, &QShortcut::activated, this, &SlideShow::quit);
    shortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_C), this);
    shortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(shortcut, &QShortcut::activated, this, &SlideShow::copy);
    shortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_E), this);
    shortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(shortcut, &QShortcut::activated, this, &SlideShow::movie);
    shortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_S), this);
    shortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(shortcut, &QShortcut::activated, this, &SlideShow::saveCurrentImage);

    installEventFilter(this);

    auto *mainLayout  = new QVBoxLayout;
    auto *toolsLayout = new QHBoxLayout;
    auto *botLayout   = new QHBoxLayout;
    auto *navLayout   = new QHBoxLayout;

    // workaround for incorrect highlight bug on macOS
    auto *dummy = new QPushButton(QIcon(), "");
    dummy->hide();
    dummy->setMinimumSize(QSize(0, 0));
    dummy->setMaximumSize(QSize(0, 0));

    auto *tomovie = new QPushButton(QIcon(":/icons/export-movie.svg"), "");
    tomovie->setToolTip("Export to movie file");
    // always enabled: movie() explains what to install if no encoder is found,
    // rather than leaving a dead, disabled button

    auto *toimage = new QPushButton(QIcon(":/icons/document-save-as.svg"), "");
    toimage->setToolTip("Export to image file");

    auto *toclip = new QPushButton(QIcon(":/icons/edit-copy.svg"), "");
    toclip->setToolTip("Copy image to clipboard");

    auto *totrash = new QPushButton(QIcon(":/icons/trash.svg"), "");
    totrash->setToolTip("Delete image files in the selected range");

    // The cache indicator is colored while the cache holds images and grayed
    // out when it is empty. Both states are given to the icon for the disabled
    // mode as well, since the icon must stay colored while the button is
    // disabled because only movie frames, which are never discarded, are held.
    const QPixmap cachePix =
        QIcon(":/icons/image-cache.svg")
            .pixmap(QSize(Cfg::TOOLBAR_ICON_SIZE, Cfg::TOOLBAR_ICON_SIZE), devicePixelRatioF());
    const QPixmap grayPix = grayscalePixmap(cachePix);
    cacheFullIcon.addPixmap(cachePix, QIcon::Normal);
    cacheFullIcon.addPixmap(cachePix, QIcon::Disabled);
    cacheEmptyIcon.addPixmap(grayPix, QIcon::Normal);
    cacheEmptyIcon.addPixmap(grayPix, QIcon::Disabled);
    cacheButton->setIcon(cacheEmptyIcon);

    // a standalone slideshow (no live simulation) has no run to stop, and must
    // not offer to delete the user's own image files
    if (!spartagui) {
        stoprun->hide();
        totrash->hide();
    }

    auto *empty = new QLabel("");
    empty->setMinimumSize(buttonhint);
    empty->setMaximumSize(buttonhint);

    auto dsize = QFontMetrics(QApplication::font()).size(Qt::TextSingleLine, "Delay:100");
    // need some extra space on Windows
#if defined(Q_OS_WIN32)
    dsize = dsize * 3 / 2;
#endif
    auto *delay = new QSpinBox;
    delay->setRange(10, 10000);
    delay->setStepType(QAbstractSpinBox::AdaptiveDecimalStepType);
    delay->setValue(timerDelay);
    delay->setObjectName("delay");
    delay->setToolTip("Set delay between images in milliseconds");
    delay->setMinimumWidth(dsize.width());
    delay->setMaximumWidth(dsize.width());
    delay->setMinimumHeight(buttonhint.height());
    delay->setMaximumHeight(buttonhint.height());

    // Start/Stop spinboxes bound the active image range for play, single
    // stepping, movie export, and deletion. Both are 1-based to match the
    // "Image N / M" counter. Stop defaults to the last image and follows the
    // growing maximum (see addImage()) until the user sets it explicitly.
    startBox->setRange(1, 1);
    startBox->setValue(1);
    startBox->setToolTip("First image of the active range for play, step, movie, and delete");
    startBox->setMinimumWidth(dsize.width());
    startBox->setMaximumWidth(dsize.width());
    startBox->setMinimumHeight(buttonhint.height());
    startBox->setMaximumHeight(buttonhint.height());

    stopBox->setRange(1, 1);
    stopBox->setValue(1);
    stopBox->setToolTip("Last image of the active range for play, step, movie, and delete");
    stopBox->setMinimumWidth(dsize.width());
    stopBox->setMaximumWidth(dsize.width());
    stopBox->setMinimumHeight(buttonhint.height());
    stopBox->setMaximumHeight(buttonhint.height());

    // keep the range ordered (Start must never exceed Stop) and mirror the
    // active range onto the navigation slider's colored band
    connect(startBox, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        if (v > stopBox->value()) stopBox->setValue(v);
        updateSliderRange();
    });
    connect(stopBox, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        if (v < startBox->value()) startBox->setValue(v);
        updateSliderRange();
    });

    auto *gofirst = new QPushButton(QIcon(":/icons/go-first.svg"), "");
    gofirst->setToolTip("Go to first image");
    auto *goprev = new QPushButton(QIcon(":/icons/go-previous-2.svg"), "");
    goprev->setToolTip("Go to previous image");
    auto *goplay = new QPushButton(QIcon(":/icons/media-playback-start-2.svg"), "");
    goplay->setToolTip("Play animation");
    goplay->setCheckable(true);
    goplay->setChecked(playtimer);
    goplay->setObjectName("play");
    auto *gonext = new QPushButton(QIcon(":/icons/go-next-2.svg"), "");
    gonext->setToolTip("Go to next image");
    auto *golast = new QPushButton(QIcon(":/icons/go-last.svg"), "");
    golast->setToolTip("Go to last image");
    auto *goloop = new QPushButton(QIcon(":/icons/media-playlist-repeat.svg"), "");
    goloop->setToolTip("Loop animation");
    goloop->setCheckable(true);
    goloop->setChecked(doLoop);

    auto *zoomin = new QPushButton(QIcon(":/icons/gtk-zoom-in.svg"), "");
    zoomin->setToolTip("Displayed image zoom in by 10 percent");
    auto *zoomout = new QPushButton(QIcon(":/icons/gtk-zoom-out.svg"), "");
    zoomout->setToolTip("Displayed image zoom out by 10 percent");
    auto *imgrotcw = new QPushButton(QIcon(":/icons/object-rotate-right.svg"), "");
    imgrotcw->setToolTip("Displayed image rotate 90<sup>o</sup> clockwise");
    auto *imgrotccw = new QPushButton(QIcon(":/icons/object-rotate-left.svg"), "");
    imgrotccw->setToolTip("Displayed image rotate 90<sup>o</sup> counter-clockwise");
    auto *imgfliph = new QPushButton(QIcon(":/icons/object-flip-horizontal.svg"), "");
    imgfliph->setToolTip("Displayed image mirror horizontally");
    auto *imgflipv = new QPushButton(QIcon(":/icons/object-flip-vertical.svg"), "");
    imgflipv->setToolTip("Displayed image mirror vertically");
    auto *normal = new QPushButton(QIcon(":/icons/gtk-zoom-fit.svg"), "");
    normal->setToolTip("Displayed image reset zoom, rotation, and mirroring");
    auto *fitwin = new QPushButton(QIcon(":/icons/fit-window.svg"), "");
    fitwin->setToolTip("Resize window to fit the displayed image size");

    // square toolbar buttons with a snug, uniform icon (shared policy)
    styleToolButtons(buttonhint,
                     {stoprun,  tomovie,   toimage,  toclip,   totrash, cacheButton, gofirst,
                      goprev,   goplay,    gonext,   golast,   goloop,  zoomin,      zoomout,
                      imgrotcw, imgrotccw, imgfliph, imgflipv, normal,  fitwin});

    connect(tomovie, &QPushButton::released, this, &SlideShow::movie);
    connect(toimage, &QPushButton::released, this, &SlideShow::saveCurrentImage);
    connect(toclip, &QPushButton::released, this, &SlideShow::copy);
    connect(totrash, &QPushButton::released, this, &SlideShow::deleteImages);
    connect(cacheButton, &QPushButton::released, this, &SlideShow::purgeCache);
    connect(delay, &QAbstractSpinBox::editingFinished, this, &SlideShow::setDelay);

    connect(gofirst, &QPushButton::released, this, &SlideShow::first);
    connect(goprev, &QPushButton::released, this, &SlideShow::prev);
    connect(goplay, &QPushButton::released, this, &SlideShow::play);
    connect(gonext, &QPushButton::released, this, &SlideShow::next);
    connect(golast, &QPushButton::released, this, &SlideShow::last);
    connect(goloop, &QPushButton::released, this, &SlideShow::loop);
    connect(zoomin, &QPushButton::released, this, &SlideShow::zoomIn);
    connect(zoomout, &QPushButton::released, this, &SlideShow::zoomOut);
    connect(imgrotcw, &QPushButton::released, this, &SlideShow::doImageRotateCw);
    connect(imgrotccw, &QPushButton::released, this, &SlideShow::doImageRotateCcw);
    connect(imgfliph, &QPushButton::released, this, &SlideShow::doImageFlipH);
    connect(imgflipv, &QPushButton::released, this, &SlideShow::doImageFlipV);
    connect(normal, &QPushButton::released, this, &SlideShow::normalSize);
    connect(fitwin, &QPushButton::released, this, &SlideShow::resetWindowSize);

    toolsLayout->addWidget(tomovie, 1);
    toolsLayout->addWidget(toimage, 1);
    toolsLayout->addWidget(toclip, 1);
    toolsLayout->addWidget(totrash, 1);
    toolsLayout->addWidget(cacheButton, 1);
    toolsLayout->addWidget(empty);
    toolsLayout->addWidget(dummy);
    // the two transform families share icons, so say which is which
    auto *dsplabel = new QLabel("Displayed image:");
    dsplabel->setToolTip("These controls change the picture on screen, not the camera");
    toolsLayout->addWidget(dsplabel, 0);
    toolsLayout->addWidget(zoomin, 1);
    toolsLayout->addWidget(zoomout, 1);
    toolsLayout->addWidget(imgrotcw, 1);
    toolsLayout->addWidget(imgrotccw, 1);
    toolsLayout->addWidget(imgfliph, 1);
    toolsLayout->addWidget(imgflipv, 1);
    toolsLayout->addWidget(normal, 1);
    toolsLayout->addWidget(fitwin, 1);
    toolsLayout->addSpacerItem(
        new QSpacerItem(10, 10, QSizePolicy::Expanding, QSizePolicy::Minimum));
    toolsLayout->addWidget(stoprun);
    toolsLayout->setSizeConstraint(QLayout::SetMinimumSize);
    toolsLayout->setSpacing(LAYOUT_SPACING);

    mainLayout->addLayout(toolsLayout);
    mainLayout->addWidget(new QHline);
    mainLayout->addWidget(display, 10);

    botLayout->addWidget(goplay, 1);
    botLayout->addWidget(goloop, 1);
    botLayout->addWidget(new QLabel("Delay:"));
    botLayout->addWidget(delay, 5);
    botLayout->addWidget(new QLabel("Start:"));
    botLayout->addWidget(startBox, 5);
    botLayout->addWidget(new QLabel("Stop:"));
    botLayout->addWidget(stopBox, 5);
    botLayout->addSpacerItem(new QSpacerItem(10, 10, QSizePolicy::Expanding, QSizePolicy::Minimum));
    botLayout->addWidget(imageCounter);
    botLayout->addWidget(imageName);
    botLayout->setStretch(8, 3);
    botLayout->setSizeConstraint(QLayout::SetMinimumSize);
    botLayout->setSpacing(LAYOUT_SPACING);

    navLayout->addWidget(gofirst, 1);
    navLayout->addWidget(goprev, 1);
    navLayout->addWidget(scrollBar, 10);
    navLayout->addWidget(gonext, 1);
    navLayout->addWidget(golast, 1);
    navLayout->setSpacing(LAYOUT_SPACING);

    mainLayout->addWidget(new QHline);
    mainLayout->addLayout(botLayout);
    mainLayout->addLayout(navLayout);
    mainLayout->setSpacing(LAYOUT_SPACING);
    goplay->setFocus();

    setWindowIcon(QIcon(Cfg::MAIN_ICON));
    emit titleChanged(QFileInfo(fileName).fileName());

    updateCacheIndicator();

    display->setVisible(true);
    setLayout(mainLayout);
    mainLayout->setSizeConstraint(QLayout::SetMinAndMaxSize);
    adjustWindowSize();

    applyWindowFlags(this);
}

void SlideShow::addImage(const QString &filename, const QString &label)
{
    if (imagefiles.contains(filename)) return;

    // update max dimensions from header only — no full decode needed
    const QSize sz = cache.imageSize(filename);
    if (sz.isValid()) {
        maxwidth  = qMax(maxwidth, sz.width());
        maxheight = qMax(maxheight, sz.height());
    }

    const int lastidx = imagefiles.size();
    imagefiles.append(filename);
    imagelabels.append(label.isEmpty() ? filename : label);
    scrollBar->setMaximum(lastidx);

    // Grow the active-range bounds with the sequence. If Stop was pinned to the
    // previous maximum, keep it tracking the last image; otherwise leave the
    // user's explicit choice untouched.
    const int total       = imagefiles.size();
    const bool followStop = (stopBox->value() >= stopBox->maximum());
    startBox->setMaximum(total);
    stopBox->setMaximum(total);
    if (followStop) stopBox->setValue(total);
    updateSliderRange();

    if (spartagui || lastidx == 0) {
        // live mode: display every incoming image; or first image in any mode
        loadImage(lastidx);
        scrollBar->setValue(lastidx);
    } else {
        // viewer mode, non-first image: dimensions already captured above;
        // update the counter total and resize without reloading the display
        imageCounter->setText(
            QString("Image %1 / %2 :").arg(current + 1, 3).arg(imagefiles.size(), 3));
        adjustWindowSize();
    }
    emit contentChanged();
}

int SlideShow::addMovie(const QString &filename)
{
    const MovieInfo info = probeMovie(filename);
    if (!info.valid) {
        warning(this, "Cannot Import Movie File",
                "\"" + QFileInfo(filename).fileName() + "\" cannot be imported:", info.error);
        return 0;
    }

    MovieImportDialog dialog(filename, info, this);
    if (dialog.exec() != QDialog::Accepted) return 0;

    const QString outdir = cache.makeSubDir(QFileInfo(filename).completeBaseName());
    if (outdir.isEmpty()) {
        warning(this, "Cannot Import Movie File",
                "Cannot create a temporary folder for the frames of \"" +
                    QFileInfo(filename).fileName() + "\"");
        return 0;
    }

    QString error;
    const int first    = dialog.firstFrame();
    const int interval = dialog.frameInterval();
    const QStringList frames =
        extractMovieFrames(this, filename, outdir, first, dialog.lastFrame(), interval, error);
    if (frames.isEmpty()) {
        warning(this, "Cannot Import Movie File",
                "No frames were extracted from \"" + QFileInfo(filename).fileName() + "\"", error);
        return 0;
    }

    cache.registerFrames(outdir);

    // the temporary file names carry no meaning, so show the movie file and
    // the number of the frame in the movie instead
    const QString base = QFileInfo(filename).fileName();
    for (int i = 0; i < frames.size(); ++i)
        addImage(frames[i], QString("%1 [frame %2]").arg(base).arg(first + i * interval));
    updateCacheIndicator();
    return frames.size();
}

void SlideShow::deleteImages()
{
    const int lo = startIdx();
    const int hi = stopIdx();
    if ((lo < 0) || (hi < lo) || (hi >= imagefiles.size())) return;

    const int count = hi - lo + 1;
    QMessageBox mb(this);
    mb.setWindowTitle("Delete Images");
    mb.setWindowIcon(QIcon(Cfg::MAIN_ICON));
    mb.setText(QString("Delete %1 image file%2 (image %3 to %4) from disk?")
                   .arg(count)
                   .arg(count == 1 ? "" : "s")
                   .arg(lo + 1)
                   .arg(hi + 1));
    mb.setInformativeText("This operation cannot be undone.");
    mb.setIconPixmap(QIcon(":/icons/warning.svg").pixmap(QSize(64, 64), mb.devicePixelRatioF()));
    mb.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
    mb.setDefaultButton(QMessageBox::No);
    mb.setEscapeButton(QMessageBox::No);

    auto *button = mb.button(QMessageBox::Yes);
    button->setIcon(QIcon(":/icons/dialog-ok.svg"));
    button = mb.button(QMessageBox::No);
    button->setIcon(QIcon(":/icons/dialog-no.svg"));

    if (mb.exec() != QMessageBox::Yes) return;

    // remove back-to-front so the lower indices stay valid while deleting
    for (int i = hi; i >= lo; --i) {
        // the conversion of a deleted file is useless and must not linger on
        cache.forget(imagefiles[i]);
        QFile::remove(imagefiles[i]);
        imagefiles.removeAt(i);
        imagelabels.removeAt(i);
    }
    updateCacheIndicator();

    // nothing left: reset to the empty state
    if (imagefiles.isEmpty()) {
        clear();
        return;
    }

    // resync navigation and reset the active range to the full remaining set
    const int total = imagefiles.size();
    scrollBar->setMaximum(total - 1);
    startBox->setMaximum(total);
    stopBox->setMaximum(total);
    startBox->setValue(1);
    stopBox->setValue(total);
    updateSliderRange();

    current = std::clamp(current, 0, total - 1);
    loadImage(current);
}

QIcon SlideShow::sourceIcon() const
{
    return QIcon(":/icons/image-x-generic.svg");
}

void SlideShow::clear()
{
    imagefiles.clear();
    imagelabels.clear();
    display->setImage(QImage());
    // forget the old sequence's dimensions so the next one sizes the window fresh
    maxwidth  = 0;
    maxheight = 0;
    display->forgetHostFit();
    imageCounter->setText("Image   0 /   0 :");
    imageName->setText("(none)");
    scrollBar->setMaximum(1);
    startBox->setRange(1, 1);
    startBox->setValue(1);
    stopBox->setRange(1, 1);
    stopBox->setValue(1);
    updateSliderRange();
    updateCacheIndicator();
    emit contentChanged();
    repaint();
}

void SlideShow::setDelay()
{
    auto *field = qobject_cast<QSpinBox *>(sender());
    if (field && (field->objectName() == "delay")) {
        timerDelay = field->value();
    }
}

int SlideShow::startIdx() const
{
    return startBox->value() - 1;
}

int SlideShow::stopIdx() const
{
    return stopBox->value() - 1;
}

void SlideShow::updateSliderRange()
{
    scrollBar->setActiveRange(startIdx(), stopIdx());
}

void SlideShow::updateCacheIndicator()
{
    const int images = cache.cachedImages();
    const int frames = cache.frameImages();

    cacheButton->setIcon(cache.isEmpty() ? cacheEmptyIcon : cacheFullIcon);
    // only the converted images can be discarded, so there is nothing to press
    // for when the cache holds movie frames alone
    cacheButton->setEnabled(images > 0);

    if (cache.isEmpty()) {
        cacheButton->setToolTip("The image cache is empty");
        return;
    }

    QStringList held;
    if (images > 0)
        held << QString("%1 converted image%2 (%3)")
                    .arg(images)
                    .arg(images == 1 ? "" : "s", locale().formattedDataSize(cache.cachedBytes()));
    if (frames > 0)
        held << QString("%1 movie frame%2 (%3)")
                    .arg(frames)
                    .arg(frames == 1 ? "" : "s", locale().formattedDataSize(cache.frameBytes()));

    QString tip = "Image cache: " + held.join(", ") + ".\n";
    if (images > 0)
        tip += "Click to discard the converted images. They are converted again when needed.";
    else
        tip += "Movie frames are kept until this window is closed.";
    cacheButton->setToolTip(tip);
}

void SlideShow::purgeCache()
{
    const int images = cache.cachedImages();
    if (images < 1) return;

    QMessageBox mb(this);
    mb.setWindowTitle("Discard Converted Images");
    mb.setWindowIcon(QIcon(Cfg::MAIN_ICON));
    mb.setText(QString("Discard %1 converted image%2 (%3) from the image cache?")
                   .arg(images)
                   .arg(images == 1 ? "" : "s", locale().formattedDataSize(cache.cachedBytes())));
    mb.setInformativeText("The original image files are not touched.  Each of them is converted "
                          "again the next time it is displayed.");
    mb.setIconPixmap(
        QIcon(":/icons/image-cache.svg").pixmap(QSize(64, 64), mb.devicePixelRatioF()));
    mb.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
    mb.setDefaultButton(QMessageBox::Yes);
    mb.setEscapeButton(QMessageBox::No);

    auto *button = mb.button(QMessageBox::Yes);
    button->setIcon(QIcon(":/icons/dialog-ok.svg"));
    button = mb.button(QMessageBox::No);
    button->setIcon(QIcon(":/icons/dialog-no.svg"));

    if (mb.exec() != QMessageBox::Yes) return;

    // the image on display was decoded into memory and stays valid
    cache.purgeConversions();
    updateCacheIndicator();
}

void SlideShow::loadImage(int idx)
{
    if ((idx < 0) || (idx >= imagefiles.size())) return;

    do {
        const QImage newImage = cache.readImage(imagefiles[idx]);

        // There was an error reading the image file. Try reading the previous image instead.
        if (newImage.isNull()) {
            --idx;
        } else {
            display->setImage(newImage);
            maxwidth  = qMax(maxwidth, newImage.width());
            maxheight = qMax(maxheight, newImage.height());
            adjustWindowSize();
            imageCounter->setText(
                QString("Image %1 / %2 :").arg(idx + 1, 3).arg(imagefiles.size(), 3));
            imageName->setText(imagelabels[idx]);
            current = idx;
            break;
        }
    } while (idx >= 0);
    scrollBar->setValue(idx);
    adjustWindowSize();
    // a display may have converted the image and thus filled the cache
    updateCacheIndicator();
}

void SlideShow::copy()
{
    copyImageToClipboard(display->displayedImage());
}

void SlideShow::quit()
{
    if (spartagui) spartagui->quit();
}

void SlideShow::stopRun()
{
    if (spartagui) spartagui->stopRun();
}

void SlideShow::saveCurrentImage()
{
    QImage shot = display->displayedImage();
    exportImage(this, &shot, "SlideShow");
}

void SlideShow::movie()
{
    // exporting a movie needs an external encoder; on macOS in particular this
    // is usually not installed by default, so give a clear explanation rather
    // than silently doing nothing
    const QString ffmpeg = findExe("ffmpeg");
    QString imconvert    = findExe("magick");
    if (imconvert.isEmpty()) imconvert = findExe("convert");
    if (ffmpeg.isEmpty() && imconvert.isEmpty()) {
        warning(this, "Export Movie",
                "Exporting a movie requires the <b>ffmpeg</b> command-line tool "
                "(or ImageMagick), which was not found on this system.",
                "Please install ffmpeg and try again. On macOS you can install it "
                "with Homebrew:\n\n    brew install ffmpeg\n\n"
                "then restart SPARTA-GUI.");
        return;
    }

    if (imagefiles.isEmpty()) {
        warning(this, "Export Movie", "There are no images to export.",
                "Run a simulation that produces a 'dump image' first, or load "
                "image files into the slide show.");
        return;
    }

    QString fileName =
        QFileDialog::getSaveFileName(this, "Export to Movie File", ".",
                                     "Movie Files (*.mp4 *.mkv *.avi *.mpg *.mpeg *.gif *.webm)");
    if (fileName.isEmpty()) return;

    // restrict the exported frames to the active [Start, Stop] range
    const int lo = startIdx();
    const int hi = stopIdx();
    if ((lo < 0) || (hi < lo) || (hi >= imagefiles.size())) return;
    const QStringList frames = imagefiles.mid(lo, hi - lo + 1);

    if (!ffmpeg.isEmpty()) {
        QDir curdir(".");
        QTemporaryFile concatfile;
        if (concatfile.open()) {
            for (const auto &img : frames) {
                concatfile.write("file '");
                concatfile.write(curdir.absoluteFilePath(img).toLocal8Bit());
                concatfile.write("'\n");
            }
            concatfile.close();

            const auto fps = QString::number(1.0 / (static_cast<double>(timerDelay) / 1000.0));
            QStringList args;
            args << "-y";
            args << "-safe"
                 << "0";
            args << "-r" << fps;
            args << "-f"
                 << "concat";
            args << "-i" << concatfile.fileName();
            // the movie must come out oriented the way the preview was
            args << ffmpegFilterArgs(display->transform());
            args << "-b:v"
                 << "2000k";
            args << "-r" << fps;
            args << fileName;

            QProcess proc;
            proc.start(ffmpeg, args);
            proc.waitForFinished(-1);
            if ((proc.exitStatus() != QProcess::NormalExit) || (proc.exitCode() != 0))
                warning(this, "Export Movie", "ffmpeg failed to create the movie file:",
                        QString::fromLocal8Bit(proc.readAllStandardError()));
        } else {
            warning(this, "Export Movie",
                    "Cannot create temporary file for generating movie:", concatfile.errorString());
        }
    } else {
        QStringList args;
        args << "-delay" << QString::number(timerDelay / 10);
        QDir curdir(".");
        for (const auto &img : frames)
            args << curdir.absoluteFilePath(img);
        args << magickTransformArgs(display->transform());
        args << fileName;

        // run the conversion command
        QProcess proc;
        proc.start(imconvert, args);
        proc.waitForFinished(-1);
        if ((proc.exitStatus() != QProcess::NormalExit) || (proc.exitCode() != 0))
            warning(this, "Export Movie", "ImageMagick failed to create the movie file:",
                    QString::fromLocal8Bit(proc.readAllStandardError()));
    }
}

void SlideShow::first()
{
    current = startIdx();
    loadImage(current);
}

void SlideShow::last()
{
    current = stopIdx();
    loadImage(current);
}

void SlideShow::play()
{
    // if we do not loop, start animation from beginning of the active range
    if (!doLoop) current = startIdx();
    auto *delay = findChild<QSpinBox *>("delay");

    if (playtimer) {
        playtimer->stop();
        delete playtimer;
        playtimer = nullptr;
        if (delay) delay->setEnabled(true);
    } else {
        playtimer = new QTimer(this);
        connect(playtimer, &QTimer::timeout, this, &SlideShow::next);
        playtimer->start(timerDelay);
        if (delay) delay->setEnabled(false);
    }

    // reset push button state. use findChild() if not triggered from button.
    auto *button = qobject_cast<QPushButton *>(sender());
    if (!button) button = findChild<QPushButton *>("play");
    if (button) button->setChecked(playtimer);
}

void SlideShow::next()
{
    const int lo = startIdx();
    const int hi = stopIdx();
    ++current;
    if (current > hi) {
        if (doLoop) {
            current = lo;
        } else {
            // stop animation at the end of the active range
            if (playtimer) play();
            current = hi;
        }
    }
    loadImage(current);
}

void SlideShow::prev()
{
    const int lo = startIdx();
    const int hi = stopIdx();
    --current;
    if (current < lo) {
        if (doLoop)
            current = hi;
        else
            current = lo;
    }
    loadImage(current);
}

void SlideShow::loop()
{
    auto *button = qobject_cast<QPushButton *>(sender());
    doLoop       = !doLoop;
    if (button) button->setChecked(doLoop);
}

void SlideShow::zoomIn()
{
    applyTransform([](DisplayTransform &t) { t.zoomIn(); });
}

void SlideShow::zoomOut()
{
    applyTransform([](DisplayTransform &t) { t.zoomOut(); });
}

void SlideShow::normalSize()
{
    applyTransform([](DisplayTransform &t) { t.reset(); });
}

void SlideShow::adjustWindowSize()
{
    if (maxwidth == 0 || maxheight == 0) return;

    // size of the largest image as displayed, i.e. with the current rotation
    // and zoom applied the same way the display applies them
    const QSize content = transformedSize(QSize(maxwidth, maxheight), display->transform());

    // make sure the scroll area is not resized beyond a certain fraction of the screen
    const QSize avail = screen()->availableSize();
    const QSize budget(avail.width() * 3 / 4, (avail.height() * 9 / 10) - EXTRA_HEIGHT);
    display->fitHostWindow(this, content, budget);
}

void SlideShow::resetWindowSize()
{
    // discard both a manual window resize and the memoized fit
    display->forgetHostFit();
    adjustWindowSize();
}

void SlideShow::showEvent(QShowEvent *event)
{
    ViewerSource::showEvent(event);
    // any fit computed while the window was hidden used unpolished style
    // metrics and was not memoized (see fitViewerWindow()); apply the fit
    // again as soon as the shown window has settled
    if (display->needsHostFit()) QTimer::singleShot(0, this, &SlideShow::adjustWindowSize);
}

// event filter to handle "Ambiguous shortcut override" issues (see LogWindow
// and ChartWindow for the same pattern): this window's own Ctrl+ shortcuts
// would otherwise be ambiguous with identically bound main-window menu
// shortcuts now that this is a docked panel sharing its shortcut context
bool SlideShow::eventFilter(QObject *watched, QEvent *event)
{
    // this panel's own Ctrl+ keys, claimed while focus is inside it
    if (dispatchCtrlShortcut(event, {{'W', [this]() { emit closeRequested(); }},
                                     {'/', [this]() { stopRun(); }},
                                     {'Q', [this]() { quit(); }},
                                     {'C', [this]() { copy(); }},
                                     {'E', [this]() { movie(); }},
                                     {'S', [this]() { saveCurrentImage(); }}}))
        return true;
    return ViewerSource::eventFilter(watched, event);
}

void SlideShow::doImageRotateCw()
{
    applyTransform([](DisplayTransform &t) { t.rotateCw(); });
}

void SlideShow::doImageRotateCcw()
{
    applyTransform([](DisplayTransform &t) { t.rotateCcw(); });
}

void SlideShow::doImageFlipH()
{
    applyTransform([](DisplayTransform &t) { t.mirrorH(); });
}

void SlideShow::doImageFlipV()
{
    applyTransform([](DisplayTransform &t) { t.mirrorV(); });
}

// Change the displayed-image transform and let the window follow it.
void SlideShow::applyTransform(const std::function<void(DisplayTransform &)> &change)
{
    DisplayTransform t = display->transform();
    change(t);
    display->setTransform(t);
    adjustWindowSize();
}

QImage SlideShow::currentImage() const
{
    return display->displayedImage();
}

// Local Variables:
// c-basic-offset: 4
// End:
