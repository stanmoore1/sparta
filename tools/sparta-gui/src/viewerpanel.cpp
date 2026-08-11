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

#include "viewerpanel.h"

#include "emptystate.h"
#include "helpers.h"
#include "imageviewer.h"
#include "slideshow.h"
#include "viewersource.h"

#include <QAction>
#include <QHBoxLayout>
#include <QIcon>
#include <QImage>
#include <QKeySequence>
#include <QMenu>
#include <QStackedWidget>
#include <QTabBar>
#include <QToolButton>
#include <QVBoxLayout>

namespace {

/**
 * @brief A stack that reports the page in front, not the biggest page.
 *
 * The same rule ViewerPanel::minimumSizeHint() applies across sources, applied
 * within one: while the placeholder card is showing, the panel must be free to
 * be as small as the card, not as small as the viewer waiting behind it.
 */
class PageStack : public QStackedWidget {
public:
    using QStackedWidget::QStackedWidget;

    [[nodiscard]] QSize minimumSizeHint() const override
    {
        if (auto *page = currentWidget()) return page->minimumSizeHint();
        return QStackedWidget::minimumSizeHint();
    }
};

/// What a tab says before its viewer exists.
///
/// Duplicating the label and the empty text that ViewerSource also carries is
/// deliberate and bounded: the tab has to describe a viewer that has not been
/// built yet, and for the snapshot viewer cannot be built until a run has got
/// far enough to render.  Once a source registers, the panel reads everything
/// from the instance instead.  ViewerPanelChrome in the tests holds the two
/// descriptions to each other so they cannot drift apart unnoticed.
struct Chrome {
    const char *label;
    const char *icon;
    const char *title;
    const char *hint;
};

const Chrome CHROME[ViewerPanel::NSources] = {
    {"Snapshot", ":/icons/image-viewer.svg", "No snapshot yet",
     "Run \u25b8 Create Image (Ctrl+I) renders the simulation as it stands right now.\n\n"
     "It needs a deck that has got as far as defining a box and a grid, so check or "
     "run the input first."},
    {"Sequence", ":/icons/media-playback-start-2.svg", "No image sequence yet",
     "Add a dump image command to your input deck and run it. The frames it writes "
     "appear here as they are produced, to step through or export as a movie.\n\n"
     "For example:\n"
     "    dump 1 image all 100 img.*.ppm type type\n\n"
     "Already have frames on disk? File \u25b8 View Image or Movie File(s) opens them."},
    {"3D", ":/icons/x-office-drawing.svg", "No 3D data yet",
     "Run \u25b8 3D Snapshot builds a scene from the simulation as it stands.\n\n"
     "For a scene that follows a run, add a VTK dump to your input deck and run it:\n"
     "    dump 1 grid/vtk all 100 grid.*.vtu\n\n"
     "Files written earlier can be opened from this panel\u2019s Open button."},
};

} // namespace

ViewerPanel::ViewerPanel(QWidget *parent) :
    QWidget(parent), tabs(new QTabBar), stack(new QStackedWidget)
{
    // A real tab bar rather than a row of toggle buttons or a combo box. It is
    // the gesture users already had, since the image viewer and the slide show
    // were separate tabs of the same dock area; it carries the "page tab"
    // accessibility role, which the dock tabs it replaces did not; and it reads
    // as structure next to toolbars that already hold a couple of dozen
    // buttons.
    tabs->setExpanding(false);
    tabs->setDrawBase(false);
    tabs->setUsesScrollButtons(false);

    // Beside the tabs rather than on a row of their own: the viewer shares a
    // window with the editor and the log, and a strip that costs nothing but
    // the height already spent on the tab bar is a strip that does not have to
    // be argued for.
    auto *save = new QToolButton;
    saveAction = new QAction(QIcon(":/icons/document-save-as.svg"), "Save Image &As...", this);
    saveAction->setToolTip("Save the picture this tab is showing to a file");
    saveAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_S));
    saveAction->setShortcutContext(Qt::WidgetWithChildrenShortcut);
    save->setDefaultAction(saveAction);
    save->setObjectName("viewersave");
    save->setAutoRaise(true);
    addAction(saveAction);

    auto *copy = new QToolButton;
    copyAction = new QAction(QIcon(":/icons/edit-copy.svg"), "&Copy Image", this);
    copyAction->setToolTip("Copy the picture this tab is showing to the clipboard");
    copyAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_C));
    copyAction->setShortcutContext(Qt::WidgetWithChildrenShortcut);
    copy->setDefaultAction(copyAction);
    copy->setObjectName("viewercopy");
    copy->setAutoRaise(true);
    addAction(copyAction);

    connect(saveAction, &QAction::triggered, this, &ViewerPanel::saveCurrentImage);
    connect(copyAction, &QAction::triggered, this, &ViewerPanel::copyCurrentImage);

    // A source with a menu of its own (the 3D scene's filters) hangs it here,
    // so "this tab has more" is in the same place whichever tab that is.
    menuButton = new QToolButton;
    menuButton->setObjectName("viewersourcemenu");
    menuButton->setText("More");
    menuButton->setPopupMode(QToolButton::InstantPopup);
    menuButton->setAutoRaise(true);
    menuButton->hide();

    auto *top = new QHBoxLayout;
    top->setContentsMargins(0, 0, 0, 0);
    top->setSpacing(2);
    top->addWidget(tabs);
    top->addStretch(1);
    top->addWidget(menuButton);
    top->addWidget(save);
    top->addWidget(copy);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addLayout(top);
    layout->addWidget(stack, 1);

    // Every tab, now, in enum order, whether or not its viewer exists.  A tab
    // that arrives only when its source produces something is a feature the
    // user has to stumble on; a tab that is there from the start, saying what
    // would fill it, is one they can act on.
    for (int i = 0; i < NSources; ++i) {
        const Source which = Source(i);
        if (!sourceAvailable(which)) continue;

        auto *slot = new PageStack;
        slot->setObjectName(QString("viewerslot%1").arg(i));
        auto *card = new EmptyState(QString::fromUtf8(CHROME[i].title),
                                    QString::fromUtf8(CHROME[i].hint));
        card->setObjectName(QString("viewerempty%1").arg(i));
        slot->addWidget(card);

        slots_[i] = slot;
        pageOf[i] = stack->addWidget(slot);

        const int index = tabs->addTab(QIcon(QString::fromUtf8(CHROME[i].icon)),
                                       QString::fromUtf8(CHROME[i].label));
        tabs->setTabData(index, i);
        tabs->setTabToolTip(index, QString::fromUtf8(CHROME[i].title) + "\n" +
                                       QString::fromUtf8(CHROME[i].hint));
    }

    connect(tabs, &QTabBar::currentChanged, this, [this](int index) {
        if (index < 0) return;
        const int which = tabs->tabData(index).toInt();
        if (which < 0 || which >= NSources) return;
        if (pageOf[which] >= 0) stack->setCurrentIndex(pageOf[which]);
        sourceLockedByUser = true;
        syncSharedControls();
        emit sourceChanged(which);
        emit titleChanged(title());
    });
}

ViewerPanel::~ViewerPanel() = default;

bool ViewerPanel::sourceAvailable(Source which)
{
#if defined(SPARTA_GUI_HAVE_VTK)
    Q_UNUSED(which)
    return true;
#else
    // Without VTK there is no 3D viewer to build, so offering the tab would be
    // offering a card describing something this build cannot do.
    return which != Scene;
#endif
}

void ViewerPanel::addSource(Source which, ViewerSource *source)
{
    if (!source) return;
    if (!slots_[which]) {   // a build without this source: nothing to put it in
        source->deleteLater();
        return;
    }

    if (sources[which]) {
        replaceSource(which, source);
        return;
    }

    sources[which] = source;
    slots_[which]->addWidget(source);

    // The tab exists already; take its wording from the instance now that
    // there is one, so the source stays the authority on how it describes
    // itself.
    const int index = tabOf(which);
    if (index >= 0) {
        tabs->setTabText(index, source->sourceLabel());
        tabs->setTabIcon(index, source->sourceIcon());
    }

    wireSource(which, source);
    updateTab(which);
}

void ViewerPanel::replaceSource(Source which, ViewerSource *source)
{
    if (!source) return;
    if (!slots_[which]) {
        source->deleteLater();
        return;
    }

    ViewerSource *old = sources[which];
    if (old) {
        slots_[which]->removeWidget(old);
        old->disconnect(this);
        old->deleteLater();
    }

    sources[which] = source;
    slots_[which]->addWidget(source);

    wireSource(which, source);
    updateTab(which);
}

void ViewerPanel::syncSharedControls()
{
    ViewerSource *src = sources[currentSource()];

    // Greyed rather than hidden: a tab with nothing in it still says what Save
    // and Copy would act on, and controls that come and go as tabs change read
    // as the panel rearranging itself.
    const bool havePicture = src && !src->currentImage().isNull();
    if (saveAction) saveAction->setEnabled(havePicture);
    if (copyAction) copyAction->setEnabled(havePicture);

    if (!menuButton) return;
    QMenu *own = src ? src->sourceMenu() : nullptr;
    menuButton->setMenu(own);
    menuButton->setVisible(own != nullptr);
    if (own) menuButton->setToolTip(own->title().isEmpty()
                                        ? QStringLiteral("More for this tab")
                                        : own->title());
}

void ViewerPanel::saveCurrentImage()
{
    ViewerSource *src = sources[currentSource()];
    if (!src) return;
    QImage shot = src->currentImage();
    if (shot.isNull()) return;
    exportImage(this, &shot, QStringLiteral("Viewer"));
}

void ViewerPanel::copyCurrentImage()
{
    ViewerSource *src = sources[currentSource()];
    if (!src) return;
    const QImage shot = src->currentImage();
    if (!shot.isNull()) copyImageToClipboard(shot);
}

int ViewerPanel::tabOf(Source which) const
{
    for (int i = 0; i < tabs->count(); ++i)
        if (tabs->tabData(i).toInt() == int(which)) return i;
    return -1;
}

void ViewerPanel::wireSource(Source which, ViewerSource *source)
{
    connect(source, &ViewerSource::contentChanged, this, [this, which]() { updateTab(which); });
    connect(source, &ViewerSource::closeRequested, this, &ViewerPanel::closeRequested);
    connect(source, &ViewerSource::titleChanged, this, [this, which](const QString &name) {
        names[which] = name;
        if (currentSource() == which) emit titleChanged(title());
    });
}

void ViewerPanel::updateTab(Source which)
{
    QStackedWidget *slot = slots_[which];
    if (!slot) return;

    ViewerSource *source = sources[which];
    const bool ready     = source && source->hasContent();

    // Page 0 is the card that says what would fill this panel; the viewer, if
    // one has been built, is behind it.  Swapping between them is what makes
    // clicking an empty tab answer a question rather than show a blank pane.
    slot->setCurrentIndex(ready ? 1 : 0);
    if (which == currentSource()) syncSharedControls();

    // Every tab stays *enabled*.  Disabling the empty ones reads better right
    // up until they are all empty, which is the state the panel opens in:
    // QTabBar then has no enabled tab to make current, and the panel looks
    // broken rather than merely waiting.
    const int index = tabOf(which);
    if (index < 0) return;

    if (ready) {
        tabs->setTabToolTip(index, source->sourceTip());
    } else if (source) {
        tabs->setTabToolTip(index, source->emptyTitle() + "\n" + source->emptyTip());
    }   // else: the tooltip set at construction still describes it
}

void ViewerPanel::refreshTabs()
{
    for (int i = 0; i < NSources; ++i)
        updateTab(Source(i));
}

ViewerPanel::Source ViewerPanel::currentSource() const
{
    const int index = tabs->currentIndex();
    if (index < 0) return Snapshot;
    const int which = tabs->tabData(index).toInt();
    return (which >= 0 && which < NSources) ? Source(which) : Snapshot;
}

void ViewerPanel::showSource(Source which, bool userAsked)
{
    // No guard on the source existing: the tab is real whether or not its
    // viewer has been built, and bringing forward the card that explains how
    // to build one is a perfectly good thing to be asked for.
    if (!slots_[which]) return;
    if (!userAsked && sourceLockedByUser) return;

    const int index = tabOf(which);
    if (index >= 0) {
        // A source being shown because it just produced something has content
        // by definition, but the tab may not have been told yet.
        updateTab(which);
        tabs->setCurrentIndex(index);
    }
    if (pageOf[which] >= 0) stack->setCurrentIndex(pageOf[which]);
    syncSharedControls();

    // Set from the argument rather than left to the currentChanged handler.
    // That handler only fires when the index actually moves, so asking for the
    // tab that is already in front recorded no choice -- and with every tab
    // present from the start, the tab a user picks first is very often already
    // the current one.  An automatic switch is not a choice and clears it.
    sourceLockedByUser = userAsked;
}

SlideShow *ViewerPanel::sequence() const
{
    return qobject_cast<SlideShow *>(sources[Sequence]);
}

ImageViewer *ViewerPanel::snapshot() const
{
    return qobject_cast<ImageViewer *>(sources[Snapshot]);
}

QString ViewerPanel::title() const
{
    const Source which = currentSource();
    ViewerSource *src  = sources[which];
    // A tab whose viewer has not been built yet still names itself, so the
    // dock title follows the tab rather than falling back to a bare "Viewer"
    // for every one of the three until a run fills it.
    if (!src) {
        const int index = tabOf(which);
        if (index < 0) return QStringLiteral("Viewer");
        return QStringLiteral("Viewer - ") + tabs->tabText(index);
    }

    QString name = QStringLiteral("Viewer - ") + src->sourceLabel();
    if (!names[which].isEmpty()) name += QStringLiteral(" - ") + names[which];
    return name;
}

QSize ViewerPanel::minimumSizeHint() const
{
    QSize hint = QWidget::minimumSizeHint();
    if (auto *page = stack->currentWidget()) {
        const QSize pageHint = page->minimumSizeHint();
        hint.setWidth(qMax(pageHint.width(), tabs->minimumSizeHint().width()));
        hint.setHeight(pageHint.height() + tabs->minimumSizeHint().height());
    }
    return hint;
}

// Local Variables:
// c-basic-offset: 4
// End:
