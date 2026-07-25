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

#include "imageviewer.h"
#include "slideshow.h"
#include "viewersource.h"

#include <QStackedWidget>
#include <QTabBar>
#include <QVBoxLayout>

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

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(tabs);
    layout->addWidget(stack, 1);

    connect(tabs, &QTabBar::currentChanged, this, [this](int index) {
        if (index < 0) return;
        const int which = tabs->tabData(index).toInt();
        if (which < 0 || which >= NSources) return;
        if (pageOf[which] >= 0) stack->setCurrentIndex(pageOf[which]);
        sourceLockedByUser = true;
        emit sourceChanged(which);
        emit titleChanged(title());
    });
}

ViewerPanel::~ViewerPanel() = default;

void ViewerPanel::addSource(Source which, ViewerSource *source)
{
    if (!source) return;

    if (sources[which]) {
        replaceSource(which, source);
        return;
    }

    sources[which] = source;
    pageOf[which]  = stack->addWidget(source);

    // Tabs stay in enum order however the sources arrive, so the bar does not
    // rearrange itself depending on which viewer happened to be used first.
    int at = tabs->count();
    for (int i = 0; i < tabs->count(); ++i) {
        if (tabs->tabData(i).toInt() > int(which)) {
            at = i;
            break;
        }
    }
    const int index = tabs->insertTab(at, source->sourceIcon(), source->sourceLabel());
    tabs->setTabData(index, int(which));

    wireSource(which, source);
    updateTab(which);
}

void ViewerPanel::replaceSource(Source which, ViewerSource *source)
{
    if (!source) return;

    ViewerSource *old = sources[which];
    if (old) {
        stack->removeWidget(old);
        old->disconnect(this);
        old->deleteLater();
    }

    sources[which] = source;
    pageOf[which]  = stack->addWidget(source);

    wireSource(which, source);
    updateTab(which);
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
    ViewerSource *source = sources[which];
    if (!source) return;

    for (int i = 0; i < tabs->count(); ++i) {
        if (tabs->tabData(i).toInt() != int(which)) continue;

        // A tab that vanishes when it is empty tells the user nothing, so an
        // empty source keeps its tab and says why in the tooltip: "No render
        // yet: use Run > Create Image" is a hint, an absent tab is a mystery.
        //
        // The tab stays *enabled* though. Disabling it reads better right up
        // until every source is empty, which is the state the panel opens in:
        // QTabBar then has no enabled tab to make current, and the panel looks
        // broken rather than merely empty. An empty page is honest and costs
        // nothing.
        const bool ready = source->hasContent();
        tabs->setTabToolTip(i, ready ? source->sourceTip() : source->emptyTip());
        break;
    }
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
    if (!sources[which]) return;
    if (!userAsked && sourceLockedByUser) return;

    for (int i = 0; i < tabs->count(); ++i) {
        if (tabs->tabData(i).toInt() != int(which)) continue;
        // A source being shown because it just produced something has content
        // by definition, but the tab may not have been told yet.
        updateTab(which);
        tabs->setCurrentIndex(i);
        break;
    }
    if (pageOf[which] >= 0) stack->setCurrentIndex(pageOf[which]);
    if (!userAsked) sourceLockedByUser = false;   // an automatic switch is not a choice
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
    if (!src) return QStringLiteral("Viewer");

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
