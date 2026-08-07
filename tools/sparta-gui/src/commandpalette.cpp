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

#include "commandpalette.h"

#include "actionscan.h"

#include <QAction>
#include <QEvent>
#include <QHeaderView>
#include <QKeyEvent>
#include <QLineEdit>
#include <QMenuBar>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>

#include <algorithm>

namespace {

// How well does @p needle match @p hay as a subsequence?  Higher is better;
// -1 is no match.  Word-shaped preferences, in order: matching at the start,
// matching at word starts, matching consecutively.  Nothing clever -- the
// lists are a hundred entries, not a million.
int fuzzyScore(const QString &needle, const QString &hay)
{
    if (needle.isEmpty()) return 0;
    int score = 0, h = 0;
    bool prevHit = false;
    for (int n = 0; n < needle.size(); ++n) {
        const QChar want = needle[n].toLower();
        bool found       = false;
        for (; h < hay.size(); ++h) {
            if (hay[h].toLower() == want) {
                const bool wordStart =
                    h == 0 || hay[h - 1] == QLatin1Char(' ') || hay[h - 1] == QLatin1Char('>');
                if (h == n) score += 3;      // still tracking the very front
                if (wordStart) score += 2;   // "cv" finds "Charts Window" nicely
                if (prevHit) score += 2;     // consecutive letters beat scattered
                prevHit = true;
                found   = true;
                ++h;
                break;
            }
            prevHit = false;
        }
        if (!found) return -1;
    }
    return score;
}

constexpr int RoleIndex = Qt::UserRole; // row -> index into the scanned list

} // namespace

CommandPalette::CommandPalette(QMenuBar *bar, QWidget *parent) : QDialog(parent), menubar(bar)
{
    setObjectName("commandpalette");
    setWindowTitle("SPARTA-GUI - Command Palette");
    setModal(true);
    resize(560, 420);

    auto *layout = new QVBoxLayout(this);
    input        = new QLineEdit(this);
    input->setObjectName("paletteinput");
    input->setPlaceholderText("Type to search every menu action...");
    input->setClearButtonEnabled(true);
    layout->addWidget(input);

    list = new QTreeWidget(this);
    list->setObjectName("palettelist");
    list->setColumnCount(2);
    list->setHeaderHidden(true);
    list->setRootIsDecorated(false);
    list->setAllColumnsShowFocus(true);
    list->setUniformRowHeights(true);
    list->header()->setStretchLastSection(false);
    list->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    list->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    layout->addWidget(list, 1);

    connect(input, &QLineEdit::textChanged, this, &CommandPalette::refilter);
    connect(list, &QTreeWidget::itemActivated, this, &CommandPalette::triggerCurrent);
    // arrow keys move the selection while typing continues in the line edit
    input->installEventFilter(this);
}

void CommandPalette::popup()
{
    rebuild();
    input->clear();
    refilter(QString());
    show();
    raise();
    activateWindow();
    input->setFocus();
}

void CommandPalette::showEvent(QShowEvent *event)
{
    QDialog::showEvent(event);
    input->setFocus();
}

void CommandPalette::rebuild()
{
    list->clear();
    const auto infos = scanMenuBar(menubar);
    int idx          = 0;
    for (const auto &info : infos) {
        if (!info.action) {
            ++idx;
            continue;
        }
        QString name = info.text;
        if (info.action->isCheckable())
            name.prepend(info.action->isChecked() ? QStringLiteral("[x] ")
                                                  : QStringLiteral("[ ] "));
        auto *item = new QTreeWidgetItem(
            {name + "   -   " + info.path,
             info.action->shortcut().toString(QKeySequence::NativeText)});
        item->setData(0, RoleIndex, idx);
        item->setToolTip(0, info.action->statusTip());
        if (!info.action->isEnabled()) item->setDisabled(true);
        list->addTopLevelItem(item);
        ++idx;
    }
}

void CommandPalette::refilter(const QString &needle)
{
    // score every row, hide the misses, order the hits by score
    struct Hit {
        QTreeWidgetItem *item;
        int score;
    };
    QList<Hit> hits;
    for (int i = 0; i < list->topLevelItemCount(); ++i) {
        auto *item      = list->topLevelItem(i);
        const int score = fuzzyScore(needle, item->text(0));
        item->setHidden(score < 0);
        if (score >= 0) hits.append({item, score});
    }
    std::stable_sort(hits.begin(), hits.end(),
                     [](const Hit &a, const Hit &b) { return a.score > b.score; });
    // QTreeWidget has no reorder-in-place; re-rank via sort keys would churn.
    // Selecting the best hit is what matters: Enter should do the right thing.
    for (const auto &h : hits) {
        if (!h.item->isDisabled()) {
            list->setCurrentItem(h.item);
            break;
        }
    }
}

void CommandPalette::triggerCurrent()
{
    auto *item = list->currentItem();
    if (!item || item->isHidden() || item->isDisabled()) return;
    const int idx = item->data(0, RoleIndex).toInt();
    // re-scan: menus may have been rebuilt since; the index is only a hint
    const auto infos = scanMenuBar(menubar);
    if (idx < 0 || idx >= infos.size() || !infos[idx].action) return;
    QAction *action = infos[idx].action;
    accept();
    // after the dialog is gone, so a modal the action opens stacks cleanly
    action->trigger();
}

void CommandPalette::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
        triggerCurrent();
        return;
    }
    QDialog::keyPressEvent(event);
}

bool CommandPalette::eventFilter(QObject *watched, QEvent *event)
{
    if (watched == input && event->type() == QEvent::KeyPress) {
        auto *ke = static_cast<QKeyEvent *>(event);
        if (ke->key() == Qt::Key_Down || ke->key() == Qt::Key_Up) {
            // move the selection over visible, enabled rows
            const int dir  = (ke->key() == Qt::Key_Down) ? 1 : -1;
            auto *current  = list->currentItem();
            int from       = current ? list->indexOfTopLevelItem(current) : -1;
            for (int i = from + dir; i >= 0 && i < list->topLevelItemCount(); i += dir) {
                auto *item = list->topLevelItem(i);
                if (!item->isHidden() && !item->isDisabled()) {
                    list->setCurrentItem(item);
                    break;
                }
            }
            return true;
        }
    }
    return QDialog::eventFilter(watched, event);
}

// Local Variables:
// c-basic-offset: 4
// End:
