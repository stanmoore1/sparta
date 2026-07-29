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

#include "setvariables.h"

#include "constants.h"
#include "helpers.h"

#include <QDialogButtonBox>
#include <QIcon>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSizePolicy>
#include <QVBoxLayout>

/* ---------------------------------------------------------------------- */

namespace {
constexpr int LAYOUT_SPACING = 6;
}

namespace {
/// Bold a value that differs from the deck's own, and say so in a tooltip.
void markOverride(QLineEdit *val)
{
    const QString scriptValue = val->property("scriptValue").toString();
    const bool overridden     = !scriptValue.isEmpty() && val->text() != scriptValue;
    val->setStyleSheet(overridden ? "font-weight: bold" : "");
    if (overridden)
        val->setToolTip(QString("Overrides the input deck value: %1").arg(scriptValue));
    else if (!scriptValue.isEmpty())
        val->setToolTip("Value from the input deck");
    else
        val->setToolTip("Not defined in the input deck; passed to the run like -var");
}
} // namespace

SetVariables::SetVariables(QList<QPair<QString, QString>> &_vars,
                           const QHash<QString, QString> &_script, QWidget *parent) :
    QDialog(parent), vars(_vars), scriptValues(_script), layout(new QVBoxLayout)
{
    auto *top = new QLabel("Set Variables:");
    layout->addWidget(top, 0, Qt::AlignHCenter);
    // Which of these actually change anything is otherwise invisible: the list
    // is seeded from the deck, so most rows repeat what the deck already says
    // and only the edited ones have any effect at run time.
    auto *hint = new QLabel("Bold values override the definition in the input deck");
    hint->setStyleSheet("font-style: italic");
    layout->addWidget(hint, 0, Qt::AlignHCenter);
    layout->setSpacing(LAYOUT_SPACING);

    // 2, not 1: the two labels above occupy indices 0 and 1, and delRow()
    // takes the row at the index stored in the button's object name.
    int i = 2;
    for (const auto &v : vars) {
        auto *row  = new QHBoxLayout;
        auto *name = new QLineEdit(v.first);
        auto *val  = new QLineEdit(v.second);
        val->setProperty("scriptValue", scriptValues.value(v.first));
        connect(val, &QLineEdit::textChanged, val, [val]() { markOverride(val); });
        markOverride(val);
        auto *del  = new QPushButton(QIcon(":/icons/edit-delete.svg"), "");
        del->setObjectName(QString::number(i));
        connect(del, &QPushButton::released, this, &SetVariables::delRow);
        row->addWidget(name);
        row->addWidget(val);
        row->addWidget(del);
        row->setSpacing(LAYOUT_SPACING);
        layout->addLayout(row);
        ++i;
    }
    layout->addSpacerItem(new QSpacerItem(10, 10, QSizePolicy::Minimum, QSizePolicy::Expanding));

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    auto *add       = new QPushButton(QIcon(":/icons/expand-text.svg"), "&Add Row");
    add->setObjectName("addRow");
    buttonBox->addButton(add, QDialogButtonBox::ActionRole);
    connect(add, &QPushButton::released, this, &SetVariables::addRow);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &SetVariables::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    layout->addWidget(buttonBox);
    setLayout(layout);
    setWindowIcon(QIcon(Cfg::MAIN_ICON));
    styleDialogButtons(buttonBox);

    setWindowTitle("SPARTA-GUI - Set Variables");
    resize(300, 200);
}

void SetVariables::accept()
{
    // store all data in variables class and then confirm accepting
    vars.clear();
    int nrows = layout->count() - 2;
    // from 2: index 0 is the title and index 1 the override hint, neither of
    // which is a row layout -- reading them as one dereferences a null layout
    for (int i = 2; i < nrows; ++i) {
        auto *row = layout->itemAt(i)->layout();
        auto *var = qobject_cast<QLineEdit *>(row->itemAt(0)->widget());
        auto *val = qobject_cast<QLineEdit *>(row->itemAt(1)->widget());
        if (var && val) vars.append(qMakePair(var->text(), val->text()));
    }

    QDialog::accept();
}

void SetVariables::addRow()
{
    int nrows  = layout->count();
    auto *row  = new QHBoxLayout;
    auto *name = new QLineEdit(QString());
    auto *val  = new QLineEdit(QString());
    auto *del  = new QPushButton(QIcon(":/icons/edit-delete.svg"), "");
    del->setObjectName(QString::number(nrows - 2));
    connect(del, &QPushButton::released, this, &SetVariables::delRow);
    row->addWidget(name);
    row->addWidget(val);
    row->addWidget(del);
    layout->insertLayout(nrows - 2, row);
}

void SetVariables::delRow()
{
    int nrows = layout->count();
    auto *who = sender();
    if (who) {
        // figure out which row was deleted and delete its layout and widgets
        int delrow = who->objectName().toInt();
        auto *row  = layout->takeAt(delrow);
        while (row->layout()->count() > 0) {
            auto *item = row->layout()->takeAt(0);
            if (item) {
                // deleteLater(): one of these widgets is the button whose slot
                // is executing right now; the sender must not be deleted here
                if (item->widget()) item->widget()->deleteLater();
                delete item;
            }
        }
        delete row->layout();

        // renumber the delete pushbutton names
        for (int i = delrow; i < nrows - 3; ++i) {
            auto *rowlayout = layout->itemAt(i)->layout();
            auto *widget    = rowlayout->itemAt(2)->widget();
            widget->setObjectName(QString::number(i));
        }
    }
}

// Local Variables:
// c-basic-offset: 4
// End:
