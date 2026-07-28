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

#include "findandreplace.h"

#include "codeeditor.h"
#include "constants.h"
#include "spartagui.h"

#include <QCheckBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QShortcut>
#include <QTextCursor>

/* ---------------------------------------------------------------------- */

namespace {
constexpr int LAYOUT_SPACING = 6;
}

FindAndReplace::FindAndReplace(CodeEditor *_editor, QWidget *parent) :
    QDialog(parent), editor(_editor), search(nullptr), replace(nullptr), withcase(nullptr),
    wrap(nullptr), whole(nullptr)
{
    auto *layout  = new QGridLayout;
    search        = new QLineEdit;
    replace       = new QLineEdit;
    withcase      = new QCheckBox("Match case");
    wrap          = new QCheckBox("Wrap around");
    whole         = new QCheckBox("Whole word");
    // named so tests and the accessibility walk can reach them; the text
    // fields are otherwise indistinguishable from one another
    search->setObjectName("search");
    replace->setObjectName("replace");
    withcase->setObjectName("withcase");
    wrap->setObjectName("wrap");
    whole->setObjectName("whole");
    auto *next    = new QPushButton("&Next");
    auto *replone = new QPushButton("&Replace");
    auto *replall = new QPushButton("Replace &All");
    auto *done    = new QPushButton("&Done");

    layout->addWidget(new QLabel("Find:"), 0, 0, Qt::AlignRight);
    layout->addWidget(search, 0, 1, 1, 2, Qt::AlignLeft);
    layout->addWidget(new QLabel("Replace with:"), 1, 0, Qt::AlignRight);
    layout->addWidget(replace, 1, 1, 1, 2, Qt::AlignLeft);
    layout->addWidget(withcase, 2, 0, Qt::AlignLeft);
    layout->addWidget(wrap, 2, 1, Qt::AlignLeft);
    layout->addWidget(whole, 2, 2, Qt::AlignLeft);
    wrap->setChecked(true);

    auto *buttons = new QHBoxLayout;
    buttons->addWidget(next);
    buttons->addWidget(replone);
    buttons->addWidget(replall);
    buttons->addWidget(done);
    buttons->setSpacing(LAYOUT_SPACING);
    layout->addLayout(buttons, 3, 0, 1, 3, Qt::AlignHCenter);
    layout->setSpacing(LAYOUT_SPACING);

    connect(next, &QPushButton::released, this, &FindAndReplace::findNext);
    connect(replone, &QPushButton::released, this, &FindAndReplace::replaceNext);
    connect(replall, &QPushButton::released, this, &FindAndReplace::replaceAll);
    connect(done, &QPushButton::released, this, &QDialog::accept);

    auto *action = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_Q), this);
    connect(action, &QShortcut::activated, this, &FindAndReplace::quit);

    setLayout(layout);
    setWindowIcon(QIcon(Cfg::MAIN_ICON));
    setWindowTitle("SPARTA-GUI - Find and Replace");
}

/* ---------------------------------------------------------------------- */

void FindAndReplace::findNext()
{
    auto text = search->text();

    auto find_flags = QTextDocument::FindFlags();
    if (withcase->isChecked()) find_flags |= QTextDocument::FindCaseSensitively;
    if (whole->isChecked()) find_flags |= QTextDocument::FindWholeWords;

    if (!text.isEmpty()) {
        if (!editor->find(text, find_flags) && wrap->isChecked()) {
            // nothing found from the current position to the end, reposition cursor at the
            // beginning
            editor->moveCursor(QTextCursor::Start, QTextCursor::MoveAnchor);
            editor->find(text, find_flags);
        }
    }
}

/* ---------------------------------------------------------------------- */

void FindAndReplace::replaceNext()
{
    auto text = search->text();
    if (text.isEmpty()) return;

    auto cursor = editor->textCursor();
    auto flag   = withcase->isChecked() ? Qt::CaseSensitive : Qt::CaseInsensitive;

    // if selected text at cursor location matches search text, replace
    if (QString::compare(cursor.selectedText(), text, flag) == 0)
        cursor.insertText(replace->text());

    findNext();
}

/* ---------------------------------------------------------------------- */

void FindAndReplace::replaceAll()
{
    const auto text = search->text();
    if (text.isEmpty()) return;

    // Deliberately not findNext(): that one wraps to the top when it reaches
    // the end, and with wrapping on -- the default -- it re-found the text this
    // loop had just inserted.  Replacing "fix" with "fix all" then never
    // terminated: the GUI hung and the document grew until memory ran out.
    //
    // Searching forward from the start of the document without wrapping is both
    // the fix and what Replace All is normally understood to mean.  Each search
    // resumes at the end of the text just inserted, so the position strictly
    // advances and the loop ends at the last match whatever the replacement
    // contains.
    auto find_flags = QTextDocument::FindFlags();
    if (withcase->isChecked()) find_flags |= QTextDocument::FindCaseSensitively;
    if (whole->isChecked()) find_flags |= QTextDocument::FindWholeWords;

    QTextDocument *doc = editor->document();

    // one undo step for the whole operation, rather than one per occurrence
    QTextCursor anchor(doc);
    anchor.beginEditBlock();
    QTextCursor found = doc->find(text, QTextCursor(doc), find_flags);
    while (!found.isNull()) {
        found.insertText(replace->text());
        found = doc->find(text, found, find_flags);
    }
    anchor.endEditBlock();
}

/* ---------------------------------------------------------------------- */

void FindAndReplace::quit()
{
    auto *main = qobject_cast<SpartaGui *>(parent());
    if (main) main->quit();
}

// Local Variables:
// c-basic-offset: 4
// End:
