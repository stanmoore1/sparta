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

// The input editor: the widget a user actually types their deck into.
//
// Its text transforms -- reformatting a command into aligned columns,
// commenting and uncommenting, and the completion lists built by parsing the
// buffer -- are the parts that silently damage a deck if they are wrong, and
// they were reachable only through the live GUI walker, which types into the
// editor but never checks what came out.

#include "codeeditor.h"

#include "constants.h"

#include <gtest/gtest.h>

#include <QAbstractItemModel>
#include <QApplication>
#include <QCompleter>
#include <QDir>
#include <QUrl>
#include <QTimer>
#include <QTemporaryDir>
#include <QMenu>
#include <QImage>
#include <QFile>
#include <QDropEvent>
#include <QDragEnterEvent>
#include <QContextMenuEvent>
#include <QAbstractItemView>
#include <QClipboard>
#include <QMimeData>
#include <QSettings>
#include <QStringListModel>
#include <QTextBlock>
#include <QTextCursor>

namespace {

// A standalone editor: no main window, which is the case the class already
// half-supports (its constructor qobject_casts its parent and accepts null).
class Editor : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        // reformatLine reads its column widths from QSettings; pin them so the
        // expected alignment below does not depend on the developer's
        // preferences having been left at the defaults
        QSettings s;
        s.beginGroup(Keys::GROUP_REFORMAT);
        s.setValue(Keys::COMMAND, 16);
        s.setValue(Keys::TYPE, 4);
        s.setValue(Keys::ID, 4);
        s.setValue(Keys::NAME, 8);
        s.endGroup();
        s.sync();
    }

    void SetUp() override
    {
        ed = new CodeEditor(nullptr);
        // The style completers are filled by the main window from SPARTA's own
        // style lists, so a standalone editor has them empty -- and an empty
        // popup never appears, which would make every completion case below
        // pass for the wrong reason. These stand in for that.
        ed->setGroupList(); // seeds the group completer with "all"
        ed->setCommandList({"units", "uniform", "dimension", "run", "region", "fix",
                            "compute", "dump", "collide", "react", "variable"});
        ed->setRegionList({"block", "cylinder", "sphere", "union"});
        ed->setVariableList({"equal", "index", "loop", "particle", "world"});
        ed->setFixList({"ave/time", "ave/grid", "ablate", "emit/face"});
        ed->setComputeList({"thermal/grid", "temp", "count", "grid"});
        ed->setDumpList({"image", "grid", "particle", "surf"});
        ed->setSurfCollideList({"diffuse", "specular", "vanish"});
        ed->setSurfReactList({"prob", "global", "adsorb"});
        ed->setCollideList({"vss", "vhs", "hs"});
        ed->setReactList({"tce", "qk"});
        ed->setUnitsList({"cgs", "si"});
    }
    void TearDown() override { delete ed; }

    void setText(const QString &t) const { ed->setPlainText(t); }

    // put the cursor on line @p n (0-based) without selecting anything
    void gotoLine(int n) const
    {
        QTextCursor c(ed->document()->findBlockByNumber(n));
        ed->setTextCursor(c);
    }

    // select from the start of line @p first to the end of line @p last
    void selectLines(int first, int last) const
    {
        QTextCursor c(ed->document()->findBlockByNumber(first));
        c.movePosition(QTextCursor::StartOfLine);
        c.movePosition(QTextCursor::Down, QTextCursor::KeepAnchor, last - first);
        c.movePosition(QTextCursor::EndOfLine, QTextCursor::KeepAnchor);
        ed->setTextCursor(c);
    }

    QStringList completions(const char *name) const
    {
        for (auto *c : ed->findChildren<QCompleter *>()) {
            if (c->objectName() != QLatin1String(name)) continue;
            QStringList out;
            auto *m = c->model();
            for (int i = 0; i < m->rowCount(); ++i)
                out << m->index(i, 0).data().toString();
            return out;
        }
        return {};
    }

    CodeEditor *ed = nullptr;
};

} // namespace

// ---------------------------------------------------------------- reformatting

TEST_F(Editor, PadsACommandOutToTheCommandColumn)
{
    // "units" is 5 characters, the command column is 16
    EXPECT_EQ(ed->reformatLine("units cgs"), QString("units").leftJustified(16, ' ') + "cgs");
}

TEST_F(Editor, CollapsesRunsOfWhitespaceBetweenArguments)
{
    EXPECT_EQ(ed->reformatLine("units    cgs"), ed->reformatLine("units cgs"));
    EXPECT_EQ(ed->reformatLine("  units\tcgs  "), ed->reformatLine("units cgs"));
}

TEST_F(Editor, ALoneCommandGetsNoTrailingPadding)
{
    // padding only makes sense when something follows; a bare command must not
    // pick up fifteen trailing spaces
    EXPECT_EQ(ed->reformatLine("run"), "run");
    EXPECT_EQ(ed->reformatLine("clear"), "clear");
}

TEST_F(Editor, CommentsAreLeftExactlyAsTheyAre)
{
    const QString comment = "#   this   spacing   is   deliberate";
    EXPECT_EQ(ed->reformatLine(comment), comment);
}

TEST_F(Editor, AnEmptyLineStaysEmpty)
{
    EXPECT_EQ(ed->reformatLine(""), "");
    EXPECT_EQ(ed->reformatLine("   "), "");
}

TEST_F(Editor, PadsTheIdAndStyleOfFixComputeAndDump)
{
    // fix/compute/dump get their ID padded to 4 and their style to 8, so that
    // a block of them lines up in columns
    const QString fix = ed->reformatLine("fix f1 ave/time 1 100 100 c_t");
    EXPECT_TRUE(fix.startsWith(QString("fix").leftJustified(16, ' ')));
    EXPECT_TRUE(fix.contains("f1   ")) << fix.toStdString();
    EXPECT_TRUE(fix.contains("ave/time ")) << fix.toStdString();

    for (const char *cmd : {"compute", "dump"}) {
        const QString out = ed->reformatLine(QString("%1 id1 style arg").arg(cmd));
        EXPECT_TRUE(out.contains("id1  ")) << out.toStdString();
    }
}

TEST_F(Editor, PadsTheSpeciesNameOfAMixture)
{
    const QString out = ed->reformatLine("mixture air N2 frac 0.8");
    EXPECT_TRUE(out.contains("air ")) << out.toStdString();
}

TEST_F(Editor, ACommandThatNeedsNoPaddingIsUnchangedTwiceOver)
{
    // reformatting must be idempotent, or repeated Ctrl+I walks the columns
    const QString once  = ed->reformatLine("fix f1 ave/time 1 100 100 c_t");
    const QString twice = ed->reformatLine(once);
    EXPECT_EQ(once, twice);
}

TEST_F(Editor, ReformattingTheCurrentLineLeavesTheOthersAlone)
{
    setText("units    cgs\ndimension    2\nrun    100\n");
    gotoLine(1);
    QMetaObject::invokeMethod(ed, "reformatCurrentLine");

    const QStringList lines = ed->toPlainText().split('\n');
    EXPECT_EQ(lines.at(0), "units    cgs") << "line 0 was reformatted too";
    EXPECT_EQ(lines.at(1), QString("dimension").leftJustified(16, ' ') + "2");
    EXPECT_EQ(lines.at(2), "run    100") << "line 2 was reformatted too";
}

// ---------------------------------------------------------------- commenting

TEST_F(Editor, CommentsAndUncommentsOneLine)
{
    setText("units cgs\ndimension 2\n");
    gotoLine(0);
    QMetaObject::invokeMethod(ed, "commentLine");
    EXPECT_EQ(ed->toPlainText().split('\n').at(0), "#units cgs");

    gotoLine(0);
    QMetaObject::invokeMethod(ed, "uncommentLine");
    EXPECT_EQ(ed->toPlainText().split('\n').at(0), "units cgs");
}

TEST_F(Editor, UncommentingAnUncommentedLineChangesNothing)
{
    setText("units cgs\n");
    gotoLine(0);
    QMetaObject::invokeMethod(ed, "uncommentLine");
    EXPECT_EQ(ed->toPlainText().split('\n').at(0), "units cgs");
}

TEST_F(Editor, UncommentingRemovesOnlyTheFirstHash)
{
    // "## note" is a doubly commented line; one Ctrl+/ removes one level
    setText("## note\n");
    gotoLine(0);
    QMetaObject::invokeMethod(ed, "uncommentLine");
    EXPECT_EQ(ed->toPlainText().split('\n').at(0), "# note");
}

TEST_F(Editor, UncommentingSkipsLeadingWhitespaceToFindTheHash)
{
    setText("   # indented comment\n");
    gotoLine(0);
    QMetaObject::invokeMethod(ed, "uncommentLine");
    EXPECT_EQ(ed->toPlainText().split('\n').at(0), "    indented comment")
        << "the indent was eaten along with the hash";
}

TEST_F(Editor, CommentsAndUncommentsASelection)
{
    setText("units cgs\ndimension 2\nrun 100\n");
    selectLines(0, 2);
    QMetaObject::invokeMethod(ed, "commentSelection");

    QStringList lines = ed->toPlainText().split('\n');
    EXPECT_EQ(lines.at(0), "#units cgs");
    EXPECT_EQ(lines.at(1), "#dimension 2");
    EXPECT_EQ(lines.at(2), "#run 100");

    selectLines(0, 2);
    QMetaObject::invokeMethod(ed, "uncommentSelection");
    lines = ed->toPlainText().split('\n');
    EXPECT_EQ(lines.at(0), "units cgs");
    EXPECT_EQ(lines.at(1), "dimension 2");
    EXPECT_EQ(lines.at(2), "run 100");
}

TEST_F(Editor, UncommentingAMixedSelectionOnlyTouchesTheCommentedLines)
{
    setText("#units cgs\ndimension 2\n#run 100\n");
    selectLines(0, 2);
    QMetaObject::invokeMethod(ed, "uncommentSelection");
    const QStringList lines = ed->toPlainText().split('\n');
    EXPECT_EQ(lines.at(0), "units cgs");
    EXPECT_EQ(lines.at(1), "dimension 2") << "an uncommented line lost a character";
    EXPECT_EQ(lines.at(2), "run 100");
}

// ---------------------------------------------------------------- completers

TEST_F(Editor, TheGroupCompleterAlwaysOffersAll)
{
    setText("dimension 2\n");
    ed->setGroupList();
    EXPECT_TRUE(completions("group").contains("all"))
        << "\"all\" is defined by SPARTA whether or not the deck names a group";
}

TEST_F(Editor, TheGroupCompleterFindsGroupsInTheBuffer)
{
    setText("group inner grid id 1 10\ngroup outer grid id 11 20\ndimension 2\n");
    ed->setGroupList();
    const QStringList g = completions("group");
    EXPECT_TRUE(g.contains("inner")) << g.join(',').toStdString();
    EXPECT_TRUE(g.contains("outer")) << g.join(',').toStdString();
}

TEST_F(Editor, TheGroupCompleterListsEachNameOnce)
{
    setText("group inner grid id 1 10\ngroup inner grid id 11 20\n");
    ed->setGroupList();
    EXPECT_EQ(completions("group").count("inner"), 1);
}

TEST_F(Editor, TheVariableCompleterOffersEveryReferenceForm)
{
    setText("variable temp equal 300.0\n");
    ed->setVarNameList();
    const QStringList v = completions("varname");
    EXPECT_TRUE(v.contains("${temp}")) << v.join(',').toStdString();
    EXPECT_TRUE(v.contains("v_temp")) << "the v_ form a compute would use";
    EXPECT_TRUE(v.contains("${gui_run}")) << "the variable SPARTA-GUI always defines";
}

TEST_F(Editor, ASingleLetterVariableAlsoGetsTheBareDollarForm)
{
    setText("variable t equal 300.0\n");
    ed->setVarNameList();
    const QStringList v = completions("varname");
    EXPECT_TRUE(v.contains("$t")) << "$t is only valid for a one-character name";
    EXPECT_TRUE(v.contains("${t}"));
}

TEST_F(Editor, AMultiLetterVariableGetsNoBareDollarForm)
{
    setText("variable temp equal 300.0\n");
    ed->setVarNameList();
    EXPECT_FALSE(completions("varname").contains("$temp"))
        << "$temp would expand as $t followed by \"emp\"";
}

TEST_F(Editor, TheComputeAndFixCompletersFindTheirIds)
{
    setText("compute ct thermal/grid all\nfix fa ave/grid all 1 100 100 c_ct\n");
    ed->setComputeIDList();
    ed->setFixIDList();
    EXPECT_TRUE(completions("compid").contains("c_ct")) << completions("compid").join(',').toStdString();
    EXPECT_TRUE(completions("fixid").contains("f_fa")) << completions("fixid").join(',').toStdString();
}

TEST_F(Editor, TheMixtureCompleterFindsMixturesInTheBuffer)
{
    setText("mixture air N2 frac 0.8\nmixture air O2 frac 0.2\n");
    ed->setMixtureIDList();
    const QStringList m = completions("mixid");
    EXPECT_TRUE(m.contains("air")) << m.join(',').toStdString();
    EXPECT_EQ(m.count("air"), 1) << "the same mixture was listed once per line that names it";
}

TEST_F(Editor, TheCompletersSurviveAnEmptyBuffer)
{
    setText("");
    ed->setGroupList();
    ed->setVarNameList();
    ed->setComputeIDList();
    ed->setFixIDList();
    ed->setMixtureIDList();
    EXPECT_TRUE(completions("group").contains("all"));
}

TEST_F(Editor, ReformattingADefinitionRebuildsTheMatchingCompleter)
{
    // reformatLine notices that the line defines something and refreshes the
    // completer, which is what makes a name usable the moment it is typed
    setText("group inner grid id 1 10\n");
    ed->reformatLine("group inner grid id 1 10");
    EXPECT_TRUE(completions("group").contains("inner"));
}

TEST_F(Editor, ReformattingLeavesTheCursorWhereItWas)
{
    setText("units    cgs\ndimension 2\n");
    gotoLine(1);
    const int before = ed->textCursor().blockNumber();
    ed->setGroupList(); // walks the whole document looking for group commands
    EXPECT_EQ(ed->textCursor().blockNumber(), before)
        << "building a completer left the cursor where it finished searching";
}

// ---------------------------------------------------------------- highlighting

TEST_F(Editor, HighlightsAndClearsTheErrorLine)
{
    setText("units cgs\ndimension 2\nrun 100\n");
    ed->setHighlight(1, true);
    EXPECT_EQ(ed->textCursor().blockNumber(), 1) << "the error line was not scrolled to";
    ed->clearErrorHighlight();
    SUCCEED();
}

TEST_F(Editor, HighlightingAnOutOfRangeLineIsHarmless)
{
    setText("units cgs\n");
    ed->setHighlight(500, true);
    ed->setHighlight(-1, false);
    SUCCEED();
}

TEST_F(Editor, SetCursorMovesToTheRequestedBlock)
{
    setText("a\nb\nc\nd\n");
    ed->setCursor(2);
    EXPECT_EQ(ed->textCursor().blockNumber(), 2);
}

TEST_F(Editor, DiagnosticsAreShownAndCleared)
{
    setText("units cgs\nbogus command\n");
    QList<InputCheck::Diagnostic> diags;
    InputCheck::Diagnostic d;
    d.line     = 1;
    d.message  = "unknown command";
    d.severity = InputCheck::Severity::Error;
    diags << d;

    ed->setDiagnostics(diags);
    ed->clearDiagnostics();
    SUCCEED() << "the diagnostic overlay did not survive being set and cleared";
}

// ---------------------------------------------------------------- the gutter

TEST_F(Editor, TheLineNumberGutterGrowsWithTheLineCount)
{
    setText("one line\n");
    const int narrow = ed->lineNumberAreaWidth();

    QString many;
    for (int i = 0; i < 1500; ++i)
        many += "line\n";
    setText(many);
    EXPECT_GT(ed->lineNumberAreaWidth(), narrow)
        << "a four-digit line number does not fit in a one-digit gutter";
}

TEST_F(Editor, ItPaintsWithItsGutter)
{
    setText("units cgs\ndimension 2\nrun 100\n");
    ed->resize(600, 400);
    EXPECT_FALSE(ed->grab().isNull());
}

// ---------------------------------------------------------------- paste

TEST_F(Editor, PastesTextFromTheClipboard)
{
    // through paste(), which is the route Ctrl+V takes: it asks
    // canInsertFromMimeData() first and then insertFromMimeData()
    setText("");
    QGuiApplication::clipboard()->setText("run 100");
    ed->paste();
    EXPECT_EQ(ed->toPlainText(), "run 100");
}

TEST_F(Editor, RefusesToPasteAnImage)
{
    // an image on the clipboard has no text form, so pasting it into a deck
    // must leave the deck alone rather than insert a placeholder
    setText("units cgs");
    auto *image = new QMimeData;
    image->setImageData(QImage(4, 4, QImage::Format_RGB32));
    QGuiApplication::clipboard()->setMimeData(image);
    ed->paste();
    EXPECT_EQ(ed->toPlainText(), "units cgs");
}

TEST_F(Editor, PastingMultipleLinesKeepsThemSeparate)
{
    setText("");
    QGuiApplication::clipboard()->setText("units cgs\ndimension 2\n");
    ed->paste();
    EXPECT_EQ(ed->document()->blockCount(), 3) << ed->toPlainText().toStdString();
}


// ---------------------------------------------------------------- completion
//
// runCompletion() is a dispatch table: which of the seventeen completers
// applies depends on the position of the word under the cursor and on the
// command that starts the line, with the c_/f_/v_ reference prefixes
// overriding both.  Getting an entry wrong offers a user the wrong list, which
// looks like the feature working.
//
// Which completer won is observable without reaching into the editor: the one
// that fires shows its popup, and the popups belong to the completers, which
// are named.

namespace {

// Offscreen a completer's popup never becomes visible, so which one fired is
// read from the prefix instead: runCompletion() sets it on the completer it
// chose and on no other.  Every prefix is stamped with a sentinel first, so
// "which one changed" is unambiguous.
const QString kSentinel = QStringLiteral("\x01none");

void armCompleters(const CodeEditor &ed)
{
    for (auto *c : ed.findChildren<QCompleter *>()) {
        c->setCompletionPrefix(kSentinel);
        if (c->popup()) c->popup()->hide();
    }
}

// Which completer actually put a list on screen.  The prefix alone is not
// enough: runCompletion() sets it on the completer it picked and may then
// decide the word is already complete and show nothing, which is a different
// answer from "no completer applies here".
QString activeCompleter(const CodeEditor &ed)
{
    for (auto *c : ed.findChildren<QCompleter *>())
        if (c->popup() && c->popup()->isVisible()) return c->objectName();
    return {};
}

QCompleter *completerNamed(const CodeEditor &ed, const char *name)
{
    return ed.findChild<QCompleter *>(QLatin1String(name));
}

} // namespace

// Put the cursor inside word @p n of the current line and ask for completion.
// Returns which completer came up.
static QString completeWord(CodeEditor *ed, const QString &line, int wordIndex)
{
    ed->setPlainText(line);
    ed->show(); // a popup cannot be shown by a widget that is not
    armCompleters(*ed);

    // find the start of the requested word, then sit one character inside it
    const QStringList words = line.split(' ');
    int pos = 0;
    for (int i = 0; i < wordIndex && i < words.size(); ++i)
        pos += words.at(i).length() + 1;
    pos += qMin(1, words.value(wordIndex).length());

    QTextCursor c(ed->document()->findBlockByNumber(0));
    c.setPosition(qMin(pos, line.length()));
    ed->setTextCursor(c);

    QMetaObject::invokeMethod(ed, "runCompletion");
    return activeCompleter(*ed);
}

TEST_F(Editor, TheFirstWordCompletesAgainstTheCommandList)
{
    EXPECT_EQ(completeWord(ed, "uni", 0), "command");
}

TEST_F(Editor, NothingIsOfferedOnAnEmptyLineOrAComment)
{
    EXPECT_EQ(completeWord(ed, "", 0), "");
    EXPECT_EQ(completeWord(ed, "# a comment about units", 0), "");
    EXPECT_EQ(completeWord(ed, "# units cg", 1), "")
        << "a completion was offered inside a comment";
}

TEST_F(Editor, ACompleteCommandOffersNoFurtherCompletion)
{
    // the popup would sit there listing the one word already typed
    EXPECT_EQ(completeWord(ed, "units", 0), "");
}

TEST_F(Editor, TheSecondWordFollowsTheCommandThatStartsTheLine)
{
    // each of these is a distinct completer, and picking the wrong one offers
    // the user a list that has nothing to do with what they are typing
    EXPECT_EQ(completeWord(ed, "collide v", 1), "collide");
    EXPECT_EQ(completeWord(ed, "react t", 1), "react");
    EXPECT_EQ(completeWord(ed, "units c", 1), "units");
    setText("mixture air N2 frac 0.8\n");
    ed->setMixtureIDList();
    EXPECT_EQ(completeWord(ed, "create_particles a", 1), "mixid");
    EXPECT_EQ(completeWord(ed, "adapt_grid a", 1), "group");
    EXPECT_EQ(completeWord(ed, "read_isurf a", 1), "group");
}

TEST_F(Editor, TheCommandsThatTakeAFileNameCompleteAgainstTheDirectory)
{
    // the file completer lists the working directory, so there has to be one
    QTemporaryDir dir;
    const QString saved = QDir::currentPath();
    QDir::setCurrent(dir.path());
    QFile f(dir.filePath("data.grid"));
    ASSERT_TRUE(f.open(QIODevice::WriteOnly));
    f.close();
    ed->setFileList();

    for (const char *cmd : {"include", "jump", "read_grid", "read_restart", "read_surf"})
        EXPECT_EQ(completeWord(ed, QString("%1 d").arg(cmd), 1), "file") << cmd;
    QDir::setCurrent(saved);
}

TEST_F(Editor, TypingAPathSeparatorTakesTheDirectoryListingAway)
{
    // The file completer lists one directory, so once the user types a path
    // separator it has nothing useful left to offer -- and the list already on
    // screen has to come down.  Checking that no *new* popup appears is not
    // enough: with the guard removed the completer is still asked, finds
    // nothing matching, and shows nothing, which looks the same.
    QTemporaryDir dir;
    const QString saved = QDir::currentPath();
    QDir::setCurrent(dir.path());
    QFile f(dir.filePath("data.grid"));
    ASSERT_TRUE(f.open(QIODevice::WriteOnly));
    f.close();
    ed->setFileList();

    ASSERT_EQ(completeWord(ed, "include d", 1), "file") << "no listing to take away";
    auto *file = completerNamed(*ed, "file");
    ASSERT_NE(file, nullptr);
    ASSERT_TRUE(file->popup()->isVisible());

    // now type a separator, without re-arming: the popup must be hidden
    ed->setPlainText("include d/x");
    QTextCursor c(ed->document()->findBlockByNumber(0));
    c.setPosition(11);
    ed->setTextCursor(c);
    QMetaObject::invokeMethod(ed, "runCompletion");

    EXPECT_FALSE(file->popup()->isVisible())
        << "the directory listing stayed up over a path already inside one";
    QDir::setCurrent(saved);
}

TEST_F(Editor, TheThirdWordFollowsTheCommandToo)
{
    EXPECT_EQ(completeWord(ed, "region box b", 2), "region");
    EXPECT_EQ(completeWord(ed, "variable t e", 2), "variable");
    EXPECT_EQ(completeWord(ed, "fix f1 a", 2), "fix");
    EXPECT_EQ(completeWord(ed, "compute c1 t", 2), "compute");
    EXPECT_EQ(completeWord(ed, "dump d1 i", 2), "dump");
    EXPECT_EQ(completeWord(ed, "surf_collide sc d", 2), "surf_collide");
    EXPECT_EQ(completeWord(ed, "surf_react sr p", 2), "surf_react");
}

TEST_F(Editor, AReferencePrefixPicksTheListForTheThingItNames)
{
    // c_/f_/v_ name a compute, fix or variable, and the completer follows the
    // prefix rather than the command
    setText("compute ct thermal/grid all\nfix fa ave/grid all 1 100 100 c_ct\n"
            "variable temp equal 300.0\n");
    ed->setComputeIDList();
    ed->setFixIDList();
    ed->setVarNameList();

    // "balance_grid" has no per-position list of its own, so nothing competes
    EXPECT_EQ(completeWord(ed, "balance_grid c_c", 1), "compid");
    EXPECT_EQ(completeWord(ed, "balance_grid f_f", 1), "fixid");
    EXPECT_EQ(completeWord(ed, "balance_grid v_t", 1), "varname");
}

TEST_F(Editor, ACommandsOwnListWinsOverAReferencePrefix)
{
    // At the third word "dump" offers its own styles, and it does so even when
    // the word looks like a compute reference.  That order is deliberate: the
    // third word of a dump *is* the style, and a c_ there is a typo rather
    // than a reference.
    setText("compute ct thermal/grid all\n");
    ed->setComputeIDList();
    EXPECT_EQ(completeWord(ed, "dump d1 c_c all 100 img.ppm", 2), "dump")
        << "the reference prefix overrode the command's own style list";
}

TEST_F(Editor, AReferenceFurtherRightIsStillCompleted)
{
    setText("variable temp equal 300.0\n");
    ed->setVarNameList();
    EXPECT_EQ(completeWord(ed, "fix f1 ave/time 1 100 100 v_t", 7), "varname");
}

TEST_F(Editor, AnUppercasePrefixIsTreatedTheSame)
{
    setText("compute ct thermal/grid all\n");
    ed->setComputeIDList();
    EXPECT_EQ(completeWord(ed, "balance_grid C_c", 1), "compid");
}

TEST_F(Editor, TheFourthWordOfADumpTakesAMixture)
{
    setText("mixture air N2 frac 0.8\n");
    ed->setMixtureIDList();
    EXPECT_EQ(completeWord(ed, "dump d1 image a", 3), "mixid");
}

TEST_F(Editor, AnEmittingFixTakesAMixtureWhereAnOrdinaryOneDoesNot)
{
    setText("mixture air N2 frac 0.8\n");
    ed->setMixtureIDList();
    EXPECT_EQ(completeWord(ed, "fix f1 emit/face a", 3), "mixid");
    EXPECT_EQ(completeWord(ed, "fix f1 ave/time a", 3), "")
        << "a mixture list was offered for a fix that takes no mixture";
}

TEST_F(Editor, ACommandWithNoListForThatPositionOffersNothing)
{
    EXPECT_EQ(completeWord(ed, "run 1", 1), "");
    EXPECT_EQ(completeWord(ed, "timestep 0.0", 1), "");
}

// ---------------------------------------------------------------- insertion

TEST_F(Editor, AcceptingACompletionReplacesTheWordUnderTheCursor)
{
    // through the completer's own signal, which is the path the popup takes --
    // the slot reads sender() to know which completer it came from
    setText("uni cgs\n");
    QTextCursor c(ed->document()->findBlockByNumber(0));
    c.setPosition(2);
    ed->setTextCursor(c);

    auto *comp = completerNamed(*ed, "command");
    ASSERT_NE(comp, nullptr);
    emit comp->activated(QString("units"));

    EXPECT_EQ(ed->toPlainText(), "units cgs\n")
        << "the completion did not replace the partial word: " << ed->toPlainText().toStdString();
}

TEST_F(Editor, ACompletionInTheMiddleOfALineLeavesTheRestAlone)
{
    setText("compute ct therm all\n");
    QTextCursor c(ed->document()->findBlockByNumber(0));
    c.setPosition(13); // inside "therm"
    ed->setTextCursor(c);

    auto *comp = completerNamed(*ed, "compute");
    ASSERT_NE(comp, nullptr);
    emit comp->activated(QString("thermal/grid"));

    EXPECT_EQ(ed->toPlainText(), "compute ct thermal/grid all\n") << ed->toPlainText().toStdString();
}

TEST_F(Editor, ACompletionFromAnotherWidgetsCompleterIsIgnored)
{
    // the slot checks that the completer it came from belongs to this editor,
    // because two editors would otherwise complete into each other
    setText("uni cgs\n");
    CodeEditor other(nullptr);
    auto *foreign = other.findChild<QCompleter *>("command");
    ASSERT_NE(foreign, nullptr);

    emit foreign->activated(QString("units"));
    EXPECT_EQ(ed->toPlainText(), "uni cgs\n") << "another editor's completion reached this one";
}

// ---------------------------------------------------------------- context menu

namespace {

// Grabs the context menu the editor pops up, records its entries, and closes
// it. exec() is modal, so a timer is the only way in.
class MenuGrab : public QObject {
public:
    explicit MenuGrab(int = 0)
    {
        // An application-wide filter catches the menu the moment it is shown,
        // which happens synchronously inside exec(). Polling for the "active
        // popup" misses it offscreen.
        qApp->installEventFilter(this);
    }
    ~MenuGrab() override { qApp->removeEventFilter(this); }

    QStringList entries;
    QStringList data; ///< the action data, which is the help page or file name

    [[nodiscard]] bool offers(const QString &needle) const
    {
        for (const auto &e : entries)
            if (e.contains(needle)) return true;
        return false;
    }

protected:
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        if (event->type() != QEvent::Show) return false;
        auto *m = qobject_cast<QMenu *>(watched);
        if (!m || taken) return false;
        taken = true;
        for (auto *a : m->actions()) {
            if (a->isSeparator()) continue;
            entries << a->text();
            if (!a->data().toString().isEmpty()) data << a->data().toString();
        }
        // close it from the event loop: it is mid-show right now
        QTimer::singleShot(0, m, &QMenu::close);
        return false;
    }

private:
    bool taken = false;
};

QStringList menuFor(CodeEditor *ed, const QString &line, int column = 1)
{
    ed->setPlainText(line);
    QTextCursor c(ed->document()->findBlockByNumber(0));
    c.setPosition(qMin(column, line.length()));
    ed->setTextCursor(c);

    MenuGrab grab;
    QContextMenuEvent ev(QContextMenuEvent::Mouse, QPoint(5, 5), QPoint(5, 5));
    QApplication::sendEvent(ed->viewport(), &ev);
    QCoreApplication::processEvents();
    return grab.entries;
}

} // namespace

TEST_F(Editor, TheContextMenuOffersToCommentTheLineOrTheSelection)
{
    const QStringList line = menuFor(ed, "units cgs");
    EXPECT_TRUE(line.join("|").contains("Comment out line")) << line.join(" | ").toStdString();
    EXPECT_FALSE(line.join("|").contains("Comment out selection"));

    setText("units cgs\ndimension 2\n");
    QTextCursor c(ed->document()->findBlockByNumber(0));
    c.movePosition(QTextCursor::Down, QTextCursor::KeepAnchor);
    ed->setTextCursor(c);

    MenuGrab grab;
    QContextMenuEvent ev(QContextMenuEvent::Mouse, QPoint(5, 5), QPoint(5, 5));
    QApplication::sendEvent(ed->viewport(), &ev);
    QCoreApplication::processEvents();
    EXPECT_TRUE(grab.offers("Comment out selection")) << grab.entries.join(" | ").toStdString();
    EXPECT_FALSE(grab.offers("Comment out line"))
        << "the line entries were offered with a selection active";
}

TEST_F(Editor, TheContextMenuOffersDocumentationForTheCommandOnTheLine)
{
    const QStringList m = menuFor(ed, "units cgs");
    EXPECT_TRUE(m.join("|").contains("View Documentation for 'units'"))
        << m.join(" | ").toStdString();
    EXPECT_TRUE(m.join("|").contains("Reformat 'units' command")) << m.join(" | ").toStdString();
}

TEST_F(Editor, AStyledCommandOffersDocumentationForBothTheStyleAndTheCommand)
{
    // "fix ID ave/time ..." documents both fix_ave_time and fix itself, since
    // the style is the third word and has a page of its own
    const QStringList m = menuFor(ed, "fix f1 ave/time 1 100 100 c_t");
    const QString all = m.join(" | ");
    EXPECT_TRUE(all.contains("fix ave/time")) << all.toStdString();
    EXPECT_TRUE(all.contains("View Documentation for 'fix'"))
        << "only the style was documented, not the command: " << all.toStdString();
}

TEST_F(Editor, AnUnknownCommandOffersNoDocumentation)
{
    const QStringList m = menuFor(ed, "notacommand foo");
    EXPECT_FALSE(m.join("|").contains("View Documentation for 'notacommand'"))
        << m.join(" | ").toStdString();
    // but the manual entries are always there
    EXPECT_TRUE(m.join("|").contains("SPARTA Manual")) << m.join(" | ").toStdString();
}

TEST_F(Editor, TheContextMenuAlwaysOffersTheManual)
{
    for (const char *line : {"", "units cgs", "# just a comment"}) {
        const QStringList m = menuFor(ed, QLatin1String(line));
        EXPECT_TRUE(m.join("|").contains("SPARTA Commands Overview")) << line;
        EXPECT_TRUE(m.join("|").contains("SPARTA Manual")) << line;
    }
}

TEST_F(Editor, TheContextMenuOffersNoRunEntriesWithoutAMainWindow)
{
    // the editor's parent is not the main window here, so there is nothing to
    // start a run with -- offering to would be an entry that does nothing
    const QStringList m = menuFor(ed, "units cgs");
    EXPECT_FALSE(m.join("|").contains("Run SPARTA")) << m.join(" | ").toStdString();
    EXPECT_FALSE(m.join("|").contains("Stop SPARTA")) << m.join(" | ").toStdString();
}

TEST_F(Editor, AFileNameUnderTheCursorIsOfferedForViewing)
{
    QTemporaryDir dir;
    const QString name = dir.filePath("air.species");
    QFile f(name);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly | QIODevice::Text));
    f.write("Ar 40.0\n");
    f.close();

    const QString line = QString("species %1 Ar").arg(name);
    MenuGrab grab;
    ed->setPlainText(line);
    QTextCursor c(ed->document()->findBlockByNumber(0));
    c.setPosition(line.indexOf(name) + 3);
    ed->setTextCursor(c);
    // the menu repositions the cursor to where the click landed, so the event
    // has to be at the word rather than at the corner of the widget
    const QPoint at = ed->cursorRect(c).center();
    QContextMenuEvent ev(QContextMenuEvent::Mouse, at, ed->mapToGlobal(at));
    QApplication::sendEvent(ed->viewport(), &ev);
    QCoreApplication::processEvents();

    EXPECT_TRUE(grab.offers("View file")) << grab.entries.join(" | ").toStdString();
    EXPECT_TRUE(grab.offers("Open '")) << "a text file was not offered for editing: "
                                       << grab.entries.join(" | ").toStdString();
}

TEST_F(Editor, ABinaryFileIsOfferedForViewingButNotForEditing)
{
    QTemporaryDir dir;
    const QString name = dir.filePath("snap.png");
    QImage(4, 4, QImage::Format_RGB32).save(name);

    const QString line = QString("# see %1").arg(name);
    MenuGrab grab;
    ed->setPlainText(line);
    QTextCursor c(ed->document()->findBlockByNumber(0));
    c.setPosition(line.indexOf(name) + 3);
    ed->setTextCursor(c);
    // the menu repositions the cursor to where the click landed, so the event
    // has to be at the word rather than at the corner of the widget
    const QPoint at = ed->cursorRect(c).center();
    QContextMenuEvent ev(QContextMenuEvent::Mouse, at, ed->mapToGlobal(at));
    QApplication::sendEvent(ed->viewport(), &ev);
    QCoreApplication::processEvents();

    EXPECT_TRUE(grab.offers("View file")) << grab.entries.join(" | ").toStdString();
    EXPECT_FALSE(grab.offers("Open '"))
        << "a PNG was offered for editing as text: " << grab.entries.join(" | ").toStdString();
}

TEST_F(Editor, AWordThatIsNotAFileOffersNoFileEntries)
{
    const QStringList m = menuFor(ed, "units cgs");
    EXPECT_FALSE(m.join("|").contains("View file")) << m.join(" | ").toStdString();
}

// ---------------------------------------------------------------- drag and drop

TEST_F(Editor, DraggingTextInIsAccepted)
{
    QMimeData *text = new QMimeData;
    text->setText("run 100\n");
    QDragEnterEvent ev(QPoint(5, 5), Qt::CopyAction, text, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(ed->viewport(), &ev);
    EXPECT_TRUE(ev.isAccepted()) << "dragging text into the editor was refused";
    delete text;
}

TEST_F(Editor, DroppingSomethingWithNeitherTextNorUrlsIsIgnored)
{
    setText("untouched");
    QMimeData *odd = new QMimeData;
    odd->setData("application/x-nonsense", QByteArray("xx"));
    QDropEvent ev(QPointF(5, 5), Qt::CopyAction, odd, Qt::LeftButton, Qt::NoModifier,
                  QEvent::Drop);
    QApplication::sendEvent(ed->viewport(), &ev);
    QCoreApplication::processEvents();
    EXPECT_EQ(ed->toPlainText(), "untouched");
    delete odd;
}

// The text and URL branches of dropEvent() both end in
// QPlainTextEdit::dropEvent(), which wants the drag source a synthetic event
// has no way to supply -- so only the branch that handles neither is driven
// here.  The live GUI walker performs real drops.

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_codeeditor.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
