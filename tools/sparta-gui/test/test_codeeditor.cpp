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

    void SetUp() override { ed = new CodeEditor(nullptr); }
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
