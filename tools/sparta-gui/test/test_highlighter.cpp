// Unit tests for the SPARTA input-script syntax highlighter (src/highlighter.cpp).
//
// The highlighter is one of the few files that was rewritten rather than
// renamed when this application was adapted, because the grammar it colours is
// SPARTA's and not the one it came with. It had no tests, and the failure mode
// it is exposed to is quiet: a rule that stops matching leaves the token
// black, which looks like an ordinary unhighlighted word rather than like a
// bug. A rule that matches too much is worse, because the colour is then
// actively wrong about what the token is.
//
// The formats a QSyntaxHighlighter applies end up on each block's text layout,
// so they can be read back out and compared: this checks both that a token is
// coloured and that categories which mean different things are coloured
// differently.

#include "highlighter.h"

#include <QApplication>
#include <QColor>
#include <QSettings>
#include <QTextBlock>
#include <QTextDocument>
#include <QTextLayout>

#include "gtest/gtest.h"

#include <memory>

namespace {

/// A document with the highlighter attached and @p text highlighted.
class Highlighted {
public:
    explicit Highlighted(const QString &text)
    {
        doc = std::make_unique<QTextDocument>();
        hl  = std::make_unique<Highlighter>(doc.get());
        doc->setPlainText(text);
        // setPlainText highlights synchronously for a document with no view
        doc->documentLayout();
    }

    /// Foreground colour applied at character @p pos of line @p line, or an
    /// invalid colour where the highlighter left the text alone.
    QColor colorAt(int line, int pos) const
    {
        const QTextBlock block = doc->findBlockByNumber(line);
        if (!block.isValid() || !block.layout()) return {};
        for (const QTextLayout::FormatRange &r : block.layout()->formats())
            if (pos >= r.start && pos < r.start + r.length && r.format.foreground().style() !=
                                                                  Qt::NoBrush)
                return r.format.foreground().color();
        return {};
    }

    /// Colour applied to the first occurrence of @p token on line @p line.
    QColor colorOf(int line, const QString &token) const
    {
        const QTextBlock block = doc->findBlockByNumber(line);
        if (!block.isValid()) return {};
        const int at = block.text().indexOf(token);
        if (at < 0) return {};
        return colorAt(line, at);
    }

    int formatCount(int line) const
    {
        const QTextBlock block = doc->findBlockByNumber(line);
        return (block.isValid() && block.layout()) ? int(block.layout()->formats().size()) : 0;
    }

private:
    std::unique_ptr<QTextDocument> doc;
    std::unique_ptr<Highlighter> hl;
};

// ---------------------------------------------------------------------------
// Tokens get coloured
// ---------------------------------------------------------------------------

TEST(Highlighter, ColorsEachCommandFamily)
{
    // One line per family the rules distinguish. Every command word has to be
    // coloured; a family whose rule stopped matching leaves it plain, which is
    // indistinguishable from an argument.
    const Highlighted h(
        "units si\n"                             // 0 lattice/setup
        "create_box 0 1 0 1 0 1\n"               // 1 box creation
        "boundary o o p\n"                       // 2 boundary
        "stats 100\n"                            // 3 output
        "dump_modify 1 pad 4\n"                  // 4 output (two-argument form)
        "read_surf data.circle\n"                // 5 input
        "compute 1 grid all n\n"                 // 6 styled definition
        "collide vss air air.vss\n"              // 7 collision model
        "variable x equal 3.0\n"                 // 8 define
        "uncompute 1\n"                          // 9 undo
        "species air.species N2 O2\n"            // 10 particle
        "run 1000\n"                             // 11 run
        "clear\n"                                // 12 setup
        "reset_timestep 0\n");                   // 13 setup with an argument

    const char *first[] = {"units",     "create_box", "boundary",  "stats",
                           "dump_modify", "read_surf", "compute",  "collide",
                           "variable",  "uncompute",  "species",   "run",
                           "clear",     "reset_timestep"};
    for (int line = 0; line < 14; ++line) {
        const QColor c = h.colorOf(line, QString::fromLatin1(first[line]));
        EXPECT_TRUE(c.isValid()) << "'" << first[line] << "' (line " << line
                                 << ") was left unhighlighted";
    }
}

TEST(Highlighter, CommandFamiliesAreToldApart)
{
    // Colouring everything the same would pass every "is it coloured" check
    // above while telling the reader nothing. These four families mean
    // genuinely different things -- setting up the box, writing output,
    // reading input, running -- and must not share a colour.
    const Highlighted h(
        "units si\n"
        "stats 100\n"
        "read_surf data.circle\n"
        "run 1000\n");

    const QColor setup  = h.colorOf(0, "units");
    const QColor output = h.colorOf(1, "stats");
    const QColor input  = h.colorOf(2, "read_surf");
    const QColor run    = h.colorOf(3, "run");
    ASSERT_TRUE(setup.isValid() && output.isValid() && input.isValid() && run.isValid());

    EXPECT_NE(setup.name(), output.name());
    EXPECT_NE(output.name(), input.name());
    EXPECT_NE(input.name(), run.name());
}

TEST(Highlighter, ColorsNumbersApartFromCommands)
{
    const Highlighted h("timestep 1.0e-7\n"
                        "run 1000\n"
                        "region box block -1 1 -1 1 -0.5 0.5\n");

    const QColor cmd = h.colorOf(0, "timestep");
    const QColor num = h.colorOf(0, "1.0e-7");
    ASSERT_TRUE(cmd.isValid());
    ASSERT_TRUE(num.isValid()) << "a floating point number in exponent form was not coloured";
    EXPECT_NE(cmd.name(), num.name());

    EXPECT_TRUE(h.colorOf(1, "1000").isValid()) << "a plain integer was not coloured";
    EXPECT_TRUE(h.colorOf(2, "-0.5").isValid()) << "a negative decimal was not coloured";
}

TEST(Highlighter, ColorsVariableUsesAndComputeReferences)
{
    const Highlighted h("variable t equal 300\n"
                        "fix 1 emit/face all $t\n"
                        "stats_style step c_temp f_ave v_t\n");

    EXPECT_TRUE(h.colorOf(1, "$t").isValid()) << "a $-substitution was not coloured";
    for (const char *ref : {"c_temp", "f_ave", "v_t"})
        EXPECT_TRUE(h.colorOf(2, QString::fromLatin1(ref)).isValid())
            << ref << " (a compute/fix/variable reference) was not coloured";
}

TEST(Highlighter, ColorsCommentsToTheEndOfTheLine)
{
    const Highlighted h("run 100   # how long to run for\n");
    const QColor comment = h.colorOf(0, "#");
    ASSERT_TRUE(comment.isValid());
    // the whole trailing text is one colour, including words that would
    // otherwise match a command rule
    EXPECT_EQ(h.colorOf(0, "how").name(), comment.name());
    EXPECT_EQ(h.colorOf(0, "run for").name(), comment.name());
    // and the command before it keeps its own
    EXPECT_NE(h.colorAt(0, 0).name(), comment.name());
}

TEST(Highlighter, AFullLineCommentIsAllComment)
{
    const Highlighted h("# SPARTA input for a circle flow\n");
    const QColor c = h.colorAt(0, 0);
    ASSERT_TRUE(c.isValid());
    EXPECT_EQ(h.colorAt(0, 20).name(), c.name());
}

TEST(Highlighter, ColorsQuotedStrings)
{
    const Highlighted h("print \"the run finished\"\n");
    const QColor s = h.colorOf(0, "\"the");
    ASSERT_TRUE(s.isValid()) << "a quoted string was not coloured";
    EXPECT_NE(s.name(), h.colorOf(0, "print").name());
}

TEST(Highlighter, AHashInsideQuotesIsNotAComment)
{
    // shell and print commands legitimately carry a '#'; treating it as a
    // comment would grey out the rest of a line that is really an argument
    const Highlighted h("print \"count # of particles\"\n");
    const QColor quoted = h.colorOf(0, "\"count");
    const Highlighted ref("# a real comment\n");
    const QColor comment = ref.colorAt(0, 0);
    ASSERT_TRUE(quoted.isValid());
    ASSERT_TRUE(comment.isValid());
    EXPECT_NE(quoted.name(), comment.name());
}

TEST(Highlighter, ColorsSpecialWords)
{
    const Highlighted h("create_particles air n 0 nrho 1.0 INF\n"
                        "region r block INF INF EDGE EDGE 0 1\n");
    EXPECT_TRUE(h.colorOf(0, "INF").isValid()) << "INF was not coloured";
    EXPECT_TRUE(h.colorOf(1, "EDGE").isValid()) << "EDGE was not coloured";
}

TEST(Highlighter, LeavesOrdinaryWordsAlone)
{
    // a word that is not part of the grammar must not pick up a colour, or
    // every misspelled command looks correct
    const Highlighted h("notacommand withanargument\n");
    EXPECT_FALSE(h.colorAt(0, 0).isValid());
}

TEST(Highlighter, IndentedCommandsAreStillCommands)
{
    // continuation blocks and loop bodies are commonly indented
    const Highlighted h("    run 1000\n");
    EXPECT_TRUE(h.colorOf(0, "run").isValid());
}

TEST(Highlighter, AnEmptyLineProducesNoFormats)
{
    const Highlighted h("\nrun 10\n");
    EXPECT_EQ(h.formatCount(0), 0);
}

// ---------------------------------------------------------------------------
// Colour schemes
// ---------------------------------------------------------------------------

TEST(HighlighterSchemes, IdsAndLabelsLineUp)
{
    const QStringList ids    = Highlighter::schemeIds();
    const QStringList labels = Highlighter::schemeLabels();
    EXPECT_FALSE(ids.isEmpty());
    EXPECT_EQ(ids.size(), labels.size())
        << "the preferences combo shows labels and stores ids by index; "
           "different lengths mean it stores the wrong one";
    for (const QString &l : labels) EXPECT_FALSE(l.isEmpty());
    EXPECT_TRUE(ids.contains(Highlighter::defaultScheme()));
}

TEST(HighlighterSchemes, EachSchemeIsDistinct)
{
    // two schemes that produce identical colours are one scheme with two names
    QStringList seen;
    for (const QString &id : Highlighter::schemeIds()) {
        QTextDocument doc;
        doc.documentLayout();   // block layouts, and so formats, need one
        Highlighter hl(&doc);
        hl.applyScheme(id);
        doc.setPlainText("units si\nstats 100\nrun 1000\n");

        QStringList colors;
        for (int line = 0; line < 3; ++line) {
            const QTextBlock b = doc.findBlockByNumber(line);
            if (b.layout() && !b.layout()->formats().isEmpty())
                colors << b.layout()->formats().first().format.foreground().color().name();
        }
        const QString signature = colors.join(',');
        EXPECT_FALSE(seen.contains(signature)) << "scheme '" << id.toStdString()
                                               << "' is a duplicate of an earlier one";
        seen << signature;
    }
}

TEST(HighlighterSchemes, AnUnknownSchemeFallsBackInsteadOfClearing)
{
    QTextDocument doc;
    doc.documentLayout();
    Highlighter hl(&doc);
    hl.applyScheme(QStringLiteral("no-such-scheme"));
    doc.setPlainText("run 1000\n");

    const QTextBlock b = doc.findBlockByNumber(0);
    ASSERT_TRUE(b.layout());
    EXPECT_FALSE(b.layout()->formats().isEmpty())
        << "an unrecognized stored preference left the editor with no highlighting at all";
}

TEST(HighlighterSchemes, BackgroundAndForegroundContrastWhereTheyAreDefined)
{
    // a scheme that defines its own editor colours has to define both, and
    // they have to differ -- text the colour of its background is invisible
    for (const QString &id : Highlighter::schemeIds()) {
        for (bool light : {true, false}) {
            const QColor bg = Highlighter::schemeBackground(id, light);
            const QColor fg = Highlighter::schemeForeground(id, light);
            if (!bg.isValid() && !fg.isValid()) continue; // follows the app theme
            EXPECT_TRUE(bg.isValid() && fg.isValid())
                << "scheme '" << id.toStdString() << "' defines one of background/foreground "
                << "but not the other, so the missing one comes from the app theme and may "
                << "match the one it is drawn on";
            if (bg.isValid() && fg.isValid())
                EXPECT_GT(std::abs(bg.lightness() - fg.lightness()), 40)
                    << "scheme '" << id.toStdString() << "' (light=" << light
                    << ") has too little contrast between text and background";
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    // The highlighter reads the stored colour-scheme preference at
    // construction; keep the tests off whatever the developer has set.
    QCoreApplication::setOrganizationName("SPARTA-GUI-tests");
    QCoreApplication::setApplicationName("highlighter-tests");
    QSettings().clear();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
