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

// The light/dark appearance.
//
// Small, untested, and applied exactly once in main() before the main window
// exists -- so anything wrong with it is wrong for the whole session and there
// is no second chance to notice. The mode is persisted as a string, which makes
// the round trip through the settings file the part that matters: a mode that
// does not survive being written and read back silently resets everyone's
// preference to System on the next launch.
//
// apply() is checked for what it is documented to do (install a palette, leave
// the style alone) rather than for particular colours, which are a design
// choice and not a contract.

#include <gtest/gtest.h>

#include <QApplication>
#include <QPalette>
#include <QStyle>
#include <QRegularExpression>
#include <QStyleFactory>

#include "theme.h"

using Theme::Mode;

// ------------------------------------------------------------- the mode name

TEST(Theme, EveryModeSurvivesTheRoundTripThroughItsName)
{
    for (Mode m : {Mode::System, Mode::Light, Mode::Dark})
        EXPECT_EQ(Theme::modeFromString(Theme::modeToString(m)), m)
            << "the mode written to the settings file does not read back as itself, so the "
               "user's choice is lost at the next launch";
}

TEST(Theme, TheNamesAreTheDocumentedOnes)
{
    // These strings are in the settings file of everyone who has ever run the
    // application; renaming one silently resets their preference.
    EXPECT_EQ(Theme::modeToString(Mode::System), QString("system"));
    EXPECT_EQ(Theme::modeToString(Mode::Light), QString("light"));
    EXPECT_EQ(Theme::modeToString(Mode::Dark), QString("dark"));
}

TEST(Theme, TheNameIsReadWithoutRegardToCaseOrSurroundingSpace)
{
    EXPECT_EQ(Theme::modeFromString("Dark"), Mode::Dark);
    EXPECT_EQ(Theme::modeFromString("  dark  "), Mode::Dark);
    EXPECT_EQ(Theme::modeFromString("LIGHT"), Mode::Light);
}

TEST(Theme, AnythingUnrecognisedIsSystemRatherThanAFailure)
{
    // A settings file from a newer version, or a hand-edited one, must not
    // leave the application with no appearance at all.
    for (const char *junk : {"", "  ", "System", "auto", "solarized", "0", "true"})
        EXPECT_EQ(Theme::modeFromString(junk), Mode::System)
            << "\"" << junk << "\" was read as something other than System";
}

// -------------------------------------------------------------- applying it

class ThemeApply : public ::testing::Test {
protected:
    void SetUp() override
    {
        savedPalette = QApplication::palette();
        savedSheet   = qApp->styleSheet();
        savedStyle   = QApplication::style()->objectName();
    }
    void TearDown() override
    {
        QApplication::setPalette(savedPalette);
        qApp->setStyleSheet(savedSheet);
    }
    QPalette savedPalette;
    QString savedSheet;
    QString savedStyle;
};

TEST_F(ThemeApply, LightAndDarkAreNotTheSameAppearance)
{
    Theme::apply(Mode::Light, false);
    const QPalette light = QApplication::palette();

    Theme::apply(Mode::Dark, false);
    const QPalette dark = QApplication::palette();

    EXPECT_NE(light.color(QPalette::Window), dark.color(QPalette::Window))
        << "the light and dark modes install the same window colour";
    EXPECT_NE(light.color(QPalette::WindowText), dark.color(QPalette::WindowText))
        << "the light and dark modes install the same text colour";
}

TEST_F(ThemeApply, DarkIsActuallyDarkAndLightIsActuallyLight)
{
    Theme::apply(Mode::Dark, false);
    const QPalette dark = QApplication::palette();
    EXPECT_LT(dark.color(QPalette::Window).lightness(),
              dark.color(QPalette::WindowText).lightness())
        << "dark mode paints light text on a lighter background";

    Theme::apply(Mode::Light, false);
    const QPalette light = QApplication::palette();
    EXPECT_GT(light.color(QPalette::Window).lightness(),
              light.color(QPalette::WindowText).lightness())
        << "light mode paints dark text on a darker background";
}

// System is not a third appearance: it resolves to one of the other two, using
// the preference captured before the palette was replaced.
TEST_F(ThemeApply, SystemResolvesToWhicheverTheOsAsksFor)
{
    Theme::apply(Mode::System, true);
    const QPalette asDark = QApplication::palette();
    Theme::apply(Mode::Dark, false);
    const QPalette dark = QApplication::palette();
    EXPECT_EQ(asDark.color(QPalette::Window), dark.color(QPalette::Window))
        << "System with the OS preferring dark did not give the dark appearance";

    Theme::apply(Mode::System, false);
    const QPalette asLight = QApplication::palette();
    Theme::apply(Mode::Light, false);
    const QPalette light = QApplication::palette();
    EXPECT_EQ(asLight.color(QPalette::Window), light.color(QPalette::Window))
        << "System with the OS preferring light did not give the light appearance";
}

// The header is explicit that apply() sets the palette and stylesheet only --
// the caller has already chosen the QStyle, and replacing it here would undo a
// -style command-line override.
//
// Installing a stylesheet makes Qt wrap the style in a QStyleSheetStyle of its
// own, so the style Qt reports afterwards is that wrapper rather than Fusion.
// What has to hold is that the base underneath is untouched: clear the
// stylesheet and Fusion is back.
TEST_F(ThemeApply, ApplyingAThemeDoesNotReplaceTheStyle)
{
    ASSERT_EQ(QApplication::style()->objectName(), QString("fusion"))
        << "this test starts from Fusion, as main() does";

    Theme::apply(Mode::Dark, false);

    qApp->setStyleSheet(QString());
    EXPECT_EQ(QApplication::style()->objectName(), QString("fusion"))
        << "applying a theme replaced the QStyle, undoing any -style override";
}

TEST_F(ThemeApply, TheStylesheetIsInstalledAndIsOnlyAboutSpacing)
{
    Theme::apply(Mode::Light, false);
    const QString sheet = qApp->styleSheet();
    EXPECT_FALSE(sheet.isEmpty()) << "no global stylesheet was installed";

    // The documented contract: spacing and density only, never fonts, because
    // the font is the user's own preference and is set from Preferences.
    //
    // Comments are stripped first -- the sheet opens with a comment saying it
    // must not pin fonts, and matching that reports the rule as the violation.
    QString rules = sheet;
    rules.remove(QRegularExpression(R"(/\*.*?\*/)", QRegularExpression::DotMatchesEverythingOption));

    EXPECT_FALSE(rules.contains("font-family", Qt::CaseInsensitive))
        << "the stylesheet pins a font family, overriding the user's choice";
    EXPECT_FALSE(rules.contains("font-size", Qt::CaseInsensitive))
        << "the stylesheet pins a font size, overriding the user's choice";
}

TEST_F(ThemeApply, ApplyingTheSameThemeTwiceIsTheSameAppearance)
{
    Theme::apply(Mode::Dark, false);
    const QPalette once  = QApplication::palette();
    const QString sheet1 = qApp->styleSheet();
    Theme::apply(Mode::Dark, false);
    const QPalette twice = QApplication::palette();

    EXPECT_EQ(once.color(QPalette::Window), twice.color(QPalette::Window));
    EXPECT_EQ(sheet1, qApp->styleSheet()) << "the stylesheet accumulated on top of itself";
}

TEST_F(ThemeApply, SwitchingBackAndForthLandsWhereItStarted)
{
    Theme::apply(Mode::Light, false);
    const QPalette first = QApplication::palette();
    Theme::apply(Mode::Dark, false);
    Theme::apply(Mode::Light, false);
    const QPalette back = QApplication::palette();

    EXPECT_EQ(first.color(QPalette::Window), back.color(QPalette::Window))
        << "going dark and back again did not restore the light appearance";
    EXPECT_EQ(first.color(QPalette::Base), back.color(QPalette::Base));
    EXPECT_EQ(first.color(QPalette::Highlight), back.color(QPalette::Highlight));
}

TEST(Theme, TheOsPreferenceIsAnswerableWithoutCrashing)
{
    // The value depends on the machine; that it can be asked at all is the
    // assertion, since main() calls it before anything else is set up.
    const bool dark = Theme::osPrefersDark();
    EXPECT_TRUE(dark == true || dark == false);
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    // main() sets Fusion before applying a theme; do the same, so the
    // "apply() leaves the style alone" check is meaningful.
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
