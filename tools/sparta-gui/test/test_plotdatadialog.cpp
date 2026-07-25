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

// The column chooser behind File > Plot Data File.
//
// The dialog decides which column becomes the x axis, which become curves, what
// they are called, and -- through the derived-column field -- lets an
// expression add a column that was not in the file. Everything it returns is an
// index into the data it hands back, so an off-by-one between the two is a
// chart of the wrong quantity drawn without complaint. That is what these check
// against: buildData() and the index accessors have to agree.
//
// The widget walker drives this dialog's controls already; what it cannot do is
// say whether the answers mean the right thing.

#include <gtest/gtest.h>

#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QLineEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QTimer>

#include <memory>
#include <vector>

#include "plotdata.h"
#include "plotdatadialog.h"

namespace {

// step / temp / press, three rows, so a derived column has something to chew on
PlotData sample()
{
    PlotData d;
    // addColumn() registers the name as well as the numbers; naming the
    // columns separately on top of it would define each one twice.
    d.addColumn("step", {0.0, 100.0, 200.0});
    d.addColumn("temp", {300.0, 310.0, 320.0});
    d.addColumn("press", {1.0, 2.0, 4.0});
    return d;
}

// The dialog builds one row per column: an x radio, a y checkbox, a name edit.
QList<QCheckBox *> yChecks(PlotDataDialog &d)
{
    return d.findChildren<QCheckBox *>();
}
QList<QRadioButton *> xRadios(PlotDataDialog &d)
{
    return d.findChildren<QRadioButton *>();
}
QList<QLineEdit *> edits(PlotDataDialog &d)
{
    return d.findChildren<QLineEdit *>();
}

// Press "Add column", dismissing whatever it complains with.
//
// computeColumn() reports a bad expression through QMessageBox::exec(), which
// runs its own event loop; with nobody to click OK that is a hang, so a timer
// closes the box from inside that loop. Returns whether anything complained.
bool addColumn(PlotDataDialog &d)
{
    bool complained = false;
    QTimer reaper;
    QObject::connect(&reaper, &QTimer::timeout, [&complained]() {
        if (QWidget *m = QApplication::activeModalWidget()) {
            complained = true;
            m->close();
        }
    });
    reaper.start(10);

    for (auto *b : d.findChildren<QPushButton *>())
        if (b->text().contains("Add column", Qt::CaseInsensitive)) b->click();

    reaper.stop();
    return complained;
}

} // namespace

TEST(PlotDataDialog, OffersARowPerColumn)
{
    PlotDataDialog d(sample());
    EXPECT_EQ(xRadios(d).size(), 3) << "one x-axis choice per column";
    EXPECT_EQ(yChecks(d).size(), 3) << "one y-axis choice per column";
    EXPECT_EQ(d.columnNames().size(), 3);
}

TEST(PlotDataDialog, TheColumnNamesComeFromTheData)
{
    PlotDataDialog d(sample());
    EXPECT_EQ(d.columnNames(), (QStringList{"step", "temp", "press"}));
}

TEST(PlotDataDialog, TheDataHandedBackMatchesTheDataGiven)
{
    PlotDataDialog d(sample());
    const PlotData out = d.buildData();

    ASSERT_EQ(out.columnCount(), 3);
    ASSERT_EQ(out.rowCount(), 3);
    EXPECT_EQ(out.column(0), (std::vector<double>{0.0, 100.0, 200.0}));
    EXPECT_EQ(out.column(2), (std::vector<double>{1.0, 2.0, 4.0}));
}

// The first column is the x axis unless the user says otherwise; a dialog that
// answered -1 here would index off the front of the data.
TEST(PlotDataDialog, TheFirstColumnIsTheDefaultXAxis)
{
    PlotDataDialog d(sample());
    EXPECT_EQ(d.xColumn(), 0);
}

TEST(PlotDataDialog, ChoosingAnXColumnIsReportedByIndex)
{
    PlotDataDialog d(sample());
    auto radios = xRadios(d);
    ASSERT_EQ(radios.size(), 3);

    radios[2]->setChecked(true);
    EXPECT_EQ(d.xColumn(), 2) << "the x-axis choice does not match the row that was picked";

    radios[1]->setChecked(true);
    EXPECT_EQ(d.xColumn(), 1) << "the x-axis radios are not exclusive, or the ids are wrong";
}

TEST(PlotDataDialog, TheCheckedColumnsAreTheOnesReported)
{
    PlotDataDialog d(sample());
    auto checks = yChecks(d);
    ASSERT_EQ(checks.size(), 3);

    for (auto *c : checks)
        c->setChecked(false);
    EXPECT_TRUE(d.yColumns().isEmpty()) << "columns were reported that nobody selected";

    checks[1]->setChecked(true);
    checks[2]->setChecked(true);
    EXPECT_EQ(d.yColumns(), (QList<int>{1, 2}))
        << "the curve selection does not match the boxes that are ticked";
}

// A renamed column has to keep its data: the chart legend reads the name and
// the curve reads the numbers, and they are looked up by the same index.
TEST(PlotDataDialog, RenamingAColumnKeepsItsNumbers)
{
    PlotDataDialog d(sample());
    auto es = edits(d);
    ASSERT_GE(es.size(), 3);

    es[1]->setText("temperature");
    EXPECT_EQ(d.columnNames().at(1), QString("temperature"));

    const PlotData out = d.buildData();
    ASSERT_EQ(out.columnCount(), 3);
    EXPECT_EQ(out.columnName(1), QString("temperature"))
        << "the rename did not reach the data handed back";
    EXPECT_EQ(out.column(1), (std::vector<double>{300.0, 310.0, 320.0}))
        << "renaming a column moved its numbers";
    EXPECT_EQ(out.column(0), (std::vector<double>{0.0, 100.0, 200.0}))
        << "renaming one column disturbed another";
}

TEST(PlotDataDialog, AnEmptyRenameDoesNotLeaveAnUnnamedColumn)
{
    PlotDataDialog d(sample());
    auto es = edits(d);
    ASSERT_GE(es.size(), 3);
    es[0]->setText("");

    const PlotData out = d.buildData();
    EXPECT_FALSE(out.columnName(0).isEmpty())
        << "clearing the name field left a column with no name at all, so the legend and the "
           "axis menu would both show a blank entry";
}

// The derived-column field is the only way to plot something the file does not
// contain, and it is the only place in this dialog that can fail.
TEST(PlotDataDialog, ADerivedColumnIsAppendedWithItsValues)
{
    PlotDataDialog d(sample());
    const int before = d.buildData().columnCount();

    auto es = edits(d);
    ASSERT_GE(es.size(), 5) << "expected a name and an expression field for the derived column";
    // the last two line edits are the derived-column name and expression
    es[es.size() - 2]->setText("double_press");
    es[es.size() - 1]->setText("press*2");

    EXPECT_FALSE(addColumn(d)) << "a valid expression was rejected";

    const PlotData out = d.buildData();
    ASSERT_EQ(out.columnCount(), before + 1) << "the derived column was not appended";
    EXPECT_EQ(out.columnName(before), QString("double_press"));
    EXPECT_EQ(out.column(before), (std::vector<double>{2.0, 4.0, 8.0}))
        << "the expression was not evaluated over the rows";
}

TEST(PlotDataDialog, ADerivedColumnCanBeSelectedLikeAnyOther)
{
    PlotDataDialog d(sample());
    auto es = edits(d);
    es[es.size() - 2]->setText("sum");
    es[es.size() - 1]->setText("temp+press");
    addColumn(d);

    ASSERT_EQ(yChecks(d).size(), 4) << "the derived column got no row of its own";
    for (auto *c : yChecks(d))
        c->setChecked(false);
    yChecks(d)[3]->setChecked(true);
    EXPECT_EQ(d.yColumns(), (QList<int>{3}));
}

TEST(PlotDataDialog, AnExpressionThatMakesNoSenseAddsNothing)
{
    PlotDataDialog d(sample());
    const int before = d.buildData().columnCount();

    auto es = edits(d);
    es[es.size() - 2]->setText("nonsense");
    es[es.size() - 1]->setText("no_such_column * 2");
    EXPECT_TRUE(addColumn(d)) << "an expression naming a column that does not exist was accepted "
                                 "without a word";

    EXPECT_EQ(d.buildData().columnCount(), before)
        << "an expression naming a column that does not exist still added a column, which would "
           "plot as a flat line rather than as an error";
}

TEST(PlotDataDialog, AnEmptyExpressionAddsNothing)
{
    PlotDataDialog d(sample());
    const int before = d.buildData().columnCount();

    EXPECT_TRUE(addColumn(d)) << "Add column with both fields empty said nothing at all";

    EXPECT_EQ(d.buildData().columnCount(), before)
        << "Add column with both fields empty added a column";
}

TEST(PlotDataDialog, SurvivesDataWithNoRows)
{
    PlotData empty;
    empty.addColumn("step", {});
    empty.addColumn("temp", {});

    PlotDataDialog d(empty);
    EXPECT_EQ(d.xColumn(), 0);
    EXPECT_EQ(d.buildData().rowCount(), 0);
}

TEST(PlotDataDialog, SurvivesDataWithNoColumns)
{
    PlotDataDialog d(PlotData{});
    EXPECT_EQ(d.xColumn(), 0) << "an empty file should still answer with a usable index";
    EXPECT_TRUE(d.yColumns().isEmpty());
    EXPECT_EQ(d.buildData().columnCount(), 0);
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
