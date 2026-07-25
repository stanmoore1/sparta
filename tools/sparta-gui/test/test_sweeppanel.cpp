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

// The parametric sweep's results table.
//
// This is the model behind the only table in the application that is filled in
// while runs are happening, one row at a time, from a driver that may be
// stopped part way. Nothing tested it. A table model that gets its row or
// column count wrong, or that forgets to bracket an insertion with
// beginInsertRows/endInsertRows, corrupts the view rather than failing --
// which is why the checks below go through QAbstractItemModelTester as well as
// asserting the contents.
//
// The driver itself (SweepController) needs a live SpartaGui and a running
// simulator, so it belongs to the application-driving suite, not here.

#include <gtest/gtest.h>

#include <QAbstractItemModelTester>
#include <QApplication>
#include <QColor>
#include <QSignalSpy>
#include <QStringList>

#include "sweeppanel.h"

namespace {

QString cell(const SweepResultsModel &m, int row, int col)
{
    return m.data(m.index(row, col), Qt::DisplayRole).toString();
}

} // namespace

TEST(SweepResults, StartsEmpty)
{
    SweepResultsModel m;
    EXPECT_EQ(m.rowCount(), 0);
    EXPECT_EQ(m.columnCount(), 0);
    EXPECT_TRUE(m.headers().isEmpty());
}

TEST(SweepResults, TheColumnCountFollowsTheHeaders)
{
    SweepResultsModel m;
    m.reset({"seed", "nsteps", "CPU"});
    EXPECT_EQ(m.columnCount(), 3);
    EXPECT_EQ(m.rowCount(), 0) << "resetting the headers should not invent rows";
    EXPECT_EQ(m.headers(), (QStringList{"seed", "nsteps", "CPU"}));
}

TEST(SweepResults, RowsArriveInOrderAndReadBack)
{
    SweepResultsModel m;
    m.reset({"seed", "CPU"});
    m.addRow({"12345", "0.81"}, true);
    m.addRow({"54321", "0.79"}, true);

    ASSERT_EQ(m.rowCount(), 2);
    EXPECT_EQ(cell(m, 0, 0), QString("12345"));
    EXPECT_EQ(cell(m, 0, 1), QString("0.81"));
    EXPECT_EQ(cell(m, 1, 0), QString("54321"));
    EXPECT_EQ(cell(m, 1, 1), QString("0.79"));
}

// A run that failed is the thing the user is scanning the table for, and the
// only way it is marked is the foreground colour.
TEST(SweepResults, AFailedRunIsColouredAndASuccessfulOneIsNot)
{
    SweepResultsModel m;
    m.reset({"seed", "CPU"});
    m.addRow({"1", "0.5"}, true);
    m.addRow({"2", ""}, false);

    EXPECT_FALSE(m.data(m.index(0, 0), Qt::ForegroundRole).isValid())
        << "a run that succeeded was given a colour of its own";
    const QVariant fg = m.data(m.index(1, 0), Qt::ForegroundRole);
    ASSERT_TRUE(fg.isValid()) << "a failed run is not marked at all";
    EXPECT_EQ(fg.value<QColor>(), QColor(Qt::red));
}

// The driver builds a row per sweep point and a header per variable, and the
// two can disagree: a deck that errors part way gives fewer cells than columns.
TEST(SweepResults, AShortRowReadsAsBlankRatherThanOutOfRange)
{
    SweepResultsModel m;
    m.reset({"seed", "nsteps", "CPU"});
    m.addRow({"12345"}, false);

    EXPECT_EQ(cell(m, 0, 0), QString("12345"));
    EXPECT_EQ(cell(m, 0, 1), QString()) << "a missing cell should read blank";
    EXPECT_EQ(cell(m, 0, 2), QString());
}

TEST(SweepResults, ALongRowIsNotTruncatedIntoTheWrongColumn)
{
    SweepResultsModel m;
    m.reset({"seed"});
    m.addRow({"12345", "extra"}, true);
    EXPECT_EQ(m.columnCount(), 1) << "an over-long row widened the table";
    EXPECT_EQ(cell(m, 0, 0), QString("12345"));
}

TEST(SweepResults, ClearingRowsKeepsTheColumns)
{
    SweepResultsModel m;
    m.reset({"seed", "CPU"});
    m.addRow({"1", "0.5"}, true);
    m.clearRows();

    EXPECT_EQ(m.rowCount(), 0);
    EXPECT_EQ(m.columnCount(), 2)
        << "clearing the rows also dropped the headers, so the next run's table has no columns";
}

TEST(SweepResults, ResettingTheHeadersDropsTheRows)
{
    SweepResultsModel m;
    m.reset({"seed", "CPU"});
    m.addRow({"1", "0.5"}, true);
    m.reset({"a", "b", "c"});

    EXPECT_EQ(m.rowCount(), 0)
        << "the previous run's rows survived into a table with different columns";
    EXPECT_EQ(m.columnCount(), 3);
}

TEST(SweepResults, TheHeadersAreWhatTheViewAsksFor)
{
    SweepResultsModel m;
    m.reset({"seed", "CPU"});
    EXPECT_EQ(m.headerData(0, Qt::Horizontal, Qt::DisplayRole).toString(), QString("seed"));
    EXPECT_EQ(m.headerData(1, Qt::Horizontal, Qt::DisplayRole).toString(), QString("CPU"));
    EXPECT_EQ(m.headerData(9, Qt::Horizontal, Qt::DisplayRole).toString(), QString())
        << "a column past the end should read blank, not crash";
    // rows are numbered from one, as a person counts runs
    EXPECT_EQ(m.headerData(0, Qt::Vertical, Qt::DisplayRole).toInt(), 1);
}

TEST(SweepResults, AnInvalidIndexReadsBlank)
{
    SweepResultsModel m;
    m.reset({"seed"});
    EXPECT_FALSE(m.data(QModelIndex(), Qt::DisplayRole).isValid());
    EXPECT_FALSE(m.data(m.index(5, 0), Qt::DisplayRole).isValid());
}

// Qt's own conformance check: it watches every signal the model emits against
// every change it makes, and fails on the mistakes that quietly corrupt a view
// rather than crashing -- an insertion not bracketed by begin/endInsertRows,
// a reset that does not announce itself, a count that changes without warning.
TEST(SweepResults, SatisfiesQtsModelContract)
{
    SweepResultsModel m;
    QAbstractItemModelTester tester(&m, QAbstractItemModelTester::FailureReportingMode::Warning);

    m.reset({"seed", "nsteps", "CPU"});
    for (int i = 0; i < 5; ++i)
        m.addRow({QString::number(i), "100", QString::number(i * 0.1)}, i != 3);
    m.clearRows();
    m.reset({"only"});
    m.addRow({"x"}, true);
    SUCCEED();
}

// The view has to be told before the rows appear, or it draws a table it
// believes is still empty.
TEST(SweepResults, AddingARowAnnouncesIt)
{
    SweepResultsModel m;
    m.reset({"seed"});

    QSignalSpy about(&m, &QAbstractItemModel::rowsAboutToBeInserted);
    QSignalSpy done(&m, &QAbstractItemModel::rowsInserted);
    m.addRow({"1"}, true);

    EXPECT_EQ(about.count(), 1) << "a row appeared without warning the view first";
    EXPECT_EQ(done.count(), 1);
}

TEST(SweepResults, ResettingAnnouncesItself)
{
    SweepResultsModel m;
    QSignalSpy about(&m, &QAbstractItemModel::modelAboutToBeReset);
    QSignalSpy done(&m, &QAbstractItemModel::modelReset);

    m.reset({"seed"});
    EXPECT_EQ(about.count(), 1);
    EXPECT_EQ(done.count(), 1);

    m.clearRows();
    EXPECT_EQ(done.count(), 2) << "clearing the rows did not tell the view";
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
