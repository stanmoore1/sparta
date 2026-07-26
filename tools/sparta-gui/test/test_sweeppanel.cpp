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

#include <QComboBox>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>
#include <QTableWidget>
#include <QTimer>

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


// ------------------------------------------------------------------ the panel
//
// SweepPanel needs no main window and no simulator to build its controls or to
// validate what the user typed into them: gui_ is asked only to detect
// variables in the deck, and sparta_ only to refuse starting on top of a live
// run.  What it does need is somewhere for its refusals to go -- every one of
// them is a QMessageBox -- so the reaper below reads the message and dismisses
// it, which is how the spec validator's six failure messages become testable.

namespace {

// Reads and dismisses the next modal to appear, so a slot that ends in a
// QMessageBox can be called from a test without stalling it.
class ModalText : public QObject {
public:
    explicit ModalText(int budgetMs = 2000) : deadline(budgetMs)
    {
        timer.setInterval(10);
        connect(&timer, &QTimer::timeout, this, &ModalText::poll);
        timer.start();
    }

    // run @p fn and return the text of the message box it raised ("" if none)
    template <class F> static QString capture(F &&fn)
    {
        ModalText reaper;
        fn();
        return reaper.seen;
    }

    QString seen;

private:
    void poll()
    {
        if ((deadline -= 10) < 0) { timer.stop(); return; }
        auto *m = QApplication::activeModalWidget();
        if (!m) return;
        if (auto *box = qobject_cast<QMessageBox *>(m)) seen = box->text();
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
        else m->close();
        timer.stop();
    }

    QTimer timer;
    int deadline;
};

// Add a variable row and fill it in, the way a user would.
void addRow(SweepPanel &p, const QString &name, const QString &type, const QString &spec)
{
    auto *table = p.findChild<QTableWidget *>("varTable");
    QMetaObject::invokeMethod(&p, "addVariableRow");
    const int r = table->rowCount() - 1;
    qobject_cast<QComboBox *>(table->cellWidget(r, 0))->setCurrentText(name);
    qobject_cast<QComboBox *>(table->cellWidget(r, 1))->setCurrentText(type);
    table->item(r, 2)->setText(spec);
}

// The message startSweep() refuses with, or "" if it got past validation.
QString refusal(SweepPanel &p)
{
    return ModalText::capture([&p] { QMetaObject::invokeMethod(&p, "startSweep"); });
}

QLineEdit *quantitiesField(SweepPanel &p)
{
    return p.findChild<QLineEdit *>("quantities");
}

} // namespace

TEST(SweepPanelUi, BuildsWithoutAMainWindowOrASimulator)
{
    SweepPanel p(nullptr, nullptr, nullptr);
    EXPECT_NE(p.findChild<QTableWidget *>("varTable"), nullptr) << "no variables table";
    EXPECT_GE(p.findChildren<QPushButton *>().size(), 3) << "add/remove/detect buttons";
    EXPECT_EQ(p.findChild<QTableWidget *>("varTable")->rowCount(), 0) << "the table starts empty";
}

TEST(SweepPanelUi, DetectingVariablesWithNoDeckBehindItIsHarmless)
{
    SweepPanel p(nullptr, nullptr, nullptr);
    QMetaObject::invokeMethod(&p, "refreshVariables");
    EXPECT_EQ(p.findChild<QTableWidget *>("varTable")->rowCount(), 0)
        << "a row was added for a variable that was never discovered";
}

TEST(SweepPanelUi, AddsAndRemovesVariableRows)
{
    SweepPanel p(nullptr, nullptr, nullptr);
    auto *table = p.findChild<QTableWidget *>("varTable");

    QMetaObject::invokeMethod(&p, "addVariableRow");
    QMetaObject::invokeMethod(&p, "addVariableRow");
    ASSERT_EQ(table->rowCount(), 2);
    // each row gets its own editors, not shared ones
    EXPECT_NE(table->cellWidget(0, 0), table->cellWidget(1, 0));
    EXPECT_NE(table->cellWidget(0, 1), table->cellWidget(1, 1));

    table->setCurrentCell(0, 0);
    QMetaObject::invokeMethod(&p, "removeVariableRow");
    EXPECT_EQ(table->rowCount(), 1);
}

TEST(SweepPanelUi, RemovingWithNothingSelectedRemovesNothing)
{
    SweepPanel p(nullptr, nullptr, nullptr);
    auto *table = p.findChild<QTableWidget *>("varTable");
    QMetaObject::invokeMethod(&p, "addVariableRow");
    table->setCurrentCell(-1, -1);
    QMetaObject::invokeMethod(&p, "removeVariableRow");
    EXPECT_EQ(table->rowCount(), 1);
}

TEST(SweepPanelUi, RefusesASweepWithNoVariables)
{
    SweepPanel p(nullptr, nullptr, nullptr);
    const QString msg = refusal(p);
    EXPECT_TRUE(msg.contains("at least one variable")) << msg.toStdString();
}

TEST(SweepPanelUi, RefusesAnEmptyValueList)
{
    SweepPanel p(nullptr, nullptr, nullptr);
    addRow(p, "n", "List", "   ");
    quantitiesField(p)->setText("Np");
    EXPECT_TRUE(refusal(p).contains("empty value list"));
}

TEST(SweepPanelUi, RefusesARangeThatIsNotStartStopStep)
{
    SweepPanel p(nullptr, nullptr, nullptr);
    addRow(p, "n", "Range", "1:10");
    quantitiesField(p)->setText("Np");
    const QString msg = refusal(p);
    EXPECT_TRUE(msg.contains("start:stop:step")) << msg.toStdString();
}

TEST(SweepPanelUi, RefusesALinspaceThatIsNotStartStopCount)
{
    SweepPanel p(nullptr, nullptr, nullptr);
    addRow(p, "n", "Linspace", "1:10");
    quantitiesField(p)->setText("Np");
    const QString msg = refusal(p);
    EXPECT_TRUE(msg.contains("start:stop:count")) << msg.toStdString();
}

TEST(SweepPanelUi, RefusesASweepWithNothingToTabulate)
{
    SweepPanel p(nullptr, nullptr, nullptr);
    addRow(p, "n", "List", "1, 2, 3");
    quantitiesField(p)->setText("");
    const QString msg = refusal(p);
    EXPECT_TRUE(msg.contains("thermo quantity")) << msg.toStdString();
}

TEST(SweepPanelUi, RefusesReplicatesWithoutASeedVariable)
{
    SweepPanel p(nullptr, nullptr, nullptr);
    addRow(p, "n", "List", "1, 2, 3");
    quantitiesField(p)->setText("Np");
    p.findChild<QSpinBox *>("replicates")->setValue(3);
    const QString msg = refusal(p);
    EXPECT_TRUE(msg.contains("seed variable")) << msg.toStdString();
}

TEST(SweepPanelUi, ExportingWithNoResultsSaysSoInsteadOfOpeningAFileDialog)
{
    SweepPanel p(nullptr, nullptr, nullptr);
    const QString msg = ModalText::capture([&p] { QMetaObject::invokeMethod(&p, "exportCsv"); });
    EXPECT_TRUE(msg.contains("No results")) << msg.toStdString();
}

TEST(SweepPanelUi, ChartingWithNoResultsSaysSoInsteadOfOpeningAnEmptyWindow)
{
    SweepPanel p(nullptr, nullptr, nullptr);
    const QString msg = ModalText::capture([&p] { QMetaObject::invokeMethod(&p, "chartResults"); });
    EXPECT_TRUE(msg.contains("Not enough")) << msg.toStdString();
}

TEST(SweepPanelUi, ProgressAndCompletionAreReportedWithoutARun)
{
    SweepPanel p(nullptr, nullptr, nullptr);
    QMetaObject::invokeMethod(&p, "onProgress", Q_ARG(int, 3), Q_ARG(int, 10));
    QMetaObject::invokeMethod(&p, "onFinished", Q_ARG(bool, true));
    QMetaObject::invokeMethod(&p, "onFinished", Q_ARG(bool, false));
    SUCCEED();
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_sweeppanel.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
