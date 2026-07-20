// Unit tests for the pure snippet-library parser (src/snippets.cpp).

#include "snippets.h"

#include <QByteArray>

#include "gtest/gtest.h"

#include <string>

using namespace Snippets;

TEST(Snippets, ParsesArrayBodyAndFields)
{
    const QByteArray json = R"([
      {"name":"A","category":"Setup","description":"first",
       "body":["line1","line2"]},
      {"name":"B","body":"single line"}
    ])";
    QString err;
    const auto list = parse(json, &err);
    ASSERT_TRUE(err.isEmpty());
    ASSERT_EQ(list.size(), 2);
    EXPECT_EQ(list[0].name.toStdString(), "A");
    EXPECT_EQ(list[0].category.toStdString(), "Setup");
    EXPECT_EQ(list[0].body.toStdString(), "line1\nline2");
    EXPECT_EQ(list[1].category.toStdString(), "General"); // default
    EXPECT_EQ(list[1].body.toStdString(), "single line");
}

TEST(Snippets, SkipsIncompleteEntries)
{
    const QByteArray json = R"([
      {"name":"","body":"x"},
      {"name":"noBody"},
      {"name":"ok","body":"y"}
    ])";
    const auto list = parse(json);
    ASSERT_EQ(list.size(), 1);
    EXPECT_EQ(list[0].name.toStdString(), "ok");
}

TEST(Snippets, MalformedJsonSetsError)
{
    QString err;
    const auto list = parse(QByteArray("{ not valid"), &err);
    EXPECT_TRUE(list.isEmpty());
    EXPECT_FALSE(err.isEmpty());
}

TEST(Snippets, NonArraySetsError)
{
    QString err;
    const auto list = parse(QByteArray("{\"a\":1}"), &err);
    EXPECT_TRUE(list.isEmpty());
    EXPECT_FALSE(err.isEmpty());
}
