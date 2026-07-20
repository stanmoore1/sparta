// Unit tests for the pure run-provenance core (src/runarchive.cpp):
// JSON round-trip of a RunRecord and the self-contained HTML report builder.

#include "runarchive.h"

#include <QByteArray>
#include <QJsonObject>

#include "gtest/gtest.h"

#include <string>

using namespace RunArchive;

namespace {
RunRecord sample()
{
    RunRecord r;
    r.id = "20260720-abc";
    r.timestamp = "2026-07-20T21:00:00";
    r.deckName = "in.flow";
    r.deckText = "run 100\nglobal nrho 1.0";
    r.logText = "Step CPU Np\n0 0 2000";
    r.thermoYaml = "keywords: [Step, Np]\ndata:\n- [0, 2000]";
    r.workDir = "/home/user/case";
    r.status = "ok";
    r.imageFiles = {"/tmp/a.png", "/tmp/b.png"};
    r.metadata.insert("SPARTA version", "24 Sep 2025");
    return r;
}
} // namespace

TEST(RunArchive, JsonRoundTrip)
{
    const RunRecord r = sample();
    const RunRecord back = RunRecord::fromJson(r.toJson());
    EXPECT_EQ(back.id.toStdString(), r.id.toStdString());
    EXPECT_EQ(back.deckName.toStdString(), r.deckName.toStdString());
    EXPECT_EQ(back.deckText.toStdString(), r.deckText.toStdString());
    EXPECT_EQ(back.imageFiles.size(), 2);
    EXPECT_EQ(back.status.toStdString(), "ok");
    EXPECT_EQ(back.metadata.value("SPARTA version").toStdString(), "24 Sep 2025");
}

TEST(RunArchive, HtmlContainsDeckLogAndMetadata)
{
    const RunRecord r = sample();
    QMap<QString, QByteArray> imgs;
    imgs.insert("/tmp/a.png", QByteArray("\x89PNG-fake-a"));
    imgs.insert("/tmp/b.png", QByteArray("\x89PNG-fake-b"));
    const std::string html = buildRunReportHtml(r, imgs).toStdString();

    EXPECT_NE(html.find("Run Report"), std::string::npos);
    EXPECT_NE(html.find("in.flow"), std::string::npos);
    EXPECT_NE(html.find("global nrho 1.0"), std::string::npos);         // deck text
    EXPECT_NE(html.find("Step CPU Np"), std::string::npos);             // log text
    EXPECT_NE(html.find("24 Sep 2025"), std::string::npos);             // metadata
    // both images inlined as base64 data URIs
    EXPECT_NE(html.find("data:image/png;base64,"), std::string::npos);
    EXPECT_EQ(html.find("<img"), html.rfind("data:image/png;base64,") == std::string::npos
                                     ? std::string::npos : html.find("<img"));
    // exactly two <img> tags
    size_t count = 0, pos = 0;
    while ((pos = html.find("<img", pos)) != std::string::npos) { ++count; pos += 4; }
    EXPECT_EQ(count, 2u);
}

TEST(RunArchive, HtmlSkipsImagesWithoutData)
{
    RunRecord r = sample();
    QMap<QString, QByteArray> imgs;
    imgs.insert("/tmp/a.png", QByteArray("data")); // only one of the two provided
    const std::string html = buildRunReportHtml(r, imgs).toStdString();
    size_t count = 0, pos = 0;
    while ((pos = html.find("<img", pos)) != std::string::npos) { ++count; pos += 4; }
    EXPECT_EQ(count, 1u);
}

TEST(RunArchive, HtmlEscapesAngleBrackets)
{
    RunRecord r = sample();
    r.deckText = "region r block <a> & <b>";
    const std::string html = buildRunReportHtml(r, {}).toStdString();
    EXPECT_NE(html.find("&lt;a&gt; &amp; &lt;b&gt;"), std::string::npos);
    EXPECT_EQ(html.find("<a>"), std::string::npos); // raw not present
}

TEST(RunArchive, ArchiveDir)
{
    EXPECT_EQ(runArchiveDir("/base/history", "abc").toStdString(), "/base/history/abc");
}
