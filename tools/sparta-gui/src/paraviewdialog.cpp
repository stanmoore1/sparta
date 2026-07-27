/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   ParaViewExportDialog implementation.  See paraviewdialog.h.
------------------------------------------------------------------------- */

#include "paraviewdialog.h"

#include "constants.h"
#include "helpers.h"
#include "paraviewexport.h"

#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTextCursor>
#include <QVBoxLayout>

using ParaviewExport::Mode;
using ParaviewExport::Settings;

namespace {

// Look for an executable: honor a saved absolute path, otherwise search the
// PATH and common install locations (helpers::findExe), and finally the
// platform-specific ParaView bundle locations that are off the PATH.
QString detectTool(const QString &exe, const QString &savedPath)
{
    if (!savedPath.isEmpty() && QFileInfo::exists(savedPath)) return savedPath;
    const QString onPath = findExe(exe);
    if (!onPath.isEmpty()) return onPath;

    QStringList globs;
#if defined(Q_OS_MACOS)
    globs << "/Applications/ParaView*.app/Contents/bin";
#elif defined(Q_OS_WIN)
    globs << "C:/Program Files/ParaView*/bin"
          << "C:/Program Files (x86)/ParaView*/bin";
#endif
    for (const QString &pattern : globs) {
        const QFileInfo pfi(pattern);
        QDir base(pfi.absolutePath());
        for (const QString &d : base.entryList({pfi.fileName()}, QDir::Dirs, QDir::Name | QDir::Reversed)) {
            const QString cand = base.absoluteFilePath(d) + "/" + exe;
#if defined(Q_OS_WIN)
            const QString candExe = cand + ".exe";
            if (QFileInfo::exists(candExe)) return candExe;
#endif
            if (QFileInfo::exists(cand)) return cand;
        }
    }
    return {};
}

} // namespace

ParaViewExportDialog::ParaViewExportDialog(QWidget *parent, const QString &deckDir)
    : QDialog(parent), deckDir_(deckDir)
{
    setWindowTitle("Export to ParaView");
    setModal(true);
    scriptsDir_ = findScriptsDir();

    QSettings settings;
    const QString pvpython =
        detectTool("pvpython", settings.value(Keys::PVPYTHON_PATH).toString());
    const QString paraview =
        detectTool("paraview", settings.value(Keys::PARAVIEW_PATH).toString());

    auto *outer = new QVBoxLayout(this);

    auto *intro = new QLabel(
        "Convert a SPARTA surface or grid to ParaView format and open it.  This runs the "
        "bundled <code>surf2paraview.py</code> / <code>grid2paraview.py</code> scripts, which "
        "require ParaView's <code>pvpython</code> interpreter.", this);
    intro->setWordWrap(true);
    outer->addWidget(intro);

    auto *form = new QFormLayout;
    outer->addLayout(form);

    mode_ = new QComboBox(this);

    mode_->setObjectName("mode");

    mode_->setAccessibleName("Conversion mode");
    mode_->addItem("Surface geometry (surf2paraview)", static_cast<int>(Mode::Surface));
    mode_->addItem("Grid (grid2paraview)", static_cast<int>(Mode::Grid));
    form->addRow("Convert:", mode_);

    // input file + browse
    inputEdit_ = new QLineEdit(this);
    inputEdit_->setObjectName("input");
    inputEdit_->setAccessibleName("Input file to convert");
    auto *inputRow = new QHBoxLayout;
    inputRow->addWidget(inputEdit_);
    auto *inBrowse = new QPushButton("Browse...", this);
    inputRow->addWidget(inBrowse);
    form->addRow("Input file:", inputRow);

    outputEdit_ = new QLineEdit(this);

    outputEdit_->setObjectName("output");

    outputEdit_->setAccessibleName("Name of the converted output");
    form->addRow("Output name:", outputEdit_);

    // optional dump result files
    resultsEdit_ = new QLineEdit(this);
    resultsEdit_->setObjectName("results");
    resultsEdit_->setAccessibleName("Results file or pattern");
    resultsEdit_->setPlaceholderText("optional: dump files, e.g. tmp_surf.* (space-separated globs)");
    auto *resRow = new QHBoxLayout;
    resRow->addWidget(resultsEdit_);
    auto *resBrowse = new QPushButton("Browse...", this);
    resRow->addWidget(resBrowse);
    form->addRow("Dump results:", resRow);

    // per-mode options, in a stacked widget
    modeOpts_ = new QStackedWidget(this);
    // page 0: surface
    auto *surfPage = new QWidget(this);
    auto *surfLay = new QHBoxLayout(surfPage);
    surfLay->setContentsMargins(0, 0, 0, 0);
    exodus_ = new QCheckBox("Write Exodus II (.ex2) instead of .pvd", surfPage);
    exodus_->setObjectName("exodus");
    exodus_->setAccessibleName("Write Exodus II instead of pvd");
    surfLay->addWidget(exodus_);
    surfLay->addStretch();
    modeOpts_->addWidget(surfPage);
    // page 1: grid
    auto *gridPage = new QWidget(this);
    auto *gridLay = new QHBoxLayout(gridPage);
    gridLay->setContentsMargins(0, 0, 0, 0);
    const char *labels[3] = {"x chunk:", "y chunk:", "z chunk:"};
    for (int i = 0; i < 3; ++i) {
        gridLay->addWidget(new QLabel(labels[i], gridPage));
        chunk_[i] = new QSpinBox(gridPage);
        chunk_[i]->setObjectName(QStringLiteral("chunk%1").arg(i));
        chunk_[i]->setAccessibleName(QStringLiteral("Chunk size %1").arg(i));
        chunk_[i]->setRange(1, 100000);
        chunk_[i]->setValue(100);
        gridLay->addWidget(chunk_[i]);
    }
    gridLay->addStretch();
    modeOpts_->addWidget(gridPage);
    form->addRow("Options:", modeOpts_);

    // tool locations
    auto *toolBox = new QGroupBox("ParaView tools", this);
    auto *toolForm = new QFormLayout(toolBox);
    pvpythonEdit_ = new QLineEdit(pvpython, toolBox);
    pvpythonEdit_->setObjectName("pvpython");
    pvpythonEdit_->setAccessibleName("Path to the pvpython interpreter");
    auto *pvRow = new QHBoxLayout;
    pvRow->addWidget(pvpythonEdit_);
    auto *pvBrowse = new QPushButton("Browse...", toolBox);
    pvRow->addWidget(pvBrowse);
    toolForm->addRow("pvpython:", pvRow);
    paraviewEdit_ = new QLineEdit(paraview, toolBox);
    paraviewEdit_->setObjectName("paraview");
    paraviewEdit_->setAccessibleName("Path to the ParaView executable");
    auto *pwRow = new QHBoxLayout;
    pwRow->addWidget(paraviewEdit_);
    auto *pwBrowse = new QPushButton("Browse...", toolBox);
    pwRow->addWidget(pwBrowse);
    toolForm->addRow("paraview:", pwRow);
    outer->addWidget(toolBox);

    openAfter_ = new QCheckBox("Open the result in ParaView when the conversion finishes", this);

    openAfter_->setObjectName("openAfter");

    openAfter_->setAccessibleName("Open the result when the conversion finishes");
    openAfter_->setChecked(true);
    outer->addWidget(openAfter_);

    preview_ = new QLabel(this);

    preview_->setObjectName("preview");

    preview_->setAccessibleName("The command that will be run");
    preview_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    preview_->setWordWrap(true);
    preview_->setStyleSheet("QLabel { font-family: monospace; }");
    outer->addWidget(preview_);

    log_ = new QPlainTextEdit(this);

    log_->setObjectName("log");

    log_->setAccessibleName("Conversion output");
    log_->setReadOnly(true);
    log_->setMinimumHeight(140);
    log_->setPlaceholderText("Conversion output appears here.");
    outer->addWidget(log_, 1);

    auto *buttons = new QDialogButtonBox(this);
    runButton_ = buttons->addButton("Convert", QDialogButtonBox::AcceptRole);
    runButton_->setObjectName("convert");
    buttons->addButton(QDialogButtonBox::Close);
    outer->addWidget(buttons);

    if (scriptsDir_.isEmpty())
        log("WARNING: could not locate the tools/paraview scripts. Set the examples/plugin "
            "path in Preferences, or run from the SPARTA source tree.");
    if (pvpython.isEmpty())
        log("WARNING: pvpython was not found. Install ParaView and/or set its path above.");

    // wire up
    connect(mode_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &ParaViewExportDialog::onModeChanged);
    connect(inBrowse, &QPushButton::clicked, this, &ParaViewExportDialog::browseInput);
    connect(resBrowse, &QPushButton::clicked, this, &ParaViewExportDialog::browseResults);
    connect(pvBrowse, &QPushButton::clicked, this, &ParaViewExportDialog::browsePvpython);
    connect(pwBrowse, &QPushButton::clicked, this, &ParaViewExportDialog::browseParaview);
    connect(inputEdit_, &QLineEdit::textChanged, this, &ParaViewExportDialog::updatePreview);
    connect(outputEdit_, &QLineEdit::textChanged, this, &ParaViewExportDialog::updatePreview);
    connect(resultsEdit_, &QLineEdit::textChanged, this, &ParaViewExportDialog::updatePreview);
    connect(exodus_, &QCheckBox::toggled, this, &ParaViewExportDialog::updatePreview);
    for (auto *sp : chunk_)
        connect(sp, QOverload<int>::of(&QSpinBox::valueChanged), this,
                &ParaViewExportDialog::updatePreview);
    // "Convert" must not close the dialog (it runs a process); intercept accept
    connect(runButton_, &QPushButton::clicked, this, &ParaViewExportDialog::runConversion);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    onModeChanged();
    resize(720, 560);
}

ParaViewExportDialog::~ParaViewExportDialog() = default;

QString ParaViewExportDialog::findScriptsDir()
{
    QStringList candidates;
    QSettings settings;
    const QString examples = settings.value(Keys::EXAMPLES_PATH).toString();
    if (!examples.isEmpty())
        candidates << QDir(examples).absoluteFilePath("../tools/paraview");
    const QString plugin = settings.value(Keys::PLUGIN_PATH).toString();
    if (!plugin.isEmpty()) {
        QDir libdir = QFileInfo(plugin).absoluteDir();
        candidates << libdir.absoluteFilePath("../../tools/paraview")
                   << libdir.absoluteFilePath("../../../tools/paraview");
    }
    const QString appdir = QCoreApplication::applicationDirPath();
    candidates << appdir + "/../Resources/tools/paraview"   // macOS app bundle
               << appdir + "/../share/sparta/tools/paraview" // Linux/Windows install
               << appdir + "/tools/paraview";
    // walk up from the current directory looking for a source checkout
    QDir dir(QDir::currentPath());
    do {
        candidates << dir.absoluteFilePath("tools/paraview");
    } while (dir.cdUp());

    for (const QString &c : candidates) {
        if (QFileInfo::exists(c + "/surf2paraview.py")) return QDir(c).absolutePath();
    }
    return {};
}

QString ParaViewExportDialog::locateScript() const
{
    if (scriptsDir_.isEmpty()) return {};
    const Mode m = static_cast<Mode>(mode_->currentData().toInt());
    return QDir(scriptsDir_).absoluteFilePath(ParaviewExport::scriptName(m));
}

void ParaViewExportDialog::onModeChanged()
{
    const Mode m = static_cast<Mode>(mode_->currentData().toInt());
    modeOpts_->setCurrentIndex(m == Mode::Grid ? 1 : 0);
    updatePreview();
}

void ParaViewExportDialog::browseInput()
{
    const Mode m = static_cast<Mode>(mode_->currentData().toInt());
    const QString filter = (m == Mode::Surface)
        ? "SPARTA surface files (*.surf data.* *);;All files (*)"
        : "Grid description files (*.txt *);;All files (*)";
    const QString start = inputEdit_->text().isEmpty() ? deckDir_ : inputEdit_->text();
    const QString f = QFileDialog::getOpenFileName(
        this, m == Mode::Surface ? "Select SPARTA surface file"
                                 : "Select grid description file",
        start, filter);
    if (f.isEmpty()) return;
    inputEdit_->setText(f);
    if (outputEdit_->text().isEmpty()) {
        const QString base = QFileInfo(f).completeBaseName();
        outputEdit_->setText(base + (m == Mode::Surface ? "_surf" : "_grid"));
    }
}

void ParaViewExportDialog::browseResults()
{
    const QString start = deckDir_.isEmpty() ? QDir::currentPath() : deckDir_;
    const QStringList files =
        QFileDialog::getOpenFileNames(this, "Select SPARTA dump result files", start);
    if (files.isEmpty()) return;
    // quote any paths that contain spaces so the field round-trips
    QStringList quoted;
    for (const QString &f : files)
        quoted << (f.contains(' ') ? '"' + f + '"' : f);
    resultsEdit_->setText(quoted.join(' '));
}

void ParaViewExportDialog::browsePvpython()
{
    const QString f = QFileDialog::getOpenFileName(this, "Locate pvpython");
    if (!f.isEmpty()) pvpythonEdit_->setText(f);
}

void ParaViewExportDialog::browseParaview()
{
    const QString f = QFileDialog::getOpenFileName(this, "Locate paraview");
    if (!f.isEmpty()) paraviewEdit_->setText(f);
}

QStringList ParaViewExportDialog::expandResultGlob() const
{
    const QString raw = resultsEdit_->text().trimmed();
    if (raw.isEmpty()) return {};

    // split on whitespace, respecting simple double-quoting
    QStringList tokens;
    QString cur;
    bool inq = false;
    for (const QChar c : raw) {
        if (c == '"') { inq = !inq; continue; }
        if (c.isSpace() && !inq) {
            if (!cur.isEmpty()) { tokens << cur; cur.clear(); }
        } else {
            cur += c;
        }
    }
    if (!cur.isEmpty()) tokens << cur;

    const QString baseDir = deckDir_.isEmpty() ? QDir::currentPath() : deckDir_;
    QStringList out;
    for (const QString &tok : tokens) {
        QFileInfo fi(tok);
        if (!fi.isAbsolute()) fi.setFile(QDir(baseDir), tok);
        if (fi.fileName().contains('*') || fi.fileName().contains('?')) {
            QDir d(fi.absolutePath());
            const QStringList matched =
                d.entryList({fi.fileName()}, QDir::Files, QDir::Name);
            for (const QString &m : matched) out << d.absoluteFilePath(m);
        } else {
            out << fi.absoluteFilePath();
        }
    }
    return out;
}

Settings ParaViewExportDialog::collectSettings() const
{
    Settings s;
    s.mode = static_cast<Mode>(mode_->currentData().toInt());
    s.inputFile = inputEdit_->text().trimmed();
    s.outputName = outputEdit_->text().trimmed();
    s.resultFiles = expandResultGlob();
    s.exodus = exodus_->isChecked();
    s.xchunk = chunk_[0]->value();
    s.ychunk = chunk_[1]->value();
    s.zchunk = chunk_[2]->value();
    return s;
}

void ParaViewExportDialog::updatePreview()
{
    const Settings s = collectSettings();
    const QString script = locateScript();
    const QString scriptShown = script.isEmpty()
        ? ParaviewExport::scriptName(s.mode)
        : script;
    const QStringList args = ParaviewExport::buildScriptArgs(s, scriptShown);
    const QString pv = pvpythonEdit_->text().trimmed();
    const QString pvShown = pv.isEmpty() ? "pvpython" : QFileInfo(pv).fileName();
    QStringList shown;
    shown << pvShown;
    for (const QString &a : args)
        shown << (a.contains(' ') ? '"' + a + '"' : a);
    preview_->setText(shown.join(' '));
}

void ParaViewExportDialog::setBusy(bool busy)
{
    runButton_->setEnabled(!busy);
    runButton_->setText(busy ? "Converting..." : "Convert");
}

void ParaViewExportDialog::log(const QString &line)
{
    log_->appendPlainText(line);
}

void ParaViewExportDialog::runConversion()
{
    if (proc_) return; // already running

    Settings s = collectSettings();
    QString err;
    if (!ParaviewExport::validate(s, err)) {
        critical(this, "Export to ParaView", "Cannot run the conversion:", err);
        return;
    }
    const QString script = locateScript();
    if (script.isEmpty() || !QFileInfo::exists(script)) {
        critical(this, "Export to ParaView",
                 "Could not find the conversion script.",
                 "Set the examples or plugin path in Preferences so the tools/paraview "
                 "directory can be located.");
        return;
    }
    const QString pvpython = pvpythonEdit_->text().trimmed();
    if (pvpython.isEmpty() || !QFileInfo::exists(pvpython)) {
        critical(this, "Export to ParaView", "pvpython was not found.",
                 "Install ParaView and set the path to its pvpython interpreter.");
        return;
    }

    // work in the input file's directory so relative output names land there
    const QString workDir = QFileInfo(s.inputFile).absolutePath();
    const QString outAbs = QDir(workDir).absoluteFilePath(ParaviewExport::expectedOutput(s));
    const QString outDir = QDir(workDir).absoluteFilePath(s.outputName);

    // both scripts refuse to overwrite an existing .pvd (and grid an existing
    // output directory); offer to remove stale output first
    if (QFileInfo::exists(outAbs) || QFileInfo::exists(outDir)) {
        const auto btn = QMessageBox::question(
            this, "Export to ParaView",
            QString("Output \"%1\" already exists. Overwrite it?").arg(s.outputName),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (btn != QMessageBox::Yes) return;
        QFile::remove(outAbs);
        QDir(outDir).removeRecursively();
    }

    // persist the tool paths for next time
    QSettings settings;
    settings.setValue(Keys::PVPYTHON_PATH, pvpython);
    if (!paraviewEdit_->text().trimmed().isEmpty())
        settings.setValue(Keys::PARAVIEW_PATH, paraviewEdit_->text().trimmed());

    const QStringList args = ParaviewExport::buildScriptArgs(s, script);
    pendingPvd_ = outAbs;

    log_->clear();
    log(QString("$ cd %1").arg(workDir));
    log(QString("$ %1 %2").arg(QFileInfo(pvpython).fileName(), args.join(' ')));
    log(QString());

    proc_ = new QProcess(this);
    proc_->setProcessChannelMode(QProcess::MergedChannels);
    proc_->setWorkingDirectory(workDir);
    connect(proc_, &QProcess::readyReadStandardOutput, this,
            &ParaViewExportDialog::onProcessOutput);
    connect(proc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
            &ParaViewExportDialog::onProcessFinished);
    setBusy(true);
    proc_->start(pvpython, args);
    if (!proc_->waitForStarted(5000)) {
        log("ERROR: failed to start pvpython.");
        setBusy(false);
        proc_->deleteLater();
        proc_ = nullptr;
    }
}

void ParaViewExportDialog::onProcessOutput()
{
    if (!proc_) return;
    const QString chunk = QString::fromLocal8Bit(proc_->readAllStandardOutput());
    // append without forcing extra newlines
    log_->moveCursor(QTextCursor::End);
    log_->insertPlainText(chunk);
    log_->moveCursor(QTextCursor::End);
}

void ParaViewExportDialog::onProcessFinished(int exitCode, int exitStatus)
{
    onProcessOutput();
    setBusy(false);
    const bool ok = (exitStatus == QProcess::NormalExit) && (exitCode == 0);
    if (proc_) {
        proc_->deleteLater();
        proc_ = nullptr;
    }
    if (!ok) {
        log(QString("\nConversion failed (exit code %1).").arg(exitCode));
        warning(this, "Export to ParaView", "The conversion did not complete successfully.",
                "See the log in the dialog for details.");
        return;
    }
    log(QString("\nDone. Wrote %1").arg(pendingPvd_));
    if (!openAfter_->isChecked()) return;

    const QString paraview = paraviewEdit_->text().trimmed();
    if (paraview.isEmpty() || !QFileInfo::exists(paraview)) {
        warning(this, "Export to ParaView",
                "The conversion succeeded, but the paraview executable was not found.",
                QString("Open this file in ParaView manually:\n%1").arg(pendingPvd_));
        return;
    }
    if (QProcess::startDetached(paraview, {pendingPvd_}))
        log(QString("Launched ParaView on %1").arg(pendingPvd_));
    else
        warning(this, "Export to ParaView", "Could not launch ParaView.",
                QString("Open this file manually:\n%1").arg(pendingPvd_));
}

// Local Variables:
// c-basic-offset: 4
// End:
