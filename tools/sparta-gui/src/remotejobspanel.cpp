/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Remote-execution UI implementation.  See remotejobspanel.h.
------------------------------------------------------------------------- */

#include "remotejobspanel.h"

#include "remotejobmanager.h"
#include "schedulerspec.h"

#include <QComboBox>
#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QSplitter>
#include <QTableView>
#include <QUrl>
#include <QVBoxLayout>

using namespace Remote;

namespace {
void addSchedulers(QComboBox *c)
{
    c->addItem("Slurm (sbatch)", static_cast<int>(Scheduler::Slurm));
    c->addItem("PBS / Torque (qsub)", static_cast<int>(Scheduler::PBS));
    c->addItem("Flux (flux batch)", static_cast<int>(Scheduler::Flux));
}
} // namespace

// ===========================================================================
// ConnectionProfilesDialog
// ===========================================================================

ConnectionProfilesDialog::ConnectionProfilesDialog(QWidget *parent, RemoteJobManager *mgr)
    : QDialog(parent), mgr_(mgr)
{
    setWindowTitle("Cluster Connection Profiles");
    setModal(true);
    resize(760, 560);

    auto *outer = new QHBoxLayout(this);

    // left: profile list + New/Delete
    auto *left = new QVBoxLayout;
    list_ = new QListWidget(this);
    left->addWidget(list_, 1);
    auto *newBtn = new QPushButton("New", this);
    auto *delBtn = new QPushButton("Delete", this);
    auto *lrow = new QHBoxLayout;
    lrow->addWidget(newBtn);
    lrow->addWidget(delBtn);
    left->addLayout(lrow);
    outer->addLayout(left, 0);

    // right: form
    auto *form = new QFormLayout;
    name_ = new QLineEdit(this);
    host_ = new QLineEdit(this);
    user_ = new QLineEdit(this);
    port_ = new QSpinBox(this);
    port_->setRange(1, 65535);
    port_->setValue(22);
    workdir_ = new QLineEdit(this);
    workdir_->setPlaceholderText("/scratch/<user>/sparta");
    scheduler_ = new QComboBox(this);
    addSchedulers(scheduler_);
    launch_ = new QLineEdit(this);
    launch_->setPlaceholderText("srun  (or: mpirun -np N)");
    exe_ = new QLineEdit(this);
    exe_->setPlaceholderText("spa_");
    modules_ = new QPlainTextEdit(this);
    modules_->setPlaceholderText("one 'module load ...' line per entry");
    modules_->setMaximumHeight(70);
    templateEdit_ = new QPlainTextEdit(this);
    templateEdit_->setPlaceholderText("leave empty to use the scheduler's default template");

    form->addRow("Profile name:", name_);
    form->addRow("Host:", host_);
    form->addRow("User:", user_);
    form->addRow("Port:", port_);
    form->addRow("Remote workdir:", workdir_);
    form->addRow("Scheduler:", scheduler_);
    form->addRow("Launcher:", launch_);
    form->addRow("SPARTA exe:", exe_);
    form->addRow("Module loads:", modules_);
    auto *tmplBtn = new QPushButton("Load default template", this);
    form->addRow("Batch template:", templateEdit_);
    form->addRow("", tmplBtn);

    auto *right = new QVBoxLayout;
    right->addLayout(form, 1);
    auto *bb = new QDialogButtonBox(this);
    auto *saveBtn = bb->addButton("Save Profile", QDialogButtonBox::ApplyRole);
    bb->addButton(QDialogButtonBox::Close);
    right->addWidget(bb);
    outer->addLayout(right, 1);

    connect(list_, &QListWidget::currentRowChanged, this, &ConnectionProfilesDialog::selectProfile);
    connect(newBtn, &QPushButton::clicked, this, &ConnectionProfilesDialog::newProfile);
    connect(delBtn, &QPushButton::clicked, this, &ConnectionProfilesDialog::deleteCurrent);
    connect(saveBtn, &QPushButton::clicked, this, &ConnectionProfilesDialog::saveCurrent);
    connect(tmplBtn, &QPushButton::clicked, this, &ConnectionProfilesDialog::loadDefaultTemplate);
    connect(bb, &QDialogButtonBox::rejected, this, &QDialog::accept);

    reloadList();
    if (list_->count() == 0) newProfile();
}

void ConnectionProfilesDialog::reloadList(const QString &select)
{
    list_->blockSignals(true);
    list_->clear();
    for (const auto &p : mgr_->profiles()) list_->addItem(p.name);
    list_->blockSignals(false);
    if (!select.isEmpty()) {
        const auto hits = list_->findItems(select, Qt::MatchExactly);
        if (!hits.isEmpty()) list_->setCurrentItem(hits.first());
    } else if (list_->count() > 0) {
        list_->setCurrentRow(0);
    }
}

void ConnectionProfilesDialog::selectProfile(int row)
{
    if (row < 0 || row >= list_->count()) return;
    fillForm(mgr_->profile(list_->item(row)->text()));
}

void ConnectionProfilesDialog::fillForm(const ConnectionProfile &p)
{
    name_->setText(p.name);
    host_->setText(p.host);
    user_->setText(p.user);
    port_->setValue(p.port);
    workdir_->setText(p.remoteWorkdir);
    scheduler_->setCurrentIndex(scheduler_->findData(static_cast<int>(p.scheduler)));
    launch_->setText(p.launchCmd);
    exe_->setText(p.spartaExe);
    modules_->setPlainText(p.moduleLoads.join('\n'));
    templateEdit_->setPlainText(p.batchTemplate);
}

ConnectionProfile ConnectionProfilesDialog::formToProfile() const
{
    ConnectionProfile p;
    p.name = name_->text().trimmed();
    p.host = host_->text().trimmed();
    p.user = user_->text().trimmed();
    p.port = port_->value();
    p.remoteWorkdir = workdir_->text().trimmed();
    p.scheduler = static_cast<Scheduler>(scheduler_->currentData().toInt());
    p.launchCmd = launch_->text().trimmed();
    p.spartaExe = exe_->text().trimmed().isEmpty() ? "spa_" : exe_->text().trimmed();
    const QStringList mods = modules_->toPlainText().split('\n', Qt::SkipEmptyParts);
    p.moduleLoads = mods;
    p.batchTemplate = templateEdit_->toPlainText();
    return p;
}

void ConnectionProfilesDialog::newProfile()
{
    ConnectionProfile p;
    p.name = "new-cluster";
    p.launchCmd = "srun";
    p.spartaExe = "spa_";
    fillForm(p);
    name_->setFocus();
    name_->selectAll();
}

void ConnectionProfilesDialog::saveCurrent()
{
    const ConnectionProfile p = formToProfile();
    if (p.name.isEmpty() || p.host.isEmpty()) {
        QMessageBox::warning(this, "Save Profile", "A profile needs at least a name and a host.");
        return;
    }
    mgr_->saveProfile(p);
    reloadList(p.name);
}

void ConnectionProfilesDialog::deleteCurrent()
{
    auto *it = list_->currentItem();
    if (!it) return;
    mgr_->removeProfile(it->text());
    reloadList();
    if (list_->count() == 0) newProfile();
}

void ConnectionProfilesDialog::loadDefaultTemplate()
{
    const auto s = static_cast<Scheduler>(scheduler_->currentData().toInt());
    templateEdit_->setPlainText(SchedulerSpec::defaultTemplate(s));
}

// ===========================================================================
// RemoteSubmitDialog
// ===========================================================================

RemoteSubmitDialog::RemoteSubmitDialog(QWidget *parent, RemoteJobManager *mgr,
                                       const QString &deckPath, const QString &deckDir)
    : QDialog(parent), mgr_(mgr), deckPath_(deckPath), deckDir_(deckDir)
{
    setWindowTitle("Submit to Cluster");
    setModal(true);
    resize(720, 620);

    auto *outer = new QVBoxLayout(this);
    auto *form = new QFormLayout;
    outer->addLayout(form);

    auto *prow = new QHBoxLayout;
    profile_ = new QComboBox(this);
    prow->addWidget(profile_, 1);
    auto *mgrBtn = new QPushButton("Manage...", this);
    prow->addWidget(mgrBtn);
    form->addRow("Profile:", prow);

    // derive a sensible job name from the deck: SPARTA decks are conventionally
    // named "in.<case>", so use the <case> part; fall back to the base name.
    QString defName = QFileInfo(deckPath).fileName();
    if (defName.startsWith("in.") && defName.size() > 3) defName = defName.mid(3);
    else defName = QFileInfo(deckPath).completeBaseName();
    if (defName.isEmpty()) defName = "sparta";
    jobName_ = new QLineEdit(defName, this);
    form->addRow("Job name:", jobName_);
    nodes_ = new QSpinBox(this);  nodes_->setRange(1, 100000); nodes_->setValue(1);
    ntasks_ = new QSpinBox(this); ntasks_->setRange(1, 1000000); ntasks_->setValue(1);
    auto *nrow = new QHBoxLayout;
    nrow->addWidget(new QLabel("Nodes:", this)); nrow->addWidget(nodes_);
    nrow->addWidget(new QLabel("Tasks:", this)); nrow->addWidget(ntasks_);
    nrow->addStretch();
    form->addRow("Resources:", nrow);
    walltime_ = new QLineEdit("01:00:00", this);
    form->addRow("Walltime:", walltime_);
    account_ = new QLineEdit(this); account_->setPlaceholderText("optional");
    form->addRow("Account:", account_);
    queue_ = new QLineEdit(this); queue_->setPlaceholderText("partition / queue (optional)");
    form->addRow("Queue:", queue_);

    deck_ = new QLineEdit(QFileInfo(deckPath).fileName(), this);
    deck_->setReadOnly(true);
    form->addRow("Input deck:", deck_);

    dataFiles_ = new QListWidget(this);
    dataFiles_->setMaximumHeight(90);
    auto *addBtn = new QPushButton("Add data files...", this);
    auto *drow = new QVBoxLayout;
    drow->addWidget(dataFiles_);
    drow->addWidget(addBtn);
    form->addRow("Extra files:", drow);

    outer->addWidget(new QLabel("Rendered batch script preview:", this));
    preview_ = new QPlainTextEdit(this);
    preview_->setReadOnly(true);
    preview_->setStyleSheet("QPlainTextEdit { font-family: monospace; }");
    outer->addWidget(preview_, 1);

    auto *bb = new QDialogButtonBox(this);
    auto *submitBtn = bb->addButton("Submit", QDialogButtonBox::AcceptRole);
    bb->addButton(QDialogButtonBox::Cancel);
    outer->addWidget(bb);
    (void)submitBtn;

    connect(mgrBtn, &QPushButton::clicked, this, &RemoteSubmitDialog::manageProfiles);
    connect(addBtn, &QPushButton::clicked, this, &RemoteSubmitDialog::addDataFiles);
    connect(profile_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &RemoteSubmitDialog::updatePreview);
    for (auto *e : {jobName_, walltime_, account_, queue_})
        connect(e, &QLineEdit::textChanged, this, &RemoteSubmitDialog::updatePreview);
    for (auto *s : {nodes_, ntasks_})
        connect(s, QOverload<int>::of(&QSpinBox::valueChanged), this,
                &RemoteSubmitDialog::updatePreview);
    connect(bb, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, this, &QDialog::reject);

    reloadProfiles();
    updatePreview();
}

void RemoteSubmitDialog::reloadProfiles(const QString &select)
{
    profile_->blockSignals(true);
    profile_->clear();
    for (const auto &p : mgr_->profiles()) profile_->addItem(p.name);
    profile_->blockSignals(false);
    if (!select.isEmpty()) {
        const int i = profile_->findText(select);
        if (i >= 0) profile_->setCurrentIndex(i);
    }
    updatePreview();
}

void RemoteSubmitDialog::manageProfiles()
{
    ConnectionProfilesDialog dlg(this, mgr_);
    dlg.exec();
    reloadProfiles(profile_->currentText());
}

void RemoteSubmitDialog::addDataFiles()
{
    const QStringList files = QFileDialog::getOpenFileNames(
        this, "Select additional files to stage", deckDir_);
    for (const QString &f : files) dataFiles_->addItem(f);
    updatePreview();
}

QString RemoteSubmitDialog::profileName() const { return profile_->currentText(); }

SubmitParams RemoteSubmitDialog::params() const
{
    SubmitParams sp;
    sp.jobName = jobName_->text().trimmed();
    sp.nodes = nodes_->value();
    sp.ntasks = ntasks_->value();
    sp.walltime = walltime_->text().trimmed();
    sp.account = account_->text().trimmed();
    sp.queue = queue_->text().trimmed();
    sp.inputDeck = QFileInfo(deckPath_).fileName();
    return sp;
}

QStringList RemoteSubmitDialog::filesToStage() const
{
    QStringList files;
    if (!deckPath_.isEmpty()) files << deckPath_;
    for (int i = 0; i < dataFiles_->count(); ++i) files << dataFiles_->item(i)->text();
    return files;
}

void RemoteSubmitDialog::updatePreview()
{
    if (profile_->currentText().isEmpty()) {
        preview_->setPlainText("# No connection profile selected. Use \"Manage...\" to add one.");
        return;
    }
    const ConnectionProfile p = mgr_->profile(profile_->currentText());
    preview_->setPlainText(SchedulerSpec::renderScript(p, params()));
}

// ===========================================================================
// RemoteJobsPanel
// ===========================================================================

RemoteJobsPanel::RemoteJobsPanel(QWidget *parent, RemoteJobManager *mgr)
    : QWidget(parent), mgr_(mgr)
{
    auto *outer = new QVBoxLayout(this);
    outer->setContentsMargins(4, 4, 4, 4);

    auto *tb = new QHBoxLayout;
    auto mkBtn = [&](const QString &txt) {
        auto *b = new QPushButton(txt, this);
        tb->addWidget(b);
        return b;
    };
    auto *submitBtn = mkBtn("Submit...");
    auto *cancelBtn = mkBtn("Cancel");
    auto *resubBtn = mkBtn("Resubmit");
    auto *pullBtn = mkBtn("Pull");
    auto *openBtn = mkBtn("Open Folder");
    auto *rmBtn = mkBtn("Remove");
    tb->addStretch();
    auto *profBtn = mkBtn("Profiles...");
    outer->addLayout(tb);

    auto *split = new QSplitter(Qt::Vertical, this);
    table_ = new QTableView(split);
    table_->setModel(mgr_->model());
    table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    table_->setSelectionMode(QAbstractItemView::SingleSelection);
    table_->horizontalHeader()->setStretchLastSection(true);
    table_->verticalHeader()->setVisible(false);
    split->addWidget(table_);

    log_ = new QPlainTextEdit(split);
    log_->setReadOnly(true);
    log_->setStyleSheet("QPlainTextEdit { font-family: monospace; }");
    log_->setPlaceholderText("Select a running job to follow its log (log.sparta).");
    split->addWidget(log_);
    split->setStretchFactor(0, 2);
    split->setStretchFactor(1, 3);
    outer->addWidget(split, 1);

    connect(submitBtn, &QPushButton::clicked, this, &RemoteJobsPanel::submitRequested);
    connect(cancelBtn, &QPushButton::clicked, this, &RemoteJobsPanel::cancelSelected);
    connect(resubBtn, &QPushButton::clicked, this, &RemoteJobsPanel::resubmitSelected);
    connect(pullBtn, &QPushButton::clicked, this, &RemoteJobsPanel::pullSelected);
    connect(openBtn, &QPushButton::clicked, this, &RemoteJobsPanel::openSelectedFolder);
    connect(rmBtn, &QPushButton::clicked, this, &RemoteJobsPanel::removeSelected);
    connect(profBtn, &QPushButton::clicked, this, &RemoteJobsPanel::manageProfiles);

    connect(table_->selectionModel(), &QItemSelectionModel::selectionChanged, this,
            &RemoteJobsPanel::onSelectionChanged);
    connect(mgr_, &RemoteJobManager::logChunk, this, &RemoteJobsPanel::onLogChunk);
}

QString RemoteJobsPanel::selectedId() const
{
    const auto rows = table_->selectionModel()->selectedRows();
    if (rows.isEmpty()) return {};
    const int r = rows.first().row();
    if (r < 0 || r >= mgr_->jobCount()) return {};
    return mgr_->jobAt(r).localId;
}

void RemoteJobsPanel::onSelectionChanged()
{
    shownId_ = selectedId();
    log_->clear();
    if (shownId_.isEmpty()) return;
    const int r = mgr_->rowForId(shownId_);
    if (r >= 0) {
        const auto &j = mgr_->jobAt(r);
        if (!j.lastError.isEmpty()) log_->appendPlainText("ERROR: " + j.lastError);
    }
}

void RemoteJobsPanel::onLogChunk(const QString &localId, const QString &text)
{
    if (localId != shownId_) return;
    log_->moveCursor(QTextCursor::End);
    log_->insertPlainText(text);
    log_->moveCursor(QTextCursor::End);
}

void RemoteJobsPanel::cancelSelected()
{
    const QString id = selectedId();
    if (!id.isEmpty()) mgr_->cancel(id);
}

void RemoteJobsPanel::resubmitSelected()
{
    const QString id = selectedId();
    if (!id.isEmpty()) mgr_->resubmit(id, {});
}

void RemoteJobsPanel::pullSelected()
{
    const QString id = selectedId();
    if (!id.isEmpty()) mgr_->pullArtifacts(id);
}

void RemoteJobsPanel::openSelectedFolder()
{
    const int r = mgr_->rowForId(selectedId());
    if (r < 0) return;
    const QString dir = mgr_->jobAt(r).localRunDir;
    if (!dir.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
}

void RemoteJobsPanel::removeSelected()
{
    const QString id = selectedId();
    if (id.isEmpty()) return;
    if (QMessageBox::question(this, "Remove Job",
                              "Remove this job from the list? (does not cancel it on the cluster)")
        == QMessageBox::Yes)
        mgr_->removeJob(id);
}

void RemoteJobsPanel::manageProfiles()
{
    ConnectionProfilesDialog dlg(this, mgr_);
    dlg.exec();
}

// Local Variables:
// c-basic-offset: 4
// End:
