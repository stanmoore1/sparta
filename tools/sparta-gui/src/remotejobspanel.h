/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   The remote-execution UI: a docked Cluster Jobs panel (job table + toolbar
   + per-job log tail) and the two dialogs it uses -- RemoteSubmitDialog
   (compose and preview a submission) and ConnectionProfilesDialog (CRUD over
   connection profiles and the batch-script template).  All the stateful
   ssh/scp/rsync work is in RemoteJobManager; these are thin views over it.
------------------------------------------------------------------------- */

#ifndef REMOTEJOBSPANEL_H
#define REMOTEJOBSPANEL_H

#include "remotejob.h"

#include <QDialog>
#include <QWidget>

class RemoteJobManager;
class QComboBox;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;
class QSpinBox;
class QTableView;
class QListWidget;

// ---------------------------------------------------------------------------

/** @brief Dialog to edit/create connection profiles and the batch template. */
class ConnectionProfilesDialog : public QDialog {
    Q_OBJECT
public:
    ConnectionProfilesDialog(QWidget *parent, RemoteJobManager *mgr);

private slots:
    void selectProfile(int row);
    void newProfile();
    void saveCurrent();
    void deleteCurrent();
    void loadDefaultTemplate();

private:
    void reloadList(const QString &select = QString());
    void fillForm(const Remote::ConnectionProfile &p);
    Remote::ConnectionProfile formToProfile() const;

    RemoteJobManager *mgr_;
    QListWidget *list_ = nullptr;
    QLineEdit *name_ = nullptr, *host_ = nullptr, *user_ = nullptr, *workdir_ = nullptr;
    QLineEdit *launch_ = nullptr, *exe_ = nullptr;
    QSpinBox *port_ = nullptr;
    QComboBox *scheduler_ = nullptr;
    QPlainTextEdit *modules_ = nullptr, *templateEdit_ = nullptr;
};

// ---------------------------------------------------------------------------

/** @brief Dialog to compose and preview a remote submission. */
class RemoteSubmitDialog : public QDialog {
    Q_OBJECT
public:
    RemoteSubmitDialog(QWidget *parent, RemoteJobManager *mgr,
                       const QString &deckPath, const QString &deckDir);

    QString profileName() const;
    Remote::SubmitParams params() const;
    QStringList filesToStage() const;

private slots:
    void manageProfiles();
    void addDataFiles();
    void updatePreview();

private:
    void reloadProfiles(const QString &select = QString());

    RemoteJobManager *mgr_;
    QString deckPath_, deckDir_;
    QComboBox *profile_ = nullptr;
    QLineEdit *jobName_ = nullptr, *walltime_ = nullptr, *account_ = nullptr, *queue_ = nullptr;
    QSpinBox *nodes_ = nullptr, *ntasks_ = nullptr;
    QLineEdit *deck_ = nullptr;
    QListWidget *dataFiles_ = nullptr;
    QPlainTextEdit *preview_ = nullptr;
};

// ---------------------------------------------------------------------------

/** @brief Docked panel: the job table, a toolbar, and the selected job's log. */
class RemoteJobsPanel : public QWidget {
    Q_OBJECT
public:
    RemoteJobsPanel(QWidget *parent, RemoteJobManager *mgr);

signals:
    void submitRequested();   ///< toolbar "Submit..." -> SpartaGui opens the dialog

private slots:
    void onSelectionChanged();
    void onLogChunk(const QString &localId, const QString &text);
    void cancelSelected();
    void resubmitSelected();
    void pullSelected();
    void openSelectedFolder();
    void removeSelected();
    void manageProfiles();

private:
    QString selectedId() const;

    RemoteJobManager *mgr_;
    QTableView *table_ = nullptr;
    QPlainTextEdit *log_ = nullptr;
    QString shownId_;
};

#endif // REMOTEJOBSPANEL_H
