/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   ParaViewExportDialog: a thin GUI shell over the pure ParaviewExport
   builders (paraviewexport.h).  It lets the user convert a SPARTA surface
   or grid to ParaView .pvd format by running the bundled pvpython scripts
   (surf2paraview.py / grid2paraview.py) as a subprocess, streams their
   output to a log, and optionally launches ParaView on the result.

   The GUI does not reimplement the conversions in C++: the scripts depend
   on ParaView's VTK Python modules and must run under pvpython/pvbatch.
------------------------------------------------------------------------- */

#ifndef PARAVIEWDIALOG_H
#define PARAVIEWDIALOG_H

#include "paraviewexport.h"

#include <QDialog>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QProcess;
class QPushButton;
class QSpinBox;
class QStackedWidget;

/**
 * @brief Dialog that converts SPARTA surface/grid data to ParaView format.
 *
 * Constructed with the directory of the current input deck so the file
 * pickers and default output name start there.  Locates pvpython, paraview
 * and the bundled tools/paraview scripts (each overridable in the dialog and
 * remembered in QSettings), then runs the conversion and opens ParaView.
 */
class ParaViewExportDialog : public QDialog {
    Q_OBJECT

public:
    /**
     * @param parch   parent widget
     * @param deckDir directory of the current deck (pre-fills the pickers)
     */
    explicit ParaViewExportDialog(QWidget *parent, const QString &deckDir);
    ~ParaViewExportDialog() override;

private slots:
    void onModeChanged();
    void browseInput();
    void browseResults();
    void browsePvpython();
    void browseParaview();
    void updatePreview();
    void runConversion();
    void onProcessOutput();
    void onProcessFinished(int exitCode, int exitStatus);

private:
    ParaviewExport::Settings collectSettings() const;
    QStringList expandResultGlob() const;   ///< turn the results field into concrete files
    QString locateScript() const;           ///< full path to the mode's bundled .py
    static QString findScriptsDir();         ///< the tools/paraview directory
    void setBusy(bool busy);
    void log(const QString &line);

    QString deckDir_;
    QString scriptsDir_;

    QComboBox *mode_ = nullptr;
    QLineEdit *inputEdit_ = nullptr;
    QLineEdit *outputEdit_ = nullptr;
    QLineEdit *resultsEdit_ = nullptr;
    QStackedWidget *modeOpts_ = nullptr;    // surface options vs grid options
    QCheckBox *exodus_ = nullptr;
    QSpinBox *chunk_[3] = {nullptr, nullptr, nullptr};
    QLineEdit *pvpythonEdit_ = nullptr;
    QLineEdit *paraviewEdit_ = nullptr;
    QCheckBox *openAfter_ = nullptr;
    QLabel *preview_ = nullptr;
    QPlainTextEdit *log_ = nullptr;
    QPushButton *runButton_ = nullptr;

    QProcess *proc_ = nullptr;
    QString pendingPvd_;                     // .pvd to open when the run succeeds
};

#endif // PARAVIEWDIALOG_H
