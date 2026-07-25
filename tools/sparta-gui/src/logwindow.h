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

#ifndef LOGWINDOW_H
#define LOGWINDOW_H

#include <QPlainTextEdit>

class FlagWarnings;
class SpartaGui;
class QLabel;

/**
 * @brief Text viewer for SPARTA log output with warning/error detection
 *
 * LogWindow specializes QPlainTextEdit for SPARTA log viewing.
 * It highlights warnings and errors, detects embedded YAML data for
 * extraction, provides navigation between warnings, and makes error
 * URLs clickable.
 */
class LogWindow : public QPlainTextEdit {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param filename Name of the input file the run belongs to (used for default save-file names)
     * @param spartagui Pointer to SpartaGui for sending signals
     * @param parent Parent widget
     */
    LogWindow(const QString &filename, SpartaGui *spartagui, QWidget *parent = nullptr);

    /**
     * @brief Destructor
     */
    ~LogWindow() override;

    LogWindow()                             = delete;
    LogWindow(const LogWindow &)            = delete;
    LogWindow(LogWindow &&)                 = delete;
    LogWindow &operator=(const LogWindow &) = delete;
    LogWindow &operator=(LogWindow &&)      = delete;

private slots:
    void extractYaml();  ///< Extract YAML data to separate file
    void quit();         ///< Quit the entire application (via SpartaGui::quit)
    void saveAs();       ///< Save log to file
    void stopRun();      ///< Stop running simulation
    void runBuffer();    ///< Start running simulation
    void nextWarning();  ///< Navigate to next warning
    void openErrorUrl(); ///< Open error documentation URL in browser

protected:
    /** @brief Keep the warning badge in the strip reserved under the text */
    void resizeEvent(QResizeEvent *event) override;
    /** @brief Place the badge once the widget has a real size */
    void showEvent(QShowEvent *event) override;

    /**
     * @brief Handle window close event
     * @param event Close event
     */
    void closeEvent(QCloseEvent *event) override;

    /**
     * @brief Handle double-click to open URLs
     * @param event Mouse event
     */
    void mouseDoubleClickEvent(QMouseEvent *event) override;

    /**
     * @brief Show context menu with log-specific actions
     * @param event Context menu event
     */
    void contextMenuEvent(QContextMenuEvent *event) override;

    /**
     * @brief Event filter for keyboard shortcuts
     * @param watched Object being watched
     * @param event Event to filter
     * @return true if event handled, false otherwise
     */
    bool eventFilter(QObject *watched, QEvent *event) override;

    /**
     * @brief Check if log contains embedded YAML data
     * @return true if YAML data detected, false otherwise
     */
    bool checkYaml();

private:
    void placeWarningBadge(); ///< centre the badge in its reserved strip
    QString filename;       ///< Input file name used to derive default save-file names
    SpartaGui *spartagui;   ///< Main widget pointer for receiving signals
    QString errorurl;       ///< URL of last detected error
    FlagWarnings *warnings; ///< Warning highlighter
    QLabel *summary;
    QWidget *warningBadge = nullptr; ///< warning count, in its own strip below the text        ///< Summary label for warning count
};

#endif
// Local Variables:
// c-basic-offset: 4
// End:
