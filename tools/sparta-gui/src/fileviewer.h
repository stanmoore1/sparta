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

#ifndef FILEVIEWER_H
#define FILEVIEWER_H

#include <QPlainTextEdit>

class SpartaGui;

/**
 * @brief Read-only text viewer for displaying file contents
 *
 * FileViewer provides a simple read-only text window for viewing
 * file contents. It's used in the context menu of the code editor
 * to view files referenced in SPARTA input scripts (data files,
 * potential files, etc.). The viewer supports keyboard shortcuts
 * for closing and stopping the simulation.
 */
class FileViewer : public QPlainTextEdit {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param filename Path to file to display
     * @param spartagui Pointer to SpartaGui for sending signals
     * @param title Window title (defaults to filename if empty)
     * @param parent Parent widget
     */
    explicit FileViewer(const QString &filename, SpartaGui *spartagui, const QString &title = "",
                        QWidget *parent = nullptr);

    /**
     * @brief Destructor
     */
    ~FileViewer() override = default;

    FileViewer()                              = delete;
    FileViewer(const FileViewer &)            = delete;
    FileViewer(FileViewer &&)                 = delete;
    FileViewer &operator=(const FileViewer &) = delete;
    FileViewer &operator=(FileViewer &&)      = delete;

private slots:
    void quit();    ///< Close the viewer window
    void stopRun(); ///< Stop the running simulation

protected:
    /**
     * @brief Event filter for keyboard shortcuts
     * @param watched Object being watched
     * @param event Event to filter
     * @return true if event handled, false otherwise
     */
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    QString fileName;     ///< Path to the displayed file
    SpartaGui *spartagui; ///< Main widget pointer for receiving signals
};

#endif
// Local Variables:
// c-basic-offset: 4
// End:
