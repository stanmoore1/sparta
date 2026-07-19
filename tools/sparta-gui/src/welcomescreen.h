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

#ifndef WELCOMESCREEN_H
#define WELCOMESCREEN_H

#include <QStringList>
#include <QWidget>

class QLabel;
class QListWidget;

/**
 * @brief Landing view shown in the editor area on startup.
 *
 * The welcome screen is a purely additive "front door": it lists the recent
 * files and shows a thumbnail gallery of the bundled SPARTA @c examples/ decks
 * so a new user can get from launch to a running simulation in one click. It
 * owns no application state -- it is fed the recent-file list and the examples
 * directory by SpartaGui and emits requests back; SpartaGui does the actual
 * opening (reusing openFile()/openExamplePath()/newDocument()).
 */
class WelcomeScreen : public QWidget {
    Q_OBJECT

public:
    explicit WelcomeScreen(QWidget *parent = nullptr);

    /** @brief Set the recent-file list (absolute paths) and rebuild that column */
    void setRecentFiles(const QStringList &files);

    /** @brief Set the SPARTA examples directory and (re)build the gallery */
    void setExamplesDir(const QString &dir);

signals:
    /** @brief The user picked a recent file (absolute path) to open */
    void openFileRequested(const QString &path);
    /** @brief The user picked an example deck (absolute in.* path) to open */
    void openExampleRequested(const QString &path);
    /** @brief The user asked to start a new (blank) input file */
    void newFileRequested();
    /** @brief The user asked to browse for a file to open */
    void browseRequested();

private:
    /** @brief The shipped :/examples thumbnail for a deck, or a null pixmap if
     *  the deck has no gallery image (such decks are omitted from the gallery). */
    QPixmap thumbnailFor(const QString &inpath) const;

    void rebuildRecents();
    void rebuildExamples();

    QListWidget *recentList;
    QListWidget *exampleList;
    QLabel *recentEmpty;
    QStringList recentFiles;
    QString examplesDir;
};

#endif // WELCOMESCREEN_H

// Local Variables:
// c-basic-offset: 4
// End:
