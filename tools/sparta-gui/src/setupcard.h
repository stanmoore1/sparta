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

#ifndef SETUPCARD_H
#define SETUPCARD_H

#include <QFrame>
#include <QString>

class QLabel;
class QPushButton;

/**
 * @brief The banner shown while SPARTA-GUI has no simulator behind it.
 *
 * Without a SPARTA shared library the application can still edit, highlight,
 * complete and save an input deck -- everything except run one.  It used to
 * refuse to reach that state: the first launch put up a modal dialog offering
 * download, browse or exit, and would not proceed past it, so a user with a
 * deck to look at and no library had the choice of downloading one or leaving.
 *
 * This says the same three things from a strip above the editor instead, so
 * the application comes up, the deck is readable, and acquiring a library is
 * something to do next rather than a toll gate.  It stays until there is a
 * library and then goes away by itself.
 *
 * It knows nothing about how a library is found or loaded -- it emits what the
 * user asked for and is told what came of it.
 */
class SetupCard : public QFrame {
    Q_OBJECT

public:
    explicit SetupCard(QWidget *parent = nullptr);

    /// @brief Show @p message as the reason the last attempt did not work.
    void setError(const QString &message);

    /// @brief Drop any error message, leaving the standing explanation.
    void clearError();

    /// @brief Whether this build can offer a pre-compiled download at all.
    [[nodiscard]] bool canDownload() const;

signals:
    void downloadRequested(); ///< fetch the pre-compiled library for this platform
    void browseRequested();   ///< pick a library file from the filesystem
    void helpRequested();     ///< explain what a SPARTA shared library is

private:
    QLabel *explanation;
    QLabel *problem;
    QPushButton *downloadButton;
};

#endif // SETUPCARD_H

// Local Variables:
// c-basic-offset: 4
// End:
