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

#ifndef VIEWERWINDOW_H
#define VIEWERWINDOW_H

#include <QMainWindow>
#include <QString>

class ImageViewer;
class SlideShow;
class SpartaGui;
class SpartaWrapper;
class ViewerSource;

/**
 * @brief A viewer source on its own, as a top-level window.
 *
 * Most of the time a source lives in the viewer panel. Three paths need one
 * outside the main window instead: inspecting a restart file, opening image or
 * movie files from the File menu, and the @c -i command-line option, which
 * runs with no main window at all.
 *
 * This is where the window-ness lives -- title, icon, platform window flags,
 * and turning the source's close request into an actual window close. Keeping
 * it out of the sources themselves is what lets the same class be docked.
 */
class ViewerWindow : public QMainWindow {
    Q_OBJECT

public:
    ~ViewerWindow() override;

    ViewerWindow(const ViewerWindow &)            = delete;
    ViewerWindow &operator=(const ViewerWindow &) = delete;

    /// @brief A window showing a SPARTA-rendered snapshot of @p file.
    static ViewerWindow *forSnapshot(const QString &file, SpartaWrapper *sparta,
                                     SpartaGui *spartagui, QWidget *parent = nullptr);

    /// @brief A window showing a sequence of image files, starting at @p file.
    static ViewerWindow *forSequence(const QString &file, SpartaGui *spartagui = nullptr,
                                     QWidget *parent = nullptr);

    [[nodiscard]] ViewerSource *source() const { return view; }

    /// @name Typed accessors, so callers keep using the source's own API
    /// @{
    [[nodiscard]] SlideShow *sequence() const;
    [[nodiscard]] ImageViewer *snapshot() const;
    /// @}

private:
    explicit ViewerWindow(ViewerSource *source, const QString &titlePrefix, QWidget *parent);

    ViewerSource *view = nullptr;
    QString prefix;   ///< what the title says before the file name
};

#endif

// Local Variables:
// c-basic-offset: 4
// End:
