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

#ifndef VIEWERPANEL_H
#define VIEWERPANEL_H

#include <QString>
#include <QWidget>

class QAction;
class QStackedWidget;
class QTabBar;
class QToolButton;

class ImageViewer;
class SlideShow;
class ViewerSource;

/**
 * @brief The one place the application shows a picture of a simulation.
 *
 * There used to be three: a docked image viewer rendering snapshots through
 * SPARTA, a docked slide show stepping through written frames, and a separate
 * 3D window. They overlapped enough to be confusing -- two of them labelled a
 * button "Zoom in by 10 percent" for operations that were not the same thing --
 * and the two docked ones competed for the same corner of the window.
 *
 * This is a tab bar over a stack. Each page is one @ref ViewerSource, whole and
 * unmodified: the panel does not rearrange anything inside a source, it only
 * decides which is in front. That is deliberate. All three carry sizing and
 * layout code that took real debugging to get right, and re-laying it out
 * wholesale would put every one of those fixes back at risk.
 *
 * What the panel does hoist out is the part that was the same in all three and
 * spelled differently in each: saving the picture and copying it. Those were
 * "Save Image As..." in one, a button tipped "Export to image file" in
 * another, and "Save the current 3D view to an image file" in the third, with
 * one of the three offering no copy at all -- three names, three places and
 * three shortcuts for one idea. They are now one pair of controls beside the
 * tabs, acting on whichever source is in front through
 * @ref ViewerSource::currentImage(), which is what that method was declared
 * for. Anything genuinely specific to a viewer -- a camera, a frame slider, a
 * filter -- stays inside it, because those are not the same operation even
 * when they share an icon.
 *
 * All three tabs are there from the start, whether or not the viewer behind
 * one has been built yet.  A tab that appears only once its source has
 * something to show cannot teach anyone that the source exists, and the panel
 * used to open with one tab, or none, and grow silently during a run.  Behind
 * a tab with nothing yet is a card naming the command or menu entry that fills
 * it, so "what do I do to get a slide show?" is answered in the place the
 * question is asked rather than in a tooltip.
 */
class ViewerPanel : public QWidget {
    Q_OBJECT

public:
    /** @brief Which of the viewers is in front */
    enum Source { Snapshot, Sequence, Scene, NSources };

    explicit ViewerPanel(QWidget *parent = nullptr);
    ~ViewerPanel() override;

    ViewerPanel(const ViewerPanel &)            = delete;
    ViewerPanel &operator=(const ViewerPanel &) = delete;

    /**
     * @brief Install a source behind its tab, replacing the placeholder card.
     *
     * The tab already exists -- the panel makes all of them up front.  The 3D
     * scene is the exception: without VTK there is no such viewer to build, so
     * that tab is not created at all and the bar comes up with two.
     */
    void addSource(Source which, ViewerSource *source);

    /** @brief Is this build able to offer this source at all? */
    [[nodiscard]] static bool sourceAvailable(Source which);

    /** @brief Replace a registered source's widget (a new run builds a new one) */
    void replaceSource(Source which, ViewerSource *source);

    [[nodiscard]] bool hasSource(Source which) const { return sources[which] != nullptr; }
    [[nodiscard]] ViewerSource *source(Source which) const { return sources[which]; }
    [[nodiscard]] Source currentSource() const;

    /// @name Typed accessors, so call sites keep using each viewer's own API
    /// @{
    [[nodiscard]] SlideShow *sequence() const;
    [[nodiscard]] ImageViewer *snapshot() const;
    /// @}

    /**
     * @brief Bring a source to the front.
     * @param userAsked true when this came from the user rather than from
     *        content arriving on its own
     *
     * Content that shows up in the background must not take the view away from
     * whatever the user chose to look at, so a run writing frames only switches
     * to them if the user has not picked something since the run began.
     */
    void showSource(Source which, bool userAsked = false);

    /** @brief Forget that the user picked a source (called when a run starts) */
    void unlockSource() { sourceLockedByUser = false; }

    /** @brief Re-check which tabs have anything behind them */
    void refreshTabs();

    /** @brief The panel's title, including which source is in front */
    [[nodiscard]] QString title() const;

signals:
    /** @brief A different source came to the front */
    void sourceChanged(int which);

    /** @brief The panel's title changed (source switched, or a source renamed) */
    void titleChanged(const QString &title);

    /** @brief A source asked to be closed; the host closes the whole panel */
    void closeRequested();

protected:
    /**
     * @brief The current page's minimum, not the largest page's.
     *
     * QStackedWidget reports the maximum over every page, so the 3D scene's
     * 320x240 floor would apply to the panel even while a different source is
     * showing, and the image viewer deliberately sets no minimum at all so a
     * multi-thousand-pixel render cannot force the window wider than the
     * screen. Reporting the visible page keeps both of those true.
     */
    [[nodiscard]] QSize minimumSizeHint() const override;

private slots:
    /// @brief Save the picture the source in front is showing.
    void saveCurrentImage();
    /// @brief Copy the picture the source in front is showing.
    void copyCurrentImage();

private:
    void wireSource(Source which, ViewerSource *source);
    void updateTab(Source which);
    /// @brief Point the shared controls at whatever is in front now.
    void syncSharedControls();
    /// @brief The tab index carrying @p which, or -1 in a build without it.
    [[nodiscard]] int tabOf(Source which) const;

    QTabBar *tabs           = nullptr;
    QStackedWidget *stack   = nullptr;
    QAction *saveAction     = nullptr;  ///< shared "Save Image As...", all sources
    QAction *copyAction     = nullptr;  ///< shared "Copy Image", all sources
    QToolButton *menuButton = nullptr;  ///< the front source's own menu, if it has one
    ViewerSource *sources[NSources] = {};
    /// Per source: a two-page stack holding the placeholder card and, once it
    /// exists, the viewer itself.  Present from construction, so a tab always
    /// has something to show.
    QStackedWidget *slots_[NSources] = {};
    int pageOf[NSources]    = {-1, -1, -1};   ///< stack index per source, -1 if absent
    QString names[NSources];                  ///< last title each source announced
    bool sourceLockedByUser = false;
};

#endif

// Local Variables:
// c-basic-offset: 4
// End:
