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

class QStackedWidget;
class QTabBar;

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
 * layout code that took real debugging to get right, and re-laying it out to
 * share one toolbar would put every one of those fixes back at risk for a
 * cosmetic gain.
 *
 * A source with nothing to show keeps its tab, disabled, with a tooltip saying
 * how to fill it -- the empty slide show panel used to give no clue at all.
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
     * @brief Install a source and give it a tab.
     *
     * Only registered sources get one, which is how the 3D scene disappears
     * cleanly from a build without VTK: it is simply never registered, and the
     * bar comes up with two tabs instead of three.
     */
    void addSource(Source which, ViewerSource *source);

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

private:
    void wireSource(Source which, ViewerSource *source);
    void updateTab(Source which);

    QTabBar *tabs           = nullptr;
    QStackedWidget *stack   = nullptr;
    ViewerSource *sources[NSources] = {};
    int pageOf[NSources]    = {-1, -1, -1};   ///< stack index per source, -1 if absent
    QString names[NSources];                  ///< last title each source announced
    bool sourceLockedByUser = false;
};

#endif

// Local Variables:
// c-basic-offset: 4
// End:
