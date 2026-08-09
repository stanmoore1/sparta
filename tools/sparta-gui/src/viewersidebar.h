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

#ifndef VIEWERSIDEBAR_H
#define VIEWERSIDEBAR_H

#include <QList>
#include <QSize>
#include <QString>
#include <QWidget>

class QAbstractButton;
class QGridLayout;
class QToolButton;
class QVBoxLayout;

/**
 * @brief The snapshot viewer's control column: one labelled row per subject.
 *
 * The viewer used to spread its controls over two places that had nothing to do
 * with each other: a row of eight unlabelled icon buttons along the top (the
 * render toggles) and a column of eight "... " buttons down the right (the
 * settings dialog tabs).  Both are per-subject -- particles, grid, surfaces,
 * box -- so the pairs belonged together.  This widget puts them on one line:
 *
 *     [on/off]  Particles...          [extra toggles]
 *
 * The buttons themselves are still built by the viewer and simply handed over,
 * so their object names, tooltips, mnemonics and connections are unchanged and
 * every existing slot keeps working.  The sidebar only owns the arrangement.
 *
 * It can also collapse to a narrow strip, which is the point of the exercise: a
 * viewer sharing a window with the editor and the log has little width to
 * spare, and the controls are worth more when they can get out of the way than
 * when they squeeze the render.
 */
class ViewerSidebar : public QWidget {
    Q_OBJECT

public:
    explicit ViewerSidebar(QWidget *parent = nullptr);

    /// @brief Add a labelled widget above the rows (the mixture chooser).
    void addHeader(const QString &label, QWidget *widget);

    /**
     * @brief Add one subject row.
     * @param toggle primary on/off button, or nullptr for a subject that has
     *               nothing to switch (grid planes, camera, color maps) -- the
     *               slot is then left empty so the names still line up
     * @param name   the button that opens the settings dialog for this subject
     * @param extra  secondary toggles, shown at the right end of the row
     */
    void addRow(QAbstractButton *toggle, QAbstractButton *name,
                const QList<QAbstractButton *> &extra = {});

    /// @brief Add a divider and a trailing widget (the Help button).
    void addFooter(QWidget *widget);

    /// @brief Is the sidebar currently collapsed to its handle?
    [[nodiscard]] bool isCollapsed() const { return collapsed; }

public slots:
    /// @brief Collapse to a narrow handle, or expand back to the full column.
    void setCollapsed(bool on);

signals:
    /// @brief Emitted whenever the collapsed state changes, however it changed.
    void collapsedChanged(bool collapsed);

private:
    /// index of the trailing stretch in @ref bodyBox; new items go in front of it
    [[nodiscard]] int tail() const;

    QWidget *body;         ///< everything that is hidden when collapsed
    QVBoxLayout *bodyBox;  ///< header widgets, the row grid, footer widgets
    QGridLayout *rows;     ///< toggle | name | extras, one line per subject
    QToolButton *handle;   ///< the strip that brings a collapsed sidebar back
    bool collapsed = false;
};

#endif // VIEWERSIDEBAR_H

// Local Variables:
// c-basic-offset: 4
// End:
