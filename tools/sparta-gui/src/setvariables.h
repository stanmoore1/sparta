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

#ifndef SET_VARIABLES_H
#define SET_VARIABLES_H

#include <QDialog>
#include <QHash>
#include <QList>
#include <QPair>
#include <QString>

/**
 * @brief Dialog for editing SPARTA index-style variable definitions
 *
 * SetVariables provides a dialog for managing name-value pairs that
 * will be used as index-style variables in SPARTA input scripts.
 * Users can add, delete, and edit variable definitions. The dialog
 * shows the variables that the input script defines or uses.
 */
class SetVariables : public QDialog {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param vars Reference to list of variable name-value pairs (modified in place)
     * @param parent Parent widget
     */
    /**
     * @param vars    name/value pairs to edit, updated in place on accept
     * @param script  the value each variable is given in the input deck, by
     *                name.  Used only to show which entries override the deck;
     *                a name absent from it is one the deck does not define.
     * @param parent  parent widget
     */
    explicit SetVariables(QList<QPair<QString, QString>> &vars,
                          const QHash<QString, QString> &script = {},
                          QWidget *parent = nullptr);

    /**
     * @brief Destructor
     */
    ~SetVariables() override = default;

    SetVariables()                                = delete;
    SetVariables(const SetVariables &)            = delete;
    SetVariables(SetVariables &&)                 = delete;
    SetVariables &operator=(const SetVariables &) = delete;
    SetVariables &operator=(SetVariables &&)      = delete;

private slots:
    /**
     * @brief Accept dialog and update variable list
     */
    void accept() override;

    /**
     * @brief Add a new empty variable row
     */
    void addRow();

    /**
     * @brief Delete the currently selected variable row
     */
    void delRow();

private:
    QList<QPair<QString, QString>> &vars; ///< Reference to variable list
    QHash<QString, QString> scriptValues; ///< value each name has in the deck
    class QVBoxLayout *layout;            ///< Dialog layout
};

#endif

// Local Variables:
// c-basic-offset: 4
// End:
