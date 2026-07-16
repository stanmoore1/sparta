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

#ifndef PREFERENCES_H
#define PREFERENCES_H

#include <QDialog>

class QDialogButtonBox;
class QFont;
class QSettings;
class QTabWidget;
class SpartaWrapper;
class SpartaGui;

/**
 * @brief Preferences/Settings dialog for SPARTA-GUI
 *
 * This dialog provides a tabbed interface for configuring various aspects
 * of SPARTA-GUI including:
 * - General settings (SPARTA library path, plugins, etc.)
 * - Accelerator package settings
 * - Image viewer defaults
 * - Editor appearance and behavior
 * - Chart viewer settings
 *
 * Settings are persisted using QSettings and loaded on startup.
 */
class Preferences : public QDialog {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param sparta Pointer to SpartaWrapper for querying SPARTA configuration
     * @param spartagui Pointer to SpartaGui for sending signals
     * @param parent Parent widget
     */
    explicit Preferences(SpartaWrapper *sparta, SpartaGui *spartagui, QWidget *parent = nullptr);

    /**
     * @brief Destructor
     */
    ~Preferences() override;

    Preferences()                               = delete;
    Preferences(const Preferences &)            = delete;
    Preferences(Preferences &&)                 = delete;
    Preferences &operator=(const Preferences &) = delete;
    Preferences &operator=(Preferences &&)      = delete;

private slots:
    /**
     * @brief Handle dialog acceptance - saves all settings
     */
    void accept() override;

public:
    /**
     * @brief Set flag indicating application needs restart
     * @param val true if restart needed, false otherwise
     *
     * Some settings require restarting the application to take effect.
     */
    void setRelaunch(bool val) { needRelaunch = val; }

private:
    QTabWidget *tabWidget;       ///< Tab widget for preference categories
    QDialogButtonBox *buttonBox; ///< Dialog buttons (OK, Cancel)
    QSettings *settings;         ///< Qt settings storage
    SpartaWrapper *sparta;       ///< SPARTA interface for configuration queries
    SpartaGui *spartagui;        ///< Main widget pointer for receiving signals
    bool needRelaunch;           ///< Flag indicating restart is needed
};

// individual tabs

/**
 * @brief Preferences Tab for General SPARTA-GUI Settings
 */
class GeneralTab : public QWidget {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param settings Pointer to QSettings for storing preferences
     * @param sparta Pointer to SpartaWrapper for querying SPARTA configuration
     * @param spartagui Pointer to SpartaGui for sending signals
     * @param parent Parent widget
     */
    explicit GeneralTab(QSettings *settings, SpartaWrapper *sparta, SpartaGui *spartagui,
                        QWidget *parent = nullptr);

private slots:
    void downloadPlugin();
    void pluginPath();
    void examplesPath();
    void newAllFont();
    void newTextFont();

private:
    void updateFonts(const QFont &all, const QFont &text);
    QSettings *settings;
    SpartaWrapper *sparta;
    SpartaGui *spartagui;
};

/**
 * @brief Preferences Tab for SPARTA Accelerator settings
 */
class AcceleratorTab : public QWidget {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param settings Pointer to QSettings for storing preferences
     * @param sparta Pointer to SpartaWrapper for querying available accelerator packages
     * @param parent Parent widget
     */
    explicit AcceleratorTab(QSettings *settings, SpartaWrapper *sparta, QWidget *parent = nullptr);
    /** Constants for selecting SPARTA accelerator package */
    enum AccelType {
        None,  ///< no accelerator
        Kokkos ///< KOKKOS package
    };

private slots:
    void updateAccel();

private:
    QSettings *settings;
    SpartaWrapper *sparta;
};

/**
 * @brief Preferences Tab for Snapshot Viewer Settings
 */
class SnapshotTab : public QWidget {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param settings Pointer to QSettings for storing preferences
     * @param parent Parent widget
     */
    explicit SnapshotTab(QSettings *settings, QWidget *parent = nullptr);

private slots:
    void chooseVdw();
    void chooseBond();

private:
    QSettings *settings;
};

/**
 * @brief Preferences Tab for SPARTA-GUI Editor Settings
 */
class EditorTab : public QWidget {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param settings Pointer to QSettings for storing preferences
     * @param parent Parent widget
     */
    explicit EditorTab(QSettings *settings, QWidget *parent = nullptr);

private:
    QSettings *settings;
};

/**
 * @brief Preferences Tab for SPARTA-GUI Charts Viewer Settings
 */
class ChartsTab : public QWidget {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param settings Pointer to QSettings for storing preferences
     * @param parent Parent widget
     */
    explicit ChartsTab(QSettings *settings, QWidget *parent = nullptr);

private:
    QSettings *settings;
};

#endif

// Local Variables:
// c-basic-offset: 4
// End:
