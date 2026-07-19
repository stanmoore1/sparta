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

#ifndef SPARTAGUI_H
#define SPARTAGUI_H

#include <QMainWindow>

#include <QList>
#include <QPair>
#include <QString>
#include <string>
#include <vector>

#include "spartawrapper.h"

// forward declarations

class QAction;
class QEvent;
class QFont;
class QLabel;
class QMenu;
class QMenuBar;
class QProgressBar;
class QSettings;
class QStatusBar;
class QTimer;
class QWidget;

class ChartWindow;
class CodeEditor;
class GeneralTab;
class Highlighter;
class ImageViewer;
class PanelManager;
class SpartaRunner;
class LogWindow;
class Preferences;
class SlideShow;
class StdCapture;
class URLDownloader;

/**
 * @brief Main application window for SPARTA-GUI
 *
 * SpartaGui is the central component of the SPARTA-GUI application, serving as the main
 * window that coordinates all other components. It manages:
 * - The code editor for SPARTA input scripts with syntax highlighting
 * - File operations (open, save, recent files)
 * - SPARTA simulation execution and control
 * - Visualization windows (images, charts, log output)
 * - Application preferences and settings
 *
 * The class integrates with Qt's main window framework and provides menu actions,
 * toolbars, and status bar components. It uses a SpartaWrapper to interface with
 * the SPARTA library and SpartaRunner to execute simulations in a separate thread.
 *
 * @see CodeEditor for the text editor component
 * @see ChartWindow for the charts window component
 * @see LogWindow for the log output window component
 * @see ImageViewer for the snapshot image window component
 * @see SlideShow for the slide show viewer window component
 * @see Preferences for the preferences window component
 * @see SpartaRunner for simulation execution in a separate thread
 * @see SpartaWrapper for SPARTA library interface
 */
class SpartaGui : public QMainWindow {
    Q_OBJECT

    friend class CodeEditor;
    friend class Preferences;
    friend class AcceleratorTab;
    friend class GeneralTab;

public:
    /**
     * @brief Construct the main application window
     * @param parent    Parent widget (typically nullptr for main window)
     * @param filename  Optional file to open on startup
     * @param width     Optional main editor window width override
     * @param height    Optional main editor window height override
     *
     * Initializes the main window, sets up the UI components, loads preferences,
     * initializes the SPARTA library, and optionally opens a file if provided.
     */
    SpartaGui(QWidget *parent = nullptr, const QString &filename = QString(), int width = 0,
              int height = 0);

    /**
     * @brief Destructor
     *
     * Cleans up resources including dynamically created widgets and SPARTA instances.
     */
    ~SpartaGui() override;

    SpartaGui()                             = delete;
    SpartaGui(const SpartaGui &)            = delete;
    SpartaGui(SpartaGui &&)                 = delete;
    SpartaGui &operator=(const SpartaGui &) = delete;
    SpartaGui &operator=(SpartaGui &&)      = delete;

protected:
    /** @brief Open a file in the editor */
    void openFile(const QString &filename);

    /** @brief Open a file in a read-only viewer dialog */
    void viewFile(const QString &filename);

    /** @brief Read a restart file into SPARTA and open the inspection windows */
    void inspectFile(const QString &filename);

    /** @brief Write current editor content to a file */
    void writeFile(const QString &filename);

    /** @brief Update the recent files list */
    void updateRecents(const QString &filename = "");

    /** @brief Delete all variables defined in the SPARTA instance */
    void clearVariables();

    /** @brief Rebuild the variables list from the editor buffer */
    void updateVariables();

    /**
     * @brief Execute a SPARTA simulation
     * @param use_buffer If true, runs from editor buffer; if false, saves and runs from file
     */
    void doRun(bool use_buffer);

    /** @brief Initialize and start a new SPARTA instance */
    void startSparta();

    /** @brief Handle completion of a SPARTA run */
    void runDone();

    /** @brief Perform an auto-save of the current file */
    void autoSave();

    /**
     * @brief Update the editor font
     * @param newfont The font to apply to the editor
     */
    void setFont(const QFont &newfont);

    /** @brief Clean up the inspect file dialog list */
    void purgeInspectList();

    /**
     * @brief Event filter for handling special events
     * @param watched Object being watched
     * @param event Event to filter
     * @return true if event was handled
     */
    bool eventFilter(QObject *watched, QEvent *event) override;

public slots:
    /** @brief Quit the application */
    void quit();

    /** @brief Stop a running SPARTA simulation */
    void stopRun();

    /** @brief Run SPARTA with content from editor buffer */
    void runBuffer() { doRun(true); }

private slots:
    /** @brief Create a new document */
    void newDocument();

    /** @brief Open an existing file */
    void open();

    /** @brief View a file in read-only mode */
    void view();

    /** @brief Open one or more image files in a standalone snapshot viewer */
    void openImages();

    /** @brief Select and inspect a restart file */
    void inspect();

    /** @brief Open a file from the recent files list */
    void openRecent();

    /** @brief Open an example input file from the File menu */
    void openExample();

    /** @brief Save the current file */
    void save();

    /** @brief Save the current file with a new name */
    void saveAs();

    /** @brief Copy selected text to clipboard */
    void copy();

    /** @brief Cut selected text to clipboard */
    void cut();

    /** @brief Paste text from clipboard */
    void paste();

    /** @brief Undo last edit action */
    void undo();

    /** @brief Redo previously undone action */
    void redo();

    /** @brief Open find and replace dialog */
    void findAndReplace();

    /** @brief Run SPARTA from saved file */
    void runFile() { doRun(false); }

    /** @brief Restart SPARTA with a new instance */
    void restartSparta();

    /** @brief Open dialog to edit index-style variables */
    void editVariables();

    /** @brief Render an image from a dump file */
    void renderImage();

    /** @brief Open an external data file and plot selected columns */
    void plotDataFile();

    /** @brief Show about dialog */
    void about();

#if defined(SPARTA_GUI_USE_PLUGIN)
    /** @brief Check for SPARTA library updates */
    void checkUpdate();
#endif

    /** @brief Show context-sensitive help */
    void help();

    /** @brief Open SPARTA manual */
    void manual();

    /** @brief Open HOWTO documentation */
    void howto();

    /** @brief Update log window with new output */
    void logUpdate();

    /** @brief Handle document modification */
    void modified();

    /** @brief Open preferences dialog */
    void preferences();

    /** @brief Reset settings to defaults */
    void defaults();

private:
    /** @brief Update CPU/progress/line/variable status while a run is active
     *  @return run completion in permille (1000 when not running) */
    int updateRunStatus();

    /** @brief Append the cached thermo columns for the current step to the charts */
    void updateChartData(int step, int ncols);

    /** @brief Append any newly rendered dump image to the slideshow */
    void updateSlideShow();

    /** @brief Append accelerator-package command-line arguments to spartaArgs */
    void appendAcceleratorArgs(int accel);

    /** @brief Locate the SPARTA examples folder from preferences or common locations
     *  @return canonical path of the examples folder or empty string if not found */
    QString findExamplesDir() const;

    /** @brief (Re-)populate the File->Open Example submenu from the examples folder */
    void buildExampleMenu();

    /** @brief Create and show/hide the output log window for a run */
    void createLogWindow(QSettings &settings);

    /** @brief Create and show/hide the thermo chart window for a run */
    void createChartWindow(QSettings &settings);

    /** @brief Warn (modal) if the stdout capture buffer usage was high */
    void warnHighBufferUsage();

    /** @brief Append the final thermo data point to the charts at run end */
    void finalizeChartData();

    /**
     * @brief Create all menu actions, menus, and status bar
     * @param settings application settings class instance
     * @param allFont global proportional font selection
     * @param monoFont global monospace font selection
     */
    void setupUi(QSettings &settings, QFont &allFont, QFont &monoFont);

    /**
     * @brief Configure, check, and download the SPARTA shared library
     * @param settings application settings class instance
     */
    void setupPlugin(QSettings &settings);

    /**
     * @brief Configure, check, and assign SPARTA accelerator package settings
     * @param settings application settings class instance
     */
    void setupAccelerators(QSettings &settings);

    /**
     * @brief Create a menu action with optional icon and shortcut and append it to a menu
     * @param menu     Menu to append the new action to
     * @param iconpath Resource path for the action icon (empty for no icon)
     * @param text     Action label text
     * @param shortcut Keyboard shortcut sequence (empty for none)
     * @param slot     Member function pointer or callable invoked on trigger
     * @return The created action, for any further configuration by the caller
     */
    template <typename Func>
    QAction *addMenuAction(QMenu *menu, const QString &iconpath, const QString &text,
                           const QString &shortcut, Func slot);

    /** @brief Create File menu actions and add them to the menu bar */
    void createFileMenu();

    /** @brief Create Edit menu actions and add them to the menu bar */
    void createEditMenu();

    /** @brief Create Run menu actions and add them to the menu bar */
    void createRunMenu();

    /** @brief Create View menu actions and add them to the menu bar */
    void createViewMenu();

    /** @brief Create About/Help menu actions and add them to the menu bar */
    void createAboutMenu();

    /** @brief Create the status bar and its widgets */
    void createStatusBar();

    /** @brief Create (or recreate) the docked "Variables" panel content.
     *
     * @c varwindow is torn down together with the other panel contents when the
     * editor is reset (newDocument()/openFile()), so it must be recreated on
     * demand -- see the PanelManager::panelOpened handler in createViewMenu(). */
    void createVariableWindow();

    // Central GUI elements
    CodeEditor *textEdit;           ///< Custom code editor widget
    QMenuBar *menubar;              ///< Menu bar with menus and actions
    QStatusBar *statusbar;          ///< status bar
    QList<QAction *> recentActions; ///< list of actions for recent files
    QMenu *exampleMenu;             ///< File menu entry with SPARTA example inputs

    Highlighter *highlighter; ///< Syntax highlighter for SPARTA input
    StdCapture *capturer;     ///< Captures stdout/stderr from SPARTA
    QLabel *status;           ///< Status bar label for general status
    QLabel *cpuuse;           ///< Status bar label for CPU usage
    int lastCpuBucket;        ///< Last applied cpuuse color bucket (-1 = none yet)
    PanelManager *panels;     ///< Docked-panel layout manager (Output/Charts/Image/Slide/Variables)
    LogWindow *logwindow;     ///< Window displaying SPARTA output log
    ImageViewer *imagewindow; ///< Window for viewing single images
    ChartWindow *chartwindow; ///< Window for displaying charts
    SlideShow *slideshow;    ///< Window for image slideshow
    QTimer *logupdater;      ///< Timer for periodic log updates
    QLabel *dirstatus;       ///< Status bar label showing current directory
    QProgressBar *progress;  ///< Progress bar for long operations
    Preferences *prefdialog; ///< Preferences dialog
    QLabel *spartastatus;    ///< Status bar label for SPARTA state
    QLabel *varwindow;       ///< Window showing variable definitions

    /**
     * @brief Container for inspect dialog widgets
     *
     * Holds references to the two windows (info, image) of an inspect dialog
     */
    struct InspectData {
        QWidget *info;  ///< Information window widget
        QWidget *image; ///< Image rendering window widget
    };
    QList<InspectData *> inspectList; ///< List of open inspect dialogs

    QString currentFile;                      ///< Path to currently opened file
    QString currentDir;                       ///< Current working directory
    QList<QString> recent;                    ///< List of recently opened files
    QList<QPair<QString, QString>> variables; ///< Index-style variable definitions

    SpartaWrapper sparta;                ///< Interface to SPARTA library
    SpartaRunner *runner;                ///< Thread for running SPARTA simulations
    QString pluginPath;                  ///< Path to SPARTA shared library (plugin mode)
    int runCounter;                      ///< Counter for simulation runs
    std::vector<std::string> spartaArgs; ///< Command-line arguments for SPARTA

protected:
    int nthreads;      ///< Number of threads for parallel execution
    bool kokkosStarted = false; ///< true once SPARTA has been launched with Kokkos in this process
                                ///< (Kokkos can be initialized only once, so its thread count is fixed)
    int mainx;         ///< Override value for main editor window width or 0
    int mainy;         ///< Override value for main editor window height or 0
    bool hasClipboard; ///< true if Qt was configured with Clipboard support, otherwise false
};

#endif // SPARTAGUI_H

// Local Variables:
// c-basic-offset: 4
// End:
