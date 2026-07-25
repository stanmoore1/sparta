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

#include "helpers.h"   // ElidedLabel
#include "inputcheck.h"
#include "spartawrapper.h"

// forward declarations

class QAbstractButton;
class QAction;
class QActionGroup;
class QEvent;
class QFont;
class QLabel;
class QListWidget;
class QMenu;
class QMenuBar;
class QProgressBar;
class QSettings;
class QStatusBar;
class QTimer;
class QWidget;

class QStackedWidget;

class ChartWindow;
class CodeEditor;
class GeneralTab;
class Highlighter;
class ImageViewer;
class ViewerPanel;
class PanelManager;
class SpartaRunner;
class LogWindow;
class Preferences;
class SlideShow;
class SweepPanel;
class RunHistory;
class HistoryPanel;
class StdCapture;
class URLDownloader;
class SceneWindow; // only defined when built with SPARTA_GUI_HAVE_VTK
class WelcomeScreen;

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

    /** @brief Tear down all docked panel content and null every pointer to it */
    void clearPanelWidgets();

    /** @brief Clean up the inspect file dialog list */
    void purgeInspectList();

    /**
     * @brief Event filter for handling special events
     * @param watched Object being watched
     * @param event Event to filter
     * @return true if event was handled
     */
    bool eventFilter(QObject *watched, QEvent *event) override;

signals:
    /** @brief Emitted at the end of runDone(); @p success is the run result.
     *  A parametric sweep uses this to advance to the next run. */
    void runFinished(bool success);

    /** @brief Emitted each logUpdate() tick so a sweep can sample thermo. */
    void thermoSampled();

public slots:
    /** @brief Quit the application */
    void quit();

    /** @brief Stop a running SPARTA simulation */
    void stopRun();

    /** @brief Run SPARTA with content from editor buffer */
    void runBuffer() { doRun(true); }

    /** @brief Set the index-variable overrides injected before the next run
     *  (used by the parametric sweep driver to vary parameters per run) */
    void setRunVariables(const QList<QPair<QString, QString>> &v) { variables = v; }

    /** @brief Re-scan the editor buffer for variables and return the set */
    QList<QPair<QString, QString>> discoverVariables()
    {
        updateVariables();
        return variables;
    }

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

    /** @brief Open an example input file by path (copying it to a writable
     *  location first if the example directory is read-only). Shared by the
     *  File->Open Example menu and the welcome screen's example gallery. */
    void openExamplePath(const QString &srcfile);

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

#if defined(SPARTA_GUI_HAVE_VTK)
    /** @brief Open the interactive VTK 3D viewer window (empty; load files with it) */
    void open3DViewer();
    /** @brief Render the current simulation state to VTK and show it in the 3D viewer */
    void renderVtkSnapshot();
    /** @brief Open the visual case-setup canvas seeded from the current deck */
#endif

    /** @brief Open an external data file and plot selected columns */
    void plotDataFile();

    /** @brief Open the STL / SPARTA-surface import wizard */
    void importSurface();

    /** @brief Open the Insert Snippet dialog and insert the chosen block */
    void insertSnippet();

    /** @brief Open the ParaView export dialog (surf2paraview / grid2paraview) */
    void exportParaview();

    /** @brief Open the surface engineering-quantity report dialog */
    void surfaceReport();


    /** @brief Show the docked Parametric Sweep panel */
    void runSweep();

    /** @brief Show the docked Run History panel */
    void showRunHistory();

    /** @brief Statically validate the current input deck and show diagnostics */
    void checkInput();

    /** @brief Auto-lint hook: re-validate after the editor cursor moves to a new
     *  line (debounced), without stealing focus by opening the panel. */
    void autoCheckInput();

    /** @brief Browse restart files and continue a run from a selected one */
    void continueRestart();

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
    /** @brief Apply the stored editor color scheme to the highlighter and editor surface */
    void applyEditorColorScheme();

    /** @brief Update CPU/progress/line/variable status while a run is active
     *  @return run completion in permille (1000 when not running) */
    int updateRunStatus();

    /** @brief Append the cached thermo columns for the current step to the charts */
    void updateChartData(int step, int ncols);

    /** @brief Build the viewer panel and dock it, the first time one is needed */
    void ensureViewerPanel();

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

    /** @brief Reflect the active workspace mode in the View menu and the
     *  status-bar mode switch (called on every mode change) */
    void syncModeControls(int mode);

    /** @brief Create Tools menu actions and add them to the menu bar */
    void createToolsMenu();

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

    /** @brief Lazily create + host the docked Parametric Sweep panel */
    void ensureSweepPanel();
    /** @brief Lazily create the RunHistory controller */
    void ensureHistory();
    /** @brief Lazily create + host the docked Run History panel */
    void ensureHistoryPanel();
    /** @brief Lazily create + host the docked Diagnostics panel */
    void ensureDiagnosticsPanel();
    /** @brief Run the static input-deck validator and refresh the diagnostics UI.
     *  @param interactive when true (manual "Check Input") the Diagnostics panel is
     *  raised and a status-bar summary is shown; the auto-lint path passes false so
     *  it updates markers quietly without stealing focus. */
    void runInputCheck(bool interactive);
    /** @brief Build the validator context from the bundled tables + live instance */
    InputCheck::Context buildCheckContext();
    /** @brief Lazily create + host the docked Project Files navigator panel */
    void ensureProjectFilesPanel();
    /** @brief Refresh the Project Files list from the working directory + deck references */
    void refreshProjectFiles();
    /** @brief Archive the just-finished run if archiving is enabled */
    void archiveFinishedRun(bool success);

    /** @brief Start the periodic crash-recovery autosave timer (if enabled) */
    void startRecoveryTimer();
    /** @brief Write the unsaved buffer to the recovery file (non-destructive) */
    void writeRecoveryFile();
    /** @brief Remove the recovery file after a clean save or exit */
    void clearRecoveryFile();
    /** @brief On startup, offer to recover a buffer left by a previous crash */
    bool maybeRecoverSession();
    /** @brief Path of the crash-recovery buffer file */
    QString recoveryFilePath() const;

    /** @brief Show the welcome screen in the central area (rebuilding its recent
     *  files list and example gallery first) */
    void showWelcome();
    /** @brief Show the code editor in the central area */
    void showEditor();

    // Central GUI elements
    CodeEditor *textEdit;           ///< Custom code editor widget
    QStackedWidget *centralStack;   ///< Hosts the welcome screen and the editor as
                                    ///< interchangeable central-area pages
    WelcomeScreen *welcome;         ///< Landing view (recent files + examples gallery)
    QMenuBar *menubar;              ///< Menu bar with menus and actions
    QActionGroup *modeGroup = nullptr;  ///< Exclusive View-menu workspace mode actions
    QList<QAbstractButton *> modeButtons; ///< Status-bar workspace mode switch buttons
    bool ranThisSession = false;    ///< True once a run started (gates the auto mode switch)
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
    ViewerPanel *viewer;      ///< The one panel showing snapshots, frames and 3D
    int keptRenderSeq = 0;    ///< numbers the renders kept when replace-on-render is off
#if defined(SPARTA_GUI_HAVE_VTK)
    SceneWindow *sceneWindow = nullptr; ///< Interactive 3D scene window (lazy)
#endif
    ChartWindow *chartwindow; ///< Window for displaying charts
    QTimer *logupdater;      ///< Timer for periodic log updates
    QTimer *recoveryTimer = nullptr; ///< Periodic crash-recovery autosave timer
    ElidedLabel *dirstatus;  ///< Status bar label showing current directory (elided when narrow)
    QProgressBar *progress;  ///< Progress bar for long operations
    Preferences *prefdialog; ///< Preferences dialog
    QLabel *spartastatus;    ///< Status bar label for SPARTA state
    QLabel *varwindow;       ///< Window showing variable definitions
    SweepPanel *sweepPanel = nullptr;        ///< Docked Parametric Sweep panel (lazy)
    RunHistory *history = nullptr;           ///< Run archive controller (lazy)
    HistoryPanel *historyPanel = nullptr;    ///< Docked Run History panel (lazy)
    QListWidget *diagnosticsList = nullptr;  ///< Docked Diagnostics panel list (lazy)
    QListWidget *projectFilesList = nullptr; ///< Docked Project Files navigator list (lazy)

    QTimer *autoLintTimer = nullptr;   ///< Debounce timer for auto-lint on line change (lazy)
    bool autoLintEnabled  = true;      ///< Auto-validate the deck on cursor line change
    int lastLintBlock     = -1;        ///< Editor block the cursor was last on (line-change detect)
    bool restoredLayout   = false;     ///< true if a saved dock layout was restored at startup
    bool startupComplete  = false;     ///< true once the constructor finished (plugin loaded, UI shown);
                                       ///< gates panelOpened side effects that must not run during
                                       ///< restoreLayout() -- e.g. auto-rendering the Image panel

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
