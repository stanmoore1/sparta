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

#ifndef SPARTARUNNER_H
#define SPARTARUNNER_H

#include <QThread>
#include <string>

class SpartaWrapper;

/**
 * @brief Worker thread for executing SPARTA simulations
 *
 * This class runs SPARTA simulations in a background thread to maintain
 * UI responsiveness during long calculations. It executes either a
 * SPARTA command string or a full input file and emits a signal upon
 * completion.
 *
 * Input is passed using std::string, ensuring clean ownership transfer.
 */
class SpartaRunner : public QThread {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param parent Parent QObject
     */
    explicit SpartaRunner(QObject *parent = nullptr);

    /**
     * @brief Destructor
     */
    ~SpartaRunner() override = default;

    SpartaRunner(const SpartaRunner &)            = delete;
    SpartaRunner(SpartaRunner &&)                 = delete;
    SpartaRunner &operator=(const SpartaRunner &) = delete;
    SpartaRunner &operator=(SpartaRunner &&)      = delete;

    /**
     * @brief Prepare the runner thread with SPARTA instance and commands
     * @param _sparta Pointer to SpartaWrapper instance
     * @param _input  String of SPARTA commands to execute (can be empty)
     * @param _file   Input file path to execute (can be empty)
     * @param _clearfirst  Issue "clear" before running, discarding the current
     *                     state.  True for running a deck from the top; false
     *                     for continuing from the state a previous run left,
     *                     which is what "Extend Run" does.
     *
     * Sets up the runner with the SPARTA instance and input. Either input or
     * file should be provided, not both.
     */
    void setupRun(SpartaWrapper *_sparta, std::string _input, std::string _file = {},
                  bool _clearfirst = true);

signals:
    /**
     * @brief Signal emitted when SPARTA execution completes
     */
    void resultReady();

protected:
    /**
     * @brief Thread execution function - runs SPARTA commands or input file
     *
     * This function executes in the worker thread. It processes either
     * a string of SPARTA commands or an input file, then signals completion.
     */
    void run() override;

private:
    SpartaWrapper *sparta; ///< Pointer to the SPARTA wrapper instance (not owned)
    std::string input;     ///< String of SPARTA commands to execute
    std::string file;      ///< Input file path to execute
};

#endif
// Local Variables:
// c-basic-offset: 4
// End:
