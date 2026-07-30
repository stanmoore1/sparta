.. _howto_18:

Variable timestep simulations
=============================

As an alternative to utilization of a user-provided constant timestep,
the variable timestep option enables SPARTA to compute global
timesteps based on the current state of the physical processes being
modeled. The timestep is global in the sense that all cells advance
their particles in time using the same timestep value.  The timestep
is adaptive in the sense that the global timestep can be recalculated
periodically throughout the simulation to account for flow state
changes.  Examples of situations where a variable timestep would be
desired are problems with highly varying density or velocity
throughout the domain and transient problems where the optimal
timestep changes throughout the simulation.

The global, variable timestep is computed at a user-specified
frequency using cell-based timesteps that are calculated using cell
mean collision and particle transit times.  These cell-based timesteps
are only used to compute the global timestep and are not used to
advance the solution locally. The benefit of the global timestep
calculation is that it will automatically reduce the timestep if the
intial value is too large, leading to higher accuracy, and it will
automatically increase the timestep if the initial value is too small,
speeding up the simulation. The overhead of using the variable
timestep option is the computational time involved in computing the
cell-based time quantities and performing parallel reductions over the
grid to construct the global minimum and average cell timesteps needed
for the global timestep calculation. For scenarios where ensembles of
similar problems are being run, one strategy to mitigate this cost is
to determine an optimal timestep using the variable timestep option
for the first run and then to utilize this timestep as a
user-specified value for the subsequent runs.

The :doc:`compute dt/grid <compute_dt_grid>` command is used to
calculate the cell-based timesteps, and the :doc:`fix dt/reset <fix_dt_reset>` command uses this data to calculate the
global timestep.  An internal time variable has been added to SPARTA
to track elapsed simulation time, and this time variable as well as
the current timestep can be output using the *time* and *dt* keywords
in the :doc:`stats\_style <stats_style>` command. These *time* and *dt*
values are also included in the :doc:`read\_restart <read_restart>` and
:doc:`write restart <write_restart>` commands.
