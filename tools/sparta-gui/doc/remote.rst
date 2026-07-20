.. _remote_execution:

************************
Remote/Cluster Execution
************************

.. index:: remote execution
.. index:: cluster
.. index:: Slurm
.. index:: PBS
.. index:: Flux
.. index:: ssh
.. index:: batch job

Real DSMC production runs happen on HPC clusters rather than a laptop.
SPARTA-GUI can submit an input deck to a cluster over SSH, track the
resulting batch job, follow its log, and pull the results back — all from
the *Run* menu, without leaving the editor.  The relevant menu entries are
*Run → Submit to Cluster...* and *Run → Manage Cluster Jobs...* (the latter
also has a *Cluster Jobs* toggle in the *View* menu).

.. TODO screenshot: capture the Submit to Cluster dialog and the Cluster
   Jobs panel as JPG/sparta-gui-remote-*.png

Requirements and authentication
===============================

.. index:: remote execution; authentication
.. index:: ssh; key-based authentication

Submission uses the ``ssh``, ``scp``/``rsync`` commands already installed on
your machine, so those must be on the ``PATH``.  **Authentication is always
key-based.**  SPARTA-GUI never asks for, stores, or transmits a password: it
simply runs ``ssh`` and inherits your existing setup — an SSH key loaded into
``ssh-agent`` and any options in ``~/.ssh/config``.

For this to work unattended you should be able to run, from a terminal,

.. code-block:: bash

   ssh <user>@<host>

and reach a shell **without being prompted** for a password or passphrase.
The usual way to achieve that is to generate a key (``ssh-keygen``), install
it on the cluster (``ssh-copy-id``), and load it into your agent
(``ssh-add``).  SPARTA-GUI adds ``-o BatchMode=yes -o ConnectTimeout=10`` so a
missing key fails immediately with a clear error instead of hanging, and it
reuses one multiplexed connection (``ControlMaster``) for the repeated status
checks.  Host keys are taken from your ``known_hosts`` — an unknown host is a
fast, visible failure rather than a silent auto-accept.

.. note::

   Sites that require a password, one-time code, or other
   keyboard-interactive authentication are not supported directly in this
   version.  Establish an ``ssh-agent`` or ``ControlMaster`` session (or use a
   jump host configured in ``~/.ssh/config``) first, and SPARTA-GUI will reuse
   that authenticated connection.

Connection profiles
====================

.. index:: remote execution; connection profile

A *connection profile* describes one cluster and is reused across
submissions.  Manage profiles from the *Profiles...* button in either the
submit dialog or the Cluster Jobs panel.  A profile has:

- **Host**, **User**, and **Port** for the SSH connection.
- **Remote workdir** — a base directory on the cluster (e.g.
  ``/scratch/<user>/sparta``) under which each job gets its own run
  directory.
- **Scheduler** — Slurm, PBS/Torque, or Flux.
- **Launcher** and **SPARTA exe** — the parallel launch command and remote
  SPARTA binary, e.g. ``srun`` and ``spa_`` (or ``mpirun -np ...``).
- **Module loads** — optional ``module load ...`` lines to emit in the
  batch script.
- **Batch template** — leave empty to use the scheduler's built-in template,
  or provide your own (see below).

Profiles are stored in the application configuration directory
(``remote_profiles.json``) and persist across sessions.

The batch-script template
=========================

.. index:: remote execution; batch template

Each submission renders a batch script from a template.  Built-in templates
are provided for Slurm (``#SBATCH`` directives, ``sbatch``), PBS/Torque
(``#PBS``, ``qsub``), and Flux (``# flux:``, ``flux batch``).  The template
uses ``${...}`` placeholders that are substituted per submission:
``${JOBNAME}``, ``${NODES}``, ``${NTASKS}``, ``${WALLTIME}``, ``${ACCOUNT}``,
``${QUEUE}``, ``${MODULES}``, ``${LAUNCH}``, ``${SPARTAEXE}``, and
``${INPUT}``.  Directive lines for an unset optional (account or queue) are
dropped automatically.

To adapt to a site's specific requirements, click *Load default template* in
the profile editor and edit it; a non-empty template overrides the built-in
one entirely.  The submit dialog shows a **live preview** of the exact script
that will be submitted.

Submitting and tracking a job
=============================

.. index:: remote execution; submit

Save the input deck first, then choose *Run → Submit to Cluster...*.  Pick a
profile, set the job name, node/task counts, walltime, and optionally an
account and queue, add any extra data files the deck reads (the deck itself
is staged automatically), review the rendered script, and press *Submit*.

SPARTA-GUI then, in the background (never blocking the interface):

1. creates the remote run directory and writes the batch script;
2. copies the deck and data files up with ``rsync`` (or ``scp``);
3. submits the script and records the scheduler's job id;
4. polls the job's state periodically (``squeue``/``qstat``/``flux jobs``);
5. follows the remote ``log.sparta`` into the panel while the job runs; and
6. copies the results back into a local run directory when it completes.

The **Cluster Jobs** panel lists every job with its profile, scheduler,
state, id, and submission time, and provides *Cancel*, *Resubmit*, *Pull*
(re-copy the artifacts), *Open Folder* (reveal the local results), and
*Remove* actions.  Submitted jobs are saved (``remote_jobs.json``) and
**reattached on the next launch**: SPARTA-GUI re-polls any job that was still
running, so closing the GUI does not lose track of a cluster run.

Once results are pulled back, the log, images, and thermodynamic output can
be viewed with the same :ref:`Charts <charts>`, :ref:`image <snapshot_viewer>`,
and *Plot Data File...* tools used for local runs.
