# Security Policy

SPARTA is designed as a user-level application to conduct direct
simulation Monte Carlo (DSMC) simulations of rarefied gas flows for
research.  As such SPARTA depends to some degree on users providing
correctly formatted input, and SPARTA needs to read and write files
based on uncontrolled user input.  As a parallel application for use in
high-performance computing environments, performance critical steps are
also done without checking data.

SPARTA also is interfaced to a number of external libraries (e.g. MPI,
Kokkos, FFTW, VTK, JPEG, PNG, and Python), that are not validated and
tested by the SPARTA developers, so it is easy to import bad behavior
from calling functions in one of those libraries.  The section [Build
Dependencies](#build-dependencies) below explains which external code
is used when compiling SPARTA.

Thus it is quite easy to crash SPARTA through malicious input and do all
kinds of file system manipulations.  A SPARTA input can also run other
programs through the `shell` command (and through `gzip` and `ffmpeg`
for compressed dump files and movies) and run Python code with the
`python` command.  An input file should therefore be regarded as a
program and not as data: only run inputs from sources that you trust.
And because of that SPARTA should **NEVER** be compiled or **run** as
superuser, either from a "root" or "administrator" account directly or
indirectly via "sudo" or "su".  The SPARTA executable prints a warning
when it is started that way, and so does CMake when configuring SPARTA.
Programs in containers often run as "root" by default.  In that case it
is recommended to create a regular user account inside the container,
or to tell the container software to run as a regular user, for example
with the `--user` flag of `docker run`.

Therefore what could be seen as a security vulnerability is usually
either a user mistake or a bug in the code.  Bugs can be reported in the
SPARTA project [issue tracker on
GitHub](https://github.com/sparta/sparta/issues).

# Reporting a Vulnerability

If you have found a problem that could be used to harm other SPARTA
users or the SPARTA project, please do **not** report it in a public
issue.  Examples for such problems are ways to get malicious changes
into the SPARTA source code, and also passwords or access keys that
have become public by accident.  Please report those problems privately
with the "Report a vulnerability" button on the [Security
page](https://github.com/sparta/sparta/security) of the SPARTA
repository on GitHub, or by email to the SPARTA developers listed in
the README file.

Please include which version of SPARTA is affected and how the problem
can be reproduced.  The SPARTA developers will then work with you to
confirm and correct the problem and will agree with you on when and how
it is made public.

# Version Updates

SPARTA follows a continuous release development model.  We aim to keep
the `master` branch always fully functional and employ automated tests
to detect failures of existing functionality from adding or modifying
features.  These tests are run on pull requests and must pass *before*
merging to the `master` branch.  Bug fixes, including fixes for security
issues, are applied to the `master` branch and appear in the next
SPARTA release.

# Supported Versions

Corrections are always applied to the `master` branch first and thus
are part of the next SPARTA release.  Older versions are in general not
updated, so please upgrade to a current version.

# Build Dependencies

The SPARTA build (with CMake or with GNU make) does not download any
source code or data from the internet.  The `lib` folder contains a
copy of the Kokkos library, which is updated by the SPARTA developers
from Kokkos releases and is reviewed and tested like any other change
to SPARTA.  All other external libraries (e.g. MPI, FFTW, VTK, JPEG,
PNG, and Python) are used as they are installed on your machine.  The
SPARTA developers have no control over how those libraries are
developed or distributed.
