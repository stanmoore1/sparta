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
from calling functions in one of those libraries.

Thus it is quite easy to crash SPARTA through malicious input and do all
kinds of file system manipulations.  And because of that SPARTA should
**NEVER** be compiled or **run** as superuser, either from a "root" or
"administrator" account directly or indirectly via "sudo" or "su".

Therefore what could be seen as a security vulnerability is usually
either a user mistake or a bug in the code.  Bugs can be reported in the
SPARTA project [issue tracker on
GitHub](https://github.com/sparta/sparta/issues).

# Version Updates

SPARTA follows a continuous release development model.  We aim to keep
the `master` branch always fully functional and employ automated tests
to detect failures of existing functionality from adding or modifying
features.  These tests are run on pull requests and must pass *before*
merging to the `master` branch.  Bug fixes, including fixes for security
issues, are applied to the `master` branch and appear in the next
SPARTA release.
