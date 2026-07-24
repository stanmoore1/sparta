# Fixtures for the GUI functional pass

Data with a known-correct answer, so a feature that mishandles it is obvious
rather than merely "something changed on screen".

| File | What it is for |
| --- | --- |
| `in.broken` | A deck where every line after the header is wrong in a different way: an unknown command, an unknown compute style, a `global` with bogus arguments, a missing surface file, a non-numeric `run`. The linter must object to each. |
| `tetra.stl` | A closed tetrahedron: watertight. |
| `open.stl` | The same mesh with one face removed, leaving 3 unmatched edges. SPARTA's `read_surf` must reject it, and does. |
| `make_field_vtk.py` | Generates `field.vtk`, a 21³ structured grid with a radial `density` field and a derived `temperature` field. Iso-surfaces of the first are spheres, so a wrong result shows up by eye; the second gives the field calculator a second operand. Generated rather than committed because 270 KB of ASCII floats is not worth versioning. |

Decks that need a real run (a per-surface compute for the surface report, index
variables for sweeps) are built from `examples/circle` at test time; see the
driver scripts in the parent directory.
