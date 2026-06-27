# Issue #436 — Hollow tube smaller than the grid cell: analysis

**Issue:** Cut-cell algorithm fails for a hollow tube when the external
diameter is smaller than the grid cell size.

**Error:**

```
ERROR on proc 0: LP: More than one positive area with a negative area
(src/cut3d.cpp:678)
```

**Conclusion:** This is **not a bug** in the usual sense. It is a deliberate,
documented limitation of SPARTA's cut-cell algorithm, triggered here because
the tube is under-resolved by the grid. The supported fix is to refine the
grid (and the surface mesh) so the tube wall is resolved by the cells.

---

## 1. Reproduction

The attached reproducer (`tube.py`, `data.tube`, `in.tube`) builds a hollow
tube and meshes a 4×4×4 box with a 5×5×5 grid:

| Parameter        | Value | Note                                  |
|------------------|-------|---------------------------------------|
| Box width        | 4     | `create_box -2 2 -2 2 -2 2`           |
| Grid             | 5×5×5 | cell size ≈ **0.8**                   |
| Tube ext. radius | 0.35  | ext. **diameter 0.70** (< cell size)  |
| Tube int. radius | 0.30  | wall thickness **0.05**               |
| `n_sides`        | 4     | very coarse wall (1 element thick)    |

- `radiusExt = 0.35` → **crashes**
- `radiusExt = 0.45` → **runs** (ring now pokes past the cell edges)

The failing-cell diagnostic confirms the whole cross-section lands in one cell:

```
Cut3d failed on proc 0 in cell ID: 62
  lo corner -1.2 -0.4 -0.4
  hi corner -0.4  0.4  0.4      # cell is 0.8 wide in y,z
  # of surfs = 24 out of 32     # nearly the whole tube in one cell
```

---

## 2. Root cause

SPARTA cuts each grid cell against the surface and decomposes the result into
disjoint flow regions (polygons in 2D faces, polyhedra in 3D). The algorithm
refuses to proceed when a single cell contains a **fully-enclosed solid
feature (a negative-area/volume loop)** together with **more than one disjoint
flow region (positive-area/volume loop)**, because it does not attempt to
determine which flow region each enclosed hole is nested inside.

The guard is explicit in the code:

**2D faces — `Cut2d::loop2pg()` (`src/cut2d.cpp:907`):**

```c
// do not allow multiple positive with one or more negative
// too difficult to figure out which positive each negative is inside of
if (positive > 1 && negative) return 5;   // -> errflag 25 -> cut3d.cpp:678
```

**3D volumes — `Cut3d::loop2ph()` (`src/cut3d.cpp:1763`):**

```c
// do not allow multiple positive with one or more negative
// too difficult to figure out which positive each negative is inside of
if (positive > 1 && negative) return 5;   // "More than one positive volume
                                          //  with a negative volume"
```

The 3D cut processes each of a cell's six faces in 2D first (`Cut3d::split`
→ `cut2d->split_face` → `loop2pg`); a face error is returned with `+20`, which
is why a 2D-face failure surfaces as the "area" message reported at
`cut3d.cpp:678`.

### Why this geometry hits it

When the annulus is fully contained in a cell, the cut of the tube's cap face
yields:

- the outer flow region — a border loop (**positive**) that contains the
  ring's outer wall as an enclosed **negative** loop, **plus**
- the inner bore — a second, disjoint **positive** loop.

That is `positive > 1 && negative > 0` → the disallowed case. With
`radiusExt = 0.45` the ring crosses the cell edges, so the outer region is no
longer a closed loop-with-hole and the configuration disappears. Hence the
behavior is a **resolution effect**, not a magic radius threshold.

This matches the documented guidance in `doc/Section_howto.txt` that a grid
cell larger than the surface elements intersecting it is not desirable because
"flow around the surface object will not be well resolved."

---

## 3. Is this case supported?

**No — it is explicitly unsupported, by design.** The restriction is general
(any fully-enclosed feature inside a single cell), and the hollow tube simply
happens to trigger it.

---

## 4. Recommended fix (no code change)

Refine the grid so no single cell encloses the whole ring, ideally fine enough
to resolve the wall:

- Wall thickness is only `0.35 − 0.30 = 0.05`, so cells near the surface should
  be substantially smaller than that (a finer base grid plus adaptive
  refinement around the tube, and a correspondingly smaller `gridcut`).
- Increase `n_sides` in the PyVista script — at `n_sides = 4` the wall is a
  single, very coarse element.

This both avoids the error and is required for a physically meaningful DSMC
result; at the current resolution the flow around and inside the tube would not
be resolved regardless of this error.

---

## 5. Removing the limitation in code (effort estimate)

The missing capability is **loop-nesting / containment classification**: for
each negative loop, find the smallest positive loop that encloses it, attach it
as a hole, and build each flow region as one outer boundary minus its contained
holes (updating `create_surfmap` / `split_point*` accordingly).

| Stage                 | Difficulty | Notes                                                                                 |
|-----------------------|------------|---------------------------------------------------------------------------------------|
| 2D (`loop2pg`)        | Moderate   | Loops already closed/oriented; add point-in-polygon nesting + hole assignment.        |
| 3D (`loop2ph`)        | Hard       | Needs robust point-in-polyhedron containment, hole assignment across shared faces, and consistent splitting. This is the part the author deliberately avoided. |

The main risk is regressing the many existing (degenerate / grazing / coplanar)
cut-cell edge cases, which is why grid refinement is the practical
recommendation.

---

## 6. Nested tubes (tube-within-tube)

A concentric nested geometry hits the **same** restriction, and more often,
since a cross-section then has alternating fluid/solid rings (multiple enclosed
holes per cell). A proper, general fix to §5 would also cover the nested case;
a minimal one-hole special-case would not. Either way, nested rings **must** be
grid-resolved to give meaningful flow results.

---

## Summary

- The error is a **deliberate, documented limitation**, not a corner-case bug.
- It is triggered by an **under-resolved grid** where a whole hollow feature
  falls inside one cell.
- **Recommended fix:** refine the grid and surface mesh so cells resolve the
  wall — this both avoids the error and is required for a valid simulation.
- Lifting the restriction in code is **feasible in 2D** but a **significant,
  risky effort in 3D** (`loop2ph`).
