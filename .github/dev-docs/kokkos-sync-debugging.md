# Debugging KOKKOS host/device sync bugs

A guide for finding a missing `sync()` / `modify()` in the KOKKOS package by
running an input deck under the split-memory debug build.  This is a developer
tool; nothing here affects an ordinary build.

## What the tool can find, and what it cannot

Kokkos turns its coherence state machine off whenever the host and device
memory spaces are the same, which is every CPU-only build: `sync()` and
`modify()` return immediately and both sides are one allocation.  A missing
declaration therefore has no effect on a CPU and silently corrupts results on
a GPU.  `-D SPARTA_KOKKOS_DEBUG_SYNC=on` gives the host side its own allocation
and drives the state machine in software, so the GPU bug reproduces on a CPU.

Three failure classes exist, and they need different detectors:

1. **Stale access** -- a read or write of the side that is not current.  Caught
   by **poison mode** at the exact faulting instruction, whatever path the
   access took: a Kokkos view, a cached view such as the `d_particles` a style
   takes at the top of a step, a subview, a plain pointer from
   `memory_kokkos.h`, or a `memcpy` inside MPI.
2. **Unclaimed write** -- a write to one side of an in-sync pair that is never
   followed by the matching `modify_*()`, so the pair silently diverges with
   clean counters.  Poison cannot catch this (an in-sync pair must stay
   writable); the **watch** and **stale** checks do it by comparing state.
3. **Undeclared write** -- a style changes a shared array without declaring it
   in `datamask_modify` and without marking it as it goes.  The coherence state
   stays perfectly clean, so neither of the above sees it; the **audit**
   compares the bytes across the call instead.

Neither detector can see a bug that the input never exercises.  A run that
behaves correctly proves nothing about the code it did not execute.

## Two builds

Configure both once; keep them side by side.  Add whatever your input needs.

```bash
# detector build -- watch / stale / audit / trace, fast enough for repeated runs
cmake -S cmake -B build-sync -G Ninja \
      -D PKG_KOKKOS=on -D BUILD_KOKKOS=on \
      -D Kokkos_ENABLE_SERIAL=on -D Kokkos_ENABLE_OPENMP=off \
      -D CMAKE_BUILD_TYPE=Release -D SPARTA_KOKKOS_DEBUG_SYNC=on
cmake --build build-sync -j 4

# poison build -- adds AddressSanitizer; slower, needed only for class 1
cmake -S cmake -B build-poison -G Ninja \
      -D PKG_KOKKOS=on -D BUILD_KOKKOS=on \
      -D Kokkos_ENABLE_SERIAL=on -D Kokkos_ENABLE_OPENMP=off \
      -D CMAKE_BUILD_TYPE=RelWithDebInfo \
      -D SPARTA_KOKKOS_DEBUG_SYNC=on -D SPARTA_KOKKOS_DEBUG_SYNC_ASAN=on
cmake --build build-poison -j 4
```

Every run below must use the KOKKOS styles, or the debug build tests nothing:

```bash
./build-sync/spa_ -k on -sf kk -in in.your_input
```

Runs use about twice the memory for the shared arrays and are considerably
slower.

## Procedure

### 0. Establish that there is a fault, and where it shows

Run the input on a stock build and on `build-sync`, and compare the stats
output.  A difference is the bug reproducing.  If the two agree, vary the
conditions the coherence paths depend on before concluding there is nothing to
find:

```bash
package kokkos comm/sort yes / no     # in the input, changes the comm path
package kokkos reduction ...          # which reduction the move kernel uses
global comm/sort yes                  # matches MPI runs, changes ordering
-np 1 / -np 4                         # migration and ghost paths differ
```

Keep the exact command line that shows the difference; every later step reuses
it.

### 1. Poison mode first

It names the root cause rather than a symptom, and it is the only detector that
sees reads through plain pointers.

```bash
SPARTA_KOKKOS_POISON=1 ASAN_OPTIONS=detect_leaks=0 \
  ./build-poison/spa_ -k on -sf kk -in in.your_input
```

The run stops at the first stale access with an ASan report whose top frames
name the function that used the data.  That function is where the missing
`sync()` belongs -- or, read the other way, the array it touched is the one
whose `modify()` is missing upstream.

To collect every fault in one run instead of stopping at the first:

```bash
ASAN_OPTIONS=detect_leaks=0:halt_on_error=0:log_path=/tmp/poison/a
```

`detect_leaks=0` silences MPI's own leaks.  With more than one rank the
`log_path` is essential: four ranks writing one stream interleave into reports
that belong to no single process.

### 2. Watch and stale for the unclaimed-write class

If poison is silent but results still differ, the write was never claimed:

```bash
SPARTA_KOKKOS_WATCH= SPARTA_KOKKOS_STALE= SPARTA_KOKKOS_STALE_STRICT=1 \
  ./build-sync/spa_ -k on -sf kk -in in.your_input
```

An empty value means "every view"; give a substring instead to follow one array
(`SPARTA_KOKKOS_WATCH=particle:particles`).  Add `SPARTA_KOKKOS_WATCH_BT=1` for
a backtrace at each report once you know which array to chase -- it is verbose.

### 3. The audit for an undeclared write

Poison and watch both work from the coherence state.  A style that writes an
array it never named in `datamask_modify`, and never marked with
`particle_kk->modify(Device,PARTICLE_MASK)` while it ran, leaves that state
looking perfectly clean, so neither can see it; the audit compares the bytes
instead:

```bash
SPARTA_KOKKOS_AUDIT=1 ./build-sync/spa_ -k on -sf kk -in in.your_input
```

It copies every watched array around every style call, which is slow, so it is
off unless the variable is set.  It reports at the end of the run, naming the
style, the array, the entry and the values.  Two things it deliberately says
out loud: an array that was already stale when the style started is not covered
by `datamask_read`, and a style that declares *every* array cannot be checked at
all -- which is what a style gets by leaving `datamask_modify` at the
`ALL_MASK` that `Fix` sets in `src/fix.cpp`.

Two limits worth knowing before you read silence as a pass:

* It brackets **fix** calls only, from `ModifyKokkos`.  SPARTA's `Compute` has
  no `datamask_read` / `datamask_modify` at all, so no compute is checked.
* It watches the particle, species, custom, grid and surf arrays.  `VREMAX_MASK`
  and `REMAIN_MASK` are not among them: they live on `CollideVSSKokkos` rather
  than on a global object, and collide is not a fix, so nothing ever brackets a
  call that touches them.  The end-of-run line says so.

### 4. Always diff against a clean run

This is the step that decides whether a report matters.  Some reports appear on
correct runs too: scratch buffers filled and thrown away on purpose, and pairs
left deliberately apart by `clear_sync_state()`.  Run the **unmodified** code
with the same flags, keep its reports, and compare:

```bash
labels() { sed -n 's/^\[stale\] \+\([a-zA-Z_:]*\): .*, from \(.*\)$/\1 <- \2/p' "$1" | sort -u; }
comm -13 <(labels clean.err) <(labels suspect.err)     # only what the fault added
```

Key the comparison on the array **and the routine that touched it**, not the
array alone: the noise and a real finding often share an array name, and a
bare-name diff throws the finding away with the noise.  Watch reports name an
element index as well; include it for the same reason.

## Reading the reports

```
[stale] particle:particles: device side read while host side is newer,
        from SPARTA_NS::UpdateKokkos::move<1, 0, 1>()
```
A missing **copy**: `move()` needs `particle_kk->sync(Device,PARTICLE_MASK)`, or
the caller that wrote the host side owes a `sync` before this point.

```
[watch] grid:cinfo: the host side was written without a claim and this
        sync_device has nothing to copy -- the device keeps stale data
        element 1 of 2 is where they part
```
A missing **claim**: something filled the host side and never called
`modify_host()`, so this sync copies nothing.  Look at whoever last wrote that
array -- typically a command outside the KOKKOS package writing through the
plain pointer that `grow_kokkos()` handed out.  This is the report to expect
when a kernel reads a device view that was cached long before the write,
because no accessor runs at the time of the stale read.

```
[watch] particle:particles: the host side was written, never claimed, and is now lost
        the write is between modify_device and sync_host, which discards it
```
The same fault caught at the moment the data is thrown away, with the two calls
it happened between.

```
WARNING: datamask audit: end_of_step ave/grid changed grid:cinfo without
declaring it in datamask_modify or marking it modified, first at entry 12 of
1000 (0 -> 4) on step 100
```
An **undeclared write**: the style changed an array and neither named it up
front nor marked it as it went.

## Fixing what you find

Three legitimate remedies, in order of how often they are right:

1. **Add the missing call** -- `k_foo.modify_host()` after a host write, or a
   `sync` before a read.  For the shared arrays prefer the mask form,
   `particle_kk->modify(Host,PARTICLE_MASK)` /
   `particle_kk->sync(Device,PARTICLE_MASK)`.
2. **Mark it in the routine that wrote it.**  SPARTA's KOKKOS styles declare
   `EMPTY_MASK` and mark what they changed as they go, which is what the audit
   checks against; a style that writes an array must say so somewhere, either
   there or in `datamask_modify`.
3. **`clear_sync_state()`** -- when the code deliberately overwrites a whole
   array and the other side's contents are irrelevant.  Follow the existing
   annotations rather than inventing a fourth remedy.

Remember that a bug in one style is rarely alone: check the sibling styles for
the same shape.

## Testing a change to the tool itself

`dual_view_kokkos.h` can be exercised without building SPARTA: compile a small
program that includes it against the Kokkos libraries from any configured build
tree, providing a stub for `SPARTA_NS::datamask_audit_note_copy`.  A test that
constructs a view, fills the host side, omits the claim and syncs takes seconds
to run and tells you at once whether a detector change still fires.  Do this
before rebuilding SPARTA, which takes far longer.

## Pitfalls

* **`-k on -sf kk` is mandatory.**  `-k on` alone leaves the styles that take a
  suffix -- collide, surf collide, surf react -- as the plain versions, and
  every detector is then correctly silent about them.  Some inputs even refuse
  to run that way ("Must use Kokkos-supported collision style"), but a fix or
  compute that has no suffix requirement will run happily unaccelerated and
  tell you nothing.  This has produced false conclusions more than once.
* **A silent audit is not the same as an armed audit.**  It needs
  `SPARTA_KOKKOS_AUDIT` set *and* `DatamaskAudit::enable(1)` to have run, which
  `UpdateKokkos::run()` does around the timestep loop only.  Confirm the
  end-of-run "datamask audit: ..." line is present before reading anything into
  an empty report; upstream once lost that call in a merge and measured several
  cases against a dead auditor.
* **Multi-rank output interleaves.**  Capture per rank and concatenate
  afterwards, and use `ASAN_OPTIONS=log_path=...` for the poison build.
* **A silent run is not a clean run.**  Check the run actually finished before
  reading anything into an empty report.
* **Compare like with like.**  Clean and suspect runs must use the same binary
  options, the same environment variables and the same rank count.
* **Poison mode does not mix with the paranoid, verify or audit modes**; poison
  short circuits the accessor checks by design, and the audit reads the device
  buffers directly, which poison has marked off limits.  `enable()` refuses to
  arm the audit when `SPARTA_KOKKOS_POISON` is set.
* **A run whose results did not change proves nothing** about a detector's
  coverage.  Establish the fault first (step 0), then ask what the tools say.

## Every switch

All are read once, at first use, and cost nothing when unset.  Each takes the
text to look for in a view's name, empty for every view, unless noted.

| variable | what it does |
| --- | --- |
| `SPARTA_KOKKOS_POISON` | ASan-poison the stale side; any use of it faults (set/unset, needs the ASAN build) |
| `SPARTA_KOKKOS_WATCH` | report a write to one side that nothing claimed and a later copy discards |
| `SPARTA_KOKKOS_WATCH_BT` | add a backtrace to each watch and stale report (set/unset) |
| `SPARTA_KOKKOS_WATCH_SKIP` | comma separated names to leave out, for buffers that really are scratch |
| `SPARTA_KOKKOS_STALE` | report a read of a side the other has moved past, with per-array totals at exit |
| `SPARTA_KOKKOS_STALE_STRICT` | extend that to a pair that differs with nothing owed, which a write through a plain pointer leaves (set/unset, needs `WATCH`) |
| `SPARTA_KOKKOS_AUDIT` | arm the datamask audit (set/unset) |
| `SPARTA_KOKKOS_TRACE` | print every copy, claim and resize with the counters behind them, and bracket each audited style call |
| `SPARTA_KOKKOS_VERIFY` | check that the two sides really hold the same bytes whenever the counters say they agree |
| `SPARTA_KOKKOS_PARANOID` | copy after every claim so the two never diverge; for bisecting, reports nothing |
| `SPARTA_KOKKOS_ALIAS` | keep one allocation for the selected arrays, as a control |
| `SPARTA_KOKKOS_NO_AUTOSYNC` | leave `auto_sync` off outside the timestep loop, where it otherwise hides faults (set/unset) |
