# GS-autosync: auto_sync claims the host over a standing device claim

Scope: `ParticleKokkos::sync()`, `GridKokkos::sync()`, `SurfKokkos::sync()` each did
`if (auto_sync) modify(Host,mask);` in the Device branch, the shape fixed in
`CollideVSSKokkos::sync()`. If the pair is device-claimed, modify_host aborts
("concurrent modification of host and device views").

Private build: worktree `$S/gpusim/GS-autosync/src` (detached at gpusim-B 0778ebfc),
build dir `$S/gpusim/GS-autosync/build` (same options as bsync_B: Serial, MPI, KISS FFT,
Release, SPARTA_KOKKOS_DEBUG_SYNC=on). Decks in `$S/gpusim/GS-autosync/decks`.

## Static analysis (where a device claim can stand when auto_sync is on)

auto_sync is 0 only inside `UpdateKokkos::run()`'s loop; everywhere else (setup, between
runs, non-Kokkos fix hooks, comm/serial migrate) it is 1, and in those regions every
`modify(Device,..)` immediately syncs the host, so a claim cannot *accumulate* there.
It must come from either (a) the run loop, or (b) a DualView `resize()` (Kokkos resizes on
the device on a tie and marks the device modified -- the tool models this), which no
`modify()` call sees.

(in progress)
