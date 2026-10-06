# GS-fixes: GPU-only fixes re-tested with split-memory detector builds
Binaries: A_sync=$S/bsync_A/src/spa_ (e071055f+tool), B_sync=$S/bsync_B/src/spa_ (39e1c1f7+tool; all listed fix commits are ancestors).
Poison builds ($S/bpoison_{A,B}/src/spa_): not present at start (checked again per item).
Detector env for "watch/stale" runs: SPARTA_KOKKOS_WATCH= SPARTA_KOKKOS_STALE= SPARTA_KOKKOS_STALE_STRICT=1.
Work dir: $S/gpusim/GS-fixes/<ID>. Stock CPU reference = A_sync binary without -k on.
Known noise (A and B identical, unrelated to these fixes): `[stale] irregular:index_send: device side read while host side is newer,
from IrregularKokkos::augment_data_uniform / ~IrregularKokkos` on every multi-rank run.

### F-G04-1 — ReactBirdKokkos::init did not zero device reaction tallies
deck: AB1 in.tw (1 cell, N2 dissociation tce, run 100 + run 100); in.tw4 (same on 2x2x2 grid, balance rcb) for 1/2/4 ranks.
positive control: A_sync np1: run1 147/178, run2 **149/220** (= 147+2, 178+42: cumulative) vs B_sync run2 **2/42** (per-run); thermo A==B identical (bug only in reported tallies). in.tw4: A np1 run2 177/223 vs B 7/48; np2 A 157/211 vs B 6/37; np4 (react/retry yes, A needs it: pre-existing react/extra overflow) A 161/224 vs B 7/39. CPU ref run2 4/50 (per-run). REPRODUCED (wrong output) under split memory — the earlier CPU A/B could not.
detector: default gpu/aware yes path (extract_tally allreduces d_tally_reactions directly) -> watch/stale silent in A and B (host array never synced, nothing to compare). With `-pk kokkos gpu/aware no`: A and B both report once `[watch] react_bird:tally_reactions: the host side was written, never claimed, and is now lost ... between sync_host and modify_device, element 8 changed 147 -> 0` (from ReactBirdKokkos::extract_tally <- Finish::end on run 2). In B this is benign (ReactBird::init zeroes host unclaimed, deep_copy zeroes device, so both sides are 0 and output is right), but it is a residual detector report: cleaner form would be `ReactBird::init(); k_tally_reactions.modify_host(); k_tally_reactions.sync_device();` (or zero device and clear_sync_state).
negative control: run 1 of every variant A==B identical (147/178 etc.); single-run decks unaffected.
necessary: YES (shown: A reports cumulative tallies on run 2 under split memory, 1/2/4 ranks, gpu/aware yes and no)
complete: YES for correctness (B per-run on all variants); minor: B still triggers one benign [watch] report under gpu/aware no (unclaimed host zero in ReactBird::init).
verdict: NECESSARY+COMPLETE (cosmetic watch residual in B, suggested modify_host/sync_device instead of deep_copy)
artifacts: $S/gpusim/GS-fixes/F-G04-1

