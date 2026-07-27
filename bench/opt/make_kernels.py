#!/usr/bin/env python3
"""Compute per-kernel arithmetic intensity and achieved performance for the
roofline, and write kernels.json.

Provenance of each input (there is no PMU in this KVM guest, so nothing here
comes from hardware counters):

  bytes  - callgrind cache simulation of the same binary/input: D1 read+write
           misses x 64 B is the traffic that leaves L1. At 1M particles the
           96 MB particle array is far past the ~1-2 MB effective private
           cache measured by micro/machine_peak, so essentially all of that
           L1-miss traffic is served by L3/DRAM at the ~20-25 GB/s plateau.
  FLOPs  - counted by hand from the source, listed per kernel below.
           Transcendentals are charged at their measured FMA-equivalent cost
           from micro/machine_peak (pow = 36, sqrt = 5.7, sin = cos = 15),
           because a roofline that counts pow() as one "flop" would put the
           collide kernel absurdly far below the compute roof for the wrong
           reason.
  time   - SPARTA's own per-section timers at 1M particles, 100 steps.

The callgrind run is at 100K particles (it is ~50x slowdown, so 1M is
impractical); miss counts are therefore expressed per operation and applied
to the 1M operation counts. This slightly *understates* misses at 1M, since
the cell hash and cells[] arrays are 10x larger there. That bias is noted in
ROOFLINE.md and does not change which side of the roof a kernel lands on.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- measured section times at 1M particles / 100 steps (seconds) ----
# baseline = optmove yes, particle/reorder 5 (best baseline config)
# tuned    = all Tier A optimizations, particle/reorder 3 (best tuned config)
TIME = {
    "baseline": {"move": 2.3289, "collide": 2.7545, "sort": 2.1682},
    "tuned":    {"move": 1.8506, "collide": 2.4501, "sort": 1.7519},
}

# ---- operation counts for the 1M / 100-step benchmark run (from the log) ----
NSTEP = 100
NPART = 1_000_000
MOVES = NPART * NSTEP              # 1e8 particle-moves
ATTEMPTS = 9_365_524               # "Collide attempts" in the benchmark run
COLLISIONS = 7_015_946             # "Collide occurs"
# sort_reorder touches every particle on a reorder step; sort() every step
REORDER_BASE, REORDER_TUNED = 5, 3

# ---- FMA-equivalent cost of transcendentals, measured by machine_peak ----
POW, SQRT, SIN, COS = 36.0, 5.7, 15.0, 15.0
SINCOS = 15.0        # glibc computes both in one __sincos_fma call
FMA = 2.0            # flops per FMA-equivalent


def flops_move_per_particle():
    """Update::move<3,0,1> fast path, per particle.

    xnew = x + dt*v                      3 FMA          =  6
    6 box bound compares                                =  0
    (xnew - boxlo)/d, three axes         3 sub + 3 div  =  6
    cell index arithmetic                integer        =  0
    """
    return 12.0


def flops_test_collision():
    """CollideVSS::test_collision, per attempt.

    du,dv,dw                             3 sub          =  3
    vr2 = du*du + dv*dv + dw*dw          3 mul + 2 add  =  5
    pow(vr2, 1-omega)                    1 pow
    vre = vro * prefactor                1 mul          =  1
    MAX against vremax                                  =  0
    vre / vremax                         1 div          =  1
    """
    return 10.0 + POW * FMA


def flops_setup_collision():
    """CollideVSS::setup_collision, per collision that passes the test."""
    n = 0.0
    n += SQRT * FMA          # sqrt(vr2)
    n += 6                   # ave_rotdof, ave_vibdof, ave_dof
    n += 2                   # etrans
    n += 4                   # erot, evib, eint, etotal
    n += 1                   # 1/(imass+jmass)
    n += 12                  # ucmf, vcmf, wcmf: 2 mul + 1 add + 1 mul each
    return n


def flops_scatter():
    """CollideVSS::SCATTER_TwoBodyScattering, per collision, alpha != 1 branch."""
    n = 0.0
    n += 1                   # eps = rn * 2pi
    n += SQRT * FMA + 2      # scale = sqrt(2*etrans/(mr*vr2))
    n += POW * FMA + 2       # cosX = 2*pow(rn, alpha_r) - 1
    n += SQRT * FMA + 2      # sinX = sqrt(1 - cosX*cosX)
    n += 3                   # vrc = vi - vj
    n += SQRT * FMA + 3      # d = sqrt(vrc1^2 + vrc2^2)
    n += SINCOS * FMA        # sin(eps), cos(eps) via one sincos
    n += 24                  # ua, vb, wc expressions
    n += 1 + 12              # divisor, then six new velocity components
    return n


# ---- bytes per operation, from callgrind D1 misses x 64 B ----
# callgrind run: 100K particles, 130 steps => 13e6 particle-moves,
#                3.84e6 collision attempts, 2.81e6 collisions
CG_MOVES = 13_000_000
CG_ATTEMPTS = 3_840_023
CG_COLLISIONS = 2_809_345

CG = {                       # (D1 read misses, D1 write misses), self only
    "move":    (28_656_334, 14_383),
    "collide": (14_865_571, 3_560),      # collisions_one inclusive
    "sort":    (18_103_690, 2_132_632),  # Particle::sort self
    "reorder": (10_437_009, 325_421),    # Particle::reorder self
}
LINE = 64.0


def bytes_per(kernel, ops):
    r, w = CG[kernel]
    return (r + w) * LINE / ops


def main():
    kernels = []

    b_move = bytes_per("move", CG_MOVES)
    f_move = flops_move_per_particle()

    f_att = flops_test_collision()
    f_col = flops_setup_collision() + flops_scatter()
    # per-attempt average, folding in the collisions that actually happen
    f_collide_per_attempt = f_att + f_col * (COLLISIONS / ATTEMPTS)
    b_collide = bytes_per("collide", CG_ATTEMPTS)

    # sort: the shipped path is sort() every step plus reorder() every Nth;
    # the tuned path is sort() plus the fused counting sort every Nth.
    # It does no floating point at all, so it cannot be placed on a FLOP
    # roofline; it is reported as achieved bandwidth instead.
    b_sort_step = bytes_per("sort", CG_MOVES)
    b_reorder = bytes_per("reorder", CG_MOVES)

    for phase, tag, period in (("before", "baseline", REORDER_BASE),
                               ("after", "tuned", REORDER_TUNED)):
        t = TIME[tag]

        gf_move = MOVES * f_move / t["move"] / 1e9
        kernels.append({
            "name": f"move ({phase})", "phase": phase,
            "ai": f_move / b_move,
            "gflops": gf_move,
            "gbs": MOVES * b_move / t["move"] / 1e9,
        })

        gf_coll = ATTEMPTS * f_collide_per_attempt / t["collide"] / 1e9
        kernels.append({
            "name": f"collide ({phase})", "phase": phase,
            "ai": f_collide_per_attempt / b_collide,
            "gflops": gf_coll,
            "gbs": ATTEMPTS * b_collide / t["collide"] / 1e9,
        })

    # sort, as a bandwidth-only entry
    sort_bytes_baseline = MOVES * b_sort_step + (MOVES / REORDER_BASE) * b_reorder
    # tuned: fused counting sort measured at 24.6 ns/particle => 3 streaming
    # passes over the 96 B record, i.e. ~288 B/particle on reorder steps
    sort_bytes_tuned = (MOVES * (1 - 1.0 / REORDER_TUNED) * b_sort_step +
                        (MOVES / REORDER_TUNED) * 288.0)

    summary = {
        "kernels": kernels,
        "sort": {
            "baseline_GBs": sort_bytes_baseline / TIME["baseline"]["sort"] / 1e9,
            "tuned_GBs": sort_bytes_tuned / TIME["tuned"]["sort"] / 1e9,
            "note": "sort does no floating point; reported as bandwidth only",
        },
        "flop_model": {
            "move_per_particle": f_move,
            "test_collision_per_attempt": f_att,
            "setup_collision_per_collision": flops_setup_collision(),
            "scatter_per_collision": flops_scatter(),
            "collide_per_attempt_avg": f_collide_per_attempt,
            "transcendental_FMA_equivalents": {
                "pow": POW, "sqrt": SQRT, "sincos": SINCOS},
        },
        "bytes_model": {
            "move_per_particle": b_move,
            "collide_per_attempt": b_collide,
            "sort_per_particle_per_step": b_sort_step,
            "reorder_per_particle": b_reorder,
        },
    }

    with open(os.path.join(HERE, "kernels.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("kernel                 AI (F/B)   achieved GF/s   achieved GB/s")
    for k in kernels:
        print(f"{k['name']:22s} {k['ai']:8.3f} {k['gflops']:13.2f} {k['gbs']:15.2f}")
    print(f"\nsort  baseline {summary['sort']['baseline_GBs']:.2f} GB/s"
          f"   tuned {summary['sort']['tuned_GBs']:.2f} GB/s  (no FLOPs)")
    print(f"\nFLOP model: move {f_move:.0f}/particle, "
          f"collide {f_collide_per_attempt:.0f}/attempt "
          f"(test {f_att:.0f} + {COLLISIONS/ATTEMPTS:.2f} x collision {f_col:.0f})")
    print(f"byte model: move {b_move:.0f} B/particle, "
          f"collide {b_collide:.0f} B/attempt, sort {b_sort_step:.0f} B/particle/step, "
          f"reorder {b_reorder:.0f} B/particle")


if __name__ == "__main__":
    main()
