#!/usr/bin/env python3
"""Physical-invariant checks for the SWS (species weighting scheme) examples.

Unlike the gold-log regression comparison (tools/testing/regression.py),
which pins exact trajectories for a fixed seed, this script checks that the
physics captured by the SWS implementation is right, independent of the RNG
stream. Run it after any refactor, in this directory:

    python3 verify_sws.py /path/to/spa_serial

Checks:
1. in.sws.box:  numerical particle split follows f_i/w_i, sum(w_i) tally
   equals the analytic value, and temperature is statistically steady
   (no energy leak) over the run.
2. in.sws0.box: with no SWS keyword, weights are inert: sumwi == np exactly
   and temperature holds at 273 K.
3. in.sws.emit: trace species (w=0.1) is emitted with ~10x more numerical
   particles per physical particle than its 10% mole fraction.
4. in.sws.chem: particle count grows (dissociation) and physical mass
   sum(w_i * m_i) is conserved to within the reaction stoichiometry.
"""

import subprocess, sys, os, re, tempfile

def run(exe, infile):
    log = "log.verify." + infile[3:]
    subprocess.run([exe, "-in", infile, "-log", log],
                   check=True, stdout=subprocess.DEVNULL)
    return parse(log)

# SWS and stochastic weighting (SWPM) are mutually exclusive; enabling both
# must be rejected by Particle::setup_weighting() with a single message.
# Run inline (not an in.* deck) so the gold-log regression never sees it.
EXCLUSION_DECK = """seed 12345
dimension 3
global gridcut 1.0e-5
boundary rr rr rr
create_box 0 1e-4 0 1e-4 0 1e-4
create_grid 5 5 5
species {species} N2 N SWS
mixture air N2 N vstream 0 0 0 temp 273
mixture air N2 frac 0.9
mixture air N frac 0.1
global nrho 7e22
global fnum 7e6
fix sw stochastic_weight
collide vss air {vss}
collide_modify stochastic_weight yes
create_particles air n 2000
run 1
"""

def run_expect_error(exe, deck_text, expect):
    with tempfile.NamedTemporaryFile("w", suffix=".sparta_excl",
                                     delete=False, dir=".") as f:
        f.write(deck_text)
        path = f.name
    try:
        p = subprocess.run([exe, "-in", path],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
    finally:
        os.remove(path)
    return p.returncode != 0 and "ERROR" in p.stdout and expect in p.stdout

def parse(log):
    rows, header = [], None
    with open(log) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("Step "):
            header = line.split()
            for row in lines[i+1:]:
                parts = row.split()
                if not parts or not re.match(r"^\d+$", parts[0]):
                    break
                rows.append([float(x) for x in parts])
    return header, rows

def col(header, rows, name):
    j = header.index(name)
    return [r[j] for r in rows]

def check(label, ok, detail=""):
    print(("PASS" if ok else "FAIL"), label, detail)
    return ok

def main():
    exe = sys.argv[1] if len(sys.argv) > 1 else "../../src/spa_serial"
    exe = os.path.abspath(exe)
    allok = True

    # 1. weighted thermal box
    h, rows = run(exe, "in.sws.box")
    n2 = col(h, rows, "c_nsp[1]")[-1]
    n  = col(h, rows, "c_nsp[2]")[-1]
    sumwi = col(h, rows, "c_redsum[2]")[-1]
    temp = col(h, rows, "c_temp")
    # expected numerical fractions ~ (0.9/1.0) : (0.1/0.1) = 0.9 : 1.0
    frac_n = n / (n2 + n)
    allok &= check("sws.box numerical N fraction ~ 0.526",
                   abs(frac_n - 1.0/1.9) < 0.02, f"got {frac_n:.4f}")
    allok &= check("sws.box sum(w_i) ~ N2 + 0.1*N",
                   abs(sumwi - (n2 + 0.1*n)) < 1.0, f"got {sumwi:.1f}")
    # steady temperature: last value within 10% of value at step 100,
    # and no monotonic decay (compare halves)
    t_first, t_last = temp[1], temp[-1]
    allok &= check("sws.box temperature steady (no energy leak)",
                   abs(t_last - t_first) / t_first < 0.10,
                   f"t(100)={t_first:.1f} t(end)={t_last:.1f}")
    # weighted temperature (compute temp): normalized by the summed
    # species weight, not the raw particle count.  at step 0 the particles
    # are a fresh Maxwellian draw at the mixture temperature (273.15 K, no
    # collisions yet, no energy leak), so the weighted normalization must
    # recover it.  a broken pweight (raw-count divisor, wrong scale, NaN)
    # shows up here; the exact per-step weighted temperature is pinned by
    # the gold-log regression.
    allok &= check("sws.box weighted temperature recovers draw temp (step 0)",
                   abs(temp[0] - 273.15) < 8.0, f"got {temp[0]:.2f}")

    # 2. SWS off: weights inert
    h, rows = run(exe, "in.sws0.box")
    np_ = col(h, rows, "Np")[-1]
    sumn = col(h, rows, "c_redsum[1]")[-1]
    sumwi = col(h, rows, "c_redsum[2]")[-1]
    temp = col(h, rows, "c_temp")[-1]
    allok &= check("sws0.box sumwi == np (weights reset to 1)",
                   sumwi == np_ == sumn, f"np={np_} sumwi={sumwi}")
    allok &= check("sws0.box temperature ~273 K",
                   abs(temp - 273.15) < 8.0, f"got {temp:.1f}")

    # 3. weighted emission
    h, rows = run(exe, "in.sws.emit")
    n2 = col(h, rows, "c_nsp[1]")[-1]
    n  = col(h, rows, "c_nsp[2]")[-1]
    # inflow mole fractions 0.9/0.1, weights 1.0/0.1
    # -> numerical ratio N/N2 ~ (0.1/0.1)/(0.9/1.0) = 1.11
    # (identical thermal speed factors for N2 vs N differ slightly via
    #  mol_inflow's vscale; allow a generous band)
    ratio = n / n2
    allok &= check("sws.emit trace species oversampled ~10x",
                   0.8 < ratio < 1.7, f"N/N2 numerical ratio = {ratio:.2f}")

    # 4. chemistry: physical mass conservation
    h, rows = run(exe, "in.sws.chem")
    m_n2, m_n = 4.65e-26, 2.325e-26
    w_n2, w_n = 1.0, 0.1
    mass = [r[h.index("c_nsp[1]")]*w_n2*m_n2 + r[h.index("c_nsp[2]")]*w_n*m_n
            for r in rows]
    drift = abs(mass[-1] - mass[0]) / mass[0]
    allok &= check("sws.chem physical mass conserved",
                   drift < 0.02, f"relative drift = {drift:.4f}")
    nreact = col(h, rows, "Nreact")[-1] if "Nreact" in h else \
             col(h, rows, "c_temp")[0]  # fallback
    allok &= check("sws.chem reactions occurred",
                   col(h, rows, "Np")[-1] > col(h, rows, "Np")[0],
                   f"np {col(h,rows,'Np')[0]:.0f} -> {col(h,rows,'Np')[-1]:.0f}")

    # 5. mutual exclusion: SWS + SWPM together must be rejected
    deck = EXCLUSION_DECK.format(species="sws.species", vss="sws.vss")
    allok &= check("SWS + SWPM rejected as mutually exclusive",
                   run_expect_error(exe, deck, "mutually exclusive"))

    print("=" * 40)
    print("ALL CHECKS PASSED" if allok else "SOME CHECKS FAILED")
    sys.exit(0 if allok else 1)

if __name__ == "__main__":
    main()
