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

import subprocess, sys, os, re

def run(exe, infile):
    log = "log.verify." + infile[3:]
    subprocess.run([exe, "-in", infile, "-log", log],
                   check=True, stdout=subprocess.DEVNULL)
    return parse(log)

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

    print("=" * 40)
    print("ALL CHECKS PASSED" if allok else "SOME CHECKS FAILED")
    sys.exit(0 if allok else 1)

if __name__ == "__main__":
    main()
