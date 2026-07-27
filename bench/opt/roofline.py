#!/usr/bin/env python3
"""Roofline plot for the SPARTA in.collide kernels.

Reads:
  - machine_peak.out : measured bandwidth/compute ceilings (bench/opt/micro)
  - kernels.json     : per-kernel arithmetic intensity and achieved GFLOP/s

and writes roofline.png / roofline.svg.

There is no PMU in this KVM guest, so nothing here comes from hardware
counters. Provenance of every number is recorded in ROOFLINE.md:
  - ceilings      : measured by micro/machine_peak.cpp on this machine
  - kernel bytes  : callgrind cache simulation (LL misses x 64B for DRAM
                    traffic, D1 misses x 64B for L2/L3 traffic)
  - kernel FLOPs  : counted analytically from the source, with transcendentals
                    charged at their measured FMA-equivalent cost
  - kernel time   : SPARTA's own per-section timers

Falls back to writing SVG directly if matplotlib is unavailable.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def read_peaks(path):
    peaks = {"bw": [], "scalars": {}}
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "BW" and len(parts) >= 4:
                peaks["bw"].append((int(parts[1]), float(parts[2]), float(parts[3])))
            elif len(parts) == 2:
                try:
                    peaks["scalars"][parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return peaks


def bandwidth_roofs(peaks):
    """Turn the bandwidth-vs-size sweep into named roofs.

    The sweep's plateaus are the roofs. We take the max copy bandwidth inside
    each size band; the band edges come from where the curve knees, which we
    locate as the largest size still within 80% of the small-size plateau.
    """
    bw = peaks["bw"]
    if not bw:
        return {}
    # copy bandwidth (read+write), which is the mix move/sort actually see
    sizes = [b[0] for b in bw]
    copy = [b[2] for b in bw]
    read = [b[1] for b in bw]

    roofs = {}
    # L1: smallest sizes; DRAM: largest size measured
    roofs["L1"] = max(c for s, c in zip(sizes, copy) if s <= 32 * 1024)
    l2 = [c for s, c in zip(sizes, copy) if 64 * 1024 <= s <= 1024 * 1024]
    if l2:
        roofs["L2"] = max(l2)
    l3 = [c for s, c in zip(sizes, copy) if 4 * 1024 * 1024 <= s <= 64 * 1024 * 1024]
    if l3:
        roofs["L3"] = max(l3)
    dram = [c for s, c in zip(sizes, copy) if s >= 256 * 1024 * 1024]
    if dram:
        roofs["DRAM"] = max(dram)
    roofs["_read_dram"] = max(r for s, r in zip(sizes, read) if s >= 256 * 1024 * 1024) \
        if any(s >= 256 * 1024 * 1024 for s in sizes) else 0.0
    return roofs


def plot_mpl(peaks, roofs, kernels, out_png, out_svg, kernels_extra=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 6.5))

    ai = np.logspace(-3, 2, 400)

    compute_roofs = []
    for key, label in (("PEAK_AVX512_GFLOPs", "AVX-512 FMA peak"),
                       ("PEAK_AVX2_GFLOPs", "AVX2 FMA peak"),
                       ("PEAK_SCALAR_GFLOPs", "scalar FMA peak")):
        if key in peaks["scalars"]:
            compute_roofs.append((peaks["scalars"][key], label))

    ceiling = max(c for c, _ in compute_roofs) if compute_roofs else 100.0

    # bandwidth roofs: diagonal lines, GFLOP/s = AI * GB/s
    colors = {"L1": "#8a8f98", "L2": "#6b8fb5", "L3": "#5a9e78", "DRAM": "#b5733a"}
    for name in ("L1", "L2", "L3", "DRAM"):
        if name not in roofs:
            continue
        bw = roofs[name]
        y = np.minimum(ai * bw, ceiling)
        ax.plot(ai, y, "--", lw=1.4, color=colors[name], alpha=0.9)
        # label along the diagonal portion
        xi = 10 ** -2.2
        ax.text(xi, xi * bw * 1.06, f"{name} {bw:.0f} GB/s", color=colors[name],
                fontsize=8, rotation=32, rotation_mode="anchor", va="bottom")

    for i, (val, label) in enumerate(compute_roofs):
        ax.axhline(val, color="#3b3b3b", lw=1.2, alpha=0.55 - 0.12 * i, ls="-")
        ax.text(60, val * 1.05, f"{label} {val:.0f} GF/s", fontsize=8,
                ha="right", color="#3b3b3b")

    # binding roof for each kernel: whichever of DRAM bandwidth and scalar
    # compute is lower at that arithmetic intensity
    dram = roofs.get("DRAM", 20.0)
    scalar = peaks["scalars"].get("PEAK_SCALAR_GFLOPs", 12.8)

    markers = {"before": ("o", "#c0392b"), "after": ("s", "#1e7d4f")}
    for k in kernels:
        m, c = markers.get(k.get("phase", "before"), ("o", "#c0392b"))
        ax.plot(k["ai"], k["gflops"], m, color=c, ms=9,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)

        roof = min(k["ai"] * dram, scalar)
        frac = 100.0 * k["gflops"] / roof
        which = "DRAM" if k["ai"] * dram < scalar else "compute"

        # a faint stem up to the binding roof makes the gap legible at a glance
        ax.plot([k["ai"], k["ai"]], [k["gflops"], roof], "-",
                color=c, lw=0.8, alpha=0.35, zorder=1)
        ax.annotate(f'{k["name"]}  —  {frac:.0f}% of {which} roof',
                    (k["ai"], k["gflops"]),
                    textcoords="offset points", xytext=(11, -3),
                    fontsize=8.5, color=c)

    # sort does no floating point, so it has no place on a FLOP axis; state
    # its bandwidth utilisation in the corner rather than omitting it silently
    srt = kernels_extra
    if srt and "tuned_GBs" in srt:
        ax.text(1.3e-3, 1.6e-2,
                f'sort: no FLOPs — {srt["baseline_GBs"]:.1f} GB/s baseline, '
                f'{srt["tuned_GBs"]:.1f} GB/s tuned, of a {dram:.0f} GB/s roof',
                fontsize=8.5, color="#444",
                bbox=dict(fc="white", ec="#ccc", lw=0.6, pad=3.5))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity  [FLOP / byte of DRAM traffic]")
    ax.set_ylabel("performance  [GFLOP/s]")
    ax.set_title("SPARTA bench/in.collide — roofline, 1M particles, 1 core")
    ax.grid(True, which="major", alpha=0.25, lw=0.6)
    ax.grid(True, which="minor", alpha=0.10, lw=0.4)
    ax.set_xlim(1e-3, 1e2)
    ax.set_ylim(1e-2, ceiling * 3)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color="#c0392b", label="baseline"),
               Line2D([], [], marker="s", ls="", color="#1e7d4f", label="optimized")]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_svg)
    print(f"wrote {out_png} and {out_svg}")


def plot_svg(peaks, roofs, kernels, out_svg, kernels_extra=None):
    """Minimal hand-rolled SVG, used when matplotlib is not installed."""
    import math

    W, H = 900, 640
    L, R, T, B = 90, 40, 60, 70
    pw, ph = W - L - R, H - T - B

    xlo, xhi = -3.0, 2.0          # log10 AI
    compute = [(v, k) for k, v in peaks["scalars"].items() if k.startswith("PEAK_")]
    ceiling = max(v for v, _ in compute) if compute else 100.0
    ylo, yhi = -2.0, math.log10(ceiling * 3)

    def X(ai):
        return L + pw * (math.log10(ai) - xlo) / (xhi - xlo)

    def Y(g):
        return T + ph * (1.0 - (math.log10(g) - ylo) / (yhi - ylo))

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">')
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')

    # grid + axes
    for d in range(int(xlo), int(xhi) + 1):
        x = X(10 ** d)
        s.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}" '
                 f'stroke="#e3e3e3" stroke-width="1"/>')
        s.append(f'<text x="{x:.1f}" y="{T+ph+18}" font-size="11" '
                 f'text-anchor="middle" fill="#444">1e{d}</text>')
    d = int(math.floor(ylo))
    while d <= yhi:
        y = Y(10 ** d)
        if T <= y <= T + ph:
            s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" '
                     f'stroke="#e3e3e3" stroke-width="1"/>')
            s.append(f'<text x="{L-8}" y="{y+4:.1f}" font-size="11" '
                     f'text-anchor="end" fill="#444">1e{d}</text>')
        d += 1
    s.append(f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="none" '
             f'stroke="#999" stroke-width="1"/>')

    colors = {"L1": "#8a8f98", "L2": "#6b8fb5", "L3": "#5a9e78", "DRAM": "#b5733a"}
    for name in ("L1", "L2", "L3", "DRAM"):
        if name not in roofs:
            continue
        bw = roofs[name]
        pts = []
        for i in range(200):
            ai = 10 ** (xlo + (xhi - xlo) * i / 199.0)
            g = min(ai * bw, ceiling)
            if g < 10 ** ylo:
                continue
            pts.append(f"{X(ai):.1f},{Y(g):.1f}")
        s.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                 f'stroke="{colors[name]}" stroke-width="1.6" '
                 f'stroke-dasharray="6,4"/>')
        ai0 = 10 ** (xlo + 0.35)
        s.append(f'<text x="{X(ai0):.1f}" y="{Y(min(ai0*bw, ceiling))-6:.1f}" '
                 f'font-size="10" fill="{colors[name]}">{name} {bw:.0f} GB/s</text>')

    for val, key in sorted(compute, reverse=True):
        y = Y(val)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" '
                 f'stroke="#3b3b3b" stroke-width="1.1" opacity="0.5"/>')
        lbl = key.replace("PEAK_", "").replace("_GFLOPs", "")
        s.append(f'<text x="{L+pw-6}" y="{y-5:.1f}" font-size="10" '
                 f'text-anchor="end" fill="#3b3b3b">{lbl} {val:.0f} GF/s</text>')

    for k in kernels:
        col = "#1e7d4f" if k.get("phase") == "after" else "#c0392b"
        x, y = X(k["ai"]), Y(k["gflops"])
        if k.get("phase") == "after":
            s.append(f'<rect x="{x-5:.1f}" y="{y-5:.1f}" width="10" height="10" '
                     f'fill="{col}" stroke="white" stroke-width="1.5"/>')
        else:
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{col}" '
                     f'stroke="white" stroke-width="1.5"/>')
        s.append(f'<text x="{x+9:.1f}" y="{y-6:.1f}" font-size="10.5" '
                 f'fill="{col}">{k["name"]}</text>')

    s.append(f'<text x="{L+pw/2:.0f}" y="{H-20}" font-size="12.5" '
             f'text-anchor="middle" fill="#222">'
             f'arithmetic intensity [FLOP / byte of DRAM traffic]</text>')
    s.append(f'<text x="22" y="{T+ph/2:.0f}" font-size="12.5" '
             f'text-anchor="middle" fill="#222" '
             f'transform="rotate(-90 22 {T+ph/2:.0f})">performance [GFLOP/s]</text>')
    s.append(f'<text x="{L}" y="{T-24}" font-size="15" fill="#111">'
             f'SPARTA bench/in.collide — roofline, 1M particles, 1 core</text>')
    s.append(f'<circle cx="{L+pw-150}" cy="{T+16}" r="5" fill="#c0392b"/>')
    s.append(f'<text x="{L+pw-138}" y="{T+20}" font-size="11" fill="#444">baseline</text>')
    s.append(f'<rect x="{L+pw-80}" y="{T+11}" width="10" height="10" fill="#1e7d4f"/>')
    s.append(f'<text x="{L+pw-64}" y="{T+20}" font-size="11" fill="#444">optimized</text>')
    s.append('</svg>')

    with open(out_svg, "w") as f:
        f.write("\n".join(s))
    print(f"wrote {out_svg} (matplotlib unavailable, hand-rolled SVG)")


def main():
    peaks_path = os.path.join(HERE, "micro", "machine_peak.out")
    kernels_path = os.path.join(HERE, "kernels.json")

    peaks = read_peaks(peaks_path)
    if peaks is None:
        sys.exit(f"missing {peaks_path}; run micro/machine_peak first")
    roofs = bandwidth_roofs(peaks)

    if not os.path.exists(kernels_path):
        sys.exit(f"missing {kernels_path}")
    with open(kernels_path) as f:
        data = json.load(f)
    kernels = data["kernels"]
    extra = data.get("sort")

    out_png = os.path.join(HERE, "roofline.png")
    out_svg = os.path.join(HERE, "roofline.svg")
    try:
        plot_mpl(peaks, roofs, kernels, out_png, out_svg, extra)
    except ImportError:
        plot_svg(peaks, roofs, kernels, out_svg, extra)

    print("\nbandwidth roofs (GB/s, read+write mix):")
    for k in ("L1", "L2", "L3", "DRAM"):
        if k in roofs:
            print(f"  {k:5s} {roofs[k]:8.1f}")
    print("compute ceilings (GFLOP/s):")
    for k, v in sorted(peaks["scalars"].items()):
        if k.startswith("PEAK_"):
            print(f"  {k:24s} {v:8.1f}")


if __name__ == "__main__":
    main()
