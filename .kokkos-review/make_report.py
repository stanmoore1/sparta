#!/usr/bin/env python3
"""Build the A/B verification report (HTML) from .kokkos-review/ab/AB*.md.
Usage: make_report.py <out.html>"""
import re, sys, glob, os, html, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]

CLUSTER_NAMES = {
  'AB1': 'Collisions and gas reactions', 'AB2': 'Surface collisions, move, particles',
  'AB3': 'Surface reactions', 'AB4': 'Grid and misc computes',
  'AB5': 'Surface tally computes, temp/rescale', 'AB6': 'Emit fixes',
  'AB7': 'fix ave/histo, ave/grid', 'AB8': 'FFT and remap',
  'AB9': 'Infrastructure, grid, surf', 'AB10': 'Follow-up fixes (round 1)',
  'AB11': 'Follow-up fixes (round 2)', 'AB12-mpi': 'Multi-rank and bounds-check addendum',
}
# incomplete verdicts that a later follow-up fix closed (finding -> follow-up + where it was re-tested)
RESOLVED_BY = {
  'G12x-F-G16-2': 'FU-1 (AB10)', 'G12x-F-G14-1': 'FU-4 (AB10)', 'F-G09-4a': 'FU-8 (AB10)',
  'F-G22-2': 'FU-10 (AB10)', 'FU-3': 'FU-3b (AB11)', 'F-G19-4': 'FU-15: accepted behaviour change',
  'F-G20-5': 'R2-2 (round 2)',
}

def cat(v):
  v = re.sub(r'^(FU|F-G)\S*\s+', '', v.upper())
  if v.startswith('NECESSARY+COMPLETE'): return 'nc'
  if 'COMPLETENESS-PARTIAL' in v: return 'np'
  if v.startswith('INCOMPLETE') or 'INCOMPLETE' in v.split('(')[0]: return 'inc'
  if v.startswith('NOT-SHOWN-NECESSARY'): return 'nsn'
  if v.startswith('NOT-CATCHABLE'): return 'gpu'
  if v.startswith('NECESSARY') and 'COMPLETE' in v and 'INCOMPLETE' not in v: return 'nc'
  if v.startswith('FIX-FAILS'): return 'fail'
  return 'other'

LABEL = {'nc': 'Necessary + complete', 'np': 'Necessary, partly complete', 'inc': 'Incomplete',
         'nsn': 'Not shown necessary', 'fail': 'Fix fails', 'other': 'Other'}

def parse(path):
  txt = open(path, errors='replace').read()
  cl = os.path.basename(path)[:-3]
  entries = []
  for blk in re.split(r'\n(?=### )', txt):
    if not blk.startswith('### '): continue
    head = blk.split('\n', 1)[0][4:].strip()
    m = re.match(r'(\S+(?:\s*/\s*\S+)*?)\s+(?:—|-|\()\s*(.*)', head)
    ident, title = (m.group(1), m.group(2)) if m else (head, '')
    f = {}
    for key in ['class', 'positive control', 'negative control', 'necessary', 'complete', 'verdict']:
      mm = re.search(r'^' + key + r':\s*(.*?)(?=\n[a-z ]+:\s|\Z)', blk, re.M | re.S)
      if mm: f[key] = ' '.join(mm.group(1).split())
    if 'verdict' not in f: continue
    ident = ident.strip('*')
    prev = next((e for e in entries if e['id'] == ident), None)
    if prev:  # a later "addendum" entry for the same ID updates the verdict
      prev['verdict'] = f['verdict']
      prev['addendum_txt'] = ' '.join(f.get(k, '') for k in ('necessary', 'complete'))
      continue
    entries.append(dict(cluster=cl, id=ident, title=title.rstrip(')'), **f))
  status = 'complete' if 'STATUS: COMPLETE' in txt else 'in progress'
  return cl, entries, status

files = sorted(glob.glob(os.path.join(ROOT, 'ab', 'AB*.md')),
               key=lambda p: (len(os.path.basename(p).split('-')[0]), os.path.basename(p)))
clusters = []
for p in files:
  clusters.append(parse(p))

# addendum verdicts override primary verdicts by ID: MPI pass, then GPU split-memory passes
addendum = {}
gpusim = {}
for gf in ['GS-fixes.md', 'GS-poison.md']:
  p = os.path.join(ROOT, 'gpusim', gf)
  if os.path.exists(p):
    for e in parse(p)[1]:
      gpusim[e['id'].split()[0]] = e
for cl, ents, _ in clusters:
  if cl.startswith('AB12'):
    for e in ents:
      addendum[e['id'].split()[0]] = e

counts = collections.Counter()
rows_by_cluster = []
for cl, ents, st in clusters:
  if cl.startswith('AB12'): continue
  rows = []
  for e in ents:
    final = e['verdict']
    ad = addendum.get(e['id'].split()[0])
    if ad: final = ad['verdict']
    gs = gpusim.get(e['id'].split()[0])
    if gs and (cat(final) in ('nsn', 'other') or (cat(final) == 'np' and cat(gs['verdict']) == 'nc')):
      final = gs['verdict']
      ad = ad or {}
      ad = dict(ad, necessary=(ad.get('necessary','') + ' GPU split-memory test: ' + gs.get('necessary','')).strip(), complete=gs.get('complete',''))
    c = cat(final)
    res = RESOLVED_BY.get(e['id'])
    if c == 'inc' and res: c = 'acc' if 'accepted' in res else 'res'
    if c == 'np' and res and res.startswith('R2'): c = 'res'
    counts[c] += 1
    rows.append((e, final, c, ad, res))
  rows_by_cluster.append((cl, rows, st))
LABEL['res'] = 'Incomplete, closed by follow-up'
LABEL['acc'] = 'Behaviour change accepted'
LABEL['gpu'] = 'Needs a real GPU to show'

def esc(s): return html.escape(s or '')
def code(s):  # backticks -> <code>
  return re.sub(r'`([^`]+)`', lambda m: '<code>' + m.group(1) + '</code>', esc(s))

BACKOUT = [
 # id, why not shown, what the fix protects against, recommendation
 ('F-G00-2', 'Needs a real GPU', 'Kernel dereferenced the host Update pointer; illegal address on CUDA/HIP', 'Keep'),
 ('F-G00-4', 'Needs a real GPU', 'Host code did atomics on device-only views; fault on a discrete GPU', 'Keep'),
 ('F-G02-2', 'Race not observable on the host backend', 'Non-atomic += on shared grow counters across GPU threads (undefined behaviour)', 'Keep'),
 ('F-G01-4 / F-G02-1', 'No valid input makes a zero-volume cell with particles', 'GPU spins ~2^31 attempts before the error surfaces', 'Keep'),
 ('F-G21-7', 'Needs a SPARTA_KOKKOS_EXACT + MPI build', 'EXACT-build migration compacted unsynced host data and never pushed it back', 'Keep'),
 ('FU-9', 'No device reader of the grid custom index arrays today', 'Same missing sync as F-G13-4 on the particle side, which the detector showed necessary', 'Keep'),
 ('F-G19-7', 'Benign with FFTW 3.3.10 here', 'Calling fftw cleanup_threads per plan is undefined per the FFTW docs and invalidates other live plans', 'Keep'),
 ('F-G00-17', 'Unreachable: vremax==0 gives zero attempts', 'Same guard the CPU collide_vss has; costs nothing', 'Keep (CPU parity)'),
 ('F-G11-4', 'nrho is already infinite when the guard fires', 'Same guard the CPU fix emit/surf has', 'Keep (CPU parity)'),
 ('F-G12-1', 'nrho is already infinite when the guard fires', 'Same guard the CPU fix emit/face has', 'Keep (CPU parity)'),
 ('F-G17-2', 'mvv2e is 1.0 in every unit system', 'int storage of a double; would truncate if units ever change', 'Keep (type correctness)'),
 ('F-G22-3', 'Not reachable end to end', 'CPU Surf::grow zeroes the new tail; unit test shows the Kokkos tail uninitialized', 'Keep (CPU parity)'),
 ('FU-10', 'Hygiene; no reader of the uninitialized tail', 'CPU Grid::grow_* zero the first allocation', 'Keep (CPU parity)'),
 ('F-G06-1', 'Only compiled with -DSPARTA_KOKKOS_FIXED_LISTS; no failing path found', 'Unused tally slots kept raw handles of computes that may be reallocated', 'Optional: keep (2 placeholder members)'),
 ('F-G00-16-note', 'Unreachable: at least one cell always counts', 'Divide by zero in temp/rescale ave mode if no cell counts', 'Optional: could be dropped (2-line guard, CPU + kk)'),
 ('F-G00-20', 'Unreachable: the parser never produces CUSTOM inputs', 'fix ave/grid/kk CUSTOM branch read etype[-1] and used the double view for int arrays', 'Optional: could be dropped (dead code), or kept so the branch is correct if the parser is extended'),
]
rows = ''.join(f'<tr><td class="id">{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td><td>{esc(d)}</td></tr>' for a,b,c,d in BACKOUT)
BACKOUT_HTML = ('<section class="cluster"><div class="chead"><h2>Back-out review: fixes not proven necessary</h2>'
  '<span class="meta">16 fixes · none harmful · none recommended for back-out</span></div>'
  '<p class="lede">Every fix below passed its negative control: the pre-fix and fixed builds give identical results off the buggy path, and the 141 example decks are identical before and after. So none changes behaviour that was correct. They could not be shown necessary because the failure needs a GPU, an EXACT build, or an input the code cannot currently receive. The three marked optional are the only candidates for dropping to keep the diff small; nothing is gained in correctness by removing them.</p>'
  '<div class="tbl"><table><thead><tr><th>Fix</th><th>Why not shown</th><th>What it guards against</th><th>Recommendation</th></tr></thead><tbody>'
  + rows + '</tbody></table></div></section>')

total = sum(counts.values())
order = ['nc', 'np', 'res', 'acc', 'inc', 'gpu', 'nsn', 'fail', 'other']

followups = open(os.path.join(ROOT, 'AB_FOLLOWUPS.md'), errors='replace').read()
fu = re.findall(r'^- (FU-\S+|Install\.sh|Notes|FU-1\.\.6)[^\n]*', followups, re.M)
fixes_md = open(os.path.join(ROOT, 'FIXES.md'), errors='replace').read()
deferred = [l for l in fixes_md.splitlines() if 'DEFERRED' in l]

o = []
o.append('''<title>KOKKOS Fix Verification</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<style>
/* layout: single reading column; summary strip, then one ledger table per test cluster */
:root{
  --bg:#f6f7f9; --panel:#ffffff; --fg:#1b2330; --muted:#5a6676; --line:#dde2e9; --accent:#2c5d8a;
  --ok:#1f7a4d; --ok-bg:#e3f3ea; --part:#8a6a12; --part-bg:#f7eed3; --bad:#a3322a; --bad-bg:#f8e3e0;
  --na:#5b5f6b; --na-bg:#eceef2; --res:#2c5d8a; --res-bg:#e2ecf6;
  --f-display:"Source Serif 4",Georgia,serif; --f-body:"IBM Plex Sans",system-ui,sans-serif; --f-mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#12161c; --panel:#1a2029; --fg:#e3e8ef; --muted:#9aa6b6; --line:#2c3440; --accent:#7fb0dc;
  --ok:#6fd19c; --ok-bg:#16301f; --part:#e2c46b; --part-bg:#352b10; --bad:#f0897e; --bad-bg:#3a1a17;
  --na:#b3b8c4; --na-bg:#262c36; --res:#8dbbe4; --res-bg:#182a3c; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#12161c; --panel:#1a2029; --fg:#e3e8ef; --muted:#9aa6b6; --line:#2c3440; --accent:#7fb0dc;
  --ok:#6fd19c; --ok-bg:#16301f; --part:#e2c46b; --part-bg:#352b10; --bad:#f0897e; --bad-bg:#3a1a17;
  --na:#b3b8c4; --na-bg:#262c36; --res:#8dbbe4; --res-bg:#182a3c; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.55 var(--f-body);padding-inline:16px;padding-block:28px 64px}
main{max-width:1080px;margin:0 auto;display:flex;flex-direction:column;gap:36px}
h1{font:600 2rem/1.15 var(--f-display);margin:0;text-wrap:balance}
h2{font:600 1.35rem/1.2 var(--f-display);margin:0 0 4px;text-wrap:balance}
p{margin:0;max-width:72ch}
.lede{color:var(--muted)}
code{font:0.88em var(--f-mono);background:var(--na-bg);padding:0 .25em;border-radius:3px;overflow-wrap:anywhere}
.eyebrow{font:500 .75rem var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px;display:flex;flex-direction:column;gap:2px}
.tile b{font:600 1.6rem var(--f-mono);font-variant-numeric:tabular-nums}
.tile span{font-size:.82rem;color:var(--muted)}
.chip{display:inline-block;font:500 .72rem var(--f-mono);letter-spacing:.02em;padding:2px 7px;border-radius:999px;white-space:nowrap}
.c-nc{color:var(--ok);background:var(--ok-bg)} .c-np{color:var(--part);background:var(--part-bg)}
.c-inc,.c-fail{color:var(--bad);background:var(--bad-bg)} .c-nsn,.c-other{color:var(--na);background:var(--na-bg)}
.c-res{color:var(--res);background:var(--res-bg)} .c-gpu{color:var(--na);background:var(--na-bg)} .c-acc{color:var(--part);background:var(--part-bg)}
.tile.k-nc b{color:var(--ok)} .tile.k-np b{color:var(--part)} .tile.k-inc b{color:var(--bad)} .tile.k-nsn b{color:var(--na)} .tile.k-res b{color:var(--res)}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.filters button{font:500 .8rem var(--f-body);border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:999px;padding:4px 12px;cursor:pointer}
.filters button[aria-pressed="true"]{border-color:var(--accent);color:var(--accent)}
.filters button:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
section.cluster{display:flex;flex-direction:column;gap:10px}
.chead{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:baseline}
.chead .meta{font:.8rem var(--f-mono);color:var(--muted)}
.tbl{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;min-width:720px;font-size:.88rem}
th,td{text-align:left;vertical-align:top;padding:9px 12px;border-bottom:1px solid var(--line)}
th{font:500 .72rem var(--f-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
tr:last-child td{border-bottom:0}
td.id{font:500 .82rem var(--f-mono);white-space:nowrap}
td.ev{min-width:0}
details summary{cursor:pointer;color:var(--accent);font-size:.82rem;margin-top:4px}
details .more{display:flex;flex-direction:column;gap:6px;margin-top:6px;color:var(--muted);font-size:.84rem}
details .more b{color:var(--fg);font-weight:500}
.note{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px 16px;display:flex;flex-direction:column;gap:8px}
.note ul{margin:0;padding-left:1.1em;display:flex;flex-direction:column;gap:6px}
.method{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.method div{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px;min-width:0}
.method h3{font:500 .75rem var(--f-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 6px}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
</style>
<main>
<header style="display:flex;flex-direction:column;gap:10px">
  <div class="eyebrow">SPARTA · src/KOKKOS · branch ccr-3dcb1e77-fem393</div>
  <h1>KOKKOS fix verification: is each fix necessary and complete?</h1>
  <p class="lede">Every bug fix from the line-by-line review of the KOKKOS package was A/B tested: the pre-fix build (A, commit e071055f) against the fixed build (B), with the CPU styles of the same binary as a reference. A fix counts as <b>necessary</b> when A demonstrably misbehaves on the buggy path, and <b>complete</b> when B is correct on every variant the fix covers and no sibling site keeps the bug.</p>
</header>
''')

o.append('<section style="display:flex;flex-direction:column;gap:12px"><h2>Outcome</h2><div class="strip">')
o.append(f'<div class="tile"><b>{total}</b><span>fix IDs tested</span></div>')
for k in order:
  if counts[k]:
    o.append(f'<div class="tile k-{k}"><b>{counts[k]}</b><span>{LABEL[k]}</span></div>')
o.append('</div><p class="lede">Incomplete fixes found by testing were completed with follow-up fixes and tested again; those rows show where. "Not shown necessary" means the bug cannot be made to fail on this machine (GPU-only memory spaces, unreachable input, or a build mode not available); for every one of them a negative control shows the fix changes nothing on the CPU.</p></section>')

o.append('<div class="filters" role="group" aria-label="Filter by verdict"><span class="eyebrow" style="color:var(--muted)">Show</span>')
o.append('<button type="button" data-f="all" aria-pressed="true">All</button>')
for k in order:
  if counts[k]:
    o.append(f'<button type="button" data-f="{k}" aria-pressed="false">{LABEL[k]}</button>')
o.append('</div>')

for cl, rows, st in rows_by_cluster:
  name = CLUSTER_NAMES.get(cl, cl)
  cc = collections.Counter(r[2] for r in rows)
  summ = ' · '.join(f'{cc[k]} {LABEL[k].lower()}' for k in order if cc[k])
  o.append(f'<section class="cluster" data-cluster="{cl}"><div class="chead"><h2>{esc(name)}</h2><span class="meta">{cl} · {len(rows)} fixes · {summ}</span></div>')
  o.append('<div class="tbl"><table><thead><tr><th>Fix</th><th>Bug</th><th>Verdict</th><th>Evidence</th></tr></thead><tbody>')
  for e, final, c, ad, res in rows:
    ev = e.get('necessary', '')
    more = []
    for k, lab in [('positive control', 'Positive control'), ('negative control', 'Negative control'),
                   ('complete', 'Complete'), ('class', 'Class')]:
      if e.get(k): more.append(f'<div><b>{lab}:</b> {code(e[k])}</div>')
    if e.get('addendum_txt'):
      more.append(f'<div><b>Addendum:</b> {code(e["addendum_txt"])}</div>')
    if ad:
      more.append(f'<div><b>Multi-rank addendum:</b> {code(ad.get("necessary",""))} {code(ad.get("complete",""))}</div>')
    if res:
      more.append(f'<div><b>Closed by:</b> {esc(res)}</div>')
    chip = f'<span class="chip c-{c}">{esc(LABEL[c])}</span>'
    o.append(f'<tr data-k="{c}"><td class="id">{esc(e["id"])}</td><td>{code(e["title"])}</td><td>{chip}<div style="font-size:.78rem;color:var(--muted);margin-top:4px">{code(final)}</div></td>'
             f'<td class="ev">{code(ev)}<details><summary>Controls and details</summary><div class="more">{"".join(more)}</div></details></td></tr>')
  o.append('</tbody></table></div></section>')


ROUND2 = [
 ('R2-1', 'Implicit-surf tally collate put tallies of cells migrated by fix balance into local cell 0 (compute isurf/grid, react/isurf/grid, fix ave/grid); collate_vector_implicit dropped ghost tallies', 'e7d8acdb',
  'A: 9.5k-29k wrong cells per run (2d/3d, np2/np4, CPU/kk); fix ave/grid 1,700-10,300; segfault when fix ablate post-processed before balance. B: 0 mismatches everywhere; no-balance and np1 runs byte-identical.', 'r2/R2-1.md'),
 ('R2-2', 'FFT remap collective mode: idle-rank crash, rank mix-up giving wrong FFTs, leaked MPI groups and communicator, uninitialised 2d plan field', '80739203',
  'A: SIGSEGV at remap3d_kokkos.cpp:689 on idle-rank collective plans; 2 groups leaked per plan. B: 1,488 harness runs match the CPU FFT bitwise, collective == point-to-point, 0 leaks; examples/fft unchanged.', 'r2/R2-2.md'),
 ('R2-3', 'Surf counts in compute react/surf and Surf::init depended on the rank count', '8df7d59e',
  'A: 25/26/28 and 600/978/1370 on 1/2/4 ranks. B: true count on every rank count; tallies bit-identical.', 'r2/R2-3.md'),
 ('auto_sync', 'Particle/Grid/SurfKokkos::sync(Device) claimed the host while the device held a claim (Kokkos abort); grid lost restart custom values before wrap', 'cc4ae24f',
  'A and pre-review: abort on custom-vector growth, adapt with custom grid vectors, and 4 stock examples/custom/*restart decks. B: all run and match the non-Kokkos run. Surf class unexercised (fixed on the LAMMPS precedent).', 'gpusim/GS-autosync.md'),
 ('GPU-coh', 'Latent host/device coherence faults: custom vectors resized on the device, surf custom values never sent to the device on the first run, handles taken before sync, stale writes in emit destructors, unclaimed scratch edits', '3859618b',
  'Split-memory detector: A shows watch/stale/poison reports on targeted decks; C is clean; 134 examples np1 + 21 np4 stats identical to stock.', 'gpusim/GS-coh.md'),
 ('R8b', 'fix custom surf set / fix surf/temp wrote surf custom values on the host during a run without claiming them', '647697c1',
  'A: watch reports on the 4 in.custom.circle.*.fix* decks at np1/np4. B: 0 reports; stats identical to stock; 130 other decks unchanged.', 'gpusim/GS-r8b.md'),
]

o.append('<section class="cluster"><div class="chead"><h2>Round 2: requested fixes and GPU split-memory findings</h2><span class="meta">6 fixes · all necessary and complete except where noted</span></div>')
o.append('<div class="tbl"><table><thead><tr><th>Fix</th><th>Bug</th><th>Commit</th><th>Evidence</th></tr></thead><tbody>')
for rid, bug, cm, ev, src in ROUND2:
  o.append(f'<tr data-k="nc"><td class="id">{esc(rid)}</td><td>{code(bug)}</td><td class="id">{cm}</td><td class="ev">{code(ev)}<div style="font-size:.78rem;color:var(--muted);margin-top:4px">Record: <code>.kokkos-review/{src}</code></div></td></tr>')
o.append('</tbody></table></div></section>')
o.append(BACKOUT_HTML)
o.append('<section class="note"><h2>Follow-ups found by A/B testing</h2><ul>')
for line in followups.splitlines():
  if line.startswith('- '):
    o.append(f'<li>{code(line[2:])}</li>')
o.append('</ul></section>')

o.append('<section class="note"><h2>Deferred, not fixed</h2><ul>')
for d in deferred:
  cells = [c.strip() for c in d.strip('|').split('|')]
  o.append(f'<li><code>{esc(cells[0])}</code> {esc(cells[1])}</li>')
o.append('</ul></section>')

o.append('''<section style="display:flex;flex-direction:column;gap:12px"><h2>Method</h2><div class="method">
<div><h3>Builds</h3><p>A: commit e071055f. B: all review fixes. C: B plus follow-ups. Each as an OpenMP build with MPI stubs, and as a real-MPI build with Kokkos <code>DEBUG_BOUNDS_CHECK</code> so out-of-bounds view accesses abort.</p></div>
<div><h3>Controls</h3><p>Positive control: an input on the buggy path where A fails (wrong value vs CPU or analytic, crash, bounds abort, NaN, hang, leak). Negative control: same code path without the trigger; A and B must agree.</p></div>
<div><h3>Coverage</h3><p>Kernel paths (atomic, duplicated, sorted), 1 and 4 threads, 1 to 4 MPI ranks, 2D and 3D, each surface collision model. Standalone unit drivers for unreachable or library-only paths.</p></div>
<div><h3>Regression</h3><p>All 141 example inputs with Kokkos on: thermo output identical between A and B except ambi, which now runs further; identical between B and the final build with all follow-up fixes, and again after round 2 (141/141).</p></div>\n<div><h3>GPU memory</h3><p>Split-memory detector builds (separate host allocation, software DualView state machine; watch, stale, audit and AddressSanitizer poison modes) of the pre-fix and fixed code. All 134 examples at 1 rank and 21 at 4 ranks match stock stats on the fixed code.</p></div>
<div><h3>Limits</h3><p>No GPU on the test machine. The split-memory tool reproduces most host/device coherence bugs, but not a kernel dereferencing a host pointer or host code touching device-only memory. No <code>SPARTA_KOKKOS_EXACT</code> build was made.</p></div>
<div><h3>Records</h3><p>Per-fix evidence: <code>.kokkos-review/ab/AB*.md</code>. Fix list and commits: <code>.kokkos-review/FIXES.md</code>.</p></div>
</div></section>
</main>
<script>
(function(){
  var btns=document.querySelectorAll('.filters button');
  function apply(f){
    btns.forEach(function(b){b.setAttribute('aria-pressed', String(b.dataset.f===f));});
    document.querySelectorAll('tr[data-k]').forEach(function(r){r.hidden = !(f==='all'||r.dataset.k===f);});
    document.querySelectorAll('section.cluster').forEach(function(s){
      s.hidden = !s.querySelector('tr[data-k]:not([hidden])');});
  }
  btns.forEach(function(b){b.addEventListener('click',function(){apply(b.dataset.f);});});
})();
</script>
''')
open(OUT, 'w').write('\n'.join(o))
print('wrote', OUT, 'entries', total, dict(counts))
