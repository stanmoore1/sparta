# stats.py <log>: print every stats block with time-like columns removed (CPU, TPS, wall, cpu, cpu/...)
import sys, re
drop = {'CPU', 'TPS', 'Wall', 'wall', 'cpu', 'tpcpu', 'spcpu', 'elapsed', 'elaplong', 'Elapsed'}
lines = open(sys.argv[1], errors='replace').read().splitlines()
i = 0
while i < len(lines):
    l = lines[i]
    if re.match(r'^\s*Step\s', l):
        hdr = l.split(); keep = [k for k, h in enumerate(hdr) if h not in drop]
        print('H', ' '.join(hdr[k] for k in keep)); i += 1
        while i < len(lines) and not lines[i].startswith('Loop') and not lines[i].startswith('ERROR'):
            w = lines[i].split()
            if len(w) == len(hdr) and re.match(r'^-?\d', w[0]):
                print(' '.join(w[k] for k in keep))
            i += 1
    else:
        i += 1
