# cmp.py <resdir>: compare kfin vs kmst per (np, example); prints SAME / DIFF / ERR lines and a summary
import sys, os, collections
R = sys.argv[1]; units = collections.defaultdict(dict)
for f in os.listdir(R):
    if not f.endswith('.res'): continue
    b, np, rest = f[:-4].split('.', 2)
    units[(np, rest)][b] = open(os.path.join(R, f)).read()
cnt = collections.Counter()
for k in sorted(units):
    u = units[k]
    if len(u) < 2: cnt['PENDING'] += 1; continue
    a, m = u['kfin'], u['kmst']
    ra, rm = a.splitlines()[0], m.splitlines()[0]
    sa = [l for l in a.splitlines() if not l.startswith('RC') and not l.startswith('ERROR')]
    sm = [l for l in m.splitlines() if not l.startswith('RC') and not l.startswith('ERROR')]
    ea = [l for l in a.splitlines() if l.startswith('ERROR')]; em = [l for l in m.splitlines() if l.startswith('ERROR')]
    if a == m: tag = 'SAME'
    elif sa == sm and ra == rm: tag = 'SAME-STATS(err text differs)'
    elif sa == sm: tag = 'SAME-STATS(rc differs)'
    else:
        n = min(len(sa), len(sm)); first = next((i for i in range(n) if sa[i] != sm[i]), n)
        tag = 'DIFF first@%d' % first
    cnt[tag.split('(')[0].split(' ')[0]] += 1
    print(tag, k[0], k[1], '| F', ra, (ea[:1] or [''])[0][:90], '| M', rm, (em[:1] or [''])[0][:90])
print('SUMMARY', dict(cnt))
