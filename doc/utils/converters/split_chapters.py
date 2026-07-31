#!/usr/bin/env python3
"""Split the Section_* chapters into per-topic pages, LAMMPS style.

Splits only at headings carrying a numbered-subsection label (howto_N,
start_N, ...), which are exactly the subsections the old Manual.txt
enumerated. Unlabelled sub-headings stay inside their parent page.

The chapter file keeps its filename and becomes a toctree landing page, so
every existing Section_*.html URL still resolves.
"""

import pathlib
import re
import sys

# chapter -> (label prefix, page-name prefix, optional caption groups)
CHAPTERS = {
    'Section_intro':      ('intro',  'Intro'),
    'Section_start':      ('start',  'Start'),
    'Section_commands':   ('cmd',    'Commands'),
    'Section_accelerate': ('acc',    'Speed'),
    'Section_howto':      ('howto',  'Howto'),
    'Section_modify':     ('mod',    'Modify'),
    'Section_python':     ('py',     'Python'),
    'Section_errors':     ('err',    'Errors'),
    'Section_history':    ('hist',   'History'),
}

STOPWORDS = {'a', 'an', 'the', 'of', 'in', 'to', 'for', 'and', 'with',
             'from', 'on', 'using', 'sparta', 'is', 'that', 'or'}

# hand-picked names where the automatic slug reads badly
OVERRIDES = {
    'howto_1':  'Howto_2d',
    'howto_2':  'Howto_axisymmetric',
    'howto_3':  'Howto_multiple',
    'howto_4':  'Howto_output',
    'howto_5':  'Howto_viz',
    'howto_6':  'Howto_library',
    'howto_7':  'Howto_couple',
    'howto_8':  'Howto_grid',
    'howto_9':  'Howto_surfaces',
    'howto_10': 'Howto_restart',
    'howto_11': 'Howto_ambipolar',
    'howto_12': 'Howto_vibrational',
    'howto_13': 'Howto_surf_elements',
    'howto_14': 'Howto_ablation',
    'howto_15': 'Howto_transparent',
    'howto_16': 'Howto_paraview',
    'howto_17': 'Howto_custom',
    'howto_18': 'Howto_variable_timestep',
    'howto_19': 'Howto_particles',
    'start_1':  'Start_distribution',
    'start_2':  'Start_build',
    'start_3':  'Start_packages',
    'start_4':  'Start_library',
    'start_5':  'Start_testing',
    'start_6':  'Start_run',
    'start_7':  'Start_cmdline',
    'start_8':  'Start_screen',
    'intro_1':  'Intro_overview',
    'intro_2':  'Intro_features',
    'intro_3':  'Intro_grids_surfaces',
    'intro_4':  'Intro_opensource',
    'intro_5':  'Intro_citing',
    'cmd_1':    'Commands_input',
    'cmd_2':    'Commands_parse',
    'cmd_3':    'Commands_structure',
    'cmd_4':    'Commands_category',
    'cmd_5':    'Commands_all',
    'acc_1':    'Speed_measure',
    'acc_2':    'Speed_packages',
    'acc_3':    'Speed_kokkos',
    'py_1':     'Python_shlib',
    'py_2':     'Python_parallel',
    'py_3':     'Python_mpi',
    'py_4':     'Python_test',
    'py_5':     'Python_run',
    'py_6':     'Python_examples',
    'py_7':     'Python_call',
    'err_1':    'Errors_common',
    'err_2':    'Errors_bugs',
    'err_3':    'Errors_messages',
    'hist_1':   'History_future',
    'hist_2':   'History_past',
    'mod_5':    'Modify_surf_collide',
    'mod_8':    'Modify_command',
}

LABEL_RE = re.compile(r'^\.\. _([A-Za-z0-9_]+):\s*$')


def slug(prefix, title):
    words = [w for w in re.findall(r'[A-Za-z0-9]+', title.lower())
             if w not in STOPWORDS]
    return '%s_%s' % (prefix, '_'.join(words[:3]) or 'section')


def find_splits(lines, label_prefix):
    """Indices of headings that start a new topic page."""
    label_re = re.compile(r'^%s_\d+$' % re.escape(label_prefix))
    splits = []
    for i, ln in enumerate(lines):
        if i + 1 >= len(lines):
            continue
        if not ln.strip() or not re.fullmatch(r'-{3,}', lines[i + 1]):
            continue
        if len(lines[i + 1]) < len(ln.rstrip()):
            continue
        # a heading can be preceded by several stacked labels
        labels = []
        for j in range(max(0, i - 6), i):
            m = LABEL_RE.match(lines[j])
            if m:
                labels.append((j, m.group(1)))
        match = next((lab for _, lab in labels if label_re.match(lab)), None)
        if match:
            splits.append((i, match, ln.strip(), labels[0][0]))
    return splits


def split_chapter(path, label_prefix, page_prefix, dry_run):
    lines = path.read_text().split('\n')
    splits = find_splits(lines, label_prefix)
    if not splits:
        print("  %s: no numbered subsections, left as-is" % path.name)
        return []

    # each block runs from its label line to just before the next label line
    starts = []
    for idx, (i, label, title, label_line) in enumerate(splits):
        starts.append((label_line, i, label, title))

    created = []
    for n, (label_line, head_i, label, title) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        block = lines[label_line:end]

        # promote the topic heading from '-' to '=' so it is the page title
        for k, ln in enumerate(block):
            if re.fullmatch(r'-{3,}', ln) and k > 0 and block[k - 1].strip() == title:
                block[k] = '=' * len(title)
                break

        # a chapter's trailing '----------' rules belong to the chapter, not
        # the last topic page
        while block and block[-1].strip() in ('', '----------'):
            block.pop()

        name = OVERRIDES.get(label) or slug(page_prefix, title)
        created.append((name, title))
        if not dry_run:
            out = path.parent / (name + '.rst')
            out.write_text('\n'.join(block).rstrip() + '\n')

    # landing page: everything before the first topic, plus a toctree
    head = lines[:starts[0][0]]
    while head and head[-1].strip() in ('', '----------'):
        head.pop()
    toc = ['', '', '.. toctree::', '   :maxdepth: 1', '']
    toc += ['   %s' % name for name, _ in created]
    toc += ['']
    if not dry_run:
        path.write_text('\n'.join(head + toc).rstrip() + '\n')

    print("  %-20s -> %2d topic pages" % (path.name, len(created)))
    return created


def main():
    dry_run = '--dry-run' in sys.argv
    src = pathlib.Path('src')
    total = 0
    for chapter, (label_prefix, page_prefix) in sorted(CHAPTERS.items()):
        p = src / (chapter + '.rst')
        if not p.exists():
            print("  missing", p)
            continue
        created = split_chapter(p, label_prefix, page_prefix, dry_run)
        total += len(created)
        for name, title in created:
            print("       %-26s %s" % (name, title[:50]))
    print("total topic pages: %d" % total)


if __name__ == '__main__':
    main()
