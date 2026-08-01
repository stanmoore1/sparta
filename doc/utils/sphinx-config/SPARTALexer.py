"""Pygments lexer for SPARTA input scripts.

Modelled on LAMMPSLexer.py from the LAMMPS documentation.  Used by the
``:sparta:`` inline role defined in ``rst_prolog`` and by
``.. code-block:: SPARTA``.
"""

from pygments.lexer import RegexLexer, words, include
from pygments.token import Comment, Keyword, Name, Number, Operator, String, Text

# Commands that take an ID as their first argument are handled separately
# below so the ID is highlighted as a name rather than as a keyword.
SPARTA_COMMANDS = (
    "adapt_grid", "balance_grid", "bound_modify", "boundary", "clear",
    "collide", "collide_modify", "create_box", "create_grid", "create_isurf",
    "create_particles", "custom", "dimension", "echo", "global", "if",
    "include", "jump", "label", "log", "mixture", "move_surf", "next",
    "package", "partition", "print", "python", "quit", "react",
    "react_modify", "read_grid", "read_isurf", "read_particles",
    "read_restart", "read_surf", "remove_surf", "reset_timestep", "restart",
    "run", "scale_particles", "seed", "shell", "species", "species_modify",
    "stats", "stats_modify", "stats_style", "suffix", "surf_collide",
    "surf_modify", "surf_react", "timestep", "units", "write_grid",
    "write_isurf", "write_restart", "write_surf",
)


class SPARTALexer(RegexLexer):
    name = 'SPARTA'
    aliases = ['sparta']

    tokens = {
        'root': [
            (r'#.*?\n', Comment.Single),
            # ID-taking commands: highlight the command, then its ID
            (r'\b(fix|compute|dump|region|group|variable)(\s+)',
             lambda lexer, match: [
                 (match.start(1), Keyword, match.group(1)),
                 (match.start(2), Text, match.group(2)),
             ], 'identifier'),
            (r'\b(unfix|uncompute|undump|dump_modify|fix_modify|compute_modify)(\s+)',
             lambda lexer, match: [
                 (match.start(1), Keyword, match.group(1)),
                 (match.start(2), Text, match.group(2)),
             ], 'identifier'),
            (words(SPARTA_COMMANDS, prefix=r'\b', suffix=r'\b'), Keyword),
            include('common'),
        ],
        'identifier': [
            (r'\s+', Text),
            (r'[\w/\[\]\*\.-]+', Name.Variable, '#pop'),
            (r'', Text, '#pop'),
        ],
        'common': [
            (r'#.*?\n', Comment.Single),
            # variable references: v_name, c_ID, f_ID, ${name}, $(expr), $x
            # a trailing [i] or [*] is part of the reference, but an operator
            # after it is not -- so the bracket group is matched explicitly
            # rather than folded into the name character class
            (r'[vcfsgp]_[\w.-]+(\[[\w*]*\])?', Name.Variable),
            (r'\$\{[^}]*\}', Name.Variable),
            (r'\$\([^)]*\)', Name.Variable),
            (r'\$\w', Name.Variable),
            (r'"[^"]*"', String.Double),
            (r"'[^']*'", String.Single),
            (r'[-+]?\d+\.\d*([eE][-+]?\d+)?', Number.Float),
            (r'[-+]?\.\d+([eE][-+]?\d+)?', Number.Float),
            (r'[-+]?\d+([eE][-+]?\d+)?', Number.Integer),
            (r'[=<>!&|^~+\-*/%]+', Operator),
            (r'[\w/\[\]\*\.-]+', Text),
            (r'\s+', Text),
            (r'.', Text),
        ],
    }
