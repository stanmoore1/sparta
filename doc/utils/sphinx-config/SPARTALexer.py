"""Pygments lexer for SPARTA input scripts.

Modelled on LAMMPSLexer.py from the LAMMPS documentation.  Used by
``.. code-block:: SPARTA`` in the manual.

The command names are not listed here.  conf.py reads them from the
source tree -- every CommandStyle() in src/*.h plus the commands
src/input.cpp handles itself -- and passes them to make_lexer(), so a new
command is highlighted as soon as it exists, and a removed one stops
being highlighted, without anyone remembering to edit this file.

Sphinx does not track this file: after changing it, run "make clean" (or
delete doc/doctrees) so every page is highlighted again, not just the ones
whose source changed.
"""

from pygments.lexer import RegexLexer, bygroups, include
from pygments.token import Comment, Keyword, Name, Number, Operator, String, Text

# Commands whose first argument is an ID.  The ID is highlighted as a
# name rather than as plain text.
ID_COMMANDS = (
    'fix', 'compute', 'dump', 'region', 'group', 'variable',
    'unfix', 'uncompute', 'undump', 'dump_modify',
)


def make_lexer(commands):
    """Return a lexer class that highlights the given command names."""
    missing = set(ID_COMMANDS) - set(commands)
    if missing:
        raise ValueError('ID_COMMANDS names commands the source does not '
                         'define: ' + ', '.join(sorted(missing)))
    plain = sorted(set(commands) - set(ID_COMMANDS))

    class SPARTALexer(RegexLexer):
        name = 'SPARTA'
        aliases = ['sparta']

        tokens = {
            # A command is the first word of a line.  The same names occur
            # as arguments ("collide vss species all.vss relax variable"),
            # and there they are plain text.
            'root': [
                (r'#.*?$', Comment.Single),
                (r'^([ \t]*)(' + '|'.join(ID_COMMANDS) + r')([ \t]+)',
                 bygroups(Text, Keyword, Text), 'identifier'),
                (r'^([ \t]*)(' + '|'.join(sorted(plain, key=len, reverse=True))
                 + r')\b', bygroups(Text, Keyword)),
                include('common'),
            ],
            'identifier': [
                (r'[\w/\[\]\*\.-]+', Name.Variable, '#pop'),
                (r'', Text, '#pop'),
            ],
            'common': [
                (r'#.*?$', Comment.Single),
                # variable references: v_name, c_ID, f_ID, ${name}, $(expr),
                # $x.  A trailing [i] or [*] is part of the reference, but
                # an operator after it is not, so the bracket is matched as
                # its own group rather than folded into the name.
                (r'\b[vcfsgp]_[\w.-]+(\[[\w*]*\])?', Name.Variable),
                (r'\$\{[^}]*\}', Name.Variable),
                (r'\$\([^)]*\)', Name.Variable),
                (r'\$\w', Name.Variable),
                (r'"[^"]*"', String.Double),
                (r"'[^']*'", String.Single),
                (r'[-+]?\d+\.\d*([eE][-+]?\d+)?\b', Number.Float),
                (r'[-+]?\.\d+([eE][-+]?\d+)?\b', Number.Float),
                (r'[-+]?\d+([eE][-+]?\d+)?\b', Number.Integer),
                (r'[=<>!&|^~+\-*/%]+', Operator),
                (r'[\w/\[\]\*\.-]+', Text),
                # a newline on its own, so the next line's indentation is
                # left for the line-start command rule in 'root'
                (r'\n', Text),
                (r'[ \t]+', Text),
                (r'.', Text),
            ],
        }

    return SPARTALexer
