"""Pygments lexer for SPARTA input scripts.

Modelled on LAMMPSLexer.py from the LAMMPS documentation.  Used by
``.. code-block:: SPARTA`` in the manual.

The command names are not listed here.  conf.py reads them from the
source tree -- every CommandStyle() in src/*.h plus the commands
src/input.cpp handles itself -- and passes them to make_lexer(), so a new
command is highlighted as soon as it exists, and a removed one stops
being highlighted, without anyone remembering to edit this file.
"""

from pygments.lexer import RegexLexer, words, include
from pygments.token import Comment, Keyword, Name, Number, Operator, String, Text

# Commands whose first argument is an ID.  The ID is highlighted as a
# name rather than as plain text.
ID_COMMANDS = (
    'fix', 'compute', 'dump', 'region', 'group', 'variable',
    'unfix', 'uncompute', 'undump', 'dump_modify',
)


def _command_then_id(lexer, match):
    yield match.start(1), Keyword, match.group(1)
    yield match.start(2), Text, match.group(2)


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
            'root': [
                (r'#.*?$', Comment.Single),
                (r'\b(' + '|'.join(ID_COMMANDS) + r')(\s+)',
                 _command_then_id, 'identifier'),
                (words(plain, prefix=r'\b', suffix=r'\b'), Keyword),
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
                (r'\s+', Text),
                (r'.', Text),
            ],
        }

    return SPARTALexer
