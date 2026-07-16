from pygments.lexer import RegexLexer, words, include, default
from pygments.token import *

# SPARTA commands that take no leading ID argument: the internal commands
# implemented in input.cpp (see resources/sparta_internal_commands.txt) plus
# the command styles registered in style_command.h.  Commands with an ID or
# name as the first argument (fix, compute, dump, region, variable, group,
# mixture, ...) are handled by dedicated lexer states below.
SPARTA_COMMANDS = (# internal commands
                   "adapt_grid", "balance_grid", "boundary", "bound_modify",
                   "clear", "collide", "collide_modify", "compute_modify",
                   "create_box", "create_grid", "create_isurf",
                   "create_particles", "custom", "dimension", "dump_modify",
                   "echo", "fix_modify", "global", "include", "log",
                   "move_surf", "next", "package", "partition", "print",
                   "python", "quit", "react", "react_modify", "read_grid",
                   "read_isurf", "read_particles", "read_restart",
                   "read_surf", "remove_surf", "reset_timestep", "restart",
                   "run", "scale_particles", "seed", "shell", "species",
                   "species_modify", "stats", "stats_modify", "stats_style",
                   "surf_collide", "surf_modify", "surf_react", "timestep",
                   "units", "write_grid", "write_isurf", "write_restart",
                   "write_surf")

#fix ID style args
#compute ID style args
#dump ID style mix-ID N file args
#region ID style args keyword arg ...
#variable name style args ...
#group ID style args
#mixture ID species-ID ... keyword args ...
#uncompute compute-ID
#undump dump-ID
#unfix fix-ID

class SPARTALexer(RegexLexer):
    name = 'SPARTA'
    tokens = {
        'root': [
            (r'fix\s+', Keyword, 'id_cmd'),
            (r'compute\s+', Keyword, 'id_cmd'),
            (r'dump\s+', Keyword, 'id_cmd'),
            (r'region\s+', Keyword, 'id_cmd'),
            (r'variable\s+', Keyword, 'id_cmd'),
            (r'group\s+', Keyword, 'id_cmd'),
            (r'mixture\s+', Keyword, 'id_cmd'),
            (r'uncompute\s+', Keyword, 'id_cmd'),
            (r'unfix\s+', Keyword, 'id_cmd'),
            (r'undump\s+', Keyword, 'id_cmd'),
            (r'jump\s+', Keyword, 'jump_cmd'),
            (r'label\s+', Keyword, 'jump_cmd'),
            include('conditionals'),
            include('keywords'),
            (r'#.*?\n', Comment),
            (r'"', String, 'string'),
            (r'\'', String, 'single_quote_string'),
            (r'[0-9]+:[0-9]+(:[0-9]+)?', Number),
            (r'[0-9]+(\.[0-9]+)?([eE]\-?[0-9]+)?', Number),
            (r'\$?\(', Name.Variable, 'expression'),
            (r'\$\{', Name.Variable, 'variable'),
            (r'[\w_\.\[\]]+', Name),
            (r'\$[\w_]+', Name.Variable),
            (r'\s+', Whitespace),
            (r'[\+\-\*\^\|\/\!%&=<>]', Operator),
            (r'[\~\.\w_:,@\-\/\\0-9]+', Text),
        ],
        'conditionals' : [
            (words(('if','else','elif','then'), suffix=r'\b', prefix=r'\b'), Keyword)
        ]
        ,
        'keywords' : [
            (words(SPARTA_COMMANDS, suffix=r'\b', prefix=r'^\s*'), Keyword)
        ]
        ,
        'variable' : [
            (r'[^\}]+', Name.Variable),
            (r'\}', Name.Variable, '#pop'),
        ],
        'string' : [
            (r'[^"]+', String),
            (r'"', String, '#pop'),
        ],
        'single_quote_string' : [
            (r'[^\']+', String),
            (r'\'', String, '#pop'),
        ],
        'expression' : [
            (r'[^\(\)]+', Name.Variable),
            (r'\(', Name.Variable, 'expression'),
            (r'\)', Name.Variable, '#pop'),
        ],
        'id_cmd' : [
            (r'[\w_\-\.\[\]]+', Name.Variable.Identifier),
            default('#pop')
        ],
        'jump_cmd' : [
            (r'[\w_\-\.\[\]]+', Literal.String.Char),
            default('#pop')
        ]
    }
