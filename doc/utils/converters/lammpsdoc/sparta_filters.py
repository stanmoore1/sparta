# SPARTA Documentation Utilities
#
# Based on lammps_filters.py from the LAMMPS Documentation Utilities
# Copyright (C) 2015 Richard Berger
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import re


def indent(content):
    indented = ""
    for line in content.splitlines():
        indented += "   %s\n" % line
    return indented


def detect_and_format_notes(paragraph):
    note_pattern = re.compile(r"(?P<type>(IMPORTANT )?NOTE):\s+(?P<content>.+)",
                              re.MULTILINE | re.DOTALL)

    if note_pattern.match(paragraph):
        m = note_pattern.match(paragraph)
        content = indent(m.group('content').strip())

        if m.group('type') == 'IMPORTANT NOTE':
            paragraph = '.. warning::\n\n' + content + '\n'
        else:
            paragraph = '.. note::\n\n' + content + '\n'
    return paragraph


def detect_and_add_command_to_index(content):
    command_pattern = re.compile(r"^(?P<command>.+) command\s*\n")
    m = command_pattern.match(content)

    if m:
        return ".. index:: %s\n\n" % m.group('command') + content

    return content


def filter_file_header_until_first_horizontal_line(content):
    """Drop the breadcrumb preamble every SPARTA doc page opens with.

    Each doc/*.txt begins with the same 7 lines -- a centered
    "SPARTA WWW Site - SPARTA Documentation - SPARTA Commands" line, the
    three link aliases it uses, and a :line rule. The aliases are re-emitted
    as reST hyperlink targets so any in-body use of them still resolves.
    """
    hr = '----------\n\n'
    first_hr = content.find(hr)

    common_links = "\n.. _sws: https://sparta.github.io\n" \
                   ".. _sd: Manual.html\n" \
                   ".. _sc: Section_commands.html\n"

    if first_hr >= 0:
        return content[first_hr + len(hr):].lstrip() + common_links
    return content


def promote_doc_keywords(content):
    """Turn txt2html's bolded section labels into real reST section headers.

    SPARTA command pages use a fixed vocabulary: Syntax, Examples,
    Description, Output info, Restrictions, Related commands, Default.
    """
    keywords = ['Syntax',
                'Examples',
                'Description',
                'Output info',
                'Restrictions',
                'Related commands',
                'Default']

    for keyword in keywords:
        underline = '"' * len(keyword)
        content = content.replace('**%s:**\n' % keyword,
                                  '%s\n%s\n' % (keyword, underline))

    return content


def filter_multiple_horizontal_rules(content):
    return re.sub(r"----------[\s\n]+----------", '', content)


def flatten_nested_inline_markup(content):
    """Collapse bold-wrapping-italic to plain bold.

    txt2html allows [{text}], which nests <B> and <I>. reST has no nested
    inline markup, so the converter emits ``**\\ *text*\\ **``, which docutils
    rejects with "Inline strong start-string without end-string". Keeping the
    bold and dropping the emphasis is the conventional workaround.
    """
    return re.sub(r'\*\*\\ \*([^*]+)\*\\ \*\*', r'**\1**', content)


def blank_line_after_directive_options(content):
    """Ensure a directive's option block is followed by a blank line.

    ``:c,image(f,l)`` emits ".. image:: f" plus ":target:"/":align:" option
    lines.  When the next paragraph follows immediately, docutils reports
    "Explicit markup ends without a blank line; unexpected unindent" and
    swallows the paragraph into the directive.
    """
    return re.sub(r'^(   :(?:align|target|width|height|scale|alt): .*\n)(?=\S)',
                  r'\1\n', content, flags=re.M)


def escape_backticks_in_literal_blocks(content):
    """Escape stray backticks inside ``.. parsed-literal::`` blocks.

    Unlike a plain literal block, parsed-literal interprets inline markup, so
    an unpaired backtick -- common in captured linker output such as
    `.rodata' -- opens an interpreted-text span that never closes.
    """
    out = []
    in_block = False
    for line in content.split('\n'):
        if line.strip().startswith('.. parsed-literal::'):
            in_block = True
            out.append(line)
            continue
        if in_block:
            # the block ends at the first non-blank, non-indented line
            if line.strip() and not line.startswith('   '):
                in_block = False
            elif line.count('`') % 2 == 1:
                line = line.replace('`', '\\`')
        out.append(line)
    return '\n'.join(out)


def merge_preformatted_sections(content):
    mergable_section_pattern = re.compile(r"\.\. parsed-literal::\n"
                                          r"\n"
                                          r"(?P<listingA>((   [^\n]+\n)|(^\n))+)\n\s*"
                                          r"^\.\. parsed-literal::\n"
                                          r"\n"
                                          r"(?P<listingB>((   [^\n]+\n)|(^\n))+)\n",
                                          re.MULTILINE | re.DOTALL)

    m = mergable_section_pattern.search(content)

    while m:
        content = mergable_section_pattern.sub(r".. parsed-literal::\n"
                                               r"\n"
                                               r"\g<listingA>"
                                               r"\g<listingB>"
                                               r"\n", content)
        m = mergable_section_pattern.search(content)

    return content
