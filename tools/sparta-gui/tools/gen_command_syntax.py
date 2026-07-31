#!/usr/bin/env python3
"""Generate the SPARTA-GUI command-syntax table from the SPARTA documentation.

Each SPARTA command has a doc page ``doc/src/<cmd>.rst`` whose ``Syntax``
section opens with a ``.. parsed-literal::`` block holding the template, e.g.::

    Syntax
    \"\"\"\"\"\"

    .. parsed-literal::

       create_box xlo xhi ylo yhi zlo zhi

followed by a bullet list describing each placeholder.

This script parses those templates into a compact table consumed by the
pre-run input validator (``src/inputcheck.cpp``).  For every *base* command
(a single token before ``command`` in the page's first title -- style pages
such as ``fix ave/grid command`` are skipped) it emits one line::

    <command> <min_required_args> <variadic 0|1>

where ``min_required_args`` counts the fixed leading placeholders up to the
first variadic/optional marker, and ``variadic`` is 1 when trailing
``args``/``keyword``/``values``/``...`` may follow (so no maximum is enforced).

The manual used to be txt2html, whose markup this converts back to before
parsing: reST emphasis ``*fnum*`` is what ``{fnum}`` was, bullets are what the
``:ulb,l`` list markers were, and ``:doc:`text <page>``` is what ``"text"_page``
was.  Normalising to that shape keeps one description-parsing implementation
rather than two.

Usage:  gen_command_syntax.py <sparta_doc_dir> <output_table>
"""
import os
import re
import sys

# tokens in a syntax template that end the fixed-argument prefix and mean
# "zero or more additional args may follow"
VARIADIC = {"args", "arg", "keyword", "keywords", "value", "values",
            "params", "arary", "..."}

# reST section underline: a run of one punctuation character
UNDERLINE_RE = re.compile(r"^([=\-`:'\"~^_*+#])\1*\s*$")

# ---------------------------------------------------------------- reST markup

_ROLE_RE = re.compile(r":[a-z:]+:`([^`<]*?)\s*(?:<[^`>]*>)?`")
_LITERAL_RE = re.compile(r"``([^`]*)``")
_EMPHASIS_RE = re.compile(r"\*([A-Za-z0-9_/][^*\n]*?)\*")


def unrest(line):
    """Return a reST body line in the markup the description parser expects.

    Emphasis becomes ``{braces}`` (what txt2html used), roles and inline
    literals collapse to their text, list bullets and backslash escapes go
    away.  Indentation is preserved: the nesting of a description list is what
    tells one placeholder's sub-values from the next placeholder.
    """
    # a list bullet, which needs whitespace after it -- "*fnum*" is emphasis
    line = re.sub(r"^(\s*)[*+-]\s+", r"\1", line)
    line = _ROLE_RE.sub(r"\1", line)
    line = _LITERAL_RE.sub(r"\1", line)
    line = _EMPHASIS_RE.sub(r"{\1}", line)
    line = re.sub(r"\\(.)", r"\1", line)
    return line.rstrip()


def is_directive(line):
    """True for a reST directive line ('.. parsed-literal::', '.. index:: x')."""
    return line.lstrip().startswith("..")


def field_names(line):
    """Return the fixed leading placeholder names of a template (before variadic)."""
    toks = line.split()
    if not toks:
        return []
    has_ellipsis = any(t == "..." or t.endswith("...") for t in toks)
    numbered = re.compile(r"^[A-Za-z_][\w-]*\d$")
    out = []
    for t in toks[1:]:
        if t == "..." or t.endswith("...") or t.lower() in VARIADIC:
            break
        out.append(t)
        if has_ellipsis and numbered.match(t):
            break
    return out


# words in a field's description that mark it as a numeric value, and words that
# mark it as a string/name/keyword.  A field is treated as "numeric" only when a
# numeric word is present and no string word is -- conservative on purpose so the
# type check never fires on a field whose kind is ambiguous.
_NUM_WORD = re.compile(
    r"\b(units\)|number|bounds?|seed|integer|coordinate|coord|ratio|count|size|"
    r"factor|fraction|temperature|density|distance|magnitude|angle|length|"
    r"timestep|frequency|probability|cutoff|weight|threshold)\b|#", re.IGNORECASE)
_STR_WORD = re.compile(
    r"\b(ID|name|style|file|filename|group|keyword|mode|string|word|expression)\b"
    r"|yes or no|=\s*\{", re.IGNORECASE)


def arg_numeric(template, body):
    """Classify each fixed positional field as numeric (True) or unknown (None).

    Only a field whose documentation clearly describes a number (and never a
    name/ID/style/keyword) is marked numeric; everything else stays None so the
    validator leaves it alone.
    """
    text = "\n".join(body)
    out = []
    for f in field_names(template):
        # find the "<field> = ..." (or combined "a,<field>,b = ...") description
        m = re.search(r"(?mi)^\s*[\w,]*\b" + re.escape(f) + r"\b[\w,]*\s*=(.*)$", text)
        desc = m.group(1) if m else ""
        if desc and not _STR_WORD.search(desc) and _NUM_WORD.search(desc):
            out.append(True)
        else:
            out.append(None)
    return out


def keyword_names(body):
    """Extract top-level keyword names from a 'keyword = {a} or {b} ...' list.

    The enumeration may wrap across several physical lines; a wrapped line ends
    with the word 'or' (e.g. read_surf lists its keywords over two lines), so we
    join following lines while the accumulated text still ends in 'or' before
    reading the ``{name}`` tokens out of the whole span.
    """
    names = []
    seen = set()
    i = 0
    while i < len(body):
        line = body[i]
        if "keyword" in line and "=" in line:
            acc = line
            while acc.strip().endswith("or") and i + 1 < len(body):
                i += 1
                acc += " " + body[i]
            for m in re.findall(r"\{([A-Za-z0-9_/]+)\}", acc):
                if m not in seen:
                    seen.add(m)
                    names.append(m)
        i += 1
    return names


KEYWORD_TOKENS = {"keyword", "keywords"}


def parse_template(line):
    """Return (min_required, variadic, keyword_led, keyword_start) from a template.

    keyword_led is True when the variadic tail is a keyword list (e.g.
    "global keyword values ..." or "compute_modify ID keyword value"), so a
    "one or more" note in the body can require at least one keyword.
    keyword_start is the number of fixed placeholders that precede the keyword
    token (0 for "global ...", 1 for "compute_modify ID ..."), i.e. the argument
    index at which the keyword list begins -- or None when not keyword_led.
    """
    toks = line.split()
    if not toks:
        return None
    # a "name1 name2 ..." series means one-or-more of that placeholder
    has_ellipsis = any(t == "..." or t.endswith("...") for t in toks)
    numbered = re.compile(r"^[A-Za-z_][\w-]*\d$")
    # toks[0] is the command name; count the fixed placeholders after it
    required = 0
    variadic = False
    keyword_led = False
    keyword_start = None
    for t in toks[1:]:
        if t == "..." or t.endswith("..."):
            variadic = True
            break
        if t.lower() in VARIADIC:
            variadic = True
            keyword_led = t.lower() in KEYWORD_TOKENS
            if keyword_led:
                keyword_start = required  # fixed placeholders before the keyword list
            break
        if has_ellipsis and numbered.match(t):
            # first element of a "one or more" series: count once, then stop
            required += 1
            variadic = True
            break
        required += 1
    return required, variadic, keyword_led, keyword_start


# phrases in the syntax body that indicate optional trailing arguments even when
# the template line itself shows a fixed argument list (e.g. write_surf)
BODY_VARIADIC = re.compile(
    r"zero or more|one or more|may be appended|optional (keyword|arg)", re.IGNORECASE)


def page_command(lines):
    """Return the base command a page documents, or None.

    The command is the first section title reading "<name> command".  A title
    naming a style ("fix ave/grid command") means a style page, which has no
    entry of its own -- the base command's own page carries the syntax.
    """
    for i in range(len(lines) - 1):
        title, under = lines[i], lines[i + 1]
        if not title.strip() or title[:1].isspace() or is_directive(title):
            continue
        if not UNDERLINE_RE.match(under) or len(under.rstrip()) < len(title.rstrip()):
            continue
        name = unrest(title).strip()
        if not name.endswith(" command"):
            continue
        name = name[: -len(" command")].strip()
        return None if (" " in name or "/" in name) else name
    return None


def syntax_section(lines):
    """Return the body lines of the page's "Syntax" section, or []."""
    for i in range(len(lines) - 1):
        if lines[i].strip() != "Syntax" or not UNDERLINE_RE.match(lines[i + 1]):
            continue
        out = []
        j = i + 2
        while j < len(lines):
            # the next section title ends the block
            if (j + 1 < len(lines) and lines[j].strip() and not lines[j][:1].isspace()
                    and UNDERLINE_RE.match(lines[j + 1])
                    and len(lines[j + 1].rstrip()) >= len(lines[j].rstrip())):
                break
            out.append(lines[j])
            j += 1
        return out
    return []


def extract(path):
    """Return the syntax record for a base-command doc page, or None."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None

    command = page_command(lines)
    if command is None:
        return None

    templates = []
    body = []
    for raw in syntax_section(lines):
        if is_directive(raw):
            continue
        text = unrest(raw)
        # A template is a literal-block line that starts with the command, and
        # they all come first: once the description list has begun, a line that
        # happens to start with the command is prose about it, not another form
        # of it (variable.rst describes "variable references = v_name" among the
        # things an equal-style expression may contain).
        if not body and raw[:1].isspace() and text.split()[0:1] == [command]:
            templates.append(text.strip())
        elif text.strip() or body:
            body.append(text)

    if not templates:
        return None
    parsed = [p for p in (parse_template(t) for t in templates) if p is not None]
    if not parsed:
        return None
    # A command may document alternative forms ("restart 0" as well as
    # "restart N root keyword value ..."). The shortest of them is what the
    # validator may demand; the fullest is what to show in a call tip, and is
    # the one the txt2html sources marked with ":pre".
    template = templates[-1]
    required = min(p[0] for p in parsed)
    variadic = any(p[1] for p in parsed)
    _, _, keyword_led, keyword_start = parsed[-1]

    bodytext = "\n".join(body)
    # the body may reveal optional trailing keywords the template line omitted
    if not variadic and BODY_VARIADIC.search(bodytext):
        variadic = True
    # a mandatory keyword list ("one or more keyword/value pairs", not "zero or
    # more") requires at least one keyword -- e.g. bare "global" is invalid
    if keyword_led and re.search(r"one or more", bodytext, re.IGNORECASE) \
            and not re.search(r"zero or more", bodytext, re.IGNORECASE):
        required += 1
    keywords = keyword_names(body)
    numeric = arg_numeric(template, body)
    rec = {
        "command": command,
        "minArgs": required,
        "variadic": variadic,
        "syntax": template,
        "args": field_names(template),
        "keywords": keywords,
        # 1-based indices of positional args documented as numeric values
        "numericArgs": [i + 1 for i, n in enumerate(numeric) if n],
    }
    # only advertise a keyword list the validator can check when we actually
    # captured the keyword names, so it never flags a valid keyword we missed
    if keyword_led and keyword_start is not None and keywords:
        rec["keywordStart"] = keyword_start
    return rec


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: gen_command_syntax.py <sparta_doc_dir> <output_table>")
    docdir, out = sys.argv[1], sys.argv[2]
    # accept either the doc directory or the reST source directory inside it
    if os.path.isdir(os.path.join(docdir, "src")):
        docdir = os.path.join(docdir, "src")
    recs = []
    for fname in sorted(os.listdir(docdir)):
        if not fname.endswith(".rst"):
            continue
        rec = extract(os.path.join(docdir, fname))
        if rec:
            recs.append(rec)
    if not recs:
        sys.exit("%s: no command pages found in %s" % (sys.argv[0], docdir))
    recs.sort(key=lambda r: r["command"])

    # compact table consumed by the validator (src/inputcheck.cpp)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# SPARTA command-syntax table -- generated by "
                 "tools/gen_command_syntax.py from the SPARTA docs.\n")
        fh.write("# columns: command  min_required_args  variadic(0|1)\n")
        for r in recs:
            fh.write("%s %d %d\n" % (r["command"], r["minArgs"], 1 if r["variadic"] else 0))

    # richer JSON (syntax template, field names, keywords) for the GUI's
    # syntax-aware autocomplete and linter error help
    import json
    jpath = os.path.splitext(out)[0] + ".json"
    catalog = {}
    for r in recs:
        entry = {"syntax": r["syntax"], "args": r["args"], "keywords": r["keywords"]}
        if "keywordStart" in r:
            entry["keywordStart"] = r["keywordStart"]
        if r.get("numericArgs"):
            entry["numericArgs"] = r["numericArgs"]
        catalog[r["command"]] = entry
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=0, sort_keys=True)
        fh.write("\n")
    sys.stderr.write("wrote %d commands to %s and %s\n" % (len(recs), out, jpath))


if __name__ == "__main__":
    main()
