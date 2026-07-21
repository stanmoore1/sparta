#!/usr/bin/env python3
"""Generate the SPARTA-GUI command-syntax table from the SPARTA documentation.

Each SPARTA command has a doc page ``doc/<cmd>.txt`` whose ``[Syntax:]`` block
contains a ``:pre`` template line, e.g.::

    create_box xlo xhi ylo yhi zlo zhi :pre
    run N keyword values ... :pre
    fix ID style args :pre

This script parses those templates into a compact table consumed by the
pre-run input validator (``src/inputcheck.cpp``).  For every *base* command
(a single token before ``command :h3`` -- style pages such as
``fix ave/grid command`` are skipped) it emits one line::

    <command> <min_required_args> <variadic 0|1>

where ``min_required_args`` counts the fixed leading placeholders up to the
first variadic/optional marker, and ``variadic`` is 1 when trailing
``args``/``keyword``/``values``/``...`` may follow (so no maximum is enforced).

Usage:  gen_command_syntax.py <sparta_doc_dir> <output_table>
"""
import os
import re
import sys

# tokens in a syntax template that end the fixed-argument prefix and mean
# "zero or more additional args may follow"
VARIADIC = {"args", "arg", "keyword", "keywords", "value", "values",
            "params", "arary", "..."}

H3_RE = re.compile(r"^(.*\S)\s+command\s+:h3\s*$")


def clean_template(line):
    """Return the syntax template text with trailing ':pre'-style markers removed."""
    return " ".join(t for t in line.split() if not t.startswith(":"))


def field_names(line):
    """Return the fixed leading placeholder names of a template (before variadic)."""
    toks = [t for t in line.split() if not t.startswith(":")]
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


def _strip_markers(s):
    """Drop a trailing txt2html list marker (':l', ':ulb,l', ...) for text tests."""
    return re.sub(r"\s*:[a-z,]+\s*$", "", s).strip()


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
            while _strip_markers(acc).endswith("or") and i + 1 < len(body):
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
    # drop the trailing ':pre' (and any ':ulb,l'-style tail markers)
    toks = [t for t in toks if not t.startswith(":")]
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


def extract(path):
    """Return (command, min_required, variadic) for a base-command doc page."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None

    command = None
    in_syntax = False
    template = None
    body = []
    for raw in lines:
        m = H3_RE.match(raw)
        if m and command is None:
            name = m.group(1).strip()
            # base command only: a single token (style pages have "fix ave/grid")
            if " " in name or "/" in name:
                return None
            command = name
            continue
        if raw.strip() == "[Syntax:]":
            in_syntax = True
            continue
        if in_syntax:
            # the next section header ends the syntax block
            if raw.startswith("[") and raw.endswith("]"):
                break
            if template is None and ":pre" in raw and raw.split()[0:1] == [command]:
                template = raw
            else:
                body.append(raw)

    if template is None:
        return None
    parsed = parse_template(template)
    if parsed is None:
        return None
    required, variadic, keyword_led, keyword_start = parsed
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
        "syntax": clean_template(template),
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
    recs = []
    for fname in sorted(os.listdir(docdir)):
        if not fname.endswith(".txt"):
            continue
        rec = extract(os.path.join(docdir, fname))
        if rec:
            recs.append(rec)
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
