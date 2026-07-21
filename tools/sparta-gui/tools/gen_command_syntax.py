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


def keyword_names(body):
    """Extract top-level keyword names from '{a} or {b} ...' enumerations."""
    names = []
    seen = set()
    for line in body:
        # lines like: "keyword = {upto} or {start} or {stop} ... :l"
        if "keyword" not in line or "=" not in line:
            continue
        for m in re.findall(r"\{([A-Za-z0-9_/]+)\}", line):
            if m not in seen:
                seen.add(m)
                names.append(m)
    return names


def parse_template(line):
    """Return (min_required, variadic) from a syntax ':pre' template line."""
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
    for t in toks[1:]:
        if t == "..." or t.endswith("..."):
            variadic = True
            break
        if t.lower() in VARIADIC:
            variadic = True
            break
        if has_ellipsis and numbered.match(t):
            # first element of a "one or more" series: count once, then stop
            required += 1
            variadic = True
            break
        required += 1
    return required, variadic


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
    required, variadic = parsed
    # the body may reveal optional trailing keywords the template line omitted
    if not variadic and BODY_VARIADIC.search("\n".join(body)):
        variadic = True
    return {
        "command": command,
        "minArgs": required,
        "variadic": variadic,
        "syntax": clean_template(template),
        "args": field_names(template),
        "keywords": keyword_names(body),
    }


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
    catalog = {r["command"]: {"syntax": r["syntax"], "args": r["args"],
                              "keywords": r["keywords"]} for r in recs}
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=0, sort_keys=True)
        fh.write("\n")
    sys.stderr.write("wrote %d commands to %s and %s\n" % (len(recs), out, jpath))


if __name__ == "__main__":
    main()
