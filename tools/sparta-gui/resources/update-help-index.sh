#!/bin/bash
# this updates the help index table

if [ $# -lt 1 ]
then
    echo "usage: $0 <sparta-doc-dir>"
    exit 1
fi

SRCDIR="$(cd "$1" && pwd)" || exit 1
# accept either the doc directory or the reST source directory inside it
[ -d "${SRCDIR}/src" ] && SRCDIR="${SRCDIR}/src"

if ! compgen -G "${SRCDIR}/*.rst" > /dev/null
then
    echo "$0: no .rst files found in ${SRCDIR}"
    exit 1
fi

cd "$(dirname $0)" || exit 1

# SPARTA doc pages are reST sources with one or more section titles of the
# form "<command> command" or "<command> <style> command", each followed by
# an "=====" underline.  Pages document accelerator variants as extra titles
# ("compute count command" and "compute count/kk command" both live in
# compute_count.rst), which is why titles rather than the single ".. index::"
# directive at the top of each page are the thing to scan.  Each <name>.rst
# file is rendered to a <name>.html page.
#
# Underscores are escaped in reST titles ("adapt\_grid command"); undo that.
#
# A few pages carry a stub title for a command documented elsewhere -- dump.rst
# titles "dump image command" and then points at dump_image.rst -- so the same
# command can be claimed by two pages.  When that happens, prefer the page whose
# file name is the command name ("dump image" -> dump_image.html); that is the
# page with the actual documentation on it.
mv help_index.table help_index.oldtable
awk '
    FNR == 1 { prev = ""; page = FILENAME
               sub(/^.*\//, "", page); sub(/\.rst$/, ".html", page) }
    # an underline at least as long as the title line above it
    /^=+$/ && prev ~ / command$/ && length($0) >= length(prev) {
        title = prev
        sub(/ +command$/, "", title)
        gsub(/\\_/, "_", title)
        print title "\t" page
    }
    { prev = $0 }
' "${SRCDIR}"/*.rst \
    | awk -F'\t' '
        { title = $1; page = $2
          canon = title; gsub(/[ \/]/, "_", canon); canon = canon ".html"
          if (!(title in best) || page == canon) best[title] = page }
        END { for (t in best) print best[t], t }
      ' \
    | sort > help_index.table
cmp help_index.table help_index.oldtable > /dev/null || touch spartagui.qrc
