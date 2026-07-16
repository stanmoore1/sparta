#!/bin/bash
# this updates the help index table

if [ $# -lt 1 ]
then
    echo "usage: $0 <sparta-doc-dir>"
    exit 1
fi

cd "$(dirname $0)" || exit 1

# SPARTA doc pages are .txt sources with one or more "<command> command :h3"
# or "<command> <style> command :h3" headings per file. Each <name>.txt file
# is translated to a <name>.html page. Lines with quotes are doc links, not
# command headings, and must be skipped.
mv help_index.table help_index.oldtable
grep ' command :h3' "$1"/*.txt | grep -v '"' | sort \
    | sed -e 's/^.*\/\([^/:]\+\)\.txt:/\1.html /' \
          -e 's/ \+command :h3.*$//' > help_index.table
cmp help_index.table help_index.oldtable > /dev/null || touch spartagui.qrc
