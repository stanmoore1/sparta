#!/bin/bash
# SPARTA-GUI mechanical rename tool
#
# Applies the LAMMPS -> SPARTA translation rules in rename.map to an
# upstream LAMMPS-GUI source tree or to a patch, so that upstream
# changes can be ported to SPARTA-GUI mechanically.  See UPSTREAM.md
# for the complete porting recipe.
#
# Usage:
#   rename.sh <upstream-tree> <output-dir>
#       Copy the upstream tree to <output-dir>, renaming files,
#       directories, and file contents according to rename.map.
#       Binary files are copied with renamed names but unchanged
#       contents.
#
#   rename.sh --patch < upstream.patch > renamed.patch
#       Filter a patch (e.g. the diff between two upstream releases)
#       through the same rules, including the file names in the
#       patch headers, so it can be applied to the SPARTA-GUI tree.

set -e

MAPFILE="$(cd "$(dirname "$0")" && pwd)/rename.map"

if [ ! -f "$MAPFILE" ]; then
    echo "rename.sh: cannot find rename.map next to this script" >&2
    exit 1
fi

# build a sed script from rename.map

build_sed_script() {
    local script=""
    while IFS=$'\t' read -r mode old new; do
        case "$mode" in
            ''|'#'*) continue ;;
        esac
        # escape regex metacharacters in OLD and replacement specials in NEW
        local old_esc new_esc
        old_esc=$(printf '%s' "$old" | sed -e 's/[][\.*^$(){}?+|/]/\\&/g')
        new_esc=$(printf '%s' "$new" | sed -e 's/[\/&]/\\&/g')
        if [ "$mode" = "w" ]; then
            script="${script}s/\\b${old_esc}\\b/${new_esc}/g"$'\n'
        elif [ "$mode" = "s" ]; then
            script="${script}s/${old_esc}/${new_esc}/g"$'\n'
        else
            echo "rename.sh: unknown mode '$mode' in rename.map" >&2
            exit 1
        fi
    done < "$MAPFILE"
    printf '%s' "$script"
}

SEDSCRIPT=$(build_sed_script)

rename_string() {
    printf '%s' "$1" | sed -E "$SEDSCRIPT"
}

# ----------------------------------------------------------------------
# patch mode
# ----------------------------------------------------------------------

if [ "$1" = "--patch" ]; then
    exec sed -E "$SEDSCRIPT"
fi

# ----------------------------------------------------------------------
# tree mode
# ----------------------------------------------------------------------

SRC="$1"
DST="$2"

if [ -z "$SRC" ] || [ -z "$DST" ]; then
    echo "usage: rename.sh <upstream-tree> <output-dir>" >&2
    echo "       rename.sh --patch < upstream.patch > renamed.patch" >&2
    exit 1
fi

if [ ! -d "$SRC" ]; then
    echo "rename.sh: upstream tree '$SRC' does not exist" >&2
    exit 1
fi

mkdir -p "$DST"

is_text_file() {
    case "$1" in
        *.png|*.jpg|*.jpeg|*.gif|*.ico|*.icns|*.xcf|*.pdf|*.ttf|*.otf|*.zip|*.gz)
            return 1 ;;
    esac
    # fall back to file(1) for anything else
    file -b --mime "$1" 2>/dev/null | grep -q "charset=binary" && return 1
    return 0
}

(cd "$SRC" && find . -not -path './.git*' -print) | while read -r path; do
    rel="${path#./}"
    [ "$rel" = "." ] && continue
    newrel=$(rename_string "$rel")
    if [ -d "$SRC/$rel" ]; then
        mkdir -p "$DST/$newrel"
    else
        mkdir -p "$DST/$(dirname "$newrel")"
        if is_text_file "$SRC/$rel"; then
            sed -E "$SEDSCRIPT" "$SRC/$rel" > "$DST/$newrel"
        else
            cp "$SRC/$rel" "$DST/$newrel"
        fi
        # preserve the executable bit
        [ -x "$SRC/$rel" ] && chmod +x "$DST/$newrel"
    fi
done

echo "rename.sh: renamed tree written to $DST" >&2
