#!/bin/bash
# Regenerate the flat UI icons in resources/icons/ from the Lucide icon set.
#
# Usage:
#   1. Download the Lucide static SVGs, e.g.
#        curl -sSL https://registry.npmjs.org/lucide-static/-/lucide-static-1.25.0.tgz \
#          | tar xz -C /tmp        # -> /tmp/package/icons/*.svg
#   2. Run:  ./regenerate-icons.sh /tmp/package/icons
#
# icon_map.txt has three columns:  <app-icon-name> <lucide-glyph> <hex-color>
# Each Lucide SVG's currentColor is replaced with the semantic category color
# (files=blue, run=green, danger=red, visualization=teal, settings=purple,
# help=blue, warnings=amber, edit/nav=neutral slate) -- chosen at a mid
# luminance so the icons read on both the light and dark application palettes.
# Output keeps the original file name so no C++ call site changes; a handful of
# names are also rasterized to PNG where a call site references a .png path.
set -e
LUC="${1:?path to lucide icons dir (…/package/icons)}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DST="$HERE/../../resources/icons"
TMP="$(mktemp --suffix=.svg)"
PNG_ONLY="show-box show-axes emblem-photos"
PNG_ALSO="rotate-up rotate-down rotate-left rotate-right"
declare -A DIM=( [show-box]=256 [show-axes]=256 [emblem-photos]=48 \
                 [rotate-up]=256 [rotate-down]=256 [rotate-left]=256 [rotate-right]=256 )
while read -r app luc color; do
  [ -z "$app" ] && continue
  sed 's/currentColor/'"$color"'/g' "$LUC/$luc.svg" > "$TMP"
  po=0; for p in $PNG_ONLY; do [ "$app" = "$p" ] && po=1; done
  [ $po -eq 0 ] && cp "$TMP" "$DST/$app.svg"
  np=0; for p in $PNG_ONLY $PNG_ALSO; do [ "$app" = "$p" ] && np=1; done
  [ $np -eq 1 ] && rsvg-convert -w "${DIM[$app]}" -h "${DIM[$app]}" -o "$DST/$app.png" "$TMP"
done < "$HERE/icon_map.txt"
rm -f "$TMP"
echo "icons regenerated from $LUC"
