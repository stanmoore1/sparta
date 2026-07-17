#!/bin/bash
# Regenerate the SPARTA-GUI logo and all derived branding assets.
#
# Requires ImageMagick (convert), librsvg (rsvg-convert), and libicns
# (png2icns).  Run from anywhere:  ./generate-logo.sh
#
# Produces, in this directory unless noted:
#   sparta-logo.svg                 vector master
#   sparta-icon-128x128.png         SPARTA library icon
#   sparta-gui-icon-128x128.png     application/window icon
#   sparta-gui-banner.png           editor splash banner
#   sparta-gui-banner.bmp           Windows installer banner
#   sparta-plugin.png               plugin illustration
#   ../../resources/sparta-gui.ico  Windows app icon        (rel. to repo)
#   ../../resources/spafile.ico     Windows input-file icon
#   ../../packaging/sparta-gui.icns macOS bundle icon
#   ../../packaging/SPARTA_DMG_Background.png  macOS dmg background

set -e
cd "$(dirname "$0")"
ICONS="$PWD"
RES="$PWD/.."
PKG="$PWD/../../packaging"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

RED="#c23b2e"
GOLD="#e0aa2e"

# ---- vector master --------------------------------------------------------
python3 - "$ICONS/sparta-logo.svg" <<'PYEOF'
import math, sys
RED, GOLD = "#c23b2e", "#e0aa2e"
def spiral(cx, cy, rmax, turns=2.2, n=60):
    return "M " + " L ".join(
        f"{cx + rmax*t*math.cos(t*turns*2*math.pi):.2f},"
        f"{cy + rmax*t*math.sin(t*turns*2*math.pi):.2f}"
        for t in (i/n for i in range(n+1)))
p = ['<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">']
p.append(f'<circle cx="256" cy="256" r="252" fill="{GOLD}"/>')
p.append(f'<circle cx="256" cy="256" r="242" fill="{RED}"/>')
for i in range(14):
    p.append(f'<g transform="rotate({i*360/14:.1f} 256 256)">'
             f'<path d="{spiral(256,51,26)}" fill="none" stroke="{GOLD}" '
             f'stroke-width="7" stroke-linecap="round"/></g>')
p.append(f'<circle cx="256" cy="256" r="168" fill="{GOLD}"/>')
p.append(f'<circle cx="256" cy="256" r="158" fill="{RED}"/>')
# large lambda, and a SPARTA wordmark small enough to stay inside the circle
p.append(f'<text x="256" y="292" font-family="DejaVu Serif" font-size="240" '
         f'font-weight="bold" fill="{GOLD}" text-anchor="middle">&#955;</text>')
p.append(f'<text x="256" y="360" font-family="DejaVu Serif" font-size="44" '
         f'fill="{GOLD}" text-anchor="middle" letter-spacing="2">SPARTA</text>')
p.append('</svg>')
open(sys.argv[1], "w").write("\n".join(p))
PYEOF

rsvg-convert -w 512 -h 512 "$ICONS/sparta-logo.svg" -o "$TMP/logo512.png"

# ---- icons ----------------------------------------------------------------
convert "$TMP/logo512.png" -resize 128x128 "$ICONS/sparta-icon-128x128.png"
cp "$ICONS/sparta-icon-128x128.png" "$ICONS/sparta-gui-icon-128x128.png"
convert "$TMP/logo512.png" -define icon:auto-resize=256,128,64,48,32,16 "$RES/sparta-gui.ico"
cp "$RES/sparta-gui.ico" "$RES/spafile.ico"
for s in 16 32 128 256 512; do convert "$TMP/logo512.png" -resize ${s}x${s} "$TMP/i$s.png"; done
png2icns "$PKG/sparta-gui.icns" "$TMP"/i16.png "$TMP"/i32.png "$TMP"/i128.png \
         "$TMP"/i256.png "$TMP"/i512.png >/dev/null

# ---- editor splash banner (wide enough for the full title) ----------------
convert -size 1120x360 xc:white \
  \( "$TMP/logo512.png" -resize 300x300 \) -geometry +30+30 -composite \
  -font DejaVu-Serif-Bold -pointsize 82 -fill "$RED" -annotate +360+180 "SPARTA-GUI" \
  -font DejaVu-Sans -pointsize 32 -fill "#555555" -annotate +364+238 "DSMC simulation GUI" \
  "$ICONS/sparta-gui-banner.png"
convert "$ICONS/sparta-gui-banner.png" -type TrueColor BMP3:"$ICONS/sparta-gui-banner.bmp"

# ---- plugin illustration --------------------------------------------------
convert -size 700x640 xc:white \
  \( "$TMP/logo512.png" -resize 380x380 \) -gravity north -geometry +0+50 -composite \
  -font DejaVu-Serif-Bold -pointsize 52 -fill "$RED" -gravity south -annotate +0+80 "libsparta plugin" \
  "$ICONS/sparta-plugin.png"

# ---- macOS dmg installer background ---------------------------------------
# Finder shows this at native resolution (no scaling), so image pixels map
# 1:1 to the AppleScript icon window coordinates in build_macos_dmg.sh:
#   app (190,216)   Applications (576,216)   README (190,400)   [64px centers]
convert -size 768x600 xc:white \
  \( -size 768x92 xc:black \) -geometry +0+0 -composite \
  \( "$TMP/logo512.png" -resize 68x68 \) -geometry +14+12 -composite \
  -font DejaVu-Serif-Bold -pointsize 42 -fill white -annotate +96+62 "SPARTA-GUI" \
  -font DejaVu-Sans -pointsize 21 -fill black -gravity North \
     -annotate +0+120 "To install, drag SPARTA-GUI onto the Applications folder:" \
  -fill "#cfcfcf" -draw "polygon 250,200 250,232 470,232 470,246 520,216 470,186 470,200" \
  -font DejaVu-Sans -pointsize 17 -fill "#333333" -gravity North \
     -annotate +0+320 "The README file explains how to run the bundled examples and how to" \
  -font DejaVu-Sans -pointsize 17 -fill "#333333" -gravity North \
     -annotate +0+344 "load your own SPARTA build. The SPARTA library is included in the app." \
  "$PKG/SPARTA_DMG_Background.png"

echo "regenerated SPARTA-GUI branding assets"
