#!/usr/bin/env python3
"""Drive features end to end and check the artifact they are supposed to produce.

The walker proves controls are reachable and the visual suite proves the ones
with a visible effect really have it. Neither says whether a feature actually
works: whether exporting a movie yields a playable file with the right number
of frames, or whether the ParaView export really invokes pvpython and gets data
back.

So each check here keys on specific evidence -- a file that exists and probes
as valid, a frame count, a row count, a named string in generated output --
never on "the screen changed", which an earlier pass on this project showed can
pass while something entirely different happened.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guidrive import Gui

OUT = os.environ.get("SHOT_DIR", "feature-shots")
FIX = os.environ.get("SPARTA_FIXTURES", "fixtures")

results = []


def note(check, verdict, detail=""):
    results.append((check, verdict, detail))
    print(f"  {verdict:4s} {check:44s} {detail}")


def have(tool):
    return subprocess.run(["which", tool], capture_output=True).returncode == 0


def probe_frames(path):
    """Frame count of a movie, via ffprobe; None if it is not readable."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    txt = (r.stdout or "").strip()
    return int(txt) if txt.isdigit() else None


def type_text(g, text):
    subprocess.run(["xdotool", "type", "--clearmodifiers", "--delay", "12", text],
                   env=g.env)
    time.sleep(0.8)


def test_movie_export():
    """Slide show -> Export to movie, then verify the file is a real movie.

    Ctrl+E is the export shortcut. The frames come from the deck's dump image
    output, so the exported movie must contain exactly the frames in the active
    [Start, Stop] range -- a file that merely exists is not evidence.
    """
    frames = sorted(f for f in os.listdir(FIX) if f.startswith("gimg.") and f.endswith(".ppm"))
    if not frames:
        note("movie export", "SKIP", "no dump-image frames in fixtures")
        return
    if not have("ffmpeg") or not have("ffprobe"):
        note("movie export", "SKIP", "ffmpeg/ffprobe not installed")
        return

    paths = [os.path.join(FIX, f) for f in frames]
    outfile = os.path.abspath(f"{OUT}/exported.mp4")
    if os.path.exists(outfile):
        os.remove(outfile)

    # -i takes one value per occurrence, so the flag has to be repeated;
    # "-i a b c" would load only a and treat b and c as input decks
    argv = []
    for p in paths:
        argv += ["-i", p]

    with Gui(display=86, outdir=OUT, args=argv) as g:
        time.sleep(3)
        ids = g._xdo("search", "--name", "Slide Show").stdout.split()
        if not ids:
            note("movie export", "FAIL", "slide show did not open")
            return
        wid = ids[-1]
        g._xdo("windowactivate", wid)
        time.sleep(1)
        g.shot("movie-01-loaded")

        g.key("ctrl+e", 3)              # Export to movie file
        type_text(g, outfile)
        g.key("Return", 4)
        # encoding runs an external ffmpeg; give it room but do not hang forever
        for _ in range(30):
            if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
                break
            time.sleep(2)
        g.shot("movie-02-after-export")

        if not os.path.exists(outfile):
            note("movie export produces a file", "FAIL", "no file was written")
            return
        note("movie export produces a file", "PASS",
             f"{os.path.getsize(outfile)} bytes")

        n = probe_frames(outfile)
        if n is None:
            note("exported movie is playable", "FAIL", "ffprobe could not read it")
        else:
            note("exported movie is playable", "PASS", f"ffprobe reports {n} frames")
            # the slide show exports the active range, which defaults to every frame
            note("exported movie has the expected frame count",
                 "PASS" if n == len(frames) else "FAIL",
                 f"{n} frames exported, {len(frames)} source frames")

        note("application alive after export",
             "PASS" if g.app.poll() is None else "FAIL")


def main():
    os.makedirs(OUT, exist_ok=True)
    test_movie_export()

    print()
    npass = sum(1 for r in results if r[1] == "PASS")
    nskip = sum(1 for r in results if r[1] == "SKIP")
    print(f"{npass}/{len(results) - nskip} feature checks passed"
          + (f" ({nskip} skipped)" if nskip else ""))
    with open(f"{OUT}/results.tsv", "w") as f:
        for c, v, d in results:
            f.write(f"{c}\t{v}\t{d}\n")
    return 0 if npass == len(results) - nskip else 1


if __name__ == "__main__":
    sys.exit(main())
