# usage: python3 revert.py <srcdir> <fix>  -- reverts one fix hunk to M code (for attribution of gold-log diffs)
import sys
src, fix = sys.argv[1], sys.argv[2]
def sub(fn, old, new):
    p = f"{src}/{fn}"; t = open(p).read()
    assert t.count(old) == 1, (fn, old[:60]); open(p, "w").write(t.replace(old, new))
if fix == "06-9":
    sub("create_particles.cpp",
"""  int flag = 0;
  if (hi[0] <= region->extent_xlo ||
      lo[0] >= region->extent_xhi) flag = 1;
  if (hi[1] <= region->extent_ylo ||
      lo[1] >= region->extent_yhi) flag = 1;
  if (dim == 3) {
    if (hi[2] <= region->extent_zlo ||
        lo[2] >= region->extent_zhi) flag = 1;
  }""",
"""  int flag = 1;
  if (hi[0] > region->extent_xlo &&
      lo[0] < region->extent_xhi) flag = 0;
  if (hi[1] > region->extent_ylo &&
      lo[1] < region->extent_yhi) flag = 0;
  if (dim == 3) {
    if (hi[2] > region->extent_zlo &&
        lo[2] < region->extent_zhi) flag = 0;
  }""")
elif fix == "12-7":
    sub("fix_emit.cpp",
"""      rflag = 0;
      if (cells[icell].hi[0] <= region->extent_xlo ||
          cells[icell].lo[0] >= region->extent_xhi) rflag = 1;
      if (cells[icell].hi[1] <= region->extent_ylo ||
          cells[icell].lo[1] >= region->extent_yhi) rflag = 1;
      if (dimension == 3) {
        if (cells[icell].hi[2] <= region->extent_zlo ||
            cells[icell].lo[2] >= region->extent_zhi) rflag = 1;
      }""",
"""      rflag = 1;
      if (cells[icell].hi[0] > region->extent_xlo &&
          cells[icell].lo[0] < region->extent_xhi) rflag = 0;
      if (cells[icell].hi[1] > region->extent_ylo &&
          cells[icell].lo[1] < region->extent_yhi) rflag = 0;
      if (dimension == 3) {
        if (cells[icell].hi[2] > region->extent_zlo &&
            cells[icell].lo[2] < region->extent_zhi) rflag = 0;
      }""")
elif fix == "19-8":
    sub("react_bird.cpp",
"""      reactions[i][j].sp2recomb = NULL; // Needed for Kokkos
      for (int m = 0; m < n; m++) {
        r = &rlist[list[m]];
        if (r->type == RECOMBINATION) {
          reactions[i][j].sp2recomb = &sp2recomb_ij[offset];
""",
"""      if (offset < nij*nspecies)
        reactions[i][j].sp2recomb = &sp2recomb_ij[offset];
      else
        reactions[i][j].sp2recomb = NULL; // Needed for Kokkos
      for (int m = 0; m < n; m++) {
        r = &rlist[list[m]];
        if (r->type == RECOMBINATION) {
""")
elif fix == "29-11":
    p = f"{src}/react_tce.cpp"; t = open(p).read()
    assert t.count("vibtemp[0]) + 0.5);") == 2
    open(p, "w").write(t.replace("vibtemp[0]) + 0.5);", "vibtemp[0]));"))
elif fix == "19-10":
    import subprocess
    open(f"{src}/react_qk.cpp", "w").write(subprocess.check_output(
        ["git", "-C", "/home/user/sparta", "show", "5702a655:src/react_qk.cpp"], text=True))
else:
    sys.exit("unknown fix")
