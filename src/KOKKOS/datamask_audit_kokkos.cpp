/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.github.io
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#include "datamask_audit_kokkos.h"

#ifdef SPARTA_KOKKOS_DEBUG_SYNC

#include "comm.h"
#include "error.h"
#include "grid_kokkos.h"
#include "kokkos_type.h"
#include "particle_kokkos.h"
#include "sparta_masks.h"
#include "surf_kokkos.h"
#include "update.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

using namespace SPARTA_NS;

enum{INT,DOUBLE};                       // several files

static int audit_enabled = 0;

// When SPARTA_KOKKOS_TRACE selects a view, bracket each audited call in the same
// stream as the dual view events, so the two can be read together.
static const char *audit_trace()
{
  static const char *f = std::getenv("SPARTA_KOKKOS_TRACE");
  return f;
}

// Asked for or not.  The audit copies every watched array on both sides of every
// style call, so it is the one detector that is worth real time even when it
// finds nothing; keep it behind a switch like the others rather than paying for
// it on every run of the debug build.
static bool audit_wanted()
{
  static const bool want = std::getenv("SPARTA_KOKKOS_AUDIT") != nullptr;
  return want;
}

// masks that the style being audited marked itself, by calling
// ParticleKokkos::modify() and friends rather than declaring them in
// datamask_modify
static unsigned int audit_self_declared = 0;

// the audit in progress, so that a sync can refresh what it is comparing against
static DatamaskAudit *audit_active = nullptr;

// one entry per style and array, so that a wrong declaration in the inner loop
// is reported once instead of on every step

static std::map<std::string, bigint> audit_found;

/* ---------------------------------------------------------------------- */

void DatamaskAudit::enable(int flag)
{
  if (!audit_wanted()) return;

  // the audit snapshots the device buffers directly, and in poison mode those
  // bytes are off limits whenever the host side is the authoritative one, so
  // the two cannot run together
  if (flag && std::getenv("SPARTA_KOKKOS_POISON")) return;
  audit_enabled = flag;
}

/* ---------------------------------------------------------------------- */

void DatamaskAudit::note_modified(unsigned int mask)
{
  audit_self_declared |= mask;
}

/* ---------------------------------------------------------------------- */

void DatamaskAudit::note_synced(unsigned int mask)
{
  if (audit_active) audit_active->rebaseline(mask);
}

/* ---------------------------------------------------------------------- */

void SPARTA_NS::datamask_audit_note_copy(const void *device_data)
{
  if (audit_active) audit_active->rebaseline_one(device_data);
}

/* ---------------------------------------------------------------------- */

void DatamaskAudit::rebaseline_one(const void *device_data)
{
  if (!active || !device_data) return;
  for (size_t i = 0; i < arrays.size(); i++) {
    if (arrays[i].data != (const char *) device_data || before[i].empty()) continue;
    before[i].assign(arrays[i].data, arrays[i].data + arrays[i].bytes);
    return;
  }
}

/* ---------------------------------------------------------------------- */

void DatamaskAudit::rebaseline(unsigned int mask)
{
  if (!active) return;
  for (size_t i = 0; i < arrays.size(); i++) {
    if (!(mask & arrays[i].bit) || before[i].empty()) continue;
    // the array may have moved or changed size since the snapshot
    if (!arrays[i].data) continue;
    before[i].assign(arrays[i].data, arrays[i].data + arrays[i].bytes);
  }
}

/* ----------------------------------------------------------------------
   The lengths the snapshots are taken against.  Collected fresh on both ends of
   the comparison: a style that adds particles or refines the grid leaves
   nothing comparable, and this is how that is noticed.
------------------------------------------------------------------------- */

static DatamaskAudit::Extents measure(SPARTA *sparta)
{
  DatamaskAudit::Extents e;
  e.nparticle = sparta->particle->nlocal;
  e.nspecies = sparta->particle->nspecies;
  e.gnlocal = sparta->grid->nlocal;
  e.gnghost = sparta->grid->nghost;
  e.gnsplitlocal = sparta->grid->nsplitlocal;
  e.gnsplitghost = sparta->grid->nsplitghost;
  e.gnparent = sparta->grid->nparent;
  e.gmaxlevel = sparta->grid->maxlevel;
  e.snlocal = sparta->surf->nlocal;
  e.snown = sparta->surf->nown;
  return e;
}

/* ----------------------------------------------------------------------
   Where the shared arrays live and how much of each belongs to the entries that
   exist.  The device side is the one to watch: that is what the kernels write,
   and in a build without a GPU it is also the copy that the host does not see.
   The allocation runs past the live entries and that tail is genuinely
   uninitialised, so each array carries its own live count rather than one
   global length.

   VREMAX_MASK and REMAIN_MASK are deliberately absent: those live on
   CollideVSSKokkos rather than on a global object, and collide is not a fix, so
   nothing here ever brackets a call that touches them.  report() says so.
------------------------------------------------------------------------- */

static void collect(SPARTA *sparta, std::vector<DatamaskAudit::Array> &out)
{
  out.clear();

  auto take = [&](unsigned int bit, const std::string &name, const char *data, size_t span,
                  size_t esz, size_t n0, int nlive, bool stale) {
    if (!data || n0 == 0 || nlive <= 0 || nlive > (int) n0) return;
    const size_t per = span * esz / n0;
    if (per == 0) return;
    out.push_back({bit, name, data, (size_t) nlive * per, (int) per, stale});
  };

#define SPARTA_AUDIT_ARRAY(BIT, NAME, KV, NLIVE)                                          \
  {                                                                                       \
    auto v = (KV).view_device();                                                          \
    take(BIT, NAME, (const char *) v.data(), v.span(),                                    \
         sizeof(typename decltype(v)::value_type), v.extent(0), (NLIVE),                  \
         (KV).need_sync_device());                                                        \
  }

  auto *particle_kk = (ParticleKokkos *) sparta->particle;
  auto *grid_kk = (GridKokkos *) sparta->grid;
  auto *surf_kk = (SurfKokkos *) sparta->surf;

  SPARTA_AUDIT_ARRAY(PARTICLE_MASK, "particle:particles", particle_kk->k_particles,
                     particle_kk->nlocal)
  SPARTA_AUDIT_ARRAY(SPECIES_MASK, "particle:species", particle_kk->k_species,
                     particle_kk->nspecies)

  SPARTA_AUDIT_ARRAY(CELL_MASK, "grid:cells", grid_kk->k_cells,
                     grid_kk->nlocal + grid_kk->nghost)
  SPARTA_AUDIT_ARRAY(CINFO_MASK, "grid:cinfo", grid_kk->k_cinfo, grid_kk->nlocal)
  SPARTA_AUDIT_ARRAY(SINFO_MASK, "grid:sinfo", grid_kk->k_sinfo,
                     grid_kk->nsplitlocal + grid_kk->nsplitghost)
  SPARTA_AUDIT_ARRAY(PCELL_MASK, "grid:pcells", grid_kk->k_pcells, grid_kk->nparent)
  SPARTA_AUDIT_ARRAY(PLEVEL_MASK, "grid:plevels", grid_kk->k_plevels, grid_kk->maxlevel + 1)

  SPARTA_AUDIT_ARRAY(LINE_MASK, "surf:lines", surf_kk->k_lines, surf_kk->nlocal)
  SPARTA_AUDIT_ARRAY(TRI_MASK, "surf:tris", surf_kk->k_tris, surf_kk->nlocal)
  SPARTA_AUDIT_ARRAY(LINE_MASK, "surf:mylines", surf_kk->k_mylines, surf_kk->nown)
  SPARTA_AUDIT_ARRAY(TRI_MASK, "surf:mytris", surf_kk->k_mytris, surf_kk->nown)

  // The custom per-particle and per-grid-cell attributes, which all share
  // CUSTOM_MASK.  Walked through the public ncustom/ename/etype/esize/ewhich
  // rather than the per-category counters, which are protected on Particle and
  // Grid: ewhich says which slot of the holder an attribute lives in, and
  // etype with esize says which holder.  This is the idiom the rest of SPARTA
  // uses, e.g. Particle::grow_custom() -- and it names each finding with the
  // attribute's own name instead of a category index.

#define SPARTA_AUDIT_CUSTOM(OWNER, WHO, NLIVE)                                            \
  for (int ic = 0; ic < (OWNER)->ncustom; ic++) {                                         \
    if ((OWNER)->ename[ic] == NULL) continue;               /* deleted attribute */       \
    const int iw = (OWNER)->ewhich[ic];                                                   \
    const std::string cname = std::string(WHO ":custom:") + (OWNER)->ename[ic];            \
    if ((OWNER)->etype[ic] == INT) {                                                      \
      if ((OWNER)->esize[ic] == 0) {                                                      \
        auto &kv = (OWNER)->k_eivec.view_host()[iw].k_view;                               \
        SPARTA_AUDIT_ARRAY(CUSTOM_MASK, cname, kv, (NLIVE))                               \
      } else {                                                                            \
        auto &kv = (OWNER)->k_eiarray.view_host()[iw].k_view;                             \
        SPARTA_AUDIT_ARRAY(CUSTOM_MASK, cname, kv, (NLIVE))                               \
      }                                                                                   \
    } else {                                                                              \
      if ((OWNER)->esize[ic] == 0) {                                                      \
        auto &kv = (OWNER)->k_edvec.view_host()[iw].k_view;                               \
        SPARTA_AUDIT_ARRAY(CUSTOM_MASK, cname, kv, (NLIVE))                               \
      } else {                                                                            \
        auto &kv = (OWNER)->k_edarray.view_host()[iw].k_view;                             \
        SPARTA_AUDIT_ARRAY(CUSTOM_MASK, cname, kv, (NLIVE))                               \
      }                                                                                   \
    }                                                                                     \
  }

  SPARTA_AUDIT_CUSTOM(particle_kk, "particle", particle_kk->nlocal)
  SPARTA_AUDIT_CUSTOM(grid_kk, "grid", grid_kk->nlocal)

#undef SPARTA_AUDIT_CUSTOM
#undef SPARTA_AUDIT_ARRAY
}

/* ---------------------------------------------------------------------- */

DatamaskAudit::DatamaskAudit(SPARTA *sparta_in, const char *what_in, const char *style_in,
                             unsigned int datamask_modify) :
    sparta(sparta_in), what(what_in), style(style_in ? style_in : "(unnamed)"),
    declared(datamask_modify), extents{}, active(false)
{
  if (!audit_enabled) return;

  extents = measure(sparta);

  audit_self_declared = 0;

  collect(sparta, arrays);
  if (arrays.empty()) return;

  before.resize(arrays.size());
  int checked = 0;
  for (size_t i = 0; i < arrays.size(); i++) {
    if (declared & arrays[i].bit) continue;    // free to change it, do not copy
    before[i].assign(arrays[i].data, arrays[i].data + arrays[i].bytes);
    checked++;
  }

  // A style that never sets datamask_modify keeps the ALL_MASK that Fix puts
  // there, which declares every array and leaves nothing to compare.  Say so,
  // rather than let the style pass as though it had been checked.

  if (checked == 0) {
    const std::string key = style + " declares every array";
    if (audit_found.count(key)) audit_found[key]++;
    else {
      audit_found[key] = 1;
      char buf[512];
      snprintf(buf, sizeof(buf),
               "datamask audit: %s %s declares every array in datamask_modify, so "
               "nothing about it can be checked, on step " BIGINT_FORMAT,
               what, style.c_str(), sparta->update->ntimestep);
      sparta->error->warning(FLERR, buf);
    }
  }

  // ModifyKokkos syncs datamask_read just before the call, so an array that is
  // still stale here is one the style did not declare.  If it then reads it, it
  // reads what the other side wrote -- the missing-sync half of the problem,
  // which no comparison of contents can see.

  for (auto &a : arrays) {
    if (!a.stale) continue;
    const std::string key = style + " reads stale " + a.name;
    if (audit_found.count(key)) { audit_found[key]++; continue; }
    audit_found[key] = 1;
    char buf[512];
    snprintf(buf, sizeof(buf),
             "datamask audit: %s %s starts with %s stale on the device, so it is not "
             "covered by datamask_read, on step " BIGINT_FORMAT,
             what, style.c_str(), a.name.c_str(), sparta->update->ntimestep);
    sparta->error->warning(FLERR, buf);
  }

  active = true;
  audit_active = this;

  if (audit_trace()) std::fprintf(stderr, "[audit] begin  %s %s\n", what, style.c_str());
}

/* ---------------------------------------------------------------------- */

DatamaskAudit::~DatamaskAudit()
{
  if (!active || !audit_enabled) return;

  trace_end(what, style.c_str());

  // whatever the style marked itself while it ran is declared just as much as
  // what it named in datamask_modify
  audit_active = nullptr;

  const unsigned int covered = declared | audit_self_declared;
  audit_self_declared = 0;

  // a migration, a reaction or a grid change in the middle leaves nothing
  // comparable
  if (measure(sparta) != extents) return;

  std::vector<Array> now;
  collect(sparta, now);

  for (size_t i = 0; i < arrays.size(); i++) {
    if (covered & arrays[i].bit) continue;
    if (before[i].empty()) continue;

    // Find the same array again rather than trusting the old pointer.  Matched
    // on the address as well as the bit: every custom attribute carries
    // CUSTOM_MASK, so the bit alone would find the wrong one.
    const Array *cur = nullptr;
    for (auto &n : now)
      if (n.bit == arrays[i].bit && n.data == arrays[i].data) { cur = &n; break; }
    if (!cur || cur->bytes != before[i].size()) continue;

    if (memcmp(cur->data, before[i].data(), before[i].size()) == 0) continue;

    int ientry = -1;
    for (size_t b = 0; b < before[i].size(); b++)
      if (cur->data[b] != before[i][b]) { ientry = (int) ((int) b / cur->stride); break; }
    if (ientry < 0) continue;

    // show the values as raw bytes read as an integer: the arrays are structs
    // of mixed types, so this names the entry that moved rather than pretending
    // to interpret it
    const size_t off = (size_t) ientry * cur->stride;
    long long ov = 0, nv = 0;
    const size_t n = (cur->stride > (int) sizeof(long long)) ? sizeof(long long) : cur->stride;
    memcpy(&ov, before[i].data() + off, n);
    memcpy(&nv, cur->data + off, n);

    const std::string key = style + " changed " + arrays[i].name;
    if (audit_found.count(key)) { audit_found[key]++; continue; }
    audit_found[key] = 1;

    char buf[768];
    snprintf(buf, sizeof(buf),
             "datamask audit: %s %s changed %s without declaring it in datamask_modify "
             "or marking it modified, first at entry %d of %d (%lld -> %lld) on step "
             BIGINT_FORMAT,
             what, style.c_str(), arrays[i].name.c_str(), ientry,
             (int) (before[i].size() / cur->stride), ov, nv, sparta->update->ntimestep);
    sparta->error->warning(FLERR, buf);
  }
}

/* ---------------------------------------------------------------------- */

void DatamaskAudit::trace_end(const char *what, const char *style)
{
  if (audit_trace()) std::fprintf(stderr, "[audit] end    %s %s\n", what, style);
}

/* ---------------------------------------------------------------------- */

void DatamaskAudit::report(SPARTA *sparta)
{
  if (!audit_wanted()) return;
  if (sparta->comm->me != 0) return;

  if (audit_found.empty()) {
    utils::logmesg(sparta,
                   "datamask audit: no undeclared changes to the arrays it watches "
                   "(vremax and remain are not among them)\n");
    return;
  }

  utils::logmesg(sparta, "datamask audit: undeclared changes to shared arrays\n");
  for (auto &f : audit_found)
    utils::logmesg(sparta, "  " + f.first + " on " + std::to_string(f.second) + " step(s)\n");
}

#endif    // SPARTA_KOKKOS_DEBUG_SYNC
