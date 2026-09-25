/* -*- c++ -*- ----------------------------------------------------------
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

// GENERATED FILE, DO NOT EDIT: run tools/kokkos_prec/gen_kk_structs.py
//   after changing a host struct or tools/kokkos_prec/precision_map.json

// KK precision copies of the host structs shared with the device, and
//   the element conversions TransformView uses on host/device sync.
//   In a mode where no field of a struct changes type, the KK struct is
//   the host struct itself.

#ifndef SPARTA_KOKKOS_STRUCTS_H
#define SPARTA_KOKKOS_STRUCTS_H

namespace SPARTA_NS {

#if defined(SPARTA_KOKKOS_DOUBLE_DOUBLE)

typedef Particle::OnePart OnePartKK;

typedef Grid::ChildCell ChildCellKK;

typedef Grid::SplitInfo SplitInfoKK;

typedef Grid::ParentCell ParentCellKK;

typedef Surf::Line LineKK;

typedef Surf::Tri TriKK;

#elif defined(SPARTA_KOKKOS_SINGLE_DOUBLE)

struct SPARTA_ALIGN(16) OnePartKK {
  int id;
  int ispecies;
  int icell;
  int flag;
  KK_POS_FLOAT x[3];
  KK_FLOAT v[3];
  KK_FLOAT erot;
  KK_FLOAT evib;
  KK_POS_FLOAT dtremain;
  KK_FLOAT weight;
};

KOKKOS_INLINE_FUNCTION
void kk_convert(OnePartKK &dst, const Particle::OnePart &src)
{
  dst.id = src.id;
  dst.ispecies = src.ispecies;
  dst.icell = src.icell;
  dst.flag = src.flag;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.x[k],src.x[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.v[k],src.v[k]);
  kk_convert(dst.erot,src.erot);
  kk_convert(dst.evib,src.evib);
  kk_convert(dst.dtremain,src.dtremain);
  kk_convert(dst.weight,src.weight);
}

KOKKOS_INLINE_FUNCTION
void kk_convert(Particle::OnePart &dst, const OnePartKK &src)
{
  dst.id = src.id;
  dst.ispecies = src.ispecies;
  dst.icell = src.icell;
  dst.flag = src.flag;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.x[k],src.x[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.v[k],src.v[k]);
  kk_convert(dst.erot,src.erot);
  kk_convert(dst.evib,src.evib);
  kk_convert(dst.dtremain,src.dtremain);
  kk_convert(dst.weight,src.weight);
}

typedef Grid::ChildCell ChildCellKK;

typedef Grid::SplitInfo SplitInfoKK;

typedef Grid::ParentCell ParentCellKK;

typedef Surf::Line LineKK;

typedef Surf::Tri TriKK;

#elif defined(SPARTA_KOKKOS_SINGLE_SINGLE)

struct SPARTA_ALIGN(16) OnePartKK {
  int id;
  int ispecies;
  int icell;
  int flag;
  KK_POS_FLOAT x[3];
  KK_FLOAT v[3];
  KK_FLOAT erot;
  KK_FLOAT evib;
  KK_POS_FLOAT dtremain;
  KK_FLOAT weight;
};

KOKKOS_INLINE_FUNCTION
void kk_convert(OnePartKK &dst, const Particle::OnePart &src)
{
  dst.id = src.id;
  dst.ispecies = src.ispecies;
  dst.icell = src.icell;
  dst.flag = src.flag;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.x[k],src.x[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.v[k],src.v[k]);
  kk_convert(dst.erot,src.erot);
  kk_convert(dst.evib,src.evib);
  kk_convert(dst.dtremain,src.dtremain);
  kk_convert(dst.weight,src.weight);
}

KOKKOS_INLINE_FUNCTION
void kk_convert(Particle::OnePart &dst, const OnePartKK &src)
{
  dst.id = src.id;
  dst.ispecies = src.ispecies;
  dst.icell = src.icell;
  dst.flag = src.flag;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.x[k],src.x[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.v[k],src.v[k]);
  kk_convert(dst.erot,src.erot);
  kk_convert(dst.evib,src.evib);
  kk_convert(dst.dtremain,src.dtremain);
  kk_convert(dst.weight,src.weight);
}

struct SPARTA_ALIGN(64) ChildCellKK {
  cellint id;
  int level;
  int proc;
  int ilocal;
  cellint neigh[6];
  int nmask;
  KK_POS_FLOAT lo[3];
  KK_POS_FLOAT hi[3];
  int nsurf;
  surfint* csurfs;
  int nsplit;
  int isplit;
};

KOKKOS_INLINE_FUNCTION
void kk_convert(ChildCellKK &dst, const Grid::ChildCell &src)
{
  dst.id = src.id;
  dst.level = src.level;
  dst.proc = src.proc;
  dst.ilocal = src.ilocal;
  for (int k = 0; k < 6; k++) dst.neigh[k] = src.neigh[k];
  dst.nmask = src.nmask;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.lo[k],src.lo[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.hi[k],src.hi[k]);
  dst.nsurf = src.nsurf;
  dst.csurfs = src.csurfs;
  dst.nsplit = src.nsplit;
  dst.isplit = src.isplit;
}

KOKKOS_INLINE_FUNCTION
void kk_convert(Grid::ChildCell &dst, const ChildCellKK &src)
{
  dst.id = src.id;
  dst.level = src.level;
  dst.proc = src.proc;
  dst.ilocal = src.ilocal;
  for (int k = 0; k < 6; k++) dst.neigh[k] = src.neigh[k];
  dst.nmask = src.nmask;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.lo[k],src.lo[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.hi[k],src.hi[k]);
  dst.nsurf = src.nsurf;
  dst.csurfs = src.csurfs;
  dst.nsplit = src.nsplit;
  dst.isplit = src.isplit;
}

struct SplitInfoKK {
  int icell;
  int xsub;
  KK_POS_FLOAT xsplit[3];
  int* csplits;
  int* csubs;
};

KOKKOS_INLINE_FUNCTION
void kk_convert(SplitInfoKK &dst, const Grid::SplitInfo &src)
{
  dst.icell = src.icell;
  dst.xsub = src.xsub;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.xsplit[k],src.xsplit[k]);
  dst.csplits = src.csplits;
  dst.csubs = src.csubs;
}

KOKKOS_INLINE_FUNCTION
void kk_convert(Grid::SplitInfo &dst, const SplitInfoKK &src)
{
  dst.icell = src.icell;
  dst.xsub = src.xsub;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.xsplit[k],src.xsplit[k]);
  dst.csplits = src.csplits;
  dst.csubs = src.csubs;
}

struct ParentCellKK {
  cellint id;
  KK_POS_FLOAT lo[3];
  KK_POS_FLOAT hi[3];
};

KOKKOS_INLINE_FUNCTION
void kk_convert(ParentCellKK &dst, const Grid::ParentCell &src)
{
  dst.id = src.id;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.lo[k],src.lo[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.hi[k],src.hi[k]);
}

KOKKOS_INLINE_FUNCTION
void kk_convert(Grid::ParentCell &dst, const ParentCellKK &src)
{
  dst.id = src.id;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.lo[k],src.lo[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.hi[k],src.hi[k]);
}

struct LineKK {
  surfint id;
  int type;
  int mask;
  int isc;
  int isr;
  KK_POS_FLOAT p1[3];
  KK_POS_FLOAT p2[3];
  KK_POS_FLOAT norm[3];
  int transparent;
};

KOKKOS_INLINE_FUNCTION
void kk_convert(LineKK &dst, const Surf::Line &src)
{
  dst.id = src.id;
  dst.type = src.type;
  dst.mask = src.mask;
  dst.isc = src.isc;
  dst.isr = src.isr;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.p1[k],src.p1[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.p2[k],src.p2[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.norm[k],src.norm[k]);
  dst.transparent = src.transparent;
}

KOKKOS_INLINE_FUNCTION
void kk_convert(Surf::Line &dst, const LineKK &src)
{
  dst.id = src.id;
  dst.type = src.type;
  dst.mask = src.mask;
  dst.isc = src.isc;
  dst.isr = src.isr;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.p1[k],src.p1[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.p2[k],src.p2[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.norm[k],src.norm[k]);
  dst.transparent = src.transparent;
}

struct TriKK {
  surfint id;
  int type;
  int mask;
  int isc;
  int isr;
  KK_POS_FLOAT p1[3];
  KK_POS_FLOAT p2[3];
  KK_POS_FLOAT p3[3];
  KK_POS_FLOAT norm[3];
  int transparent;
};

KOKKOS_INLINE_FUNCTION
void kk_convert(TriKK &dst, const Surf::Tri &src)
{
  dst.id = src.id;
  dst.type = src.type;
  dst.mask = src.mask;
  dst.isc = src.isc;
  dst.isr = src.isr;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.p1[k],src.p1[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.p2[k],src.p2[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.p3[k],src.p3[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.norm[k],src.norm[k]);
  dst.transparent = src.transparent;
}

KOKKOS_INLINE_FUNCTION
void kk_convert(Surf::Tri &dst, const TriKK &src)
{
  dst.id = src.id;
  dst.type = src.type;
  dst.mask = src.mask;
  dst.isc = src.isc;
  dst.isr = src.isr;
  for (int k = 0; k < 3; k++)
    kk_convert(dst.p1[k],src.p1[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.p2[k],src.p2[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.p3[k],src.p3[k]);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.norm[k],src.norm[k]);
  dst.transparent = src.transparent;
}

#endif

// in a double precision build the KK structs are the host structs

#if defined(SPARTA_KOKKOS_DOUBLE_DOUBLE)
static_assert(std::is_same_v<OnePartKK,Particle::OnePart>);
static_assert(std::is_same_v<ChildCellKK,Grid::ChildCell>);
static_assert(std::is_same_v<SplitInfoKK,Grid::SplitInfo>);
static_assert(std::is_same_v<ParentCellKK,Grid::ParentCell>);
static_assert(std::is_same_v<LineKK,Surf::Line>);
static_assert(std::is_same_v<TriKK,Surf::Tri>);
#endif

}

#endif
