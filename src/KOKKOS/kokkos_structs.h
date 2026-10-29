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

typedef Particle::Species SpeciesKK;

typedef CollideVSS::Params ParamsKK;

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

struct SpeciesKK {
  char id[16];
  double molwt;
  double mass;
  KK_FLOAT specwt;
  KK_FLOAT charge;
  KK_FLOAT rotrel;
  KK_FLOAT rottemp[3];
  KK_FLOAT vibtemp[Particle::MAXVIBMODE];
  KK_FLOAT vibrel[Particle::MAXVIBMODE];
  int vibdegen[Particle::MAXVIBMODE];
  int rotdof;
  int vibdof;
  int nrottemp;
  int nvibmode;
  int internaldof;
  int vibdiscrete_read;
  KK_FLOAT magmoment;
};

KOKKOS_INLINE_FUNCTION
void kk_convert(SpeciesKK &dst, const Particle::Species &src)
{
  for (int k = 0; k < 16; k++) dst.id[k] = src.id[k];
  kk_convert(dst.molwt,src.molwt);
  kk_convert(dst.mass,src.mass);
  kk_convert(dst.specwt,src.specwt);
  kk_convert(dst.charge,src.charge);
  kk_convert(dst.rotrel,src.rotrel);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.rottemp[k],src.rottemp[k]);
  for (int k = 0; k < Particle::MAXVIBMODE; k++)
    kk_convert(dst.vibtemp[k],src.vibtemp[k]);
  for (int k = 0; k < Particle::MAXVIBMODE; k++)
    kk_convert(dst.vibrel[k],src.vibrel[k]);
  for (int k = 0; k < Particle::MAXVIBMODE; k++) dst.vibdegen[k] = src.vibdegen[k];
  dst.rotdof = src.rotdof;
  dst.vibdof = src.vibdof;
  dst.nrottemp = src.nrottemp;
  dst.nvibmode = src.nvibmode;
  dst.internaldof = src.internaldof;
  dst.vibdiscrete_read = src.vibdiscrete_read;
  kk_convert(dst.magmoment,src.magmoment);
}

KOKKOS_INLINE_FUNCTION
void kk_convert(Particle::Species &dst, const SpeciesKK &src)
{
  for (int k = 0; k < 16; k++) dst.id[k] = src.id[k];
  kk_convert(dst.molwt,src.molwt);
  kk_convert(dst.mass,src.mass);
  kk_convert(dst.specwt,src.specwt);
  kk_convert(dst.charge,src.charge);
  kk_convert(dst.rotrel,src.rotrel);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.rottemp[k],src.rottemp[k]);
  for (int k = 0; k < Particle::MAXVIBMODE; k++)
    kk_convert(dst.vibtemp[k],src.vibtemp[k]);
  for (int k = 0; k < Particle::MAXVIBMODE; k++)
    kk_convert(dst.vibrel[k],src.vibrel[k]);
  for (int k = 0; k < Particle::MAXVIBMODE; k++) dst.vibdegen[k] = src.vibdegen[k];
  dst.rotdof = src.rotdof;
  dst.vibdof = src.vibdof;
  dst.nrottemp = src.nrottemp;
  dst.nvibmode = src.nvibmode;
  dst.internaldof = src.internaldof;
  dst.vibdiscrete_read = src.vibdiscrete_read;
  kk_convert(dst.magmoment,src.magmoment);
}

struct ParamsKK {
  KK_FLOAT diam;
  KK_FLOAT omega;
  KK_FLOAT tref;
  KK_FLOAT alpha;
  KK_FLOAT rotc1;
  KK_FLOAT rotc2;
  KK_FLOAT rotc3;
  KK_FLOAT vibc1;
  KK_FLOAT vibc2;
  KK_FLOAT mr;
};

KOKKOS_INLINE_FUNCTION
void kk_convert(ParamsKK &dst, const CollideVSS::Params &src)
{
  kk_convert(dst.diam,src.diam);
  kk_convert(dst.omega,src.omega);
  kk_convert(dst.tref,src.tref);
  kk_convert(dst.alpha,src.alpha);
  kk_convert(dst.rotc1,src.rotc1);
  kk_convert(dst.rotc2,src.rotc2);
  kk_convert(dst.rotc3,src.rotc3);
  kk_convert(dst.vibc1,src.vibc1);
  kk_convert(dst.vibc2,src.vibc2);
  kk_convert(dst.mr,src.mr);
}

KOKKOS_INLINE_FUNCTION
void kk_convert(CollideVSS::Params &dst, const ParamsKK &src)
{
  kk_convert(dst.diam,src.diam);
  kk_convert(dst.omega,src.omega);
  kk_convert(dst.tref,src.tref);
  kk_convert(dst.alpha,src.alpha);
  kk_convert(dst.rotc1,src.rotc1);
  kk_convert(dst.rotc2,src.rotc2);
  kk_convert(dst.rotc3,src.rotc3);
  kk_convert(dst.vibc1,src.vibc1);
  kk_convert(dst.vibc2,src.vibc2);
  kk_convert(dst.mr,src.mr);
}

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

struct SpeciesKK {
  char id[16];
  double molwt;
  double mass;
  KK_FLOAT specwt;
  KK_FLOAT charge;
  KK_FLOAT rotrel;
  KK_FLOAT rottemp[3];
  KK_FLOAT vibtemp[Particle::MAXVIBMODE];
  KK_FLOAT vibrel[Particle::MAXVIBMODE];
  int vibdegen[Particle::MAXVIBMODE];
  int rotdof;
  int vibdof;
  int nrottemp;
  int nvibmode;
  int internaldof;
  int vibdiscrete_read;
  KK_FLOAT magmoment;
};

KOKKOS_INLINE_FUNCTION
void kk_convert(SpeciesKK &dst, const Particle::Species &src)
{
  for (int k = 0; k < 16; k++) dst.id[k] = src.id[k];
  kk_convert(dst.molwt,src.molwt);
  kk_convert(dst.mass,src.mass);
  kk_convert(dst.specwt,src.specwt);
  kk_convert(dst.charge,src.charge);
  kk_convert(dst.rotrel,src.rotrel);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.rottemp[k],src.rottemp[k]);
  for (int k = 0; k < Particle::MAXVIBMODE; k++)
    kk_convert(dst.vibtemp[k],src.vibtemp[k]);
  for (int k = 0; k < Particle::MAXVIBMODE; k++)
    kk_convert(dst.vibrel[k],src.vibrel[k]);
  for (int k = 0; k < Particle::MAXVIBMODE; k++) dst.vibdegen[k] = src.vibdegen[k];
  dst.rotdof = src.rotdof;
  dst.vibdof = src.vibdof;
  dst.nrottemp = src.nrottemp;
  dst.nvibmode = src.nvibmode;
  dst.internaldof = src.internaldof;
  dst.vibdiscrete_read = src.vibdiscrete_read;
  kk_convert(dst.magmoment,src.magmoment);
}

KOKKOS_INLINE_FUNCTION
void kk_convert(Particle::Species &dst, const SpeciesKK &src)
{
  for (int k = 0; k < 16; k++) dst.id[k] = src.id[k];
  kk_convert(dst.molwt,src.molwt);
  kk_convert(dst.mass,src.mass);
  kk_convert(dst.specwt,src.specwt);
  kk_convert(dst.charge,src.charge);
  kk_convert(dst.rotrel,src.rotrel);
  for (int k = 0; k < 3; k++)
    kk_convert(dst.rottemp[k],src.rottemp[k]);
  for (int k = 0; k < Particle::MAXVIBMODE; k++)
    kk_convert(dst.vibtemp[k],src.vibtemp[k]);
  for (int k = 0; k < Particle::MAXVIBMODE; k++)
    kk_convert(dst.vibrel[k],src.vibrel[k]);
  for (int k = 0; k < Particle::MAXVIBMODE; k++) dst.vibdegen[k] = src.vibdegen[k];
  dst.rotdof = src.rotdof;
  dst.vibdof = src.vibdof;
  dst.nrottemp = src.nrottemp;
  dst.nvibmode = src.nvibmode;
  dst.internaldof = src.internaldof;
  dst.vibdiscrete_read = src.vibdiscrete_read;
  kk_convert(dst.magmoment,src.magmoment);
}

struct ParamsKK {
  KK_FLOAT diam;
  KK_FLOAT omega;
  KK_FLOAT tref;
  KK_FLOAT alpha;
  KK_FLOAT rotc1;
  KK_FLOAT rotc2;
  KK_FLOAT rotc3;
  KK_FLOAT vibc1;
  KK_FLOAT vibc2;
  KK_FLOAT mr;
};

KOKKOS_INLINE_FUNCTION
void kk_convert(ParamsKK &dst, const CollideVSS::Params &src)
{
  kk_convert(dst.diam,src.diam);
  kk_convert(dst.omega,src.omega);
  kk_convert(dst.tref,src.tref);
  kk_convert(dst.alpha,src.alpha);
  kk_convert(dst.rotc1,src.rotc1);
  kk_convert(dst.rotc2,src.rotc2);
  kk_convert(dst.rotc3,src.rotc3);
  kk_convert(dst.vibc1,src.vibc1);
  kk_convert(dst.vibc2,src.vibc2);
  kk_convert(dst.mr,src.mr);
}

KOKKOS_INLINE_FUNCTION
void kk_convert(CollideVSS::Params &dst, const ParamsKK &src)
{
  kk_convert(dst.diam,src.diam);
  kk_convert(dst.omega,src.omega);
  kk_convert(dst.tref,src.tref);
  kk_convert(dst.alpha,src.alpha);
  kk_convert(dst.rotc1,src.rotc1);
  kk_convert(dst.rotc2,src.rotc2);
  kk_convert(dst.rotc3,src.rotc3);
  kk_convert(dst.vibc1,src.vibc1);
  kk_convert(dst.vibc2,src.vibc2);
  kk_convert(dst.mr,src.mr);
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
static_assert(std::is_same_v<SpeciesKK,Particle::Species>);
static_assert(std::is_same_v<ParamsKK,CollideVSS::Params>);
#endif

}

#endif
