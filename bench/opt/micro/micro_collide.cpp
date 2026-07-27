/* Microbenchmark of the SPARTA NTC collision kernel.
 *
 * This is a faithful transcription of the code path that in.collide actually
 * executes for a single monatomic species (Ar): Collide::collisions_one ->
 * CollideVSS::{attempt_collision,test_collision,setup_collision,
 * perform_collision} -> SCATTER_TwoBodyScattering, with RanKnuth as the RNG.
 * EEXCHANGE_NonReactingEDisposal is skipped exactly as it is in the real run,
 * because Ar has ave_dof == 0.
 *
 * The point is to try structural variants in seconds rather than minutes.
 * Variants are cumulative:
 *   V0 baseline    - virtual dispatch, out-of-line RNG, plist gather via next[]
 *   V1 +devirt     - collision math inlined into the cell loop
 *   V2 +inline RNG - RanKnuth::uniform() inlinable
 *   V3 +hoist      - params/prefactor/vremax/volume invariants hoisted per cell
 *   V4 +contig     - cell particles known contiguous, plist gather skipped
 *   V5 +fastpow    - pow() replaced by exp2/log2 (CHANGES NUMERICS)
 *
 * V0..V4 must produce bit-identical results; the harness checks that.
 *
 * build: g++ -O3 -std=c++11 -o micro_collide micro_collide.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

#define MY_PI 3.14159265358979323846
#define EPSZERO 1.0e-12
#define MAX(A,B) ((A) > (B) ? (A) : (B))

static double wtime()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1.0e-9 * ts.tv_nsec;
}

/* ---------------- data structures, matching src/particle.h ---------------- */

struct alignas(16) OnePart {
  int id;
  int ispecies;
  int icell;
  int flag;
  double x[3];
  double v[3];
  double erot;
  double evib;
  double dtremain;
  double weight;
};

struct Species {
  double mass;
  double rotdof, vibdof;
};

struct Params {
  double diam, omega, tref, alpha;
  double rotc1, rotc2, rotc3, vibc1, vibc2;
  double mr;
};

struct State {
  double vr2, vr, imass, jmass;
  double ave_rotdof, ave_vibdof, ave_dof;
  double etrans, erot, evib, eexchange, eint, etotal;
  double ucmf, vcmf, wcmf;
};

/* ---------------- RanKnuth, matching src/random_knuth.cpp ---------------- */

#define MBIG 1000000000
#define MSEED 161803398
#define FAC (1.0/MBIG)

class RanKnuth {
 public:
  RanKnuth(int seed_) { seed = seed_; save = 0; initflag = 0; }

  /* out-of-line, as it is in the shipped build (separate translation unit) */
  double uniform_outofline();

  /* the same arithmetic, inlinable */
  inline double uniform_inline() {
    int mj;
    double rn;
    if (!initflag) init();
    while (1) {
      if (++inext == 56) inext = 1;
      if (++inextp == 56) inextp = 1;
      mj = ma[inext] - ma[inextp];
      if (mj < 0) mj += MBIG;
      ma[inext] = mj;
      rn = mj*FAC;
      if (rn > 0.0 && rn < 1.0) break;
    }
    return rn;
  }

  void init();
  void reset(int s) { seed = s; initflag = 0; save = 0; }

 private:
  int seed, save;
  double second;
  int initflag, inext, inextp;
  int ma[56];
};

void RanKnuth::init()
{
  int i, ii, k, mj, mk;
  initflag = 1;
  mj = labs(MSEED - labs(seed));
  mj %= MBIG;
  ma[55] = mj;
  mk = 1;
  for (i = 1; i <= 54; i++) {
    ii = (21*i) % 55;
    ma[ii] = mk;
    mk = mj - mk;
    if (mk < 0) mk += MBIG;
    mj = ma[ii];
  }
  for (k = 0; k < 4; k++)
    for (i = 1; i <= 55; i++) {
      ma[i] -= ma[1 + (i+30) % 55];
      if (ma[i] < 0) ma[i] += MBIG;
    }
  inext = 0;
  inextp = 31;
}

/* deliberately in its own noinline body to model the cross-TU call */
__attribute__((noinline)) double RanKnuth::uniform_outofline()
{
  return uniform_inline();
}

/* ---------------- fast pow, for V5 ---------------- */

/* pow(x,p) for x>0 via exp2(p*log2(x)). glibc's log2/exp2 are correctly
   rounded and much cheaper than the generic pow path. */
static inline double fast_pow(double x, double p)
{
  return exp2(p * log2(x));
}

/* ---------------- shared problem setup ---------------- */

struct Problem {
  int ncell;
  int npercell;
  std::vector<OnePart> particles;
  std::vector<int> next;          /* per-cell linked list, as Particle::next */
  std::vector<int> first;         /* cinfo.first */
  std::vector<int> count;         /* cinfo.count */
  std::vector<double> volume;     /* cinfo.volume */
  std::vector<double> vremax;     /* vremax[icell][0][0] */
  std::vector<double> remain;
  Species species[1];
  Params params_storage[1];
  Params **params;
  double *prefactor_storage;
  double **prefactor;
  double dt, fnum;

  /* scramble: 0 = cell particles contiguous and in order (post-reorder),
     1 = shuffled through memory (the un-reordered steady state) */
  void build(int ncell_, int npercell_, int scramble, unsigned seed);
};

void Problem::build(int ncell_, int npercell_, int scramble, unsigned seed)
{
  ncell = ncell_;
  npercell = npercell_;
  int n = ncell * npercell;

  particles.resize(n);
  next.resize(n);
  first.resize(ncell);
  count.resize(ncell);
  volume.resize(ncell);
  vremax.resize(ncell);
  remain.resize(ncell);

  /* Ar, from bench/ar.species and bench/ar.vss */
  species[0].mass = 6.63e-26;
  species[0].rotdof = 0.0;
  species[0].vibdof = 0.0;

  params_storage[0].diam  = 4.11e-10;
  params_storage[0].omega = 0.81;
  params_storage[0].tref  = 273.15;
  params_storage[0].alpha = 1.4;
  params_storage[0].mr    = species[0].mass * species[0].mass /
                            (species[0].mass + species[0].mass);
  static Params *prow;
  prow = &params_storage[0];
  params = &prow;

  /* prefactor as CollideVSS::init computes it, collapsed to one species */
  const double kb = 1.380658e-23;
  double d = params_storage[0].diam;
  double om = params_storage[0].omega;
  double tr = params_storage[0].tref;
  double mr = params_storage[0].mr;
  static double pf;
  pf = MY_PI * d*d * pow(4.0*kb*tr/mr, om-0.5) / tgamma(2.5-om);
  static double *pfp;
  pfp = &pf;
  prefactor = &pfp;

  dt = 7.0e-9;
  fnum = 7.07043e6;
  double cellL = 1.0e-5;
  double vol = cellL*cellL*cellL;

  /* Maxwellian at 273.15 K, generated deterministically */
  RanKnuth rng((int)seed);
  double vth = sqrt(2.0 * kb * 273.15 / species[0].mass);

  std::vector<int> perm(n);
  for (int i = 0; i < n; i++) perm[i] = i;
  if (scramble) {
    /* deterministic shuffle standing in for particles that have drifted out
       of cell order since the last reorder */
    for (int i = n-1; i > 0; i--) {
      int j = (int)(rng.uniform_inline() * (i+1));
      std::swap(perm[i], perm[j]);
    }
  }

  /* vremax is sigma*v_r, not a velocity: CollideVSS::vremax_init sets it to
     2 * pi*d^2 * vscale, which for Ar at 273.15 K is ~3.6e-16 and yields
     ~0.9 collision attempts per cell per step, matching the real benchmark */
  double cxs = MY_PI * d * d;
  double beta = sqrt(2.0 * kb * 273.15 / species[0].mass);
  double vrm_init = 2.0 * cxs * beta;

  for (int icell = 0; icell < ncell; icell++) {
    volume[icell] = vol;
    vremax[icell] = vrm_init;
    remain[icell] = 0.0;
    count[icell] = npercell;
    first[icell] = perm[icell*npercell];
    for (int k = 0; k < npercell; k++) {
      int idx = perm[icell*npercell + k];
      OnePart &p = particles[idx];
      p.id = idx; p.ispecies = 0; p.icell = icell; p.flag = 0;
      p.x[0] = p.x[1] = p.x[2] = 0.0;
      /* Box-Muller from the same stream, so V0..V5 see identical data */
      for (int c = 0; c < 3; c++) {
        double u1 = rng.uniform_inline(), u2 = rng.uniform_inline();
        p.v[c] = vth * sqrt(-log(u1)) * cos(2.0*MY_PI*u2) * 0.7071067811865476;
      }
      p.erot = p.evib = 0.0;
      p.dtremain = 0.0; p.weight = 1.0;
      next[idx] = (k == npercell-1) ? -1 : perm[icell*npercell + k + 1];
    }
  }
}

/* ---------------- V0: the shipped structure ---------------- */

class CollideBase {
 public:
  Problem *pr;
  RanKnuth *random;
  State precoln, postcoln;
  std::vector<int> plist;
  long ncollide_one, nattempt_one;

  CollideBase(Problem *p, RanKnuth *r) : pr(p), random(r) {
    plist.resize(p->npercell * 4);
    ncollide_one = nattempt_one = 0;
  }
  virtual ~CollideBase() {}

  virtual double attempt_collision(int icell, int np, double volume) = 0;
  virtual int test_collision(int icell, int ig, int jg,
                             OnePart *ip, OnePart *jp) = 0;
  virtual void setup_collision(OnePart *ip, OnePart *jp) = 0;
  virtual int perform_collision(OnePart *&ip, OnePart *&jp, OnePart *&kp) = 0;

  void collisions_one();
};

class CollideVSSm : public CollideBase {
 public:
  CollideVSSm(Problem *p, RanKnuth *r) : CollideBase(p, r) {}

  double attempt_collision(int icell, int np, double volume) override
  {
    double fnum = pr->fnum;
    double dt = pr->dt;
    double nattempt = 0.5 * np * (np-1) *
      pr->vremax[icell] * dt * fnum / volume + random->uniform_outofline();
    return nattempt;
  }

  int test_collision(int icell, int igroup, int jgroup,
                     OnePart *ip, OnePart *jp) override
  {
    double *vi = ip->v;
    double *vj = jp->v;
    int ispecies = ip->ispecies;
    int jspecies = jp->ispecies;
    double du = vi[0]-vj[0], dv = vi[1]-vj[1], dw = vi[2]-vj[2];
    double vr2 = du*du + dv*dv + dw*dw;
    if (vr2 < EPSZERO && pr->params[ispecies][jspecies].omega >= 1.0) return 0;
    double vro = pow(vr2, 1.0 - pr->params[ispecies][jspecies].omega);
    double vre = vro * pr->prefactor[ispecies][jspecies];
    pr->vremax[icell] = MAX(vre, pr->vremax[icell]);
    if (vre / pr->vremax[icell] < random->uniform_outofline()) return 0;
    precoln.vr2 = vr2;
    return 1;
  }

  void setup_collision(OnePart *ip, OnePart *jp) override
  {
    Species *species = pr->species;
    int isp = ip->ispecies, jsp = jp->ispecies;
    precoln.vr = sqrt(precoln.vr2);
    precoln.ave_rotdof = 0.5 * (species[isp].rotdof + species[jsp].rotdof);
    precoln.ave_vibdof = 0.5 * (species[isp].vibdof + species[jsp].vibdof);
    precoln.ave_dof = (precoln.ave_rotdof + precoln.ave_vibdof)/2.0;
    double imass = precoln.imass = species[isp].mass;
    double jmass = precoln.jmass = species[jsp].mass;
    precoln.etrans = 0.5 * pr->params[isp][jsp].mr * precoln.vr2;
    precoln.erot = ip->erot + jp->erot;
    precoln.evib = ip->evib + jp->evib;
    precoln.eint = precoln.erot + precoln.evib;
    precoln.etotal = precoln.etrans + precoln.eint;
    double divisor = 1.0 / (imass + jmass);
    double *vi = ip->v, *vj = jp->v;
    precoln.ucmf = ((imass*vi[0]) + (jmass*vj[0])) * divisor;
    precoln.vcmf = ((imass*vi[1]) + (jmass*vj[1])) * divisor;
    precoln.wcmf = ((imass*vi[2]) + (jmass*vj[2])) * divisor;
    postcoln.etrans = precoln.etrans;
    postcoln.erot = 0.0;
    postcoln.evib = 0.0;
    postcoln.eint = 0.0;
    postcoln.etotal = precoln.etotal;
  }

  int perform_collision(OnePart *&ip, OnePart *&jp, OnePart *&kp) override
  {
    /* no react, and Ar has ave_dof == 0, so this is the whole body */
    SCATTER_TwoBodyScattering(ip, jp);
    return 0;
  }

  void SCATTER_TwoBodyScattering(OnePart *ip, OnePart *jp)
  {
    double ua, vb, wc;
    double vrc[3];
    Species *species = pr->species;
    double *vi = ip->v, *vj = jp->v;
    int isp = ip->ispecies, jsp = jp->ispecies;
    double mass_i = species[isp].mass, mass_j = species[jsp].mass;
    double alpha_r = 1.0 / pr->params[isp][jsp].alpha;
    double eps = random->uniform_outofline() * 2*MY_PI;
    if (fabs(alpha_r - 1.0) < 0.001) {
      double vr = sqrt(2.0 * postcoln.etrans / pr->params[isp][jsp].mr);
      double cosX = 2.0*random->uniform_outofline() - 1.0;
      double sinX = sqrt(1.0 - cosX*cosX);
      ua = vr*cosX; vb = vr*sinX*cos(eps); wc = vr*sinX*sin(eps);
    } else {
      double scale = sqrt((2.0 * postcoln.etrans) /
                          (pr->params[isp][jsp].mr * precoln.vr2));
      double cosX = 2.0*pow(random->uniform_outofline(), alpha_r) - 1.0;
      double sinX = sqrt(1.0 - cosX*cosX);
      vrc[0] = vi[0]-vj[0]; vrc[1] = vi[1]-vj[1]; vrc[2] = vi[2]-vj[2];
      double d = sqrt(vrc[1]*vrc[1] + vrc[2]*vrc[2]);
      if (d > 1.0e-6) {
        ua = scale * ( cosX*vrc[0] + sinX*d*sin(eps) );
        vb = scale * ( cosX*vrc[1] + sinX*(precoln.vr*vrc[2]*cos(eps) -
                                           vrc[0]*vrc[1]*sin(eps))/d );
        wc = scale * ( cosX*vrc[2] - sinX*(precoln.vr*vrc[1]*cos(eps) +
                                           vrc[0]*vrc[2]*sin(eps))/d );
      } else {
        ua = scale * ( cosX*vrc[0] );
        vb = scale * ( sinX*vrc[0]*cos(eps) );
        wc = scale * ( sinX*vrc[0]*sin(eps) );
      }
    }
    double divisor = 1.0 / (mass_i + mass_j);
    vi[0] = precoln.ucmf + (mass_j*divisor)*ua;
    vi[1] = precoln.vcmf + (mass_j*divisor)*vb;
    vi[2] = precoln.wcmf + (mass_j*divisor)*wc;
    vj[0] = precoln.ucmf - (mass_i*divisor)*ua;
    vj[1] = precoln.vcmf - (mass_i*divisor)*vb;
    vj[2] = precoln.wcmf - (mass_i*divisor)*wc;
  }
};

void CollideBase::collisions_one()
{
  OnePart *particles = pr->particles.data();
  int *next = pr->next.data();
  OnePart *ipart, *jpart, *kpart;

  for (int icell = 0; icell < pr->ncell; icell++) {
    int np = pr->count[icell];
    if (np <= 1) continue;
    int ip = pr->first[icell];
    double volume = pr->volume[icell];

    int n = 0;
    while (ip >= 0) { plist[n++] = ip; ip = next[ip]; }

    double attempt = attempt_collision(icell, np, volume);
    int nattempt = (int) attempt;
    if (!nattempt) continue;
    nattempt_one += nattempt;

    for (int iattempt = 0; iattempt < nattempt; iattempt++) {
      int i = np * random->uniform_outofline();
      int j = np * random->uniform_outofline();
      while (i == j) j = np * random->uniform_outofline();
      ipart = &particles[plist[i]];
      jpart = &particles[plist[j]];
      if (!test_collision(icell, 0, 0, ipart, jpart)) continue;
      setup_collision(ipart, jpart);
      perform_collision(ipart, jpart, kpart);
      ncollide_one++;
    }
  }
}

/* ---------------- V1-V5: one fused, inlinable kernel ---------------- */

/* Template flags select the cumulative variants so each version is compiled
   as its own straight-line kernel with no runtime branching. */
template <int INLINE_RNG, int HOIST, int CONTIG, int FASTPOW>
struct FusedCollide {
  Problem *pr;
  RanKnuth *random;
  State precoln, postcoln;
  std::vector<int> plist;
  long ncollide_one, nattempt_one;

  FusedCollide(Problem *p, RanKnuth *r) : pr(p), random(r) {
    plist.resize(p->npercell * 4);
    ncollide_one = nattempt_one = 0;
  }

  inline double rn() {
    return INLINE_RNG ? random->uniform_inline() : random->uniform_outofline();
  }

  inline double vpow(double x, double p) {
    return FASTPOW ? fast_pow(x, p) : pow(x, p);
  }

  void collisions_one()
  {
    OnePart *particles = pr->particles.data();
    int *next = pr->next.data();
    Species *species = pr->species;

    /* hoisted out of the cell loop: single species, so these are constants */
    const Params &prm = pr->params[0][0];
    const double pref = pr->prefactor[0][0];
    const double omega1 = 1.0 - prm.omega;
    const double mr = prm.mr;
    const double alpha_r = 1.0 / prm.alpha;
    const double mass_i = species[0].mass, mass_j = species[0].mass;
    const double divisor = 1.0 / (mass_i + mass_j);
    const double dtfnum = pr->dt * pr->fnum;

    for (int icell = 0; icell < pr->ncell; icell++) {
      int np = pr->count[icell];
      if (np <= 1) continue;
      double volume = pr->volume[icell];

      const int *pl;
      int firstidx = pr->first[icell];
      if (CONTIG) {
        pl = NULL;                        /* indices are firstidx..firstidx+np-1 */
      } else {
        int ip = firstidx, n = 0;
        while (ip >= 0) { plist[n++] = ip; ip = next[ip]; }
        pl = plist.data();
      }

      /* vremax kept in a register across the cell, written back once */
      double vremax_c = pr->vremax[icell];

      double attempt;
      if (HOIST)
        attempt = 0.5 * np * (np-1) * vremax_c * dtfnum / volume + rn();
      else
        attempt = 0.5 * np * (np-1) * pr->vremax[icell] * pr->dt * pr->fnum
                  / volume + rn();
      int nattempt = (int) attempt;
      if (!nattempt) { if (HOIST) pr->vremax[icell] = vremax_c; continue; }
      nattempt_one += nattempt;

      for (int iattempt = 0; iattempt < nattempt; iattempt++) {
        int i = np * rn();
        int j = np * rn();
        while (i == j) j = np * rn();

        OnePart *ipart, *jpart;
        if (CONTIG) {
          ipart = &particles[firstidx + i];
          jpart = &particles[firstidx + j];
        } else {
          ipart = &particles[pl[i]];
          jpart = &particles[pl[j]];
        }

        /* --- test_collision --- */
        double *vi = ipart->v, *vj = jpart->v;
        double du = vi[0]-vj[0], dv = vi[1]-vj[1], dw = vi[2]-vj[2];
        double vr2 = du*du + dv*dv + dw*dw;
        if (vr2 < EPSZERO && prm.omega >= 1.0) continue;
        double vro = vpow(vr2, omega1);
        double vre = vro * pref;
        if (HOIST) {
          vremax_c = MAX(vre, vremax_c);
          if (vre / vremax_c < rn()) continue;
        } else {
          pr->vremax[icell] = MAX(vre, pr->vremax[icell]);
          if (vre / pr->vremax[icell] < rn()) continue;
        }
        precoln.vr2 = vr2;

        /* --- setup_collision --- */
        precoln.vr = sqrt(vr2);
        precoln.ave_rotdof = 0.0;
        precoln.ave_vibdof = 0.0;
        precoln.ave_dof = 0.0;
        precoln.imass = mass_i;
        precoln.jmass = mass_j;
        precoln.etrans = 0.5 * mr * vr2;
        precoln.erot = ipart->erot + jpart->erot;
        precoln.evib = ipart->evib + jpart->evib;
        precoln.eint = precoln.erot + precoln.evib;
        precoln.etotal = precoln.etrans + precoln.eint;
        precoln.ucmf = ((mass_i*vi[0]) + (mass_j*vj[0])) * divisor;
        precoln.vcmf = ((mass_i*vi[1]) + (mass_j*vj[1])) * divisor;
        precoln.wcmf = ((mass_i*vi[2]) + (mass_j*vj[2])) * divisor;
        postcoln.etrans = precoln.etrans;
        postcoln.erot = 0.0;
        postcoln.evib = 0.0;
        postcoln.eint = 0.0;
        postcoln.etotal = precoln.etotal;

        /* --- SCATTER_TwoBodyScattering (alpha != 1 branch for Ar) --- */
        double ua, vb, wc, vrc[3];
        double eps = rn() * 2*MY_PI;
        double scale = sqrt((2.0 * postcoln.etrans) / (mr * vr2));
        double cosX = 2.0*vpow(rn(), alpha_r) - 1.0;
        double sinX = sqrt(1.0 - cosX*cosX);
        vrc[0] = du; vrc[1] = dv; vrc[2] = dw;
        double d = sqrt(vrc[1]*vrc[1] + vrc[2]*vrc[2]);
        double sineps = sin(eps), coseps = cos(eps);
        if (d > 1.0e-6) {
          ua = scale * ( cosX*vrc[0] + sinX*d*sineps );
          vb = scale * ( cosX*vrc[1] + sinX*(precoln.vr*vrc[2]*coseps -
                                             vrc[0]*vrc[1]*sineps)/d );
          wc = scale * ( cosX*vrc[2] - sinX*(precoln.vr*vrc[1]*coseps +
                                             vrc[0]*vrc[2]*sineps)/d );
        } else {
          ua = scale * ( cosX*vrc[0] );
          vb = scale * ( sinX*vrc[0]*coseps );
          wc = scale * ( sinX*vrc[0]*sineps );
        }
        vi[0] = precoln.ucmf + (mass_j*divisor)*ua;
        vi[1] = precoln.vcmf + (mass_j*divisor)*vb;
        vi[2] = precoln.wcmf + (mass_j*divisor)*wc;
        vj[0] = precoln.ucmf - (mass_i*divisor)*ua;
        vj[1] = precoln.vcmf - (mass_i*divisor)*vb;
        vj[2] = precoln.wcmf - (mass_i*divisor)*wc;

        ncollide_one++;
      }

      if (HOIST) pr->vremax[icell] = vremax_c;
    }
  }
};

/* ---------------- harness ---------------- */

struct Result { double secs; long ncoll, natt; double checksum; };

static double checksum_of(const Problem &pr)
{
  double s = 0.0;
  for (size_t i = 0; i < pr.particles.size(); i++) {
    const OnePart &p = pr.particles[i];
    s += p.v[0]*1.0 + p.v[1]*2.0 + p.v[2]*3.0;
  }
  return s;
}

int main(int argc, char **argv)
{
  int ncell     = (argc > 1) ? atoi(argv[1]) : 10000;
  int npercell  = (argc > 2) ? atoi(argv[2]) : 10;
  int nsteps    = (argc > 3) ? atoi(argv[3]) : 20;
  int scramble  = (argc > 4) ? atoi(argv[4]) : 0;

  printf("# micro_collide: ncell=%d npercell=%d nsteps=%d scramble=%d "
         "(%d particles, %.1f MB)\n",
         ncell, npercell, nsteps, scramble, ncell*npercell,
         ncell*npercell*sizeof(OnePart)/1048576.0);

  Result res[6];
  const char *names[6] = {"V0 baseline (virtual, outofline RNG, plist)",
                          "V1 +devirtualized/inlined",
                          "V2 +inline RNG",
                          "V3 +hoisted invariants",
                          "V4 +contiguous (no plist)",
                          "V5 +fast pow (NUMERICS CHANGE)"};

  for (int v = 0; v < 6; v++) {
    Problem pr;
    pr.build(ncell, npercell, scramble, 12345);
    RanKnuth rng(54321);
    rng.init();

    double t0 = wtime();
    long nc = 0, na = 0;
    if (v == 0) {
      CollideVSSm c(&pr, &rng);
      for (int s = 0; s < nsteps; s++) c.collisions_one();
      nc = c.ncollide_one; na = c.nattempt_one;
    } else if (v == 1) {
      FusedCollide<0,0,0,0> c(&pr, &rng);
      for (int s = 0; s < nsteps; s++) c.collisions_one();
      nc = c.ncollide_one; na = c.nattempt_one;
    } else if (v == 2) {
      FusedCollide<1,0,0,0> c(&pr, &rng);
      for (int s = 0; s < nsteps; s++) c.collisions_one();
      nc = c.ncollide_one; na = c.nattempt_one;
    } else if (v == 3) {
      FusedCollide<1,1,0,0> c(&pr, &rng);
      for (int s = 0; s < nsteps; s++) c.collisions_one();
      nc = c.ncollide_one; na = c.nattempt_one;
    } else if (v == 4) {
      if (scramble) { res[v].secs = -1; continue; }   /* only valid if contiguous */
      FusedCollide<1,1,1,0> c(&pr, &rng);
      for (int s = 0; s < nsteps; s++) c.collisions_one();
      nc = c.ncollide_one; na = c.nattempt_one;
    } else {
      FusedCollide<1,1,(0),1> c(&pr, &rng);
      for (int s = 0; s < nsteps; s++) c.collisions_one();
      nc = c.ncollide_one; na = c.nattempt_one;
    }
    double t1 = wtime();
    res[v].secs = t1 - t0;
    res[v].ncoll = nc;
    res[v].natt = na;
    res[v].checksum = checksum_of(pr);
  }

  printf("%-46s  %10s  %8s  %10s  %s\n",
         "variant", "seconds", "speedup", "ncollide", "identical");
  for (int v = 0; v < 6; v++) {
    if (res[v].secs < 0) { printf("%-46s  %10s\n", names[v], "n/a"); continue; }
    bool same = (res[v].ncoll == res[0].ncoll) &&
                (res[v].natt == res[0].natt) &&
                (res[v].checksum == res[0].checksum);
    printf("%-46s  %10.4f  %8.3fx  %10ld  %s\n",
           names[v], res[v].secs, res[0].secs/res[v].secs, res[v].ncoll,
           same ? "yes" : "NO");
  }

  /* rate, for the roofline */
  printf("\ncollide attempts: %ld, collisions: %ld over %d steps\n",
         res[0].natt, res[0].ncoll, nsteps);
  printf("V0 ns/attempt: %.2f   best ns/attempt: %.2f\n",
         1e9*res[0].secs/res[0].natt,
         1e9*res[4].secs/(res[4].secs > 0 ? res[4].natt : 1));
  return 0;
}
