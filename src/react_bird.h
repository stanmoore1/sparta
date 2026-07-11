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

#ifndef SPARTA_REACT_BIRD_H
#define SPARTA_REACT_BIRD_H

#include "stdio.h"
#include "math.h"
#include "react.h"
#include "particle.h"

namespace SPARTA_NS {

class ReactBird : public React {
 public:
  ReactBird(class SPARTA *, int, char **);
  ReactBird(class SPARTA *);
  virtual ~ReactBird();
  virtual void init();
  int recomb_exist(int, int);
  void ambi_check();
  virtual int attempt(Particle::OnePart *, Particle::OnePart *,
                      double, double, double, double, double &, int &) = 0;
  char *reactionID(int);
  virtual double extract_tally(int);

  // per-reaction tables of the microcanonical TCE energy factor over the
  // joint vibrational (x electronic) ladder, for react_modify
  // vib_energy micro; built by build_micro_tables() at init, shared by
  // the CPU and Kokkos TCE variants

  double **mtab;            // factor vs total collision energy, per rlist
                            // entry; NULL if the reactants have no ladders
  double *mtab_du;          // energy grid spacing (J) per table
  int *mtab_n;              // # of grid points per table
  int mtab_nlist;           // # of rlist entries tables were built for

  // 3-body detailed-balance table of a reverse recombination: the
  // calibrated forward numerator divided by the flat measure of the
  // (pair energy, third-body energy) decomposition, as a function of
  // the total available energy w = u + e3; the pair density of states
  // x_AB(u) is stored in the reaction's mtab slot; both share the mtab
  // grid; NULL for all other reactions (see build_db3_table)

  double **mtab_num;

  void build_micro_tables();
  void free_micro_tables();

  inline double vib_micro_factor(int ireact, double ecc) const {
    const double *tab = mtab[ireact];
    const int n = mtab_n[ireact];
    const double x = ecc / mtab_du[ireact];
    const int k = (int) x;
    if (k >= n-1) return tab[n-1];
    const double f = x - k;
    return (1.0-f)*tab[k] + f*tab[k+1];
  }

  inline double db3_num_factor(int ireact, double ew) const {
    const double *tab = mtab_num[ireact];
    const int n = mtab_n[ireact];
    const double x = ew / mtab_du[ireact];
    const int k = (int) x;
    if (k >= n-1) return tab[n-1];
    const double f = x - k;
    return (1.0-f)*tab[k] + f*tab[k+1];
  }

 protected:
  FILE *fp;

  // tallies for reactions

  bigint *tally_reactions,*tally_reactions_all;
  int tally_flag;

  struct OneReaction {
    int active;                    // 1 if reaction is active
    int initflag;                  // 1 if reaction params have been init
    int type;                      // reaction type = DISSOCIATION, etc
    int style;                     // reaction style = ARRHENIUS, etc
    int ncoeff;                    // # of numerical coeffs
    int nreactant,nproduct;        // # of reactants and products
    char **id_reactants,**id_products;  // species IDs of reactants/products
    int *reactants,*products;      // species indices of reactants/products
    double *coeff;                 // numerical coeffs for reaction
    char *id;                      // reaction ID (formula)
    int reverse;                   // 1 if backward rate is derived from a
                                   //   forward reaction by detailed balance
                                   //   (reaction style 'B'), else 0
    int reverse_partner;           // rlist index of the forward reaction whose
                                   //   Arrhenius rate this reverse rate derives
                                   //   from, or -1 if none found
    double reverse_bf;             // temperature exponent of the forward
                                   //   reaction, applied at the cell
                                   //   temperature in the backward
                                   //   recombination prefactor
    double reverse_A;              // raw Arrhenius prefactor of the forward
                                   //   reaction (stashed before the TCE
                                   //   transform), used to calibrate the
                                   //   detailed-balance table of a B-style
                                   //   exchange reaction
    int generated;                 // 1 if this reverse reaction was
                                   //   auto-generated (react_modify
                                   //   reverse auto), else 0
    int keq_flag;                  // 1 if this reverse reaction uses an
                                   //   external equilibrium-constant curve
                                   //   fit (react_modify keq_file), else 0
    double keq_coeff[5];           // Park-form fit coefficients:
                                   //   ln Keq = c0/Z + c1 + c2 ln(Z) +
                                   //   c3 Z + c4 Z^2 with Z = 10000 K / T
    double reverse_dEa;            // Ea_F - seeded Ea_B: exponential shift
                                   //   between the forward barrier and the
                                   //   (clamped) backward barrier, used by
                                   //   the external-Keq prefactor
  };

  OneReaction *rlist;              // list of all reactions read from file
  int maxlist;                     // max # of reactions in rlist

  // all reactions a pair of reactant species is part of

  struct ReactionIJ {
    int *list;       // N-length list of rlist indices
                     //   for reactions defined for this IJ pair,
                     //   just a ptr into sub-section of long list_ij vector
                     //   for all pairs
    int *sp2recomb;  // Nspecies-length list of rlist indices
                     //   for recomb reactions defined for this IJ pair,
                     //   one index for all 3rd particle species,
                     //   just a ptr into sub-section of long sp2recomb_ij
                     //   vector for all pairs which have recomb reactions
    int n;           // # of reactions in list
  };

  ReactionIJ **reactions;     // reaction info for all IJ pairs of species
  int *list_ij;               // chunks of rlist indices,
                              //   one chunk per IJ pair,
                              //   stored in contiguous vector,
                              //   length of each chunk is # of IJ reactions
                              // pointed into by reactions[i][k].list
  int *sp2recomb_ij;          // chunks of rlist indices,
                              //   one chunk per IJ pair that has
                              //     recombination reactions,
                              //   stored in contiguous vector
                              //   length of each chunk is # of species
                              // pointed into by reactions[i][k].sp2recomb

  // equilibrium-constant curve fits read from react_modify keq_file

  struct KeqFit {
    int reactants[2];                // species indices of the FORWARD
    int products[3];                 //   reaction the fit belongs to
    int nreactant,nproduct;
    double coeff[5];                 // Park-form coefficients
    int used;                        // 1 once matched to a reaction
  };

  KeqFit *keqfits;
  int nkeqfits;

  int generated_flag;              // 1 once auto-reverses were generated

  void readfile(char *);
  int readone(char *, char *, int &, int &);
  void check_duplicate();
  void check_tce_bounds();
  virtual void grow_tallies();
  double partition_function(int, double);
  void build_db_table(int);
  void build_db3_table(int);
  void generate_reverses();
  void read_keq_file();
  void assign_keq_fits();

  inline double keq_eval(const double *c, double T) const {
    double z = 10000.0/T;
    return exp(c[0]/z + c[1] + c[2]*log(z) + c[3]*z + c[4]*z*z);
  }
  void print_reaction(char *, char *);
  void print_reaction(OneReaction *);
  void print_reaction_ambipolar(OneReaction *);
};

}

#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: React tce can only be used with collide vss

Self-explanatory.

E: Ionization and recombination reactions are not yet implemented

This error conditions will be removed after those reaction styles are
fully implemented.

E: Unknown outcome in reaction

The specified type of the reaction is not encoded in the reaction
style.

E: Cannot open reaction file %s

Self-explanatory.

E: Invalid reaction formula in file

Self-explanatory.

E: Invalid reaction type in file

Self-explanatory.

E: Invalid reaction style in file

Self-explanatory.

E: Invalid reaction coefficients in file

Self-explanatory.

*/
