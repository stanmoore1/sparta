/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
   Steve Plimpton, sjplimp@sandia.gov, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#include "string.h"
#include "stdlib.h"
#include "compute_fft_grid_kokkos.h"
#include "update.h"
#include "domain.h"
#include "grid.h"
#include "modify.h"
#include "fix.h"
#include "input.h"
#include "variable.h"
#include "irregular.h"
#include "fft3d_wrap.h"
#include "fft2d_wrap.h"
#include "memory.h"
#include "error.h"

#include <string>
#include <fstream>
#include <cassert>
#include <iostream>

#ifdef SPARTA_MAP
#include <map>
#elif defined SPARTA_UNORDERED_MAP
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif

using namespace SPARTA_NS;

enum{COMPUTE,FIX,VARIABLE};

#define INVOKED_PER_GRID 16

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF  1.0f
#else
#define ZEROF 0.0
#define ONEF  1.0
#endif


/* ---------------------------------------------------------------------- */

ComputeFFTGridKokkos::ComputeFFTGridKokkos(SPARTA *sparta, int narg, char **arg) :
  ComputeFFTGrid(sparta, narg, arg){

  // kokkos dualview allocator instead
  memory->destroy(fft);
  memory->destroy(fftwork);

  memory->create_kokkos(k_fft, fft, 2*nfft, "fft/grid:fft");
  memory->create_kokkos(k_fftwork, fftwork, nfft, "fft/grid:fftwork");

  nglocal = 0;
  reallocate();

}

/* ---------------------------------------------------------------------- */

ComputeFFTGridKokkos::~ComputeFFTGridKokkos()
{

  if (copymode) return;

  memory->destroy_kokkos(k_vector_grid, vector_grid);
  memory->destroy_kokkos(k_array_grid, array_grid);
  vector_grid = NULL;
  array_grid = NULL;

  memory->destroy_kokkos(k_fft, fft);
  memory->destroy_kokkos(k_fftwork, fftwork);
  fft = NULL;
  fftwork = NULL;

  memory->destroy_kokkos(k_ingrid, ingrid);
  memory->destroy_kokkos(k_gridwork, gridwork);
  memory->destroy_kokkos(k_gridworkcomplex, gridworkcomplex);
  ingrid = NULL;
  gridwork = NULL;
  gridworkcomplex = NULL;

  memory->destroy_kokkos(k_map1, map1);
  memory->destroy_kokkos(k_map2, map2);
  map1 = NULL;
  map2 = NULL;

}

/* ---------------------------------------------------------------------- */

void ComputeFFTGridKokkos::compute_per_grid()
{
  int i,j,m;
  double *ingridptr;

  invoked_per_grid = update->ntimestep;

  // check that grid has not adapted
  // NOTE: also need to check it has not been re-balanced?

  if (grid->maxlevel != 1) 
    error->all(FLERR,"Compute fft/grid require uniform one-level grid");

  // if sumflag set, zero output vector/array, but not K-space indices
  // so can sum each value's result into it

  if (sumflag) {
    if (ncol == 1) {
      for (i = 0; i < nglocal; i++)
        vector_grid[i] = 0.0;
    } else {
      for (i = 0; i < nglocal; i++)
        for (m = startcol; m < ncol; m++)
          array_grid[i][m] = 0.0;
    }
  }

  // process values, one FFT per value
  for (m = 0; m < nvalues; m++) {

    int vidx = value2index[m];
    int aidx = argindex[m];

    // invoke compute if not previously invoked
    // for per-grid compute, invoke post_process_grid() if necessary

    if (which[m] == COMPUTE) {
      Compute *c = modify->compute[vidx];

      if (!(c->invoked_flag & INVOKED_PER_GRID)) {
        c->compute_per_grid();
        c->invoked_flag |= INVOKED_PER_GRID;
      }

      if (c->post_process_grid_flag) 
        c->post_process_grid(aidx,-1,1,NULL,NULL,NULL,1);
      
      if (aidx == 0 || c->post_process_grid_flag) {
        ingridptr = c->vector_grid;
      } else {
        k_tmp = c->d_array_grid;
        offset = aidx - 1;

        copymode = 1;
        Kokkos::parallel_for(copy_from_tmp(0, grid->nlocal), *this);
        SPADeviceType::fence();
        copymode = 0;

        k_ingrid.modify< SPADeviceType >();
        k_ingrid.sync< SPAHostType >();

        ingridptr = ingrid;

      }

    // access fix fields, check if fix frequency is a match

    } else if (which[m] == FIX) {
      Fix *fix = modify->fix[vidx];

      if (update->ntimestep % modify->fix[vidx]->per_grid_freq)
        error->all(FLERR,"Fix used in compute fft/grid not "
                   "computed at compatible time");
      if (aidx == 0) {
        ingridptr = fix->vector_grid;
      } else {
        k_tmp = fix->k_array_grid.d_view;
        offset = aidx - 1;

        copymode = 1;
        Kokkos::parallel_for(copy_from_tmp(0, grid->nlocal), *this);
        SPADeviceType::fence();
        copymode = 0;

        k_ingrid.modify< SPADeviceType >();
        k_ingrid.sync< SPAHostType >();
        
        ingridptr = ingrid;

      }

    // evaluate particle-style or grid-style variable

    } else if (which[m] == VARIABLE) {
      input->variable->compute_grid(vidx,ingrid,1,0);
      ingridptr = ingrid;
    }

    // ------------------------------
    // perform FFT on inbufptr values
    // ------------------------------

    // irregular comm to move grid values from SPARTA owners -> FFT owners

    irregular1->exchange_uniform((char *) ingridptr,sizeof(double),
                                 (char *) fftwork);

    // copy new fftwork values to device
    k_fftwork.modify< SPAHostType >();
    k_fftwork.sync< SPADeviceType >();

    // use them to fill the fft buffer on the device
    copymode = 1;
    Kokkos::parallel_for(fill_fft(0, nfft), *this);
    SPADeviceType::fence();
    copymode = 0;

    // then copy the buffer contents back to the host
    k_fft.modify< SPADeviceType >();
    k_fft.sync< SPAHostType >();

    // perform FFT
    if (dimension == 3) fft3d->compute(fft,fft,1);
    else fft2d->compute(fft,fft,1);

    // update device values
    k_fft.modify< SPAHostType >();
    k_fft.sync< SPADeviceType >();

    // reverse irregular comm to move results from FFT grid -> SPARTA grid
    // if conjugate set:
    //   convert complex FFT datums back to real via c times c*
    //   comm single floating point value per grid cell
    // else: comm entire complex value
    // copy or sum received values into output vec or array via map2

    if (conjugate) {

      copymode = 1;
      Kokkos::parallel_for(compute_norm_sq(0, nfft), *this);
      SPADeviceType::fence();
      copymode = 0;

      // update device values
      k_fftwork.modify< SPAHostType >();
      k_fftwork.sync< SPADeviceType >();

      irregular2->exchange_uniform((char *) fftwork,sizeof(double),
                                   (char *) gridwork);

      // update device values
      k_fftwork.modify< SPAHostType >();
      k_fftwork.sync< SPADeviceType >();

      copymode = 1;
      offset = m;
      Kokkos::parallel_for(update_conjugate(0, grid->nlocal), *this);
      SPADeviceType::fence();
      copymode = 0;

      if (ncol == 1) {
        k_vector_grid.modify< SPADeviceType >();
        k_vector_grid.sync< SPAHostType >();
      } else {
        k_array_grid.modify< SPADeviceType >();
        k_array_grid.sync< SPAHostType >();
      }

    } else {

      irregular2->exchange_uniform((char *) fft, 2*sizeof(FFT_SCALAR),
                                   (char *) gridworkcomplex);

      // update device values
      k_gridworkcomplex.modify< SPAHostType >();
      k_gridworkcomplex.sync< SPADeviceType >();

      copymode = 1;
      offset = m;
      Kokkos::parallel_for(update_complex(0, grid->nlocal), *this);
      SPADeviceType::fence();
      copymode = 0;

      // copy back to host
      k_array_grid.modify< SPADeviceType >();
      k_array_grid.sync< SPAHostType >();

    }

  }

  // scale results if requested
  if (scalefactor != 1.0) {

    copymode = 1;
    Kokkos::parallel_for(scale_grid(0, nglocal), *this);
    SPADeviceType::fence();
    copymode = 0;

    if (ncol == 1) {
      k_vector_grid.modify< SPADeviceType >();
      k_vector_grid.sync< SPAHostType >();
    } else {
      k_array_grid.modify< SPADeviceType >();
      k_array_grid.sync< SPAHostType >();
    }


  }

}

/* ----------------------------------------------------------------------
   reallocate irregular comm patterns if local grid storage changes
   called by init() and whenever grid is rebalanced
------------------------------------------------------------------------- */

void ComputeFFTGridKokkos::reallocate()
{

  delete irregular1;
  delete irregular2;

  memory->destroy_kokkos(k_map1, map1);
  memory->destroy_kokkos(k_map2, map2);
 
  irregular_create();

  k_map2.modify< SPAHostType >();
  k_map2.sync< SPADeviceType >();

  if (grid->nlocal == nglocal) return;

  memory->destroy_kokkos(k_vector_grid, vector_grid);
  memory->destroy_kokkos(k_array_grid, array_grid);
  memory->destroy_kokkos(k_ingrid, ingrid);
  memory->destroy_kokkos(k_gridwork, gridwork);
  memory->destroy_kokkos(k_gridworkcomplex, gridworkcomplex);
  
  nglocal = grid->nlocal;

  memory->create_kokkos(k_ingrid,ingrid,nglocal,"fft/grid:ingrid");

  gridwork = NULL;
  gridworkcomplex = NULL;

  if (startcol || conjugate) {
    memory->create_kokkos(
        k_gridwork, 
        gridwork,
        nglocal,
        "fft/grid:gridwork");
  }
  if (!conjugate) {
    memory->create_kokkos(
        k_gridworkcomplex, 
        gridworkcomplex,
        2*nglocal, 
        "fft/grid:gridworkcomplex");
  }

  if (ncol == 1) {
    memory->create_kokkos(
      k_vector_grid, 
      vector_grid, 
      nglocal,
      "fft/grid:vector_grid");
  } else {
    memory->create_kokkos(
      k_array_grid,
      array_grid,
      nglocal, ncol,
      "fft/grid:array_grid");
  }

  // one-time setup of vector of K-space vector magnitudes if requested
  // compute values in K-space, irregular comm to grid decomposition
  // kx,ky,kz = indices of FFT grid cell in K-space
  // convert to distance from (0,0,0) cell using PBC
  // klen = length of K-space vector

  if (!startcol) return;

  int i,j,k;
  double ikx,iky,ikz;
  double klen;

  int nxhalf = nx/2;
  int nyhalf = ny/2;
  int nzhalf = nz/2;
  if (dimension == 2) nzhalf = 1;

  int icol = 0;

  for (int m = 0; m < 4; m++) {
    if (m == 0 && !kx) continue;
    if (m == 1 && !ky) continue;
    if (m == 2 && !kz) continue;
    if (m == 3 && !kmag) continue;

    int n = 0;
    for (k = nzlo; k <= nzhi; k++) {
      if (k < nzhalf) ikz = k;
      else ikz = nz - k;
      for (j = nylo; j <= nyhi; j++) {
        if (j < nyhalf) iky = j;
        else iky = ny - j;
        for (i = nxlo; i <= nxhi; i++) {
          if (i < nxhalf) ikx = i;
          else ikx = nx - i;

          if (m == 0) klen = ikx;
          else if (m == 1) klen = iky;
          else if (m == 2) klen = ikz;
          else if (m == 3) klen = sqrt(ikx*ikx + iky*iky + ikz*ikz);

          fftwork[n++] = klen;

        }
      }
    }

    irregular2->exchange_uniform((char *) fftwork,sizeof(double),
                                 (char *) gridwork);
 
    for (i = 0; i < nglocal; i++)
      array_grid[map2[i]][icol] = gridwork[i];

    icol++;
  }
}

void ComputeFFTGridKokkos::irregular_create()
{
  int i,ix,iy,iz,ipy,ipz;
  cellint gid;
  int *proclist1,*proclist2,*proclist3;
  char *sbuf1,*rbuf1,*sbuf2,*rbuf2;

  // plan for moving data from SPARTA grid -> FFT grid
  // send my cell IDs in SPARTA grid->cells order
  //   to proc who owns each in FFT partition
  // create map1 = how to reorder them on receiving FFT proc

  irregular1 = new Irregular(sparta);

  Grid::ChildCell *cells = grid->cells;
  int nglocal = grid->nlocal;

  memory->create(proclist1,nglocal,"fft/grid:proclist1");

  // use cell ID to determine which proc owns it in FFT partitioning
  // while loops are for case where some procs own nothing in FFT partition,
  //   due to more procs that rows/columns of FFT partition
  //   in this case formula (iy/ny * npy) may undershoot correct ipy proc
  //   increment until find ipy consistent with
  //   ipy*ny/npy to (ipy+1)*ny/npy bounds of that proc's FFT partition

  for (i = 0; i < nglocal; i++) {
    gid = cells[i].id;
    iy = ((gid-1) / nx) % ny;
    iz = (gid-1) / (nx*ny);

    ipy = static_cast<int> (1.0*iy/ny * npy);
    while (1) {
      if (iy >= ipy*ny/npy && iy < (ipy+1)*ny/npy) break;
      ipy++;
    }
    ipz = static_cast<int> (1.0*iz/nz * npz);
    while (1) {
      if (iz >= ipz*nz/npz && iz < (ipz+1)*nz/npz) break;
      ipz++;
    }

    proclist1[i] = ipz*npy + ipy;
  }

  int nrecv = irregular1->create_data_uniform(nglocal,proclist1);

  if (nrecv != nfft) 
    error->one(FLERR,"Compute fft/grid FFT mapping is inconsistent");

  memory->create(sbuf1,nglocal*sizeof(cellint),"fft/grid:sbuf1");
  memory->create(rbuf1,nfft*sizeof(cellint),"fft/grid:rbuf1");

  cellint *idsend = (cellint *) sbuf1;
  for (i = 0; i < nglocal; i++) idsend[i] = cells[i].id;

  irregular1->exchange_uniform(sbuf1,sizeof(cellint),rbuf1);

  memory->create_kokkos(k_map1,map1,nfft,"fft/grid:map1");

  cellint *idrecv = (cellint *) rbuf1;

  for (i = 0; i < nfft; i++) {
    gid = idrecv[i];
    ix = (gid-1) % nx;
    iy = ((gid-1) / nx) % ny;
    iz = (gid-1) / (nx*ny);
    map1[i] = (iz-nzlo)*nxfft*nyfft + (iy-nylo)*nxfft + (ix-nxlo);
  }

  k_map1.modify< SPAHostType >();
  k_map1.sync< SPADeviceType >();

  // plan for moving data from FFT grid -> SPARTA grid
  // send my cell IDs in FFT grid order
  //   back to proc who owns each in SPARTA grid->cells
  // proclist3 generated from irregular1->reverse()
  //   must permute proclist3 -> proclist2 via map1
  //   same way that received grid data will be permuted by map1 to FFT data
  // create map2 = how to reorder received FFT data on SPARTA grid proc

  irregular2 = new Irregular(sparta);

  memory->create(proclist3,nfft,"fft/grid:proclist2");
  irregular1->reverse(nrecv,proclist3);

  memory->create(proclist2,nfft,"fft/grid:proclist2");
  for (i = 0; i < nfft; i++) proclist2[map1[i]] = proclist3[i];

  nrecv = irregular2->create_data_uniform(nfft,proclist2);
  if (nrecv != nglocal) 
    error->one(FLERR,"Compute fft/grid FFT mapping is inconsistent");

  memory->create(sbuf2,nfft*sizeof(cellint),"fft/grid:sbuf2");
  memory->create(rbuf2,nglocal*sizeof(cellint),"fft/grid:rbuf2");

  idsend = (cellint *) sbuf2;
  for (i = 0; i < nfft; i++) idsend[map1[i]] = idrecv[i];

  irregular2->exchange_uniform(sbuf2,sizeof(cellint),rbuf2);

  // insure grid cell IDs are hashed, so can use them to build map2

  if (!grid->hashfilled) grid->rehash();

#ifdef SPARTA_MAP
  std::map<cellint,int> *hash = grid->hash;
#elif defined SPARTA_UNORDERED_MAP
  std::unordered_map<cellint,int> *hash = grid->hash;
#else
  std::tr1::unordered_map<cellint,int> *hash = grid->hash;
#endif

  idrecv = (cellint *) rbuf2;

  memory->create_kokkos(k_map2,map2,nfft,"fft/grid:map2");
  for (i = 0; i < nglocal; i++) {
    gid = idrecv[i];
    map2[i] = (*hash)[gid] - 1;
  }

  k_map2.modify< SPAHostType >();
  k_map2.sync< SPADeviceType >();

  // clean up
  memory->destroy(proclist1);
  memory->destroy(proclist2);
  memory->destroy(proclist3);
  memory->destroy(sbuf1);
  memory->destroy(rbuf1);
  memory->destroy(sbuf2);
  memory->destroy(rbuf2);
}

KOKKOS_INLINE_FUNCTION
void ComputeFFTGridKokkos::operator()(TagComputeFFTGrid_copy_from_tmp, const int &i) const {

  k_ingrid.d_view(i) = k_tmp(i, offset);

}

KOKKOS_INLINE_FUNCTION
void ComputeFFTGridKokkos::operator()(TagComputeFFTGrid_compute_norm_sq, const int &i) const {

  FFT_SCALAR real = k_fft.d_view(2*i);
  FFT_SCALAR imag = k_fft.d_view(2*i+1);

  k_fftwork.d_view(i) = real*real + imag*imag;

}

KOKKOS_INLINE_FUNCTION
void ComputeFFTGridKokkos::operator()(TagComputeFFTGrid_scale_grid, const int &i) const {

  if (ncol == 1) {
    k_vector_grid.d_view(i) *= scalefactor;
  } else {
    for (int c = startcol; c < ncol; c++) {
      k_array_grid.d_view(i, c) *= scalefactor;
    }
  }

}

KOKKOS_INLINE_FUNCTION
void ComputeFFTGridKokkos::operator()(TagComputeFFTGrid_fill_fft, const int &i) const {

  int id = k_map1.d_view(i);

  k_fft.d_view(2*id) = k_fftwork.d_view(i);
  k_fft.d_view(2*id+1) = ZEROF;

}

KOKKOS_INLINE_FUNCTION
void ComputeFFTGridKokkos::operator()(TagComputeFFTGrid_update_conjugate, const int &i) const {

  int id = k_map2.d_view(i);

  if (sumflag) {
    if (ncol == 1) {
      k_vector_grid.d_view(id) += k_gridwork.d_view(i);
    } else {
      k_array_grid.d_view(id, startcol) += k_gridwork.d_view(i);
    }
  } else {
    if (ncol == 1) {
      k_vector_grid.d_view(id) = k_gridwork.d_view(i);
    } else {
      k_array_grid.d_view(id, startcol + offset) = k_gridwork.d_view(i);
    }
  }

}

KOKKOS_INLINE_FUNCTION
void ComputeFFTGridKokkos::operator()(TagComputeFFTGrid_update_complex, const int &i) const {

  int id = k_map2.d_view(i);

  if (sumflag) {
    k_array_grid.d_view(id, startcol) += k_gridworkcomplex.d_view(2*i);
    k_array_grid.d_view(id, startcol+1) += k_gridworkcomplex.d_view(2*i+1);
  } else {
    k_array_grid.d_view(id, 2*offset+startcol) = k_gridworkcomplex.d_view(2*i);
    k_array_grid.d_view(id, 2*offset+startcol+1) = k_gridworkcomplex.d_view(2*i+1);
  }

}
