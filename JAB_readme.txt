For Stan:

1) Basic subcell method stuff
-collisions_one_subcell() stuff is stored in the overwrriten collide.*
-everything should be in order there

2) basic subcell method with KOKKOS
-stored in src/jab_collide_vss_kokkos.* 
-everything should be good and working, but we haven't done a review of it yet
-requires the updated kokkos_type.h to allow for 3d arrays (needed for subcell method) which is stored in src/jab_fft_kokkos/

3) FFT stuff
-all stored in src/jab_fft_kokkos/
-does not currently compile. most files should be good with 2 exceptions:
-pack2d_kokkos.h likely having incorrect indexing
-compute_fft_grid_kokkos.* needing changes

4) Test problem (using subcells)
-adapted from provided collide/ test problem, also added a 2d version of it
-stored in src/jab_testproblem/ as in.collide*
