# FX-surfcollide checkpoint

- F-G00-18 (part in my files) | src/KOKKOS/surf_collide_specular_kokkos.h | wrapper_kokkos now honors noslip (flags[0] if flags else noslip_flag) -> negate3 vs reflect3, matching CPU SurfCollideSpecular::wrapper; scatter_cmodel part in surf_react_adsorb_kokkos.h belongs to another fixer | compile OK
