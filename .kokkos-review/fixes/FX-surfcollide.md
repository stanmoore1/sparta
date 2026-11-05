# FX-surfcollide checkpoint

- F-G00-18 (part in my files) | src/KOKKOS/surf_collide_specular_kokkos.h | wrapper_kokkos now honors noslip (flags[0] if flags else noslip_flag) -> negate3 vs reflect3, matching CPU SurfCollideSpecular::wrapper; scatter_cmodel part in surf_react_adsorb_kokkos.h belongs to another fixer | compile OK
- F-G00-10 / F-G09-1 | src/KOKKOS/surf_collide_{cll,td,impulsive,adiabatic}_kokkos.cpp | backup() now refreshes afix_kk/vfix_kk (pre_update_custom_kokkos + kk_copy.copy) after d_particles, following the diffuse/specular/piston pattern | compile OK
- F-G08-1 / F-G09-3 | src/KOKKOS/surf_collide_{diffuse,specular,piston,cll,td,impulsive,adiabatic}_kokkos.cpp | init() fix lookup accepts "ambipolar"/"ambipolar/kk" and "vibmode"/"vibmode/kk" | compile OK
- F-G09-4 (surf_collide part only) | src/KOKKOS/surf_collide_{diffuse,specular,piston,cll,td,impulsive,adiabatic}_kokkos.cpp | surf_react dispatch in pre_collide/backup/restore accepts "global|prob|adsorb" and the "/kk" spelling; update_kokkos.cpp surf_collide_style_tag and compute_surf_kokkos.cpp parts left to other fixers | compile OK
- F-G08-2 | src/KOKKOS/surf_collide_piston_kokkos.h, src/surf_collide_piston.cpp | piston reflection wrapped in `if (ip)` so a particle destroyed by chemistry is not dereferenced (surf_react fixes still called with i=-1, as in diffuse); on the two outside-box early returns a reaction product jp is discarded (KK: flag PDISCARD; CPU: nlocal--) instead of orphaned | compile OK
- F-G08-3 | src/KOKKOS/surf_collide_{vanish,transparent}_kokkos.h, src/surf_collide_{vanish,transparent}.cpp | collide()/collide_kokkos() now name the out-param and set `reaction = 0` (optional update.cpp/update_kokkos.cpp init left to their owners) | compile OK (also update_kokkos.cpp, which includes the headers)
- F-G00-18 note: diffuse wrapper_kokkos already matches CPU (no flags used); no change
## STATUS: COMPLETE
