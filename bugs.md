# SPARTA Bug Fixes

All bugs were found by static analysis of the source tree and fixed in-place. Fixes are listed in discovery order. Each entry includes the file, line(s), the buggy code, the replacement, and the reason.

---


## 1. Double memory allocation — `src/comm.cpp:549`

**Buggy code:**
```cpp
memory->sfree(rbuf);
maxrecvbuf = recvsize;
rbuf = (char *) memory->smalloc(maxrecvbuf,"comm:rbuf");
memory->create(rbuf,maxrecvbuf,"comm:rbuf");   // spurious second allocation
memset(rbuf,0,maxrecvbuf);
```

**Fixed code:**
```cpp
memory->sfree(rbuf);
maxrecvbuf = recvsize;
rbuf = (char *) memory->smalloc(maxrecvbuf,"comm:rbuf");
memset(rbuf,0,maxrecvbuf);
```

**Reason:** `memory->create()` internally calls `smalloc` and reassigns the pointer, leaking the allocation on the previous line. Additionally, `memory->create()` takes `int n` while `maxrecvbuf` is a `bigint`, so for buffers >2 GB the size would be silently truncated. The comment on the preceding line explicitly states "must use smalloc since rbuf can be larger than 2 GB". Every other `rbuf` reallocation in the file uses only `smalloc`.

---


## 2. Dead code: reaction probability validation inside switch — `src/react_tce.cpp:182-183`

**Buggy code:**
```cpp
    case RECOMBINATION:
      { ...
        break;
      }

    if (react_prob < 0) error->warning(FLERR,"Negative reaction probability");
    else if (react_prob > 1) error->warning(FLERR,"Reaction probability greater than 1");

    default:
      error->one(FLERR,"Unknown outcome in reaction");
      break;
    }
```

**Fixed code:**
```cpp
    case RECOMBINATION:
      { ...
        break;
      }

    default:
      error->one(FLERR,"Unknown outcome in reaction");
      break;
    }

    if (react_prob < 0) error->warning(FLERR,"Negative reaction probability");
    else if (react_prob > 1) error->warning(FLERR,"Reaction probability greater than 1");
```

**Reason:** The two validation lines were placed inside the `switch` block after the last `break`, before `default:`. They are unreachable — every case has a `break` so control never falls through to them. Moved to after the closing `}` of the switch so they execute after any case computes `react_prob`.

---


## 3. Wrong variable in bounds check — `src/compute_reduce.cpp:161`

**Buggy code:**
```cpp
while (iarg < nargnew) {
    if (strcmp(arg[iarg],"replace") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal compute reduce command");
```

**Fixed code:**
```cpp
while (iarg < nargnew) {
    if (strcmp(arg[iarg],"replace") == 0) {
      if (iarg+3 > nargnew) error->all(FLERR,"Illegal compute reduce command");
...
    } else if (strcmp(arg[iarg],"subset") == 0) {
      if (iarg+2 > nargnew) error->all(FLERR,"Illegal compute reduce command");
```

**Reason:** The enclosing loop iterates up to `nargnew`. The bounds check inside must also use `nargnew` to verify that `arg[iarg+1]` and `arg[iarg+2]` are in range. When `nargnew > narg`, using `narg` allows the check to pass while the subsequent `atoi(arg[iarg+1])` and `atoi(arg[iarg+2])` access out-of-bounds memory. This error was present in both the `replace` and `subset` conditional branches.

---


## 4. Stale `react_prob` in do-while loop — `src/react_qk.cpp:125`

**Buggy code:**
```cpp
prob = 0.0;
do {
    iv = static_cast<int>(random->uniform()*(maxlev+0.99999999));
    evib = static_cast<double>(iv / inverse_kT);
    if (evib < ecc) react_prob = pow(1.0-evib/ecc,1.5-omega);
} while (random->uniform() < react_prob);
```

**Fixed code:**
```cpp
react_prob = 0.0;
do {
    iv = static_cast<int>(random->uniform()*(maxlev+0.99999999));
    evib = static_cast<double>(iv / inverse_kT);
    if (evib < ecc) react_prob = pow(1.0-evib/ecc,1.5-omega);
} while (random->uniform() < react_prob);

...
prob = 0.0;
do {
    iv = random->uniform()*(maxlev+0.99999999);
    evib = static_cast<double>(iv * update->boltz*species[mspec].vibtemp[0]);
    if (evib < ecc) prob = pow(1.0-evib/ecc,1.5 - r->coeff[6]);
} while (random->uniform() < prob);
```

**Reason:** For the first loop, `prob` is set to 0 but `react_prob` is used in the loop condition. If `evib >= ecc` on the first iteration the assignment inside the loop is skipped and `react_prob` retains its value from a prior reaction iteration. `prob = 0.0` is dead code — the zero initialisation must be applied to `react_prob`.

For the second loop, an identical logical flaw existed where `prob` was used for the condition but was never initialized prior to the loop. `prob = 0.0` was added to prevent using uninitialized memory when `evib == ecc` on the first iteration.

---


## 5. `memset` triggered on shrink causing huge write — `src/grid_custom.cpp:179`

**Buggy code:**
```cpp
if (nnew - nold)
    memset(&darray[nold][0],0,
           (nnew-nold)*edcol[ewhich[ic]]*sizeof(double));
```

**Fixed code:**
```cpp
if (nnew > nold)
    memset(&darray[nold][0],0,
           (nnew-nold)*edcol[ewhich[ic]]*sizeof(double));
```

**Reason:** When `nnew < nold` (array shrinking), `nnew - nold` is negative but non-zero, so the condition is true. The negative value is then cast to `size_t` producing a huge count, and `memset` writes far past the buffer. Every other case in the same function (`int` vector, `int` array, `double` vector) correctly uses `if (nnew > nold)`.

---


## 6. `pack_id` 3D iterates `nsown` but `cglobal` is sized `nchoose` — `src/compute_property_surf.cpp:250`

**Buggy code:**
```cpp
} else if (dimension == 3) {
    ...
    for (int i = 0; i < nsown; i++) {
        m = cglobal[i];
```

**Fixed code:**
```cpp
} else if (dimension == 3) {
    ...
    for (int i = 0; i < nchoose; i++) {
        m = cglobal[i];
```

**Reason:** `cglobal` is allocated with `nchoose` entries (the number of surfaces matching the group bitmask, which is ≤ `nsown`). The 2D branch of the same function correctly iterates `i < nchoose`. Iterating `i < nsown` in the 3D branch accesses `cglobal` out of bounds whenever a non-default surface group is used.

---


## 7. `sprintf` into fixed 128-byte buffer with unbounded input — `src/utils.cpp:126`

**Buggy code:**
```cpp
char msg[128];
sprintf(msg,"Illegal %s command: missing argument(s)",cmd.c_str());
```

**Fixed code:**
```cpp
char msg[128];
snprintf(msg,sizeof(msg),"Illegal %s command: missing argument(s)",cmd.c_str());
```

**Reason:** `cmd` is a `std::string` with no enforced length limit. A command name longer than ~90 characters overflows the 128-byte buffer.

---


## 8. `fopen` return values unchecked before use — `src/variable.cpp:655-659`

**Buggy code:**
```cpp
FILE *fp = fopen("tmp.sparta.variable.lock","r");
int tmp = fscanf(fp,"%d",&nextindex);
fclose(fp);
fp = fopen("tmp.sparta.variable.lock","w");
fprintf(fp,"%d\n",nextindex+1);
fclose(fp);
```

**Fixed code:**
```cpp
FILE *fp = fopen("tmp.sparta.variable.lock","r");
if (fp == NULL) error->one(FLERR,"Could not open variable lock file for reading");
int tmp = fscanf(fp,"%d",&nextindex);
fclose(fp);
fp = fopen("tmp.sparta.variable.lock","w");
if (fp == NULL) error->one(FLERR,"Could not open variable lock file for writing");
fprintf(fp,"%d\n",nextindex+1);
fclose(fp);
```

**Reason:** Both `fopen` calls are in the critical file-locking path coordinating parallel universe partitions. A NULL return (e.g. from a permissions or I/O error) causes an immediate null-pointer dereference in `fscanf`/`fprintf`.

---


## 9. Wrong variable checked after `find_group` — `src/compute_count.cpp:50`

**Buggy code:**
```cpp
int igroup = particle->mixture[imix]->find_group(ptr+1);
if (imix < 0)
    error->all(FLERR,"Unknown mixture group in compute count command");
```

**Fixed code:**
```cpp
int igroup = particle->mixture[imix]->find_group(ptr+1);
if (igroup < 0)
    error->all(FLERR,"Unknown mixture group in compute count command");
```

**Reason:** `imix` was already validated on the preceding line. The check that should guard against an unknown group name must test `igroup` (the return value of `find_group`), not `imix`. An invalid group name silently passes, and `igroup = -1` is stored in `indexgroup[]` for later use.

---


## 10. Copy-paste: `TYPE1` assigned for `"type2"` — `src/compute_gas_collision_tally.cpp:58`

**Buggy code:**
```cpp
else if (strcmp(arg[iarg],"type2") == 0) which[nvalue++] = TYPE1;
```

**Fixed code:**
```cpp
else if (strcmp(arg[iarg],"type2") == 0) which[nvalue++] = TYPE2;
```

**Reason:** Requesting the `type2` output column silently records `type1` data instead.

---


## 11. Two wrong velocity enum values — `src/compute_gas_reaction_tally.cpp:71-72`

**Buggy code:**
```cpp
else if (strcmp(arg[iarg],"vy2/pre") == 0) which[nvalue++] = VX2PRE;
else if (strcmp(arg[iarg],"vz2/pre") == 0) which[nvalue++] = VY2PRE;
```

**Fixed code:**
```cpp
else if (strcmp(arg[iarg],"vy2/pre") == 0) which[nvalue++] = VY2PRE;
else if (strcmp(arg[iarg],"vz2/pre") == 0) which[nvalue++] = VZ2PRE;
```

**Reason:** Copy-paste error shifts both the y and z pre-collision velocity components of particle 2 by one position, causing users requesting `vy2/pre` to receive `vx2/pre` data, and `vz2/pre` to receive `vy2/pre` data.

---


## 12. `ID2POST` checked twice, `ID1POST` never checked — `src/compute_surf_reaction_tally.cpp:291`

**Buggy code:**
```cpp
if (which[icol-1] == ID2POST || which[icol-1] == ID2POST) return INT;
```

**Fixed code:**
```cpp
if (which[icol-1] == ID1POST || which[icol-1] == ID2POST) return INT;
```

**Reason:** The first operand is a duplicate of the second. `ID1POST` is never tested, so columns of that type return `DOUBLE` instead of `INT`.

---


## 13. Out-of-bounds access inside error message — `src/fix_grid_check.cpp:121-124`

**Buggy code:**
```cpp
if (icell < 0 || icell >= nglocal) {
    if (outflag == ERROR) {
        char str[128];
        sprintf(str,
                "Particle %d,%d on proc %d is in invalid cell " CELLINT_FORMAT
                " on timestep " BIGINT_FORMAT,
                i,particles[i].id,comm->me,cells[icell].id,update->ntimestep);
```

**Fixed code:**
```cpp
if (icell < 0 || icell >= nglocal) {
    if (outflag == ERROR) {
        char str[128];
        sprintf(str,
                "Particle %d,%d on proc %d is in invalid cell index %d"
                " on timestep " BIGINT_FORMAT,
                i,particles[i].id,comm->me,icell,update->ntimestep);
        error->one(FLERR,str);
    }
    nflag++;
    continue;
}
```

**Reason:** The code accesses `cells[icell]` after explicitly confirming that `icell` is either negative or beyond the array bounds. The error message was changed to report the raw (invalid) index directly. Crucially, a `continue;` statement was added. Without it, if `outflag` was not `ERROR` (e.g. `WARNING`), the code would bypass the exception but then immediately access `cells[icell].lo` on the next line, still causing a fatal out-of-bounds crash.

---


## 14. `timeout_start` never initialized; stray debug `printf` — `src/timer.cpp`

**Buggy code (`init_timeout`):**
```cpp
void Timer::init_timeout()
{
    _s_timeout = _timeout;
    if (_timeout < 0)
        _nextcheck = -1;
    else
        _nextcheck = _checkfreq;
}
```

**Fixed code:**
```cpp
void Timer::init_timeout()
{
    _s_timeout = _timeout;
    if (_timeout < 0)
        _nextcheck = -1;
    else {
        _nextcheck = _checkfreq;
        timeout_start = MPI_Wtime();
    }
}
```

**Buggy code (`_check_timeout`):**
```cpp
MPI_Bcast(&walltime, 1, MPI_DOUBLE, 0, world);

printf("%g %g\n",walltime,_timeout);

if (walltime < _timeout) {
```

**Fixed code:**
```cpp
MPI_Bcast(&walltime, 1, MPI_DOUBLE, 0, world);

if (walltime < _timeout) {
```

**Reason:** `timeout_start` is a `double` member declared in `timer.h` but never assigned — not in the constructor nor in `init_timeout()`. All three functions that compute elapsed/remaining time (`_check_timeout`, `get_timeout_remain`, `print_timeout`) subtract it from `MPI_Wtime()`, producing undefined behaviour. It must be set to `MPI_Wtime()` when the timeout begins (i.e. when `_timeout >= 0` in `init_timeout`). The `printf` is leftover debug output.

---


## 15. `wrapper()` ignores `noslip_flag`, always does specular reflection — `src/surf_collide_specular.cpp:163`

**Buggy code:**
```cpp
void SurfCollideSpecular::wrapper(Particle::OnePart *p, double *norm,
                                  int *flags, double *coeffs)
{
    if (flags)
        noslip_flag = flags[0];

    MathExtra::reflect3(p->v,norm);
}
```

**Fixed code:**
```cpp
void SurfCollideSpecular::wrapper(Particle::OnePart *p, double *norm,
                                  int *flags, double *coeffs)
{
    if (flags)
        noslip_flag = flags[0];

    if (noslip_flag) MathExtra::negate3(p->v);
    else MathExtra::reflect3(p->v,norm);
}
```

**Reason:** `collide()` correctly branches on `noslip_flag` between `negate3` (noslip, negates all three velocity components) and `reflect3` (specular). The `wrapper()` function updates `noslip_flag` from the caller-supplied flags but then always calls `reflect3`, ignoring the flag it just set.

---


## 16. Missing `if (copy) return` guard before `delete random` — `src/surf_collide_cll.cpp:139`

**Buggy code:**
```cpp
SurfCollideCLL::~SurfCollideCLL()
{
    delete random;
}
```

**Fixed code:**
```cpp
SurfCollideCLL::~SurfCollideCLL()
{
    if (copy) return;

    delete random;
}
```

**Reason:** `surf_collide_diffuse.cpp` and `surf_collide_adiabatic.cpp` both guard `delete random` with `if (copy) return`. When an object is a copy, the `random` pointer is shared with the original; deleting it here causes a double-free.

---


## 17. Missing `if (copy) return` guard before `delete random` — `src/surf_collide_impulsive.cpp:144`

**Buggy code:**
```cpp
SurfCollideImpulsive::~SurfCollideImpulsive()
{
    delete random;
}
```

**Fixed code:**
```cpp
SurfCollideImpulsive::~SurfCollideImpulsive()
{
    if (copy) return;

    delete random;
}
```

**Reason:** Same as bug 16 — missing copy guard causes double-free of the shared `random` pointer.

---


## 18. Out-of-bounds GPU memory access after invalid cell check — `src/KOKKOS/fix_grid_check_kokkos.cpp:82`

**Buggy code:**
```cpp
if (icell < 0 || icell >= nglocal) {
    d_particle_problems(i) |= IS_IN_INVALID_CELL;
    local_nflag++;
}

// does particle coord match icell bounds
double* lo = d_cells[icell].lo;   // accessed unconditionally
double* hi = d_cells[icell].hi;
```

**Fixed code:**
```cpp
if (icell < 0 || icell >= nglocal) {
    d_particle_problems(i) |= IS_IN_INVALID_CELL;
    local_nflag++;
    return;
}

// does particle coord match icell bounds
double* lo = d_cells[icell].lo;
double* hi = d_cells[icell].hi;
```

**Reason:** Same logic error as bug 13, but in a Kokkos parallel kernel. Out-of-bounds device memory access causes undefined behaviour (or a hard GPU fault) on the four subsequent `d_cells[icell]` and `d_cinfo[icell]` accesses.

---


## 19. Missing `else` for KNY and KNZ output branches — `src/KOKKOS/compute_lambda_grid_kokkos.cpp:329,334`

**Buggy code:**
```cpp
if (l_knyflag) {
    if (l_noutputs == 1) l_vector_grid[i] = lambda / sizey;
    l_array_grid(i,l_output_order[KNY]) = lambda / sizey;   // no else
}

if (l_knzflag) {
    if (l_noutputs == 1) l_vector_grid[i] = lambda / sizez;
    l_array_grid(i,l_output_order[KNZ]) = lambda / sizez;   // no else
}
```

**Fixed code:**
```cpp
if (l_knyflag) {
    if (l_noutputs == 1) l_vector_grid[i] = lambda / sizey;
    else l_array_grid(i,l_output_order[KNY]) = lambda / sizey;
}

if (l_knzflag) {
    if (l_noutputs == 1) l_vector_grid[i] = lambda / sizez;
    else l_array_grid(i,l_output_order[KNZ]) = lambda / sizez;
}
```

**Reason:** The KNX and KNALL blocks immediately above both use `if ... else` correctly. For KNY and KNZ, the `l_array_grid` write executes unconditionally, including when `noutputs == 1` (where only `l_vector_grid` should be written). This mirrors the KNX pattern which correctly has `else`.

---


## 20. Realloc condition inverted — `src/KOKKOS/fix_ave_histo_weight_kokkos.cpp:379`

**Buggy code:**
```cpp
if (k_match.extent(0) > nmax)
    MemKK::realloc_kokkos(k_match,"fix_ave_histo_weight:match",nmax);
```

**Fixed code:**
```cpp
if (k_match.extent(0) < nmax)
    MemKK::realloc_kokkos(k_match,"fix_ave_histo_weight:match",nmax);
```

**Reason:** The identical reallocation pattern at line 315 of the same file correctly uses `< nmax` (grow when the buffer is too small). This second site uses `>` (shrink when too large), which means `match_all_kokkos` is called with a buffer that may be smaller than the required `nmax` entries.

---


## 21. Dead code accesses particle array with a cell index — `src/KOKKOS/compute_sonine_grid_kokkos.cpp:228-229`

**Buggy code:**
```cpp
KOKKOS_INLINE_FUNCTION
void ComputeSonineGridKokkos::operator()(TagComputeSonineGrid_normalize_vcom, const int &icell) const {
    const int ispecies = d_particles[icell].ispecies;   // cell index used as particle index
    const int igroup = d_s2g(imix,ispecies);

    double norm;
    for (int j=0; j<ngroup; j++) { ...
```

**Fixed code:**
```cpp
KOKKOS_INLINE_FUNCTION
void ComputeSonineGridKokkos::operator()(TagComputeSonineGrid_normalize_vcom, const int &icell) const {
    double norm;
    for (int j=0; j<ngroup; j++) { ...
```

**Reason:** `icell` is a cell index (0 to `nglocal-1`), not a particle index. Indexing into `d_particles` with it reads unrelated memory. `ispecies` and `igroup` are never used within this function — they are entirely dead code that happens to perform an out-of-bounds (or at best meaningless) array access.

---


## 22. Stray debug `printf` in production code — `src/KOKKOS/fix_ave_histo_weight_kokkos.cpp:271`

**Buggy code:**
```cpp
} else {
    printf("%d, %d\n", which[i] == VARIABLE, kind == PERGRID);
    error->all(FLERR,"Fix ave/histo/weight/kokkos option not yet supported");
}
```

**Fixed code:**
```cpp
} else {
    error->all(FLERR,"Fix ave/histo/weight/kokkos option not yet supported");
}
```

**Reason:** Debug output left in production code that prints to stdout on every unsupported code path hit.

---


## 23. Shadowing of member variable `idsource` — `src/fix_ablate.cpp:118`

**Buggy code:**
```cpp
  } else if (strncmp(arg[5],"v_",2) == 0) {
    which = VARIABLE;

    int n = strlen(arg[5]);
    char *idsource = new char[n];
    strcpy(idsource,&arg[5][2]);
```

**Fixed code:**
```cpp
  } else if (strncmp(arg[5],"v_",2) == 0) {
    which = VARIABLE;

    int n = strlen(arg[5]);
    idsource = new char[n];
    strcpy(idsource,&arg[5][2]);
```

**Reason:** By declaring a local `char *idsource`, the `idsource` class member is not updated. This causes a null pointer dereference later when the class member is used.

---


## 24. Incorrect constant `MY_PI` instead of `MY_2PI` for azimuthal angle — `src/fix_emit_face_file.cpp:475`

**Buggy code:**
```cpp
        v[ndim] = beta_un*vscale[isp]*normal[ndim] + vstream[ndim];

        theta = MY_PI * random->uniform();
        vr = vscale[isp] * sqrt(-log(random->uniform()));
```

**Fixed code:**
```cpp
        v[ndim] = beta_un*vscale[isp]*normal[ndim] + vstream[ndim];

        theta = MY_2PI * random->uniform();
        vr = vscale[isp] * sqrt(-log(random->uniform()));
```

**Reason:** Azimuthal angle `theta` for isotropic emission should be sampled over `[0, 2pi]`, not `[0, pi]`. `MY_2PI` is the correct constant.

---


## 25. Memory leak in `init()` on re-initialization — `src/fix_emit_face_file.cpp:241`

**Buggy code:**
```cpp
  // per-species vectors for mesh setting of species fractions
  // initialize to mixture settings

  fflag = new int[nspecies];
  fuser = new double[nspecies];
```

**Fixed code:**
```cpp
  // per-species vectors for mesh setting of species fractions
  // initialize to mixture settings

  delete [] fflag;
  delete [] fuser;
  fflag = new int[nspecies];
  fuser = new double[nspecies];
```

**Reason:** `init()` can be called multiple times during a simulation (e.g. on consecutive `run` commands). Without `delete []`, the previous arrays are leaked.

---


## 26. Formatting specifier mismatch for `BIGINT_FORMAT` — `src/fix_halt.cpp:218`

**Buggy code:**
```cpp
  char message[128];
  sprintf(message, "Fix halt condition for fix-id %s met on step %ld with value %g",
                                    id, update->ntimestep, attvalue);
```

**Fixed code:**
```cpp
  char message[128];
  sprintf(message, "Fix halt condition for fix-id %s met on step " BIGINT_FORMAT " with value %g",
                                    id, update->ntimestep, attvalue);
```

**Reason:** `update->ntimestep` is of type `bigint`, which may be defined as a 64-bit integer even on 32-bit platforms. Hardcoding `%ld` is non-portable and can cause runtime issues/warnings. `BIGINT_FORMAT` correctly maps to `%ld` or `PRId64` depending on the build configuration.

---


## 27. Missing format specifier for `x[2]` in error message — `src/fix_grid_check.cpp:222`

**Buggy code:**
```cpp
        sprintf(str,
                "Particle %d,%d on proc %d at %g %g %d is inside surfs in cell "
                CELLINT_FORMAT " on timestep " BIGINT_FORMAT,
                i,particles[i].id,comm->me,x[0],x[1],icell,cells[icell].id,
                update->ntimestep);
```

**Fixed code:**
```cpp
        sprintf(str,
                "Particle %d,%d on proc %d at %g %g %g is inside surfs in cell "
                CELLINT_FORMAT " on timestep " BIGINT_FORMAT,
                i,particles[i].id,comm->me,x[0],x[1],x[2],cells[icell].id,
                update->ntimestep);
```

**Reason:** The message text says "at %g %g %d", taking `x[0], x[1], icell`. It is clearly trying to print the particle's 3D coordinates `x, y, z`. The third formatter should be `%g` and the third coordinate `x[2]` should be passed instead of `icell` (the cell index is already printed directly afterwards using `CELLINT_FORMAT`).

---


## 28. Stack Buffer Overflow in `sprintf` — `src/input.cpp`, `src/move_surf.cpp`, `src/particle.cpp`

**Buggy code:**
```cpp
    if (infile == NULL) {
      char str[128];
      sprintf(str,"Cannot open input script %s",filename);
      error->one(FLERR,str);
    }
```

**Fixed code:**
```cpp
    if (infile == NULL) {
      char str[128];
      snprintf(str,128,"Cannot open input script %s",filename);
      error->one(FLERR,str);
    }
```

**Reason:** When formatting error messages for unopenable files, a fixed-size `char str[128]` buffer is used. If the user provides a file path or script name longer than approximately 100 characters, `sprintf` will write past the end of the 128-byte array, causing a stack buffer overflow. It was replaced with `snprintf` everywhere strings accept dynamic user file paths.

---


## 29. Memory Leaks During Error Handling — `src/input.cpp:241`

**Buggy code:**
```cpp
    if (execute_command()) {
      char *str = new char[maxline+32];
      sprintf(str,"Unknown command: %s",line);
      error->all(FLERR,str);
    }
```

**Fixed code:**
```cpp
    if (execute_command()) {
      char *str = new char[maxline+32];
      sprintf(str,"Unknown command: %s",line);
      error->all(FLERR,str);
      delete [] str;
    }
```

**Reason:** Memory is dynamically allocated for the `str` pointer using `new char[maxline+32]`, but it is passed directly to `error->all()` and never `delete[]`'d. Since SPARTA can be run as a library (where exceptions might be caught instead of terminating the process immediately), this guarantees a memory leak on syntax or unrecognized command errors.

---


## 30. Precision Loss / Unexpected Integer Truncation — `src/marching_cubes.h:36`

**Buggy code:**
```cpp
  double *lo,*hi;
  int v000,v001,v010,v011,v100,v101,v110,v111;
  double v000iso,v001iso,v010iso,v011iso,v100iso,v101iso,v110iso,v111iso;
```

**Fixed code:**
```cpp
  double *lo,*hi;
  double v000,v001,v010,v011,v100,v101,v110,v111;
  double v000iso,v001iso,v010iso,v011iso,v100iso,v101iso,v110iso,v111iso;
```

**Reason:** The variables `v000`, `v001`, `v010`, etc. are strictly defined as `int` in `marching_cubes.h`. However, the code in `marching_cubes.cpp` accumulates double precision floating-point values from `inval` (which maps to `mvalues`) into them, and then divides by `6.0`. Because they are integers, `int /= 6.0` invokes integer division and silent truncation, destroying fractional precision prior to threshold calculation (`v000iso = v000 - thresh;`). This behavior effectively corrupts the iso-surface generation. They were changed to `double`.

---


## 31. Missing `fopen` NULL pointer check — `src/read_isurf.cpp:556`

**Buggy code:**
```cpp
  fp = fopen(gridfile,"rb");
  if (precision == INT) {
    fseek(fp,offset*sizeof(uint8_t)+dim*sizeof(int),SEEK_SET);
```

**Fixed code:**
```cpp
  fp = fopen(gridfile,"rb");
  if (fp == NULL) {
    char str[128];
    snprintf(str,128,"Cannot open read_isurf grid corner point file %s",
             gridfile);
    error->one(FLERR,str);
  }
  if (precision == INT) {
```

**Reason:** When `gridfile` is re-opened by every MPI worker rank, the result of `fopen` is not checked for `NULL`. In parallel environments, it's common for `fopen` to fail on a single compute node. If `fopen` returns `NULL` on any rank, the immediate call to `fseek(fp, ...)` will dereference the NULL file pointer, causing a crash.

---


## 32. Stack Buffer Overflows via `sprintf` in file I/O error messages — `write_grid.cpp`, `write_isurf.cpp`, `write_surf.cpp`, `write_restart.cpp`

**Buggy code:**
```cpp
    if (!fp) {
      char str[128];
      sprintf(str,"Cannot open file %s",arg[0]);
      error->one(FLERR,str);
    }
```

**Fixed code:**
```cpp
    if (!fp) {
      char str[128];
      snprintf(str,128,"Cannot open file %s",arg[0]);
      error->one(FLERR,str);
    }
```

**Reason:** Any user-supplied or autogenerated filename exceeding ~100 characters will overflow the `char str[128]` buffer when an open fails, corrupting the stack right before the program aborts. Replaced `sprintf` with `snprintf` in multiple files to mitigate this.

---


## 33. Buffer Overflow when Combining Surface Styles with Suffixes — `src/surf.cpp:2033`

**Buggy code:**
```cpp
    if (sparta->suffix) {
      char estyle[256];
      sprintf(estyle,"%s/%s",arg[1],sparta->suffix);
```

**Fixed code:**
```cpp
    if (sparta->suffix) {
      char estyle[256];
      snprintf(estyle,256,"%s/%s",arg[1],sparta->suffix);
```

**Reason:** Appending user arguments directly into the fixed size `estyle[256]` array via `sprintf` without bounds-checking risks a stack overflow. If `arg[1]` + `sparta->suffix` combined are greater than 255 chars, it overflows.

---


## 34. Uninitialized Variable Usage due to Unchecked `fscanf` — `src/variable.cpp:658`

**Buggy code:**
```cpp
      FILE *fp = fopen("tmp.sparta.variable.lock","r");
      if (fp == NULL) error->one(FLERR,"Could not open variable lock file for reading");
      int tmp = fscanf(fp,"%d",&nextindex);
```

**Fixed code:**
```cpp
      FILE *fp = fopen("tmp.sparta.variable.lock","r");
      if (fp == NULL) error->one(FLERR,"Could not open variable lock file for reading");
      int tmp = fscanf(fp,"%d",&nextindex);
      if (tmp != 1) error->one(FLERR,"Failed to read index from variable lock file");
```

**Reason:** The variable `tmp` is assigned but its value (indicating the number of items successfully parsed) is completely ignored. If `fscanf` fails (e.g. if the lockfile is empty or corrupted), `nextindex` remains uninitialized. `nextindex+1` is then evaluated using uninitialized memory and written back to the file.

---


## 35. Memory Leak and Destructive Mutation of User Argument — `src/write_isurf.cpp:65`

**Buggy code:**
```cpp
  char *ptr;
  int n = strlen(arg[4]) + 16;
  char *file = new char[n];

  if ((ptr = strchr(arg[4],'*'))) {
    *ptr = '\0';
    sprintf(file,"%s" BIGINT_FORMAT "%s",arg[4],update->ntimestep,ptr+1);
  } else strcpy(file,arg[4]);
```

**Fixed code:**
```cpp
  char *ptr;
  int n = strlen(arg[4]) + 16;
  char *file = new char[n];
  char *arg4_copy = new char[strlen(arg[4]) + 1];
  strcpy(arg4_copy, arg[4]);

  if ((ptr = strchr(arg4_copy,'*'))) {
    *ptr = '\0';
    sprintf(file,"%s" BIGINT_FORMAT "%s",arg4_copy,update->ntimestep,ptr+1);
  } else strcpy(file,arg4_copy);
  delete [] arg4_copy;
  
  ...
  
  memory->destroy(dbuf);
  memory->destroy(dbufall);
  delete [] file;
```

**Reason:** The dynamically allocated `char *file` was utilized correctly but was never freed. Additionally, `strchr` combined with `*ptr = '\0'` destructively mutated the `arg[4]` variable which points to memory managed by the command parser. It was fixed by creating a temporary copy of `arg[4]`, modifying the copy, and freeing both the copy and `file` at the end of the method.

---


## 36. Memory Leak for `FILECOARSE` custom action types — `src/custom.cpp:59`

**Buggy code:**
```cpp
Custom::~Custom()
{
  for (int i = 0; i < naction; i++) {
    int action = actions[i].action;
    if (action == FILESTYLE) {
      delete [] actions[i].fname;
...
```

**Fixed code:**
```cpp
Custom::~Custom()
{
  for (int i = 0; i < naction; i++) {
    int action = actions[i].action;
    if (action == FILESTYLE || action == FILECOARSE) {
      delete [] actions[i].fname;
...
```

**Reason:** When the `custom` command uses the `file/coarse` action, `process_actions()` dynamically allocates string and integer arrays (`fname`, `cindex_file`, etc.). In the destructor `Custom::~Custom()`, these were only being freed if the action was `FILESTYLE`, completely omitting `FILECOARSE`. This caused a severe memory leak for any simulation utilizing a `file/coarse` action.

---


## 37. Out-of-bounds Memory Access on Error Path — `src/KOKKOS/fix_grid_check_kokkos.cpp:186`

**Buggy code:**
```cpp
      if (h_particle_problems(i) & IS_IN_INVALID_CELL) {
        sprintf(str,
                "Particle %d,%d on proc %d is in invalid cell " CELLINT_FORMAT
                " on timestep " BIGINT_FORMAT,
                i,particles[i].id,comm->me,cells[icell].id,update->ntimestep);
        error->one(FLERR,str);
      }
```

**Fixed code:**
```cpp
      if (h_particle_problems(i) & IS_IN_INVALID_CELL) {
        sprintf(str,
                "Particle %d,%d on proc %d is in invalid cell index %d"
                " on timestep " BIGINT_FORMAT,
                i,particles[i].id,comm->me,icell,update->ntimestep);
        error->one(FLERR,str);
      }
```

**Reason:** When checking grid cells, the code correctly flags particles that are in an invalid cell (`icell < 0 || icell >= nglocal`). However, when it goes to print the error message for `IS_IN_INVALID_CELL`, it uses the invalid `icell` to index into the `cells` array (`cells[icell].id`). This results in an out-of-bounds memory access. Replaced `cells[icell].id` with just printing the raw invalid `icell`.

---


## 38. Divide by Zero in Knudsen Calculation — `src/KOKKOS/compute_lambda_grid_kokkos.cpp:323`

**Buggy code:**
```cpp
      if (l_knxflag) {
        if (l_noutputs == 1) l_vector_grid[i] = lambda / sizex;
        else l_array_grid(i,l_output_order[KNX]) = lambda / sizex;
      }
```

**Fixed code:**
```cpp
      if (l_knxflag) {
        if (sizex > 0.0) {
          if (l_noutputs == 1) l_vector_grid[i] = lambda / sizex;
          else l_array_grid(i,l_output_order[KNX]) = lambda / sizex;
        } else {
          if (l_noutputs == 1) l_vector_grid[i] = 0.0;
          else l_array_grid(i,l_output_order[KNX]) = 0.0;
        }
      }
```

**Reason:** When calculating per-cell Knudsen numbers, the code divides by `sizex`, `sizey`, `sizez`, or `sizeall` unconditionally. If a cell has a zero dimension (e.g., zero volume cell or lower dimensions), division by zero occurs causing NaNs/Infs to be written. Added explicit checks to zero out the values instead of dividing if the dimension size is 0.

---


## 39. Out-of-bounds Read due to Short-Circuit Logic Error — `src/KOKKOS/compute_surf_kokkos.cpp:215`

**Buggy code:**
```cpp
  while (1) {
    while (h_surf2tally[istart] != -1 && istart < nsurf-2) istart++;
    while (h_surf2tally[iend] == -1 && iend > 0) iend--;
```

**Fixed code:**
```cpp
  if (nsurf == 0) return 0;

  while (1) {
    while (istart < nsurf-2 && h_surf2tally[istart] != -1) istart++;
    while (iend > 0 && h_surf2tally[iend] == -1) iend--;
```

**Reason:** During `array_surf_tally` compression, if `nsurf` is `0`, `iend = -1`. The `while (h_surf2tally[iend] == -1 && iend > 0)` checks `h_surf2tally[-1]` BEFORE checking if `iend > 0`, causing an out-of-bounds memory read. Also needed an early exit if `nsurf == 0`.

---


## 40. Scaling Logic Inversion in Timing Routine — `src/KOKKOS/fft2d_kokkos.cpp:768`

**Buggy code:**
```cpp
  if (flag == 1 && plan->scaled) {
    FFT_SCALAR norm = plan->norm;
```

**Fixed code:**
```cpp
  if (flag == -1 && plan->scaled) {
    FFT_SCALAR norm = plan->norm;
```

**Reason:** In the `fft_2d_1d_only_kokkos` timing function, scaling is mistakenly applied on the forward transform (`flag == 1`) instead of the backward transform (`flag == -1`). This skews the timing results by benchmarking scaling alongside the forward transform instead of the backward transform.

---


## 41. Copy/Paste Error in Surface Reaction Mapping — `src/KOKKOS/surf_collide_specular_kokkos.cpp:129`

**Buggy code:**
```cpp
      if (strcmp(surf->sr[n]->style,"global") == 0) {
        sr_kk_global_copy[nglob].copy((SurfReactGlobalKokkos*)(surf->sr[n]));
        sr_kk_global_copy[nglob].obj.pre_react();
        sr_type_list[n] = 0;
        sr_map[n] = nprob;
        nglob++;
```

**Fixed code:**
```cpp
      if (strcmp(surf->sr[n]->style,"global") == 0) {
        sr_kk_global_copy[nglob].copy((SurfReactGlobalKokkos*)(surf->sr[n]));
        sr_kk_global_copy[nglob].obj.pre_react();
        sr_type_list[n] = 0;
        sr_map[n] = nglob;
        nglob++;
```

**Reason:** Inside the "global" reaction method block, the surface reaction index `n` is mapped to the variable `nprob` (which is meant for "prob" methods) instead of `nglob`. This maps all global surface reactions to the 0th element, corrupting memory lookup mappings.

---


## 42. Uninitialized Dual View / Mismatched Allocator — `src/KOKKOS/fix_ave_histo_weight_kokkos.cpp:259`

**Buggy code:**
```cpp
      if (grid->maxlocal > maxvectorwt) {
        memoryKK->destroy_kokkos(k_vectorwt,vectorwt);
        maxvectorwt = grid->maxlocal;
        memory->create(vectorwt,maxvectorwt,"ave/histo/weight:vectorwt");
      }
```

**Fixed code:**
```cpp
      if (grid->maxlocal > maxvectorwt) {
        memoryKK->destroy_kokkos(k_vectorwt,vectorwt);
        maxvectorwt = grid->maxlocal;
        memoryKK->create_kokkos(k_vectorwt,vectorwt,maxvectorwt,"ave/histo/weight:vectorwt");
      }
```

**Reason:** The code destroys the Kokkos DualView `k_vectorwt` via `memoryKK->destroy_kokkos()`, but then reallocates the space using the host-only standard allocator `memory->create()`. The DualView `k_vectorwt` remains uninitialized, leading to a crash when `k_vectorwt.modify_host()` is called immediately after.

---


## 43. Incorrect Loop Index Used for Update Custom — `src/KOKKOS/create_particles_kokkos.cpp:426`

**Buggy code:**
```cpp
      if (nfix_update_custom)
        modify->update_custom(particle->nlocal-1,tempscale*temp_thermal,
                              tempscale*temp_rot,tempscale*temp_vib,
                              vstream_update_custom);
```

**Fixed code:**
```cpp
      if (nfix_update_custom)
        modify->update_custom(inew,tempscale*temp_thermal,
                              tempscale*temp_rot,tempscale*temp_vib,
                              vstream_update_custom);
```

**Reason:** In the loop iterating over newly created particles, the newly calculated index `inew` is ignored. Instead, `particle->nlocal-1` is hardcoded. Because `particle->nlocal` was already incremented entirely before this loop started, `particle->nlocal-1` points statically to the very last particle in the system for every iteration, meaning all custom attributes for newly created particles (except the last one) remain uninitialized.

---


## 44. Incorrect Loop Termination Logic (`return` vs `continue`) — `src/KOKKOS/compute_thermal_grid_kokkos.cpp` & `src/KOKKOS/compute_eflux_grid_kokkos.cpp`

**Buggy code:**
```cpp
    const int ispecies = d_particles[i].ispecies;
    const int igroup = d_s2g(imix,ispecies);
    if (igroup < 0) return;
```

**Fixed code:**
```cpp
    const int ispecies = d_particles[i].ispecies;
    const int igroup = d_s2g(imix,ispecies);
    if (igroup < 0) continue;
```

**Reason:** Inside Kokkos parallel execution blocks iterating over particles within a grid cell, an invalid group triggers `return;`. Because `return;` exits the lambda operator entirely, it aborts processing for *all remaining particles in the entire grid cell*, rather than simply skipping the current particle using `continue;`. This fundamentally breaks the physics tallies whenever a cell contains any unmapped species.

---


## 45. Memory Leaks on `malloc` Failure Paths — `src/KOKKOS/remap3d_kokkos.cpp`

**Buggy code:**
```cpp
  inarray = (struct extent_3d *) malloc(nprocs*sizeof(struct extent_3d));
  if (inarray == nullptr) return nullptr;

  outarray = (struct extent_3d *) malloc(nprocs*sizeof(struct extent_3d));
  if (outarray == nullptr) return nullptr;
```

**Fixed code:**
```cpp
  inarray = (struct extent_3d *) malloc(nprocs*sizeof(struct extent_3d));
  if (inarray == nullptr) { delete plan; return nullptr; }

  outarray = (struct extent_3d *) malloc(nprocs*sizeof(struct extent_3d));
  if (outarray == nullptr) { free(inarray); delete plan; return nullptr; }
```

**Reason:** In the `remap_3d_create_plan_kokkos` initialization routine, `malloc` return values are rigorously checked for `nullptr`, but on failure the code simply returns without freeing previously allocated resources. Over a dozen distinct `nullptr` checks across both point-to-point and collective communication branches were updated to cascade their respective `free()` and `delete` calls to prevent memory leaks during plan creation failures.

---


## 46. Divide-by-Zero Crash at 0K in VSS Collision — `src/collide_vss.cpp:205`

**Buggy code:**
```cpp
  double vre = vro*prefactor[ispecies][jspecies];
  vremax[icell][igroup][jgroup] = MAX(vre,vremax[icell][igroup][jgroup]);
  if (vre/vremax[icell][igroup][jgroup] < random->uniform()) return 0;
```

**Fixed code:**
```cpp
  double vre = vro*prefactor[ispecies][jspecies];
  vremax[icell][igroup][jgroup] = MAX(vre,vremax[icell][igroup][jgroup]);
  if (vremax[icell][igroup][jgroup] == 0.0) return 0;
  if (vre/vremax[icell][igroup][jgroup] < random->uniform()) return 0;
```

**Reason:** If the simulated gas is at absolute zero (e.g. perfectly cold uniform beam), all particles have identical velocities. `vremax` starts at `0.0`, and relative velocity squared (`vr2`) is `0.0`, resulting in `vre = 0.0`. `vremax` becomes `0.0`, and `vre / vremax` evaluates to `0.0 / 0.0` -> `NaN`. This `NaN` is falsely accepted by the `< random` check, and is subsequently used to inject `NaN`s directly into the particle velocity arrays via division by `precoln.vr2` in `setup_collision`, corrupting the simulation.

---


## 47. Ignored Explicit 0.0 in Cross-Species VSS Parameters — `src/collide_vss.cpp:895`

**Buggy code:**
```cpp
      if (relaxflag == VARIABLE) {
        params[isp][jsp].rotc1 = params[jsp][isp].rotc1 = atof(words[6]);
        params[isp][jsp].rotc2 = atof(words[7]);
```

**Fixed code:**
```cpp
      if (relaxflag == VARIABLE) {
        params[isp][jsp].rotc1 = params[jsp][isp].rotc1 = atof(words[6]);
        params[isp][jsp].rotc2 = params[jsp][isp].rotc2 = atof(words[7]);
```

**Reason:** When parsing cross-species rotational variables, `params[jsp][isp].rotc2` (the symmetric counterpart) was not assigned `atof(words[7])`. It relied on a subsequent `if (params[isp][jsp].rotc2 > 0)` block to mirror the value. If a user explicitly passed `0.0` for this parameter, the block was skipped, leaving `[jsp][isp]` at its initialized `-1.0` state. Later, a cleanup loop would see `-1.0` and unconditionally overwrite both `[isp][jsp]` and `[jsp][isp]` with an arithmetic average of the two individual species, quietly overriding the user's explicit `0.0` specification.

---


## 48. Hardcoded SI Boltzmann Constant Breaking CGS Polyatomic Chemistry — `src/react_tce.cpp:256`

**Buggy code:**
```cpp
  double f = -Evib;
  double kb = 1.38064852e-23;
```

**Fixed code:**
```cpp
  double f = -Evib;
  double kb = update->boltz;
```

**Reason:** The discrete vibrational energy root-finding functions (`bird_Evib` and `bird_dEvib`) statically hardcoded the Boltzmann constant in SI units ($1.38 \times 10^{-23}$). SPARTA supports both `si` and `cgs` units. If a simulation utilized `cgs`, incoming collision energies (`Evib`) were evaluated in ergs ($\sim 10^{-13}$), but the internal temperature conversion solved against the hardcoded SI constant, resulting in wildly unphysical chemistry temperatures and mathematical faults. It was swapped to use the simulation's dynamic `update->boltz` pointer.

---


## 49. Cumulative Tally Corruption during `compute_chem_rates` Dry-Runs — `src/react_tce.cpp:239`

**Buggy code:**
```cpp
        // return reaction from 1 to N

        return list[i] + 1;
      }
    }
  }
```

**Fixed code:**
```cpp
        // return reaction from 1 to N

        return list[i] + 1;
      }
      
      break;
    }
  }
```

**Reason:** In the reaction evaluation loop, probabilities are accumulated (`react_prob += ...`) into a roulette wheel. If `react_prob` overtakes the static `random_prob`, the reaction executes and returns out. However, if the `computeChemRates` flag is enabled (to tally rates without altering particles), the code skips the execution block and DOES NOT return. Because it doesn't break, the loop continues, and the accumulating `react_prob` will unconditionally satisfy `> random_prob` for all subsequent reactions in the array, artificially inflating their statistical tallies. A `break;` statement resolves this by safely exiting the pair evaluation after the first valid outcome is tallied.

---


## 50. Missing Unit Style Catch-All Resulting in Uninitialized Memory — `src/fix_surf_temp.cpp:148`

**Buggy code:**
```cpp
  if (strcmp(update->unit_style,"si") == 0) {
    prefactor = 1.0 / (emi * SB_SI);
    threshold = 1.0e-6;
  } else if (strcmp(update->unit_style,"cgs") == 0) {
    prefactor = 1.0 / (emi * SB_CGS);
    threshold = 1.0e-3;
  }
```

**Fixed code:**
```cpp
  if (strcmp(update->unit_style,"si") == 0) {
    prefactor = 1.0 / (emi * SB_SI);
    threshold = 1.0e-6;
  } else if (strcmp(update->unit_style,"cgs") == 0) {
    prefactor = 1.0 / (emi * SB_CGS);
    threshold = 1.0e-3;
  } else error->all(FLERR,"Fix surf/temp requires si or cgs units");
```

**Reason:** If the simulation was compiled or executed with any unit system other than "si" or "cgs" (e.g. "lj", "real", or a developer's custom unit module), the `if/else if` block was entirely bypassed. This left `prefactor` and `threshold` as uninitialized garbage memory which was subsequently fed directly into the `pow()` calculation for the Stefan-Boltzmann equation, completely corrupting surface temperatures without throwing any errors. An explicit fallback error was added.

---


## 51. Stale Pointers from Caching Compute/Fix Memory in Constructor — `src/fix_surf_temp.cpp:177`

**Buggy code:**
```cpp
void FixSurfTemp::init()
{
  if (!firstflag) return;
  firstflag = 0;
```

**Fixed code:**
```cpp
void FixSurfTemp::init()
{
  // resolve source pointers in case they were modified/reordered

  if (source == COMPUTE) {
    icompute = modify->find_compute(id_qw);
    if (icompute < 0) error->all(FLERR,"Could not find fix surf/temp compute ID");
    cqw = modify->compute[icompute];
  } else if (source == FIX) {
    ifix = modify->find_fix(id_qw);
    if (ifix < 0) error->all(FLERR,"Could not find fix surf/temp fix ID");
    fqw = modify->fix[ifix];
  }

  if (!firstflag) return;
  firstflag = 0;
```

**Reason:** The pointers to the dependency `Compute` (`cqw`) and `Fix` (`fqw`) were being cached strictly inside the constructor. In SPARTA, users frequently execute multiple runs, deleting and redefining computes between runs. If the dependency was deleted and redefined, the memory location changed, and `FixSurfTemp` was left holding a dangling pointer to freed memory. Attempting to invoke `cqw->post_process_surf()` during `end_of_step()` would instantly trigger a segmentation fault. Standard SPARTA protocol mandates resolving dependencies by ID dynamically inside `init()`, which is executed immediately before every new simulation run.

---


## 52. Missing Division-by-Zero Checks and `else` Branches in C++ `compute_lambda_grid.cpp`

**Location:** `src/compute_lambda_grid.cpp:660`

**Reason:** This file contained the exact same bugs as its Kokkos counterpart (Bugs 19 & 38). Specifically:
1. `knyflag` and `knzflag` lacked the `else` statement before `array_grid` assignment, meaning the code would unconditionally attempt to write to 2D array output memory even if the user only requested a 1D vector (`noutputs == 1`), leading to memory corruption.
2. The Knudsen calculations divided `lambda` directly by grid dimensions (`sizex`, `sizey`, `sizez`, `sizeall`) without validating that the dimensions were non-zero. For zero-volume cells or lower-dimensional systems, this caused immediate NaN output generation. It was patched to enforce a non-zero check, outputting `0.0` if the dimension is empty.

---


## 53. Double-Free Segfault from Dangling Pointers in `Update` class — `src/update.cpp:1534`

**Buggy code:**
```cpp
  delete [] glist_active;
  delete [] slist_active;
  delete [] blist_active;

  glist_compute = slist_compute = blist_compute = NULL;
```

**Fixed code:**
```cpp
  delete [] glist_active;
  delete [] slist_active;
  delete [] blist_active;

  glist_compute = slist_compute = blist_compute = NULL;
  glist_active = slist_active = blist_active = NULL;
```

**Reason:** Inside `Update::tally_setup()`, the `glist_active` array is `delete[]`'d. Following this, the code loops to count the active computes. If the count is greater than zero, it reallocates the pointer `glist_active = new Compute*[]`. However, if the user explicitly deleted computes between `run` commands such that the count evaluates to `0`, the reallocation is bypassed. Because the pointer was never `NULL`'ed out after deletion, it remains a dangling pointer pointing to freed memory. When the user executes the next `run` command or cleanly exits the simulation, `Update::~Update` or a subsequent `tally_setup()` unconditionally calls `delete[] glist_active;` again, causing an immediate double-free segmentation fault that kills the entire MPI cluster.

---


## 54. Uninitialized Pointer Segfault in `ReactBird` default constructor — `src/react_bird.cpp:68`

**Buggy code:**
```cpp
ReactBird::ReactBird(SPARTA *sparta) : React(sparta)
{
  rlist = NULL;
  reactions = NULL;
  list_ij = NULL;
  sp2recomb_ij = NULL;
}
```

**Fixed code:**
```cpp
ReactBird::ReactBird(SPARTA *sparta) : React(sparta)
{
  rlist = NULL;
  reactions = NULL;
  list_ij = NULL;
  sp2recomb_ij = NULL;
  tally_reactions = NULL;
  tally_reactions_all = NULL;
}
```

**Reason:** The base constructor `ReactBird(SPARTA*)` omitted the initialization of `tally_reactions` and `tally_reactions_all`. When instantiated this way, these pointers contain random garbage heap data. When the class destructs, it unconditionally calls `delete[] tally_reactions`, which attempts to free unallocated/garbage memory, triggering a segmentation fault. Setting them to `NULL` allows `delete[]` to safely bypass them.

---


## 55. Integer Overflow in `Particle::size_restart()` — `src/particle.cpp:1412`

**Buggy code:**
```cpp
int Particle::size_restart()
{
  int n = sizeof(int);
  n = IROUNDUP(n);
  n += nlocal * sizeof(OnePartRestart);
  n += nlocal * sizeof_custom();
  n = IROUNDUP(n);
  return n;
}
```

**Fixed code:**
```cpp
int Particle::size_restart()
{
  bigint n = sizeof(int);
  n = BIROUNDUP(n);
  n += nlocal * sizeof(OnePartRestart);
  n += nlocal * sizeof_custom();
  n = BIROUNDUP(n);
  if (n > MAXSMALLINT)
    error->one(FLERR,"Per-processor particle count is too big for restart chunk");
  return static_cast<int>(n);
}
```

**Reason:** In simulations with extreme particle densities (e.g. >26 million particles per processor), the product `nlocal * sizeof(OnePartRestart)` evaluates to a 64-bit `size_t` value exceeding 2.14 billion. When this value was added to the 32-bit `int n`, it immediately forced a signed integer overflow. This would corrupt memory alignment and trigger segfaults during restart chunk mapping. The variable `n` was upgraded to `bigint` with a `MAXSMALLINT` bounds-check before the final return cast.

---


## 56. Array Initialization Overflows in `Grid` Collation — `src/grid_collate.cpp`

**Buggy code:**
```cpp
  if (nlocal) memset(&out[0][0],0,nlocal*ncol*sizeof(double));
// ...
  memory->create(in_rvous,2*nsend,"grid:in_rvous");
```

**Fixed code:**
```cpp
  if (nlocal) memset(&out[0][0],0,(size_t)nlocal*ncol*sizeof(double));
// ...
  memory->create(in_rvous,(bigint)2*nsend,"grid:in_rvous");
```

**Reason:** In high-density or clumped decompositions, memory allocation buffers and initialization `memset` commands utilize complex scaling parameters (e.g. `nlocal * ncol`). Because these operands are evaluated as 32-bit signed integers *before* being passed to `memset` or cast by `memory->create`, they can silently overflow 2.14 billion on large architectures. The resulting negative array size limits corrupt buffer headers and cause segmentation faults. A `(size_t)` cast was added to `memset` parameters and a `(bigint)` cast was applied inside `memory->create`.

---


## 57. Array Expansion Overflows in `Grid` Custom Values — `src/grid_custom.cpp`

**Buggy code:**
```cpp
        if (nnew > nold)
          memset(&iarray[nold][0],0,
                 (nnew-nold)*eicol[ewhich[ic]]*sizeof(int));
```

**Fixed code:**
```cpp
        if (nnew > nold)
          memset(&iarray[nold][0],0,
                 (size_t)(nnew-nold)*eicol[ewhich[ic]]*sizeof(int));
```

**Reason:** Similar to bug #57, the array expansion logic for custom dynamic properties (during particle adaptation or creation bursts) evaluates multi-term expansions. Casting the leading integer to `(size_t)` ensures that evaluating `N * width * byte_size` scales in 64-bit bounds and avoids catastrophic memory buffer overruns.

---


## 58. Array Allocation Overflows in `Grid` Constructor — `src/grid.cpp`

**Buggy code:**
```cpp
  memory->create(set1,nlocal*nface,"grid:set1");
```

**Fixed code:**
```cpp
  memory->create(set1,(bigint)nlocal*nface,"grid:set1");
```

**Reason:** Core topological lists mapping cells to face geometries (e.g., `set1` and `set2`) scale heavily with grid resolution. Because `nlocal * nface` is evaluated as a 32-bit `int`, a fine grid mesh can easily overflow this multiplier. Adding a `(bigint)` cast prevents standard 2GB allocation failures.

---


## 59. Array Initialization Overflows in `Surf` Custom Values — `src/surf_custom.cpp`

**Buggy code:**
```cpp
        int **iarray = memory->grow(eiarray[ewhich[index]],
                                    nnew,eicol[ewhich[index]],"surf:eiarray");
        if (nnew > nold)
        memset(iarray[nold],0,(nnew-nold)*eicol[ewhich[index]]*sizeof(int));
```

**Fixed code:**
```cpp
        int **iarray = memory->grow(eiarray[ewhich[index]],
                                    nnew,eicol[ewhich[index]],"surf:eiarray");
        if (nnew > nold)
        memset(iarray[nold],0,(size_t)(nnew-nold)*eicol[ewhich[index]]*sizeof(int));
```

**Reason:** A parallel vulnerability to `grid_custom.cpp`. The `surf` array expanses utilize identical 32-bit `memset` buffer sizes during reallocation bounds, meaning highly faceted internal boundaries could exceed max integer capacities. Applied `(size_t)` bounds.

---


## 60. Array Spread Overflows in `Surf` Communication — `src/surf_comm.cpp`

**Buggy code:**
```cpp
    memory->create(myvec,nlocal*n,"surf/spread:myvec");
// ...
    memory->create(ibuf,(n+1)*nunique,"spread/local2own:ibuf");
```

**Fixed code:**
```cpp
    memory->create(myvec,(bigint)nlocal*n,"surf/spread:myvec");
// ...
    memory->create(ibuf,(bigint)(n+1)*nunique,"spread/local2own:ibuf");
```

**Reason:** In cross-processor boundary exchanges, standard 32-bit allocations using multi-node scale factors (e.g. `nlocal * n`) can overflow, leaving parallel nodes attempting to write to null or partial arrays and crashing with segmentation faults. Applying `(bigint)` type-casts secures these boundaries for exascale execution.

---


## 61. Array Sizing Overflows in `Surf` Compressions — `src/surf.cpp`

**Buggy code:**
```cpp
  memory->create(proclist,n*6,"surf:proclist");
// ...
    memset(&tris[old],0,(nmax-old)*sizeof(Tri));
```

**Fixed code:**
```cpp
  memory->create(proclist,(bigint)n*6,"surf:proclist");
// ...
    memset(&tris[old],0,(size_t)(nmax-old)*sizeof(Tri));
```

**Reason:** The core memory management functions for boundary surface primitives (Lines and Triangles) calculated bounds identically to the `grid` components. `n * 6` and `(nmax - old) * sizeof(...)` evaluate as 32-bit values and silently wrap negative. By typing to `(bigint)` and `(size_t)`, the array bounds remain protected against structural overflows.

---


## 62. Divide-by-Zero (NaN) in Axisymmetric Vector Remapping — `src/update.h` & `src/geometry.cpp`

**Buggy code:**
```cpp
    double ynew = x[1] + t1*v[1];
    double znew = x[2] + t1*v[2];
    xc[1] = sqrt(ynew*ynew + znew*znew);
    xc[2] = 0.0;
    
    double rn = ynew / xc[1];
    double wn = znew / xc[1];
    vc[0] = v[0];
    vc[1] = v[1]*rn + v[2]*wn;
    vc[2] = -v[1]*wn + v[2]*rn;
```

**Fixed code:**
```cpp
    double ynew = x[1] + t1*v[1];
    double znew = x[2] + t1*v[2];
    xc[1] = sqrt(ynew*ynew + znew*znew);
    xc[2] = 0.0;

    vc[0] = v[0];
    if (xc[1] > 0.0) {
      double rn = ynew / xc[1];
      double wn = znew / xc[1];
      vc[1] = v[1]*rn + v[2]*wn;
      vc[2] = -v[1]*wn + v[2]*rn;
    } else {
      vc[1] = v[1];
      vc[2] = v[2];
    }
```

**Reason:** In the DSMC axisymmetric trajectory mapping, the `axi_remap` and `axi_line_intersect` routines compute a cylindrical vector rotation by calculating the rotational angle parameters `rn` and `wn` from the updated radial coordinate `xc[1]`. If a particle perfectly intersects the centerline axis of symmetry (`y = 0` and `z = 0`), the new radius `xc[1]` evaluates exactly to `0.0`. Dividing by this radius triggers a fatal divide-by-zero, creating `NaN` velocity vectors that cascade and silently destroy particle coordinate integrity. A bounding check was added across all instances, including the Kokkos GPU accelerated ports (`src/KOKKOS/update_kokkos.h` and `src/KOKKOS/geometry_kokkos.h`).

---


## 63. `sprintf` Buffer Overflow Risk in Fix and Compute Subsystems — `src/fix*.cpp`, `src/compute*.cpp`, `src/fix*.h`, `src/compute*.h` (Multiple files)

**Reason:** Widespread use of deprecated `sprintf` for string formatting, lacking buffer size limits, similar to the vulnerabilities found in the `grid` and `surf` subsystems. This triggers AppleClang security warnings and creates potential buffer overflow vectors. Executed a Python script to systematically replace all `sprintf(buffer, ...)` calls with secure `snprintf(buffer, sizeof(buffer), ...)` across all fix and compute headers and source files.

**Buggy code:**
```cpp
sprintf(buffer, "some format %d", value);
```

**Fixed code:**
```cpp
snprintf(buffer, sizeof(buffer), "some format %d", value);
```

---

## 64. 32-bit Integer Overflow in `compute_gas_reaction_grid.cpp` Memory Allocation — `src/compute_gas_reaction_grid.cpp`

**Reason:** In the `reallocate()` and `init()` methods, `memset` was clearing memory using the size calculation `nglocal * ncol * sizeof(double)`. Both `nglocal` (local grid cells) and `ncol` (number of columns) were 32-bit integers. Their product could overflow a 32-bit integer before implicit promotion to `size_t`, leading to a negative or undersized memory wipe if a rank contained a massive number of grid cells. Cast the first operand to `size_t` to force 64-bit precision during the multiplication: `((size_t)nglocal) * ncol * sizeof(double)`. Similar `((size_t)nglocal)` casts were applied to `compute_gas_collision_grid.cpp`.

**Buggy code:**
```cpp
memset(array, 0, nglocal * ncol * sizeof(double));
```

**Fixed code:**
```cpp
memset(array, 0, ((size_t)nglocal) * ncol * sizeof(double));
```

---

## 65. 32-bit Integer Overflow in `fix_emit` Subsystem Memory Allocations — `src/fix_emit_face.cpp`, `src/fix_emit_face_file.cpp`, `src/fix_emit_surf.cpp`

**Reason:** Variables tracking the maximum number of active cells (`maxactive`) and tasks (`ntaskmax`) were scaling memory allocations (`memset(activecell,0,maxactive*sizeof(int))`). Because `maxactive` tracks `grid->nlocal`, it could theoretically exceed safe bounds for 32-bit integer multiplication, corrupting memory space. Applied a safe explicit cast: `((size_t)maxactive)*sizeof(int)` and `((size_t)ntaskmax-oldmax)*sizeof(Task)`.

**Buggy code:**
```cpp
memset(activecell, 0, maxactive * sizeof(int));
```

**Fixed code:**
```cpp
memset(activecell, 0, ((size_t)maxactive) * sizeof(int));
```

---

## 66. Fatal Divide-by-Zero in `fix_ablate_multi_inner.cpp` — `src/fix_ablate_multi_inner.cpp`

**Reason:** The logic calculated the parameter `perout = total/Ninterface;` where `Ninterface` was the return value of `find_ninter()`. If `nsurf` was non-zero but topological edge cases caused `find_ninter()` to return 0, the program would crash due to a floating-point exception (SIGFPE). Added a bounding check to ensure `Ninterface > 0` before performing the division, safely setting `perout = 0.0` otherwise.

**Buggy code:**
```cpp
perout = total / Ninterface;
```

**Fixed code:**
```cpp
if (Ninterface > 0) perout = total / Ninterface;
else perout = 0.0;
```

---

## 67. Local Variable Shadowing and Null Dereference — `src/fix_ablate.cpp`

**Reason:** The variable `idsource` was locally declared (`char *idsource = new char[n];`), shadowing the class member variable. This caused a memory leak and left the class member as `NULL`, which resulted in a guaranteed segmentation fault when `Variable::find()` later attempted to call `strcmp()` on it. Removed the `char *` type declaration to correctly assign the allocated memory to the class member.

**Buggy code:**
```cpp
char *idsource = new char[n];
strcpy(idsource, arg[i+1]);
```

**Fixed code:**
```cpp
idsource = new char[n];
strcpy(idsource, arg[i+1]);
```

---

## 68. Severe Out-of-Bounds Array Access — `src/compute_tvib_grid.cpp`

**Reason:** In `modeflag == 2`, the loop `index` was a combination of group ID and mode, iterating up to `ngroup * maxmode - 1`. It was used directly to access `groupspecies[index]`, which is only dimensioned to `ngroup`. This caused an out-of-bounds read into unallocated memory. Changed the array indexing to use integer division `groupspecies[index / maxmode]` to correctly isolate the group ID.

**Buggy code:**
```cpp
if (groupspecies[index] < 0) continue;
```

**Fixed code:**
```cpp
if (groupspecies[index / maxmode] < 0) continue;
```

---

## 69. String Tokenization Logic Error — `src/compute_react_surf.cpp`, `src/compute_react_boundary.cpp`, `src/compute_react_isurf_grid.cpp`

**Reason:** Inside the `strtok` loop for parsing slash-separated reactant strings (e.g., `r:O/N`), `reaction2col[ireaction][icol] = 0;` wiped out the array on every iteration. This inadvertently erased valid matches found by previous tokens. Removed the destructive reset line (the array is already properly zero-initialized before the loop).

**Buggy code:**
```cpp
while (word != NULL) {
  reaction2col[ireaction][icol] = 0; // Wipes array inside loop
  // ... token logic ...
}
```

**Fixed code:**
```cpp
while (word != NULL) {
  // Array initialization removed, token logic remains
  // ... token logic ...
}
```

---

## 70. Fatal Divide-by-Zero in Temperature Rescaling — `src/fix_temp_rescale.cpp`

**Reason:** The velocity scaling factor was calculated via `vscale = sqrt(t_target / t_current);` without safety bounds. If particles shared the exact same velocity or cooled to absolute zero, `t_current` became `0.0`, resulting in a division by zero and `NaN` propagation. Added conditional bounds to default `vscale = 1.0` if `t_current <= 0.0`, and secured a similar `/= n_current` division.

**Buggy code:**
```cpp
vscale = sqrt(t_target / t_current);
```

**Fixed code:**
```cpp
if (t_current <= 0.0) vscale = 1.0;
else vscale = sqrt(t_target / t_current);
```

---

## 71. Under-reported Memory Usage Metrics — `src/compute_eflux_grid.cpp`, `src/compute_grid.cpp`, `src/compute_pflux_grid.cpp`, `src/compute_sonine_grid.cpp`, `src/compute_thermal_grid.cpp`, `src/compute_tvib_grid.cpp`, `src/compute_lambda_grid.cpp`

**Reason:** In the `memory_usage()` functions, the accumulator variable `bytes` was repeatedly overwritten with `=` instead of accumulated with `+=`. Additionally, `bytes` was often declared uninitialized (`bigint bytes;`). Explicitly initialized `bigint bytes = 0;` across all affected files and refactored the assignments to correctly use `bytes +=`.

**Buggy code:**
```cpp
bigint bytes;
bytes = ngrid * sizeof(double);
bytes = ngrid * ncol * sizeof(double);
```

**Fixed code:**
```cpp
bigint bytes = 0;
bytes += ngrid * sizeof(double);
bytes += ngrid * ncol * sizeof(double);
```

---

## 72. Rejection Sampling Probability Variable Mismatch — `src/react_qk.cpp`, `src/react_tce_qk.cpp`

**Reason:** In the quantum-kinetic (QK) reaction attempt logic, the local state probability was assigned to the wrong variable (`react_prob` instead of `prob`) inside the rejection sampling loop `do { ... } while()`. This fundamentally broke the rejection sampling algorithm because `react_prob` was not zeroed out on rejected states, artificially inflating the overall reaction probability and violating detailed balance.

**Buggy code:**
```cpp
do {
  iv = static_cast<int> (random->uniform()*(maxlev+0.99999999));
  evib = static_cast<double> (iv / inverse_kT);
  if (evib < ecc) react_prob = pow(1.0-evib/ecc,1.5-omega);
} while (random->uniform() < react_prob);
```

**Fixed code:**
```cpp
prob = 0.0;
do {
  iv = static_cast<int> (random->uniform()*(maxlev+0.99999999));
  evib = static_cast<double> (iv / inverse_kT);
  if (evib < ecc) prob = pow(1.0-evib/ecc,1.5-omega);
} while (random->uniform() < prob);
```

---

## 73. Massive PRNG State Race Condition — `src/KOKKOS/collide_vss_kokkos.cpp`

**Reason:** In the NTC collision algorithms, the Kokkos PRNG state (`rand_gen`) was incorrectly yielded back to the pool (`rand_pool.free_state(rand_gen)`) prior to a `continue;` statement inside the main collision loop. This caused the same thread to immediately re-use a freed PRNG state, creating a massive race condition where multiple threads simultaneously mutated identical PRNG state blocks, generating perfectly correlated numbers and violating stochastic validity.

**Buggy code:**
```cpp
if (precoln.vr2 < vremax*vremax*random_prob) {
  rand_pool.free_state(rand_gen);
  continue;
}
```

**Fixed code:**
```cpp
if (precoln.vr2 < vremax*vremax*random_prob) {
  continue;
}
```

---

## 74. Hardcoded SI Constants vs Dynamic Units — `src/KOKKOS/react_tce_kokkos.h`

**Reason:** In the Newton-Raphson search for the instantaneous vibrational temperature, the Kokkos implementation hardcoded the Boltzmann constant strictly to SI units (`const double kb = 1.38064852e-23;`), while the base C++ class dynamically pulled the defined simulation units (`update->boltz`). When using non-SI units, the Kokkos solver evaluated unphysical target vibrational temperatures, silently corrupting polyatomic chemistry. Replaced with `boltz`.

**Buggy code:**
```cpp
const double kb = 1.38064852e-23;
```

**Fixed code:**
```cpp
const double kb = boltz;
```

---

## 75. Flawed RNG Initial State (Pool Correlation) — `src/KOKKOS/react_bird_kokkos.cpp`, `src/KOKKOS/react_tce_kokkos.cpp`

**Reason:** Both the collision classes and the reaction classes initialized their independent `Kokkos::Random_XorShift64_Pool` structures using the exact same random seed formula (`12345 + comm->me`). Because thread processing overlaps, this meant early-step PRNG streams were perfectly correlated between collisions and chemical evaluations. Changed the seed offset to break the correlation.

**Buggy code:**
```cpp
rand_pool = rand_pool_type(12345 + comm->me);
```

**Fixed code:**
```cpp
rand_pool = rand_pool_type(54321 + comm->me);
```

---

## 76. Divide-by-Zero Vulnerabilities in Reaction/Scattering Math — `src/KOKKOS/react_tce_kokkos.h`, `src/KOKKOS/collide_vss_kokkos.cpp`, `src/react_tce.cpp`

**Reason:** Multiple physical formulas lacked bounds checks against zero-energy states. In `react_tce`, `pow(1.0-r->d_coeff[1]/ecc, ...)` would divide by zero if collision energy `ecc` was zero. In `collide_vss_kokkos`, `sqrt( ... / (mr * precoln.vr2))` would divide by zero if relative velocity `vr2` was perfectly zero (especially common in newly generated recombination particles). Added `ecc <= 0.0` and `vr2 > 0.0` bounds checks.

**Buggy code:**
```cpp
// Missing bounds checks
react_prob += ... * pow(1.0-r->d_coeff[1]/ecc, ...);
```

**Fixed code:**
```cpp
if (e_excess <= 0.0 || ecc <= 0.0) continue;
react_prob += ... * pow(1.0-r->d_coeff[1]/ecc, ...);
```

---

## 77. Divide-By-Zero in Particle Kinematics (`frac`) — `src/update.cpp`, `src/KOKKOS/update_kokkos.cpp`

**Reason:** When checking if a particle crossed a bounding box face, the timestep fraction `frac` was calculated as `(lo - x) / (xnew - x)`. If the velocity vector was exactly zero (`xnew == x`) but `x` was infinitesimally outside the boundary (`x < lo`) due to floating-point drift, this resulted in a division-by-zero. This turned `frac` to `NaN` and silently removed the particle from the simulation entirely. Added an explicit `xnew != x` check.

**Buggy code:**
```cpp
if (xnew[0] < lo[0]) {
  frac = (lo[0]-x[0]) / (xnew[0]-x[0]);
  outface = XLO;
}
```

**Fixed code:**
```cpp
if (xnew[0] < lo[0]) {
  if (xnew[0] != x[0]) frac = (lo[0]-x[0]) / (xnew[0]-x[0]);
  else frac = 0.0;
  // ... followed by clamping logic ...
  outface = XLO;
}
```

---

## 78. Infinite Energy & Time Travel via Unclamped `frac` — `src/update.cpp`, `src/KOKKOS/update_kokkos.cpp`

**Reason:** The timestep fraction `frac` used for trajectory bounding intersections was never bounded. Due to floating-point noise and particles starting slightly outside bounds, `frac` could evaluate to negative values or values `> 1.0`. Since the remaining timestep is updated as `dtremain *= 1.0 - frac`, a negative `frac` artificially increased `dtremain` (injecting free energy/distance), while `frac > 1.0` made `dtremain` negative (causing the particle to physically advect backward in time and space). Added explicit clamps to enforce a strict `[0.0, 1.0]` range.

**Buggy code:**
```cpp
frac = (lo[0]-x[0]) / (xnew[0]-x[0]);
dtremain *= 1.0 - frac; // If frac < 0, dtremain increases
```

**Fixed code:**
```cpp
frac = (lo[0]-x[0]) / (xnew[0]-x[0]);
if (frac < 0.0) frac = 0.0;
else if (frac > 1.0) frac = 1.0;
dtremain *= 1.0 - frac;
```

---

## 79. Zeno's Paradox / Infinite Advection Loop — `src/update.cpp`, `src/KOKKOS/update_kokkos.cpp`

**Reason:** To prevent particles from getting permanently wedged in acute surface corners, the code tracked a `stuck_iterate` counter to discard particles hitting `MAXSTUCK`. However, it only incremented if the collision parameter was strictly zero (`if (minparam == 0.0)`). In practice, trapped particles often compute an infinitesimally small float (e.g. `1.0e-15`), resetting the counter and causing an infinite `while` loop that hangs the thread. Changed the threshold to `minparam <= 1.0e-14`.

**Buggy code:**
```cpp
if (minparam == 0.0) stuck_iterate++;
else stuck_iterate = 0;
```

**Fixed code:**
```cpp
if (minparam <= 1.0e-14) stuck_iterate++;
else stuck_iterate = 0;
```

---

## 80. Catastrophic Cancellation in Axisymmetric Intersections — `src/geometry.cpp`, `src/KOKKOS/geometry_kokkos.h`

**Reason:** `Geometry::axi_horizontal_line` utilized the standard algebraic quadratic formula `(b - sarg) / a`. For grazing trajectories where `b` and `sarg` share the same sign and magnitude, this caused catastrophic cancellation (subtracting two nearly identical floats), yielding extreme precision noise and entirely incorrect intersection times. Rewrote the quadratic derivation using Vieta's formulas (e.g., `c / (b + sarg)`) for numerically stable root selection.

**Buggy code:**
```cpp
double arg = yhoriz*yhoriz*a - v[2]*v[2]*x[1]*x[1];
double sarg = sqrt(arg);
nc = 2;
double tone = (b - sarg) / a;
double ttwo = (b + sarg) / a;
```

**Fixed code:**
```cpp
double arg = yhoriz*yhoriz*a - v[2]*v[2]*x[1]*x[1];
double sarg = sqrt(arg);
double c = x[1]*x[1] - yhoriz*yhoriz;
nc = 2;
double tone, ttwo;
if (b > 0.0) {
  ttwo = (b + sarg) / a;
  tone = c / (b + sarg);
} else if (b < 0.0) {
// ...
```

---


## 81. Algorithmic State Corruption in `perform_coarsen()` — `src/adapt_grid.cpp`

**Reason:** When coarsening a grid cell that intersects surfaces, the `grid->coarsen_cell()` method invokes `surf2grid_one(splitflag=1)`, appending multiple sub-cells to the `grid->cells` list. However, `newcell` was incorrectly assigned to `grid->nlocal - 1`, pointing to the last appended sub-cell rather than the actual parent cell. This misdirected subsequent updates to the cell type and group mask, leaving the parent cell and other sub-cells with uninitialized or corrupted state masks (e.g., holding a temporary particle index).

**Buggy code:**
```cpp
    grid->coarsen_cell(...);
    // ...
    newcell = grid->nlocal - 1;
```

**Fixed code:**
```cpp
    int nglocalprev = grid->nlocal;
    grid->coarsen_cell(...);
    // ...
    newcell = nglocalprev;
```

---

## 82. 32-bit Integer Overflow in Rendezvous Network Buffers — `src/adapt_grid.cpp`

**Reason:** In `particle_surf_comm()`, the size of the `SendAdapt` rendezvous buffer was calculated as `nsend * sizeof(SendAdapt)`. Both `nsend` and the struct size were evaluated as 32-bit integers. During massive parallel grid adaptations, this product could overflow, resulting in a negative sign-extended allocation request to `memory->smalloc()`, causing a segmentation fault.

**Buggy code:**
```cpp
  SendAdapt *sadapt = (SendAdapt *) memory->smalloc(nsend*sizeof(SendAdapt),
                                                    "adapt_grid:sadapt");
```

**Fixed code:**
```cpp
  SendAdapt *sadapt = (SendAdapt *) memory->smalloc((bigint) nsend*sizeof(SendAdapt),
                                                    "adapt_grid:sadapt");
```

---

## 83. 32-bit Integer Overflow in Adaptation Lists — `src/adapt_grid.cpp`

**Reason:** Memory reallocations for `clist` and `alist` used 32-bit arithmetic `cnummax * sizeof(CList)` and `anummax * sizeof(ActionList)`. Similar to the rendezvous buffers, high levels of refinement/coarsening across large domain sizes caused these arrays to request sizes exceeding 2GB, wrapping into negative values and corrupting the heap.

**Buggy code:**
```cpp
clist = (CList *) memory->srealloc(clist,cnummax*sizeof(CList), "adapt_grid:clist");
// ...
alist = (ActionList *) memory->srealloc(alist,anummax*sizeof(ActionList), "adapt_grid:alist");
```

**Fixed code:**
```cpp
clist = (CList *) memory->srealloc(clist,(bigint) cnummax*sizeof(CList), "adapt_grid:clist");
// ...
alist = (ActionList *) memory->srealloc(alist,(bigint) anummax*sizeof(ActionList), "adapt_grid:alist");
```

---

## 84. Multi-Dimensional Volume (`nxyz`) Overflows — `src/adapt_grid.cpp`, `src/grid_adapt.cpp`, `src/adapt_grid.h`

**Reason:** Loop limits and geometric counts related to 3D cell subdivision (`nxyz`, `nmax`, `m`, `offset`, `nchild`) were typed as 32-bit `int`s. If users requested dense grid generation (e.g., `adapt_grid ... cells 2000 2000 2000`), the local volumetric bounding variables would overflow. This triggered negative loop limits, out-of-bounds array accesses, and corrupted child-parent ID mapping logic.

**Buggy code:**
```cpp
  int m,n,level,nxyz,nchild;
  // ...
  nxyz = plevels[level-1].nxyz;
```

**Fixed code:**
```cpp
  int level;
  bigint m,n,nxyz,nchild;
  // ...
  nxyz = plevels[level-1].nxyz;
```

## Bug 85: Kokkos PRNG Logarithm Infinite Generation (Ambipolar)
- **File:** `src/KOKKOS/fix_ambipolar_kokkos.h`
- **Bug:** `Kokkos::drand()` returns values in `[0.0, 1.0)`, meaning it can return exactly `0.0`. When computing the normal and radial velocity of an emitted electron using `sqrt(-log(rand_gen.drand()))`, a roll of `0.0` resulted in `+Infinity`, permanently corrupting the electron's trajectory and breaking the bounding box.
- **Fix:** Mapped the interval to `(0.0, 1.0]` by subtracting the random number from `1.0`, perfectly aligning it with the CPU `RanKnuth` behavior and preventing infinity evaluations.
- **Buggy code:**
```cpp
const double vn = vscale * sqrt(-log(rand_gen.drand()));
const double vr = vscale * sqrt(-log(rand_gen.drand()));
```
- **Fixed code:**
```cpp
const double vn = vscale * sqrt(-log(1.0 - rand_gen.drand()));
const double vr = vscale * sqrt(-log(1.0 - rand_gen.drand()));
```

## Bug 86: Kokkos PRNG Logarithm Quantum State Corruption (Vibmode)
- **File:** `src/KOKKOS/fix_vibmode_kokkos.h`
- **Bug:** Similar to Bug 85, computing the discrete quantum vibrational mode `ivib` using `-log(rand_gen.drand())`. A roll of `0.0` caused the expression to evaluate to `+Infinity`. Casting `+Infinity` to an `int` via `static_cast<int>` is undefined behavior in C++, immediately corrupting the particle's quantum energy state.
- **Fix:** Subtracted the Kokkos random roll from `1.0` to prevent taking the logarithm of zero.
- **Buggy code:**
```cpp
const int ivib = static_cast<int> (-log(rand_gen.drand()) * temp_vib / d_species[isp].vibtemp[imode]);
```
- **Fixed code:**
```cpp
const int ivib = static_cast<int> (-log(1.0 - rand_gen.drand()) * temp_vib / d_species[isp].vibtemp[imode]);
```

## Bug 87: Kokkos PRNG Bounds Corruption in Emission, Collision and Particle Initializers
- **Files:** `src/KOKKOS/fix_emit_surf_kokkos.cpp`, `src/KOKKOS/fix_emit_face_kokkos.cpp`, `src/KOKKOS/surf_collide_diffuse_kokkos.h`, `src/KOKKOS/particle_kokkos.h`
- **Bug:** The same Kokkos PRNG `drand()` boundary mapping flaw `[0.0, 1.0)` was used 13 times across surface emissions, face emissions, diffuse scattering, and thermal state initializers. Taking `-log(drand())` or `-log(1.0 - drand())` without ensuring it's not `0.0` could evaluate to `+Infinity`, permanently corrupting particle velocities and rotational/vibrational energy states.
- **Fix:** Standardized all instances across the codebase to `drand()` or `(1.0 - drand())` carefully to prevent evaluating `-log(0.0)`. Used `(1.0 - rand_gen.drand())` inside `-log()` to map correctly to `(0.0, 1.0]`.


## Bug 88: Array Allocation Integer Overflow — `src/create_isurf.cpp`
- **File:** `src/create_isurf.cpp:1893`
- **Bug:** `nsurf * nbytes` evaluates as a 32-bit integer. For high-resolution grid surface extractions, this exceeds 2.14 billion and wraps to negative, throwing a bad allocation or corrupting memory.
- **Fix:** Cast to `(bigint)` and `(size_t)` to preserve 64-bit geometry sizing.
- **Buggy code:**
```cpp
llines = (Surf::Line *) memory->smalloc(nsurf*nbytes,"createisurf:lines");
```
- **Fixed code:**
```cpp
llines = (Surf::Line *) memory->smalloc((bigint)nsurf*nbytes,"createisurf:lines");
memcpy(llines,surf->mylines,(size_t)nsurf*nbytes);
```

## Bug 89: Stack Buffer Overflows (`sprintf` limits) — `src/create_*.cpp`, `src/dump_movie.cpp`
- **Files:** `src/create_grid.cpp`, `src/create_isurf.cpp`, `src/create_particles.cpp`, `src/dump_movie.cpp`
- **Bug:** Widespread usage of `sprintf(str, ...)` to write format sequences into a fixed 128-byte array.
- **Fix:** Replaced with `snprintf(str, 128, ...)` and `snprintf(moviecmd, 1024, ...)` to guard against stack overflows.

## Bug 90: Critical MPI Buffer Integer Overflow — `src/create_isurf.cpp`
- **File:** `src/create_isurf.cpp` (around line 991)
- **Bug:** The maximum MPI send buffer size `maxsbuf` (and receive buffer `maxrbuf`) were defined as 32-bit `int`s. The dynamic array reallocation computed `nsend * ncomm` as a 32-bit operation. For grids larger than ~9 million cells per proc, this integer math overflowed, wrapping into a massive negative value. The negative value was then passed to `memory->create`, corrupting memory bounds.
- **Fix:** Upgraded `maxsbuf` and `maxrbuf` to `bigint` in `create_isurf.h`, and cast the equation limits: `if ((bigint)nsend*ncomm > maxsbuf)`.

## Bug 91: Mathematical Division-by-Zero in Voxel Cell Volumes — `src/create_isurf.cpp`
- **File:** `src/create_isurf.cpp:1554`
- **Bug:** Calculating corner point averages from voxel fractions evaluated `sfrac = (dx*dy*dz - cvol) / (dx*dy*dz);`. If the geometric box bounds `dx*dy*dz` calculate exactly to `0.0` due to spatial intersections or 2D edge cases, this threw a division by zero.
- **Fix:** Abstracted `boxvol` and guarded the division: `if (boxvol > 0.0) sfrac = ...; else sfrac = 0.0;`.

## Bug 92: `Infinity` State Corruption in Surface Corner Values — `src/create_isurf.cpp`
- **File:** `src/create_isurf.cpp:1851`
- **Bug:** `param2cval()` maps geometric intersects. The intersection scaling formula `v0 = (thresh - v1*param) / (1.0 - param)` suffers a division-by-zero if floating-point noise or a tangent geometric intersection pushes `param` exactly to `1.0`. The result evaluates as `Infinity` and gets propagated across the spatial arrays.
- **Fix:** Implemented a boundary clamp: `if (param >= 1.0) param = 0.999999;` to preserve numerical limits for surface thresholds.

## Bug 93: Fix Memory Allocation Integer Overflows — `src/fix_move_surf.cpp`, `src/fix_emit_face.cpp`, `src/fix_emit_face_file.cpp`, `src/fix_emit_surf.cpp`
- **Files:** `src/fix_move_surf.cpp` (lines 87, 91), `src/fix_emit_face.cpp` (line 1108), `src/fix_emit_face_file.cpp` (line 1261), `src/fix_emit_surf.cpp` (line 1491)
- **Bug:** `nsurf*sizeof(Surf::Line)` and `ntaskmax*sizeof(Task)` evaluated as 32-bit `int` multiplication prior to passing to `memory->smalloc` and `memory->srealloc`. If `nsurf` or `ntaskmax` were large, this would overflow and allocate corrupted memory sizes, or crash the process.
- **Fix:** Explicitly cast to `(bigint)` inside `memory->smalloc`/`srealloc` and `(size_t)` for `memcpy` bounds.

## Bug 94: Stack Buffer Overflows via `sprintf` — `src/KOKKOS/fix_grid_check_kokkos.cpp`
- **File:** `src/KOKKOS/fix_grid_check_kokkos.cpp` (lines 170-225)
- **Bug:** Multiple instances wrote variable-length error messages to a fixed 128-byte `char str[128]` using `sprintf`.
- **Fix:** Replaced all instances with `snprintf(str, 128, ...)` to guard the stack.

## Bug 95: Extreme Memory Loop Typo & Corruption — `src/fix_emit_face_file.cpp`
- **File:** `src/fix_emit_face_file.cpp:1237`
- **Bug:** The inner loop mapping thermal mixtures iterated `for (m = 0; m < nspecies; i++)`. By mistakenly incrementing the outer loop index `i` instead of the inner loop index `m`, `m` never advanced (causing an infinite inner loop constraint), while `i` aggressively incremented until it flew out-of-bounds of the `tasks[i]` array, corrupting memory.
- **Fix:** Fixed the typo: `for (m = 0; m < nspecies; m++)`.

## Bug 96: Fatal Division-By-Zero via `massrho_cell` — `src/fix_emit_face.cpp`, `src/fix_emit_face_file.cpp`, `src/fix_emit_surf.cpp`
- **Files:** `src/fix_emit_face.cpp` (line 1073), `src/fix_emit_face_file.cpp` (line 1229), `src/fix_emit_surf.cpp` (line 1459)
- **Bug:** Subsonic stream velocity modifications divide by `massrho_cell * soundspeed_cell`. The code defined `massrho_cell = 0.0` if a grid cell had a volume of exactly `0.0`. If a particle existed in a zero-volume cell (edge case geometries), the math attempted to divide by `0.0`, propagating `NaN` back to the particles.
- **Fix:** Added guards to `vstream` accumulations: `if (np && massrho_cell * soundspeed_cell > 0.0)`. And safely handled `tasks[i].nrho` with bounds checks on `temp_thermal_cell`.

## Bug 97: Fatal Division-by-Zero in Temperature Rescaling — `src/fix_temp_rescale.cpp`
- **File:** `src/fix_temp_rescale.cpp:295`
- **Bug:** The rescaling calculated `t_current /= n_current;` and `vscale = sqrt(t_target/t_current);`. If the total cells were zero, or more critically, if the current global thermal temperature `t_current` was zero, `vscale` generated `Infinity`.
- **Fix:** Inserted an early-exit check identical to global limits: `if (n_current == 0 || t_current == 0.0) return;`.

## Bug 98: Scope Shadowing & Memory Leak — `src/fix_ablate.cpp`
- **File:** `src/fix_ablate.cpp` (around line 118)
- **Bug:** An inline variable `char *idsource = new char[n];` was instantiated instead of utilizing the class member variable `idsource`. This effectively leaked the allocated memory instantly, while leaving the class-level `idsource` initialized to `NULL`. Later, when the `find()` function was invoked on the `NULL` pointer, it would segment fault.
- **Fix:** Removed the local `char *` instantiation and assigned the memory directly to the member variable: `idsource = new char[n];`.

## Bug 99: Array Parser Memory Leaks — `src/fix_ablate.cpp`, `src/fix_ave_*.cpp`
- **Files:** `src/fix_ablate.cpp`, `src/fix_ave_grid.cpp`, `src/fix_ave_histo.cpp`, `src/fix_ave_surf.cpp`, `src/fix_ave_time.cpp`
- **Bug:** Extracted `char *suffix = new char[n];` allocations were orphaned if the command argument syntax was invalid (e.g., missing bracket). SPARTA would throw an `error->all()` exception, abandoning the pointer without deallocation. 
- **Fix:** Pre-emptively freed `delete [] suffix;` just prior to the exception throw.

## Bug 100: Divide-by-Zero via Maximum Most Probable Speed — `src/compute_dt_grid.cpp`, `src/KOKKOS/compute_dt_grid_kokkos.cpp`
- **Files:** `src/compute_dt_grid.cpp` (line 645), `src/KOKKOS/compute_dt_grid_kokkos.cpp` (line 359)
- **Bug:** The cell timestep limit calculation derived `vrm_max = sqrt(2.0*update->boltz * temp[i] / min_species_mass);`. Without guarding if `vrm_max` was `0.0` (which occurs if cell temperature is 0, such as empty space initialized or early equilibrium), the next calculation `dt_candidate = transit_fraction*dx/vrm_max;` resulted in a division by exactly zero, generating `Infinity` and ruining the simulation timestep logic.
- **Fix:** Wrapped the logic bounds: `if (vrm_max > 0.0) { dt_candidate = ... }`.

## Bug 101: Divide-by-Zero via Empty Cell Volume Math — `src/compute_thermal_grid.cpp`, `src/compute_eflux_grid.cpp`, `src/compute_pflux_grid.cpp` (and KOKKOS)
- **Files:** `src/compute_thermal_grid.cpp`, `src/KOKKOS/compute_thermal_grid_kokkos.cpp`, `src/compute_eflux_grid.cpp`, `src/KOKKOS/compute_eflux_grid_kokkos.cpp`, `src/compute_pflux_grid.cpp`, `src/KOKKOS/compute_pflux_grid_kokkos.cpp`
- **Bug:** Energy, thermal, and momentum flux tallies accumulated `vec[k] *= cinfo[icell].weight / cinfo[icell].volume / nsample;`. Although SPARTA guards `nsample` and usually maintains `weight`, a 2D edge case or floating-point anomaly yielding a `volume` of exactly `0.0` threw an immediate division by zero, creating a `NaN` state array on grid outputs.
- **Fix:** Added guards to explicitly enforce `if (cinfo[icell].volume > 0.0)` before dividing, else storing `0.0`.

## Bug 102: Stack Buffer Overflow via `sprintf` — `src/KOKKOS/compute_fft_grid_kokkos.cpp`
- **File:** `src/KOKKOS/compute_fft_grid_kokkos.cpp`
- **Bug:** `print_FFT_info()` utilized a rigid 64-byte `char str[64]` to output compiler macros `SPARTA_FFT_PREC` and `SPARTA_FFT_KOKKOS_LIB` using `sprintf(str, ...)`. If users compiled with customized, longer library prefix macro strings, this would silently overflow the stack buffer during initialization output.
- **Fix:** Swapped `sprintf(str, ...)` to `snprintf(str, 64, ...)`.

## Bug 103: Array Parser Memory Leaks — `src/compute_lambda_grid.cpp`, `src/compute_reduce.cpp`
- **Files:** `src/compute_lambda_grid.cpp` (line 195), `src/compute_reduce.cpp` (line 134)
- **Bug:** Like the `fix` modules, parsing grid arguments cloned strings into `char *suffix = new char[n];`. If parsing encountered an error (e.g., mismatched brackets), SPARTA's `error->all()` was called, aborting the procedure and leaving the `suffix` string pointer orphaned and memory leaked. 
- **Fix:** Handled memory cleanup with `delete [] suffix;` before calling `error->all()`.

## 105. Divide-by-zero in collision volume scaling — `src/collide_vss.cpp:144, 175`

**Buggy code:**
```cpp
  double nattempt;

  if (remainflag) {
    nattempt = 0.5 * np * (np-1) *
      vremax[icell][0][0] * dt * fnum / volume + remain[icell][0][0];
```

**Fixed code:**
```cpp
  double nattempt = 0.0;

  if (volume > 0.0) {
    if (remainflag) {
      nattempt = 0.5 * np * (np-1) *
        vremax[icell][0][0] * dt * fnum / volume + remain[icell][0][0];
```

**Reason:** In VSS collision attempts, if a cell's collision volume is zero (which can happen under some transient conditions or edge cases), dividing by `volume` throws a floating-point exception or results in `NaN/Inf`. Added `volume > 0.0` guards.

---

## 106. Divide-by-zero on `ecc == 0.0` in reaction probabilities — `src/react_qk.cpp:130, 160`, `src/react_tce.cpp:106, 155`, `src/react_tce_qk.cpp:144, 234`

**Buggy code:**
```cpp
            do {
              iv =  static_cast<int> (random->uniform()*(maxlev+0.99999999));
              evib = static_cast<double> (iv / inverse_kT);
              if (evib < ecc) react_prob = pow(1.0-evib/ecc,1.5-omega);
            } while (random->uniform() < react_prob);
```

**Fixed code:**
```cpp
            do {
              iv =  static_cast<int> (random->uniform()*(maxlev+0.99999999));
              evib = static_cast<double> (iv / inverse_kT);
              if (evib < ecc && ecc > 0.0) prob = pow(1.0-evib/ecc,1.5-omega);
              else prob = 0.0;
            } while (random->uniform() < prob);
```

**Reason:** Mathematical equations simulating reaction probabilities utilize the total center-of-mass energy `ecc` as a denominator. If colliding particles happen to have near-zero energies (e.g. extremely low temperatures or exact thermal minimums), `ecc` equates to `0.0`. Taking `pow(1.0 - evib / 0.0, ...)` leads to `NaN` propagation, crashing the simulation.

---

## 107. Zero division for surface tangency/velocity magnitudes — `src/surf_react_adsorb.cpp:807, 830`

**Buggy code:**
```cpp
        double dot = MathExtra::dot3(ip->v,norm);

        if (r->nreactant == 1) {
          prob_value[i] = 2.0 * r->k_react *
            (maxstick - total_state[isurf]) * ms_inv / fabs(dot);
```
and
```cpp
          double cos_theta = abs(dot) / sqrt(vmag_sq);
```

**Fixed code:**
```cpp
        double dot = MathExtra::dot3(ip->v,norm);

        if (dot == 0.0) {
          prob_value[i] = 0.0;
        } else {
...
```
and
```cpp
          if (vmag_sq > 0.0) {
            double cos_theta = abs(dot) / sqrt(vmag_sq);
```

**Reason:** If a particle strikes a surface tangentially or has `0.0` velocity vector, `fabs(dot)` or `sqrt(vmag_sq)` respectively become zero. Dividing by this value results in a divide-by-zero floating point error.

---

## 108. Memory leak in string parsing exception — `src/variable.cpp:1286`

**Buggy code:**
```cpp
        int icompute = modify->find_compute(id);
        if (icompute < 0)
          error->all(FLERR,"Invalid compute ID in variable formula");
        Compute *compute = modify->compute[icompute];
        delete [] id;
```

**Fixed code:**
```cpp
        int icompute = modify->find_compute(id);
        if (icompute < 0) {
          delete [] id;
          error->all(FLERR,"Invalid compute ID in variable formula");
        }
        Compute *compute = modify->compute[icompute];
        delete [] id;
```

**Reason:** The dynamically allocated string `id` is deleted *after* `error->all()` if the command parser finds valid execution. If parsing encounters an exception `error->all()` is thrown *before* `delete [] id`, resulting in a memory leak.

## 109. Stack buffer overflow vulnerabilities — `src/grid.cpp:1285, 1888, 1939, 1949, 1963, 2432` and `src/input.cpp:268, 518, 770, 892, 1170` etc.

**Buggy code:**
```cpp
    char str[128];
    sprintf(str,"Grid cell interior corner points marked as unknown "
            "(volume will be wrong if cell is effectively outside) = %d",
            insideall);
```

**Fixed code:**
```cpp
    char str[128];
    snprintf(str, 128,"Grid cell interior corner points marked as unknown "
            "(volume will be wrong if cell is effectively outside) = %d",
            insideall);
```

**Reason:** Replaced numerous instances of `sprintf` into fixed 128/256-byte stack buffers with `snprintf` to prevent buffer overflows if arguments (like long integers, or strings parsed from user inputs) cause the message to exceed the buffer length.

---

## 110. Potential buffer overflow in deep grid IDs — `src/dump.cpp:438`, `src/dump_grid.cpp:368`, `src/grid_id.cpp:544`

**Buggy code:**
```cpp
// in Dump and DumpGrid
  char str[32];
// in Grid::id_num2str
  sprintf(&str[offset],CELLINT_FORMAT,ichild);
```

**Fixed code:**
```cpp
// in Dump and DumpGrid
  char str[128];
// in Grid::id_num2str
  if (offset >= 110) break; // prevent buffer overflow
  snprintf(&str[offset], 128 - offset, CELLINT_FORMAT, ichild);
```

**Reason:** `id_num2str` writes cell ID strings level by level separated by hyphens. A 64-bit integer ID could be deeply nested. A maximum string length could be up to 64 bytes for deep trees, which would overflow the `char str[32]` allocation in `dump.cpp` and `dump_grid.cpp`. Increased allocation to `128` and added length guards in `grid_id.cpp`.

---

## 111. Memory leak in invalid `if` commands — `src/input.cpp:1003, 1058`

**Buggy code:**
```cpp
    char **commands = new char*[ncommands];
    ncommands = 0;
    for (int i = first; i <= last; i++) {
      int n = strlen(arg[i]) + 1;
      if (n == 1) error->all(FLERR,"Illegal if command");
      commands[ncommands] = new char[n];
      ...
```

**Fixed code:**
```cpp
    char **commands = new char*[ncommands];
    ncommands = 0;
    for (int i = first; i <= last; i++) {
      int n = strlen(arg[i]) + 1;
      if (n == 1) {
        for (int j = 0; j < ncommands; j++) delete [] commands[j];
        delete [] commands;
        error->all(FLERR,"Illegal if command");
      }
      commands[ncommands] = new char[n];
      ...
```

**Reason:** If an `if` command parsed an empty string (`n == 1`), `error->all()` was called, leaking the previously populated strings in the `commands` array and the `commands` array itself.

## 112. Memory leak in invalid grid group operations — `src/grid.cpp:2240, 2275, 2302`

**Buggy code:**
```cpp
    int length = narg-3;
    int *list = new int[length];

    int jgroup;
    for (int iarg = 3; iarg < narg; iarg++) {
      jgroup = find_group(arg[iarg]);
      if (jgroup == -1) error->all(FLERR,"Group ID does not exist");
      list[iarg-3] = jgroup;
    }
```

**Fixed code:**
```cpp
    int length = narg-3;
    int *list = new int[length];

    int jgroup;
    for (int iarg = 3; iarg < narg; iarg++) {
      jgroup = find_group(arg[iarg]);
      if (jgroup == -1) {
        delete [] list;
        error->all(FLERR,"Group ID does not exist");
      }
      list[iarg-3] = jgroup;
    }
```

**Reason:** For `subtract`, `union`, and `intersect` grid group commands, an array `list` is dynamically allocated. If any parsed group ID is invalid, an exception is thrown without first deallocating `list`, creating a memory leak. Added `delete [] list;`.

---

## Open Questions / For Investigation

### The original Bug 53 (`fix_ablate.cpp` `epsilon_adjust`)
During a code review of `src/fix_ablate.cpp` (specifically the `epsilon_adjust()` method around line 1110), an agent flagged the following code as a bug:
```cpp
    for (i = 0; i < ncorner; i++)
      if (cvalues[icell][i] >= thresh && cvalues[icell][i] < thresh + EPSILON)
        cvalues[icell][i] = thresh - EPSILON;
      else if (cvalues[icell][i] < thresh && cvalues[icell][i] > thresh - EPSILON)
        cvalues[icell][i] = thresh - EPSILON;
```
The agent incorrectly assumed that pushing a value from `>= thresh` to `thresh - EPSILON` was a topological inversion typo, and changed it to `thresh + EPSILON`. 

This "fix" introduced a catastrophic regression in the `exp2imp.sphere.3d` test (causing a `Calculated solid fraction above one or negative` crash). We have reverted the agent's change because the original logic was deliberately enforcing a single-sided topological boundary for Marching Cubes to prevent zero-area degenerate triangles. 

**Question for investigation:** This incident highlights the fragility of the `epsilon_adjust` logic and its tight coupling to the core grid volume calculation (`cinfo[icell].volume`). The original logic is correct, but is it robust against all edge cases? Can this method be documented or fortified to prevent future developers from making the same intuitive (but mathematically fatal) assumption?