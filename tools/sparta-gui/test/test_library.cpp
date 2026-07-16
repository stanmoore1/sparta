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

// test harness for the SPARTA library interface as used by SPARTA-GUI
//
// loads libsparta.so via dlopen() like the SPARTA-GUI plugin loader and
// exercises the library functions the GUI depends on:
//   - instance lifecycle, version and configuration queries
//   - running commands from a string buffer
//   - polling the stats cache (sparta_last_thermo) from a second
//     thread while a simulation is running
//   - interrupting a run with sparta_force_timeout and running again
//   - error capture and recovery after a failed command
//   - style and ID enumeration
//
// usage: test_library <path/to/libsparta.so> <workdir>
// workdir must contain the data files of examples/circle
// (data.circle, air.species, air.vss)

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include <dlfcn.h>
#include <unistd.h>

// function pointer types for the subset of the library used here

typedef void *(*open_no_mpi_t)(int, char **, void **);
typedef void (*close_t)(void *);
typedef char *(*command_t)(void *, const char *);
typedef void (*commands_string_t)(void *, const char *);
typedef void *(*extract_global_t)(void *, const char *);
typedef int (*extract_setting_t)(void *, const char *);
typedef double (*get_thermo_t)(void *, const char *);
typedef void *(*last_thermo_t)(void *, const char *, int);
typedef int (*version_t)(void *);
typedef int (*style_count_t)(void *, const char *);
typedef int (*style_name_t)(void *, const char *, int, char *, int);
typedef int (*id_count_t)(void *, const char *);
typedef int (*id_name_t)(void *, const char *, int, char *, int);
typedef int (*has_id_t)(void *, const char *, const char *);
typedef int (*variable_info_t)(void *, int, char *, int);
typedef int (*is_running_t)(void *);
typedef void (*force_timeout_t)(void *);
typedef int (*has_error_t)(void *);
typedef int (*get_last_error_message_t)(void *, char *, int);
typedef int (*config_int_t)();
typedef int (*config_str_t)(const char *);

#define RESOLVE(name) \
  name = (name##_t) dlsym(handle,"sparta_" #name); \
  if (!name) { \
    fprintf(stderr,"FAIL: could not resolve symbol sparta_%s\n",#name); \
    return 1; \
  }

static int nfail = 0;
static int npass = 0;

static void check(bool cond, const char *msg)
{
  if (cond) {
    ++npass;
    printf("PASS: %s\n",msg);
  } else {
    ++nfail;
    printf("FAIL: %s\n",msg);
  }
  fflush(stdout);
}

// setup portion of the circle example (examples/circle/in.circle)

static const char *setup_deck =
  "seed             12345\n"
  "dimension        2\n"
  "global           gridcut 0.0 comm/sort yes\n"
  "boundary         o r p\n"
  "create_box       0 10 0 10 -0.5 0.5\n"
  "create_grid      20 20 1\n"
  "balance_grid     rcb cell\n"
  "global           nrho 1.0 fnum 0.001\n"
  "species          air.species N O\n"
  "mixture          air N O vstream 100.0 0 0\n"
  "read_surf        data.circle\n"
  "surf_collide     1 diffuse 300.0 0.0\n"
  "surf_modify      all collide 1\n"
  "collide          vss air air.vss\n"
  "create_particles air n 0 twopass\n"
  "fix              in emit/face air xlo twopass\n"
  "timestep         0.001\n"
  "compute          myturb temp\n"
  "stats            10\n"
  "stats_style      step cpu np nscoll c_myturb\n";

int main(int argc, char **argv)
{
  if (argc != 3) {
    fprintf(stderr,"usage: %s <path/to/libsparta.so> <workdir>\n",argv[0]);
    return 1;
  }

  if (chdir(argv[2]) != 0) {
    fprintf(stderr,"FAIL: cannot chdir to %s\n",argv[2]);
    return 1;
  }

  void *handle = dlopen(argv[1],RTLD_NOW|RTLD_GLOBAL);
  if (!handle) {
    fprintf(stderr,"FAIL: dlopen: %s\n",dlerror());
    return 1;
  }
  printf("PASS: dlopen %s\n",argv[1]);

  open_no_mpi_t open_no_mpi;
  close_t close;
  command_t command;
  commands_string_t commands_string;
  extract_global_t extract_global;
  extract_setting_t extract_setting;
  get_thermo_t get_thermo;
  last_thermo_t last_thermo;
  version_t version;
  style_count_t style_count;
  style_name_t style_name;
  id_count_t id_count;
  id_name_t id_name;
  has_id_t has_id;
  variable_info_t variable_info;
  is_running_t is_running;
  force_timeout_t force_timeout;
  has_error_t has_error;
  get_last_error_message_t get_last_error_message;

  RESOLVE(open_no_mpi)
  RESOLVE(close)
  RESOLVE(command)
  RESOLVE(commands_string)
  RESOLVE(extract_global)
  RESOLVE(extract_setting)
  RESOLVE(get_thermo)
  RESOLVE(last_thermo)
  RESOLVE(version)
  RESOLVE(style_count)
  RESOLVE(style_name)
  RESOLVE(id_count)
  RESOLVE(id_name)
  RESOLVE(has_id)
  RESOLVE(variable_info)
  RESOLVE(is_running)
  RESOLVE(force_timeout)
  RESOLVE(has_error)
  RESOLVE(get_last_error_message)

  config_int_t config_has_exceptions =
    (config_int_t) dlsym(handle,"sparta_config_has_exceptions");
  config_str_t config_has_package =
    (config_str_t) dlsym(handle,"sparta_config_has_package");
  check(config_has_exceptions && config_has_exceptions() == 1,
        "config_has_exceptions");

  // create an instance; suppress screen output via -screen none

  const char *args[] = {"sparta","-screen","none","-log","test_library.log"};
  void *spa = NULL;
  open_no_mpi(5,(char **) args,&spa);
  check(spa != NULL,"open_no_mpi creates instance");
  if (!spa) return 1;

  check(version(spa) >= 20250101,"version returns plausible date");

  // style enumeration

  int ncmd = style_count(spa,"command");
  check(ncmd > 10,"style_count(command) > 10");
  check(style_count(spa,"compute") > 20,"style_count(compute) > 20");
  check(style_count(spa,"fix") > 20,"style_count(fix) > 20");
  check(style_count(spa,"dump") >= 6,"style_count(dump) >= 6");
  check(style_count(spa,"surf_collide") >= 5,"style_count(surf_collide) >= 5");

  char buffer[256];
  bool found_run = false;
  bool found_image = false;
  for (int i = 0; i < ncmd; i++)
    if (style_name(spa,"command",i,buffer,256) &&
        strcmp(buffer,"run") == 0) found_run = true;
  for (int i = 0; i < style_count(spa,"dump"); i++)
    if (style_name(spa,"dump",i,buffer,256) &&
        strcmp(buffer,"image") == 0) found_image = true;
  check(found_run,"style_name finds command style 'run'");
  check(found_image,"style_name finds dump style 'image'");

  // set up the circle example

  commands_string(spa,setup_deck);
  check(has_error(spa) == 0,"setup deck runs without error");

  check(extract_setting(spa,"dimension") == 2,"extract_setting(dimension)");
  check(extract_setting(spa,"box_exist") == 1,"extract_setting(box_exist)");
  check(extract_setting(spa,"grid_exist") == 1,"extract_setting(grid_exist)");
  check(extract_setting(spa,"surf_exist") == 1,"extract_setting(surf_exist)");
  check(extract_setting(spa,"nspecies") == 2,"extract_setting(nspecies)");
  check(extract_setting(spa,"bigint") == 8 ||
        extract_setting(spa,"bigint") == 4,"extract_setting(bigint)");

  double *boxhi = (double *) extract_global(spa,"boxhi");
  check(boxhi && boxhi[0] == 10.0 && boxhi[1] == 10.0,
        "extract_global(boxhi)");
  char *units = (char *) extract_global(spa,"units");
  check(units && strcmp(units,"si") == 0,"extract_global(units)");

  // ID enumeration

  check(id_count(spa,"compute") == 1,"id_count(compute) == 1");
  check(id_name(spa,"compute",0,buffer,256) == 1 &&
        strcmp(buffer,"myturb") == 0,"id_name(compute,0)");
  check(has_id(spa,"fix","in") == 1,"has_id(fix,in)");
  check(has_id(spa,"fix","bogus") == 0,"has_id(fix,bogus)");
  // SPARTA pre-defines the mixtures "all" and "species"
  check(id_count(spa,"mixture") == 3,"id_count(mixture) == 3");
  check(has_id(spa,"mixture","air") == 1,"has_id(mixture,air)");
  check(id_count(spa,"species") == 2,"id_count(species) == 2");
  check(id_count(spa,"surf_collide") == 1,"id_count(surf_collide) == 1");

  command(spa,"variable foo equal 2*3");
  check(id_count(spa,"variable") == 1,"id_count(variable) == 1");
  check(variable_info(spa,0,buffer,256) == 1 && strcmp(buffer,"foo") == 0,
        "variable_info(0)");

  // run in a worker thread and poll the stats cache from this thread,
  // like the GUI does

  std::atomic<bool> done(false);
  std::thread runner([&]() {
    commands_string(spa,"run 2000\n");
    done = true;
  });

  int64_t last_step = -1;
  int max_num = 0;
  bool saw_keywords = false;
  bool saw_running = false;

  while (!done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (is_running(spa)) saw_running = true;

    last_thermo(spa,"lock",0);
    int *setup = (int *) last_thermo(spa,"setup",0);
    if (setup && *setup == 0) {
      int64_t *step = (int64_t *) last_thermo(spa,"step",0);
      int *num = (int *) last_thermo(spa,"num",0);
      if (step && num && *num > 0) {
        if (*step > last_step) last_step = *step;
        if (*num > max_num) max_num = *num;
        char *kw = (char *) last_thermo(spa,"keyword",2);
        int *type2 = (int *) last_thermo(spa,"type",2);
        void *data2 = last_thermo(spa,"data",2);
        if (kw && strcmp(kw,"Np") == 0 && type2 && data2) saw_keywords = true;
      }
    }
    last_thermo(spa,"unlock",0);
  }
  runner.join();

  check(saw_running,"is_running true during run");
  check(last_step > 0,"last_thermo step advances during run");
  check(max_num == 5,"last_thermo num == 5 stats columns");
  check(saw_keywords,"last_thermo keyword/type/data for Np column");
  check(is_running(spa) == 0,"is_running false after run");
  check(get_thermo(spa,"step") == 2000.0,"get_thermo(step) == 2000");
  check(get_thermo(spa,"np") > 10000.0,"get_thermo(np) > 10000");

  // force_timeout stops a long run early; the following run works again

  std::thread runner2([&]() { commands_string(spa,"run 100000\n"); });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  force_timeout(spa);
  runner2.join();
  double stopstep = get_thermo(spa,"step");
  check(stopstep < 102000.0,"force_timeout stops run early");

  commands_string(spa,"run 100\n");
  check(has_error(spa) == 0,"run after force_timeout works");
  check(get_thermo(spa,"step") == stopstep + 100.0,
        "run after force_timeout completes all steps");

  // failed command sets the error status and the instance recovers

  command(spa,"this_is_not_a_command 42");
  check(has_error(spa) == 1,"bad command sets has_error");
  int type = get_last_error_message(spa,buffer,256);
  check(type == 1,"error type is recoverable");
  check(strstr(buffer,"Unknown command") != NULL,
        "error message mentions unknown command");
  check(has_error(spa) == 0,"error status cleared after retrieval");

  command(spa,"fix bad emit/face air xhi bogus_keyword");
  check(has_error(spa) == 1,"bad fix command sets has_error");
  get_last_error_message(spa,buffer,256);

  commands_string(spa,"run 100\n");
  check(has_error(spa) == 0,"instance recovers after failed commands");

  // dump image render + imagename in the stats cache

  commands_string(spa,
    "dump img image all 100 test_render.*.ppm type type pdiam 0.08 "
    "surf proc 0.02 gline yes 0.005 box yes 0.01\n"
    "dump_modify img pad 4 backcolor gray\n"
    "run 100\n"
    "undump img\n");
  check(has_error(spa) == 0,"dump image render runs without error");
  last_thermo(spa,"lock",0);
  char *imagename = (char *) last_thermo(spa,"imagename",0);
  bool has_image = imagename && strstr(imagename,"test_render") != NULL;
  last_thermo(spa,"unlock",0);
  check(has_image,"last_thermo imagename set after dump image");

  close(spa);
  printf("\n%d passed, %d failed\n",npass,nfail);
  return (nfail > 0) ? 1 : 0;
}
