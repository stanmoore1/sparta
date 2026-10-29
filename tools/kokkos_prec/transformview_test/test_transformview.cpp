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

// unit test of the TransformView sync state machine (kokkos_type.h):
//   every edge of the triangle legacy host (H) / Kokkos host (K) /
//   device (D), need_sync_*(), resize in each modified state, copies
//   sharing their sync state, the keep rule of kk_convert(), the struct
//   conversion, and the abort on concurrent modification
// built and run by ctest (test KokkosTransformView) when the KOKKOS
//   package and SPARTA_ENABLE_TESTING are enabled

#include "kokkos_type.h"

#include <cstdio>
#include <sys/wait.h>
#include <unistd.h>

using namespace SPARTA_NS;

static int nfail = 0;
static int ncheck = 0;

#define CHECK(cond)                                                     \
  do {                                                                  \
    ncheck++;                                                           \
    if (!(cond)) {                                                      \
      nfail++;                                                          \
      printf("FAILED line %d: %s\n", __LINE__, #cond);                  \
    }                                                                   \
  } while (0)

// a sync that is needed, which a DualView without transform does not
//   report when the device is the host

#define CHECK_SYNC(cond) CHECK(!(TV::NEED_TRANSFORM || !TV::SINGLE_DEVICE) || (cond))

// a TransformView with a transform in every precision mode, and the KK
//   precision one SPARTA uses (a DualView in a double precision build)

typedef TransformView<float*, double*, Kokkos::LayoutRight> tv_float;
typedef TransformView<float**, double**, Kokkos::LayoutRight> tv_float_2d;
typedef DAT::ttransform_kkfloat_1d tv_kk;

static_assert(tv_float::NEED_TRANSFORM);
static_assert(tv_kk::NEED_TRANSFORM == !std::is_same_v<KK_FLOAT,double>);
// no transform: the legacy view is the Kokkos host view
static_assert(tv_kk::NEED_TRANSFORM ||
              std::is_same_v<tv_kk::t_host,tv_kk::kk_view::t_host>);

// device view values, copied to the host

template<class TV>
auto device_values(const TV &tv)
{
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),tv.view_device());
}

// run a kernel on the device view: d(i) = a*d(i) + b

template<class TV>
void device_update(const TV &tv, const double a, const double b)
{
  auto d = tv.view_device();
  typedef typename TV::kk_value_type T;
  const T ta = static_cast<T>(a), tb = static_cast<T>(b);
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,d.extent(0)),
                       KOKKOS_LAMBDA(const int i) { d(i) = ta*d(i) + tb; });
  Kokkos::fence();
}

template<class TV>
void test_1d(const char *name)
{
  typedef typename TV::kk_value_type T;
  const int n = 17;
  TV tv(std::string(name),n);
  auto h = tv.view_host();

  // a new view is in sync

  CHECK(!tv.need_sync_host());
  CHECK(!tv.need_sync_hostkk());
  CHECK(!tv.need_sync_device());

  // H -> D

  for (int i = 0; i < n; i++) h(i) = i + 0.1;
  tv.modify_host();
  CHECK_SYNC(tv.need_sync_device());
  CHECK_SYNC(tv.need_sync_hostkk());
  CHECK(!tv.need_sync_host());
  tv.sync_device();
  CHECK(!tv.need_sync_device());
  CHECK(!tv.need_sync_hostkk());
  CHECK(!tv.need_sync_host());
  {
    auto d = device_values(tv);
    auto k = tv.view_hostkk();
    for (int i = 0; i < n; i++) {
      CHECK(d(i) == static_cast<T>(i + 0.1));
      CHECK(k(i) == static_cast<T>(i + 0.1));
    }
  }

  // a device modification that leaves the KK values unchanged keeps the
  //   exact legacy values (kk_convert keep rule)

  tv.modify_device();
  CHECK_SYNC(tv.need_sync_host());
  tv.sync_host();
  CHECK(!tv.need_sync_host());
  for (int i = 0; i < n; i++) CHECK(h(i) == i + 0.1);

  // D -> H

  device_update(tv,2.0,1.0);
  tv.modify_device();
  CHECK_SYNC(tv.need_sync_host());
  CHECK(!tv.need_sync_device());
  tv.sync_host();
  CHECK(!tv.need_sync_host());
  CHECK(!tv.need_sync_hostkk());
  {
    auto d = device_values(tv);
    for (int i = 0; i < n; i++) CHECK(h(i) == static_cast<double>(d(i)));
  }

  // D -> K, then K -> H

  device_update(tv,0.0,7.0);
  tv.modify_device();
  tv.sync_hostkk();
  CHECK_SYNC(tv.need_sync_host());
  for (int i = 0; i < n; i++) CHECK(tv.view_hostkk()(i) == static_cast<T>(7.0));
  tv.sync_host();
  CHECK(!tv.need_sync_host());
  for (int i = 0; i < n; i++) CHECK(h(i) == 7.0);

  // H -> K, then K -> D

  for (int i = 0; i < n; i++) h(i) = 3.0*i;
  tv.modify_host();
  tv.sync_hostkk();
  CHECK(!tv.need_sync_hostkk());
  CHECK(!tv.need_sync_host());
  for (int i = 0; i < n; i++) CHECK(tv.view_hostkk()(i) == static_cast<T>(3.0*i));
  tv.sync_device();
  CHECK(!tv.need_sync_device());
  {
    auto d = device_values(tv);
    for (int i = 0; i < n; i++) CHECK(d(i) == static_cast<T>(3.0*i));
  }

  // K -> H and K -> D

  for (int i = 0; i < n; i++) tv.view_hostkk()(i) = static_cast<T>(5.0);
  tv.modify_hostkk();
  CHECK_SYNC(tv.need_sync_host());
  tv.sync_device();
  {
    auto d = device_values(tv);
    for (int i = 0; i < n; i++) CHECK(d(i) == static_cast<T>(5.0));
  }
  CHECK_SYNC(tv.need_sync_host());
  tv.sync_host();
  CHECK(!tv.need_sync_host());
  for (int i = 0; i < n; i++) CHECK(h(i) == 5.0);

  // modify<>()/sync<>() on the host space refer to the Kokkos host view

  device_update(tv,1.0,1.0);
  tv.template modify<DeviceType>();
  tv.template sync<SPAHostType>();
  for (int i = 0; i < n; i++) CHECK(tv.view_hostkk()(i) == static_cast<T>(6.0));
  CHECK_SYNC(tv.need_sync_host());
  tv.sync_host();
  for (int i = 0; i < n; i++) CHECK(h(i) == 6.0);

  // copies share their sync state, as copies of a DualView do

  {
    TV copy = tv;
    for (int i = 0; i < n; i++) copy.view_host()(i) = 11.0;
    copy.modify_host();
    CHECK_SYNC(tv.need_sync_device());
    tv.sync_device();
    CHECK(!copy.need_sync_device());
    auto d = device_values(tv);
    for (int i = 0; i < n; i++) CHECK(d(i) == static_cast<T>(11.0));
    device_update(copy,1.0,1.0);
    copy.modify_device();
    CHECK_SYNC(tv.need_sync_host());
    tv.sync_host();
    CHECK(!copy.need_sync_host());
    for (int i = 0; i < n; i++) CHECK(copy.view_host()(i) == 12.0);
  }

  // clear_sync_state

  tv.modify_host();
  tv.clear_sync_state();
  CHECK(!tv.need_sync_device());
  CHECK(!tv.need_sync_hostkk());
  tv.modify_device();
  tv.clear_sync_state();
  CHECK(!tv.need_sync_host());

  // resize with a modified legacy view: contents reach the device

  for (int i = 0; i < n; i++) tv.view_host()(i) = i + 0.5;
  tv.modify_host();
  tv.resize(2*n);
  CHECK(tv.view_host().extent(0) == (size_t) 2*n);
  CHECK(tv.extent(0) == (size_t) 2*n);
  CHECK_SYNC(tv.need_sync_device());
  tv.sync_device();
  {
    auto d = device_values(tv);
    for (int i = 0; i < n; i++) CHECK(d(i) == static_cast<T>(i + 0.5));
    for (int i = n; i < 2*n; i++) CHECK(d(i) == static_cast<T>(0.0));
  }

  // resize with a modified device view: contents reach the legacy view

  device_update(tv,1.0,1.0);
  tv.modify_device();
  tv.resize(3*n);
  CHECK_SYNC(tv.need_sync_host());
  tv.sync_host();
  for (int i = 0; i < n; i++) CHECK(tv.view_host()(i) == static_cast<double>(static_cast<T>(i + 0.5) + static_cast<T>(1.0)));
  for (int i = n; i < 2*n; i++) CHECK(tv.view_host()(i) == 1.0);
  for (int i = 2*n; i < 3*n; i++) CHECK(tv.view_host()(i) == 0.0);

  // resize with a modified Kokkos host view

  for (int i = 0; i < 3*n; i++) tv.view_hostkk()(i) = static_cast<T>(2.0);
  tv.modify_hostkk();
  tv.resize(4*n);
  tv.sync_host();
  tv.sync_device();
  {
    auto d = device_values(tv);
    for (int i = 0; i < 3*n; i++) {
      CHECK(tv.view_host()(i) == 2.0);
      CHECK(d(i) == static_cast<T>(2.0));
    }
  }

  // resize of an unmodified view keeps all three copies

  tv.resize(Kokkos::view_alloc(Kokkos::WithoutInitializing),3*n);
  // (a DualView marks the preserved side modified)
  CHECK(!TV::NEED_TRANSFORM || !tv.need_sync_host());
  tv.sync_device();
  tv.sync_hostkk();
  {
    auto d = device_values(tv);
    for (int i = 0; i < 3*n; i++) {
      CHECK(tv.view_host()(i) == 2.0);
      CHECK(tv.view_hostkk()(i) == static_cast<T>(2.0));
      CHECK(d(i) == static_cast<T>(2.0));
    }
  }

  // a view constructed with an initializing view_alloc() is zero on all
  //   three copies, one constructed WithoutInitializing is valid

  {
    TV z(Kokkos::view_alloc(std::string(name) + "_z"),n);
    auto dz = device_values(z);
    for (int i = 0; i < n; i++) {
      CHECK(z.view_host()(i) == 0.0);
      CHECK(z.view_hostkk()(i) == static_cast<T>(0.0));
      CHECK(dz(i) == static_cast<T>(0.0));
    }
    CHECK(!z.need_sync_host());
    TV u(Kokkos::view_alloc(Kokkos::WithoutInitializing,std::string(name) + "_u"),n);
    CHECK(u.view_host().extent(0) == (size_t) n);
    for (int i = 0; i < n; i++) u.view_host()(i) = 4.0;
    u.modify_host();
    u.sync_device();
    auto du = device_values(u);
    for (int i = 0; i < n; i++) CHECK(du(i) == static_cast<T>(4.0));
  }

  // a default constructed view has nothing to sync, and becomes a
  //   working view when resized

  {
    TV e;
    e.modify_host();
    e.sync_device();
    e.modify_device();
    e.sync_host();
    CHECK(!e.need_sync_host());
    e.resize(n);
    for (int i = 0; i < n; i++) e.view_host()(i) = 9.0;
    e.modify_host();
    CHECK_SYNC(e.need_sync_device());
    e.sync_device();
    auto de = device_values(e);
    for (int i = 0; i < n; i++) CHECK(de(i) == static_cast<T>(9.0));
  }
}

// 2d LayoutRight view: rows of the legacy view can be aliased by double**

void test_2d()
{
  const int n = 5, m = 3;
  tv_float_2d tv("tv_float_2d",n,m);
  auto h = tv.view_host();
  for (int i = 0; i < n; i++) {
    CHECK(&h(i,0) == h.data() + i*m);
    for (int j = 0; j < m; j++) h(i,j) = 10.0*i + j + 0.25;
  }
  tv.modify_host();
  tv.sync_device();
  auto d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),tv.view_device());
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) CHECK(d(i,j) == static_cast<float>(10.0*i + j + 0.25));
  auto dv = tv.view_device();
  Kokkos::parallel_for(Kokkos::MDRangePolicy<DeviceType,Kokkos::Rank<2>>({0,0},{n,m}),
                       KOKKOS_LAMBDA(const int i, const int j) { dv(i,j) += 1.0f; });
  Kokkos::fence();
  tv.modify_device();
  tv.sync_host();
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      CHECK(h(i,j) == static_cast<double>(static_cast<float>(10.0*i + j + 0.25) + 1.0f));
}

// per-particle struct: KK precision on the device, double on the host

void test_particles()
{
  const int n = 9;
  tdual_particle_1d tv("particles",n);
  auto h = tv.view_host();
  for (int i = 0; i < n; i++) {
    h(i).id = 100 + i;
    h(i).ispecies = i % 3;
    for (int k = 0; k < 3; k++) {
      h(i).x[k] = 0.1*i + 0.01*k + 1.0e-9;
      h(i).v[k] = 300.0 + i + 0.3*k;
    }
    h(i).erot = 1.0e-21*(i+1);
    h(i).evib = 0.0;
    h(i).dtremain = 0.0;
  }
  tv.modify_host();
  tv.sync_device();

  // the device changes velocities only

  auto d = tv.view_device();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,n),
                       KOKKOS_LAMBDA(const int i) { d(i).v[0] = -d(i).v[0]; });
  Kokkos::fence();
  tv.modify_device();
  tv.sync_host();

  for (int i = 0; i < n; i++) {
    CHECK(h(i).id == 100 + i);
    CHECK(h(i).ispecies == i % 3);
    // unchanged on the device: exact double values kept
    for (int k = 0; k < 3; k++) CHECK(h(i).x[k] == 0.1*i + 0.01*k + 1.0e-9);
    CHECK(h(i).v[1] == 300.0 + i + 0.3);
    CHECK(h(i).erot == 1.0e-21*(i+1));
    // changed on the device: the KK value
    CHECK(h(i).v[0] == -static_cast<double>(static_cast<KK_FLOAT>(300.0 + i)));
  }
}

// a legacy and a Kokkos modification with no sync in between abort, in
//   any order; checked in a child process, which must not launch a
//   parallel kernel (an OpenMP runtime is not usable after fork())

template<class F>
bool aborts(F f)
{
  fflush(stdout);
  pid_t pid = fork();
  if (pid == 0) {
    // silence the abort message of the expected failure
    if (!freopen("/dev/null","w",stderr)) _exit(2);
    f();
    _exit(0);
  }
  int status = 0;
  waitpid(pid,&status,0);
  return !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

void test_concurrent()
{
  tv_float tv("concurrent",4);
  CHECK(aborts([&]() { tv.modify_host(); tv.modify_device(); }));
  CHECK(aborts([&]() { tv.modify_device(); tv.modify_host(); }));
  CHECK(aborts([&]() { tv.modify_host(); tv.modify_hostkk(); }));
  CHECK(aborts([&]() { tv.modify_hostkk(); tv.modify_host(); }));
  // sequential modifications with a sync in between, and repeated
  //   modification of the same side, are fine; run in this process, since
  //   a sync launches a host parallel_for, which an OpenMP runtime cannot
  //   run in a forked child
  tv.modify_host(); tv.sync_device(); tv.modify_device();
  tv.sync_host(); tv.modify_host(); tv.sync_hostkk();
  tv.modify_hostkk(); tv.sync_host(); tv.modify_host();
  tv.sync_device();
  tv.modify_device(); tv.modify_device();
  tv.sync_host();
  tv.modify_host(); tv.modify_host();
  tv.sync_device();
  CHECK(!tv.need_sync_host());
  CHECK(!tv.need_sync_device());
}

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc,argv);

  test_1d<tv_float>("tv_float");
  test_1d<tv_kk>("tv_kk");
  test_2d();
  test_particles();
  test_concurrent();

  printf("TransformView test: %d of %d checks passed (KK_FLOAT %s, "
         "KK_POS_FLOAT %s, device %s the host)\n",
         ncheck - nfail, ncheck, std::is_same_v<KK_FLOAT,float> ? "float" : "double",
         std::is_same_v<KK_POS_FLOAT,float> ? "float" : "double",
         tv_float::SINGLE_DEVICE ? "is" : "is not");
  return nfail ? 1 : 0;
}
