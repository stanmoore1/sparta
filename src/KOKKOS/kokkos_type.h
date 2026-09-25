/* -*- c++ -*- ----------------------------------------------------------
   SPARTA - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://sparta.github.io, Sandia National Laboratories
   Steve Plimpton, sjplimp@gmail.com

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#ifndef SPARTA_STYPE_KOKKOS_H
#define SPARTA_STYPE_KOKKOS_H

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_ScatterView.hpp>

#include "particle.h"
#include "grid.h"
#include "surf.h"
#include "spatype.h"
#include "accelerator_kokkos_defs.h"

#include <cstring>
#include <type_traits>

// offset type for the Kokkos::Crs per-cell surf/split/sub lists.
// Under BIGBIG the total number of flattened entries on one rank can exceed
//   2^31, so the row_map offsets must be a bigint.  Under BIG they cannot, and
//   a 32-bit row_map halves the bytes touched by the per-particle surf
//   lookups in the move kernel.  All declarations of these Crs objects must
//   use this typedef so the device and host mirrors stay assignable.

#ifdef SPARTA_BIGBIG
typedef SPARTA_NS::bigint crs_size_type;
#else
typedef int crs_size_type;
#endif

#define NeighClusterSize 8

// SPARTA_KOKKOS_FIXED_LISTS restores the original fixed-size KKCopy arrays for
//   the per-type tally compute lists and for the per-style surf react lists,
//   in place of the runtime-sized device buffers that replaced them.  The
//   buffers exist to lift the instance caps below; the fixed arrays keep every
//   compute and surf react model inside the functor that is handed by value to
//   each kernel.
// Which is faster is a GPU question with no obvious answer: a smaller functor
//   can raise occupancy while costing data locality, and higher occupancy is
//   not the same thing as higher throughput.  Neither path has been measured
//   on an accelerator.  Both are kept so the two can be compared on real
//   hardware by rebuilding with -DSPARTA_KOKKOS_FIXED_LISTS, rather than by
//   reverting commits.

#ifdef SPARTA_KOKKOS_FIXED_LISTS
#define KOKKOS_MAX_SLIST 2
#define KOKKOS_MAX_BLIST 2
#define KOKKOS_MAX_GLIST 4
#define KOKKOS_MAX_SURF_REACT_PER_TYPE 2
#define KOKKOS_MAX_TOT_SURF_REACT 4
#endif

// architectures where the move kernel is dispatched with ATOMIC_REDUCTION = -1
//   (parallel_reduce) rather than atomics.  KokkosSPARTA::accelerator() clears
//   atomic_reduction for exactly these, and UpdateKokkos::move() reads the
//   per-step counters back from the reduction result only when it is clear.
//   The two decisions must agree or the counters come from the wrong place, so
//   the condition is spelled out once here instead of in both files.

#if defined(KOKKOS_ARCH_AMD_GFX940) || defined(KOKKOS_ARCH_AMD_GFX942) || \
    defined(KOKKOS_ARCH_AMD_GFX942_APU)
#define SPARTA_KOKKOS_REDUCE_ARCH 1
#else
#define SPARTA_KOKKOS_REDUCE_ARCH 0
#endif

// the active surf react models, as named by the eight classes that dispatch to
//   them: compute surf, and the seven surf collide models that support surface
//   chemistry.  Both representations are reached through these accessors, so
//   the device dispatch site in each of those classes is written exactly once
//   and the two modes cannot drift:
//
//     KK_SR_*    read on device, from the model's collide_kokkos()
//     KK_SR_H_*  the host image of the same models, used by the
//                pre_react()/post_react()/backup()/restore() lifecycle
//
//   under SPARTA_KOKKOS_FIXED_LISTS both are the same fixed KKCopy arrays, held
//   by value in the class; otherwise the device side reads the per-style device
//   buffer and the host side the host half of the same DualView.
// the accessors expand to member accesses only, and all eight classes spell
//   those members identically, so one definition here serves every one of them.

#ifdef SPARTA_KOKKOS_FIXED_LISTS

#define KK_SR_TYPE(n)     sr_type_list[n]
#define KK_SR_MAP(n)      sr_map[n]
#define KK_SR_GLOBAL(m)   sr_kk_global_copy[m].obj
#define KK_SR_PROB(m)     sr_kk_prob_copy[m].obj
#define KK_SR_ADSORB(m)   sr_kk_adsorb_copy[m].obj

#define KK_SR_H_TYPE(n)   sr_type_list[n]
#define KK_SR_H_MAP(n)    sr_map[n]
#define KK_SR_H_GLOBAL(m) sr_kk_global_copy[m].obj
#define KK_SR_H_PROB(m)   sr_kk_prob_copy[m].obj
#define KK_SR_H_ADSORB(m) sr_kk_adsorb_copy[m].obj

#else

#define KK_SR_TYPE(n)     d_sr_type_list[n]
#define KK_SR_MAP(n)      d_sr_map[n]
#define KK_SR_GLOBAL(m)   ((const SurfReactGlobalKokkos *) d_sr_global.data())[m]
#define KK_SR_PROB(m)     ((const SurfReactProbKokkos *) d_sr_prob.data())[m]
#define KK_SR_ADSORB(m)   ((const SurfReactAdsorbKokkos *) d_sr_adsorb.data())[m]

#define KK_SR_H_TYPE(n)   k_sr_type_list.view_host()[n]
#define KK_SR_H_MAP(n)    k_sr_map.view_host()[n]
#define KK_SR_H_GLOBAL(m) ((SurfReactGlobalKokkos *) k_sr_global.view_host().data())[m]
#define KK_SR_H_PROB(m)   ((SurfReactProbKokkos *) k_sr_prob.view_host().data())[m]
#define KK_SR_H_ADSORB(m) ((SurfReactAdsorbKokkos *) k_sr_adsorb.view_host().data())[m]

#endif

namespace Kokkos {
  static auto NoInit = [](std::string const& label) {
    return Kokkos::view_alloc(Kokkos::WithoutInitializing, label);
  };
}

  struct sparta_float3 {
    float x,y,z;
    KOKKOS_INLINE_FUNCTION
    sparta_float3():x(0.0f),y(0.0f),z(0.0f) {}

    KOKKOS_INLINE_FUNCTION
    void operator += (const sparta_float3& tmp) {
      x+=tmp.x;
      y+=tmp.y;
      z+=tmp.z;
    }
    KOKKOS_INLINE_FUNCTION
    void operator = (const sparta_float3& tmp) {
      x=tmp.x;
      y=tmp.y;
      z=tmp.z;
    }
  };


// set SPAHostype and DeviceType from Kokkos Default Types
typedef Kokkos::DefaultExecutionSpace SPADeviceType;
typedef Kokkos::HostSpace::execution_space SPAHostType;

typedef SPADeviceType DeviceType;

// set ExecutionSpace stuct with variable "space"

template<class Device>
struct ExecutionSpaceFromDevice;

template<>
struct ExecutionSpaceFromDevice<SPAHostType> {
  static const SPARTA_NS::ExecutionSpace space = SPARTA_NS::Host;
};

#ifdef KOKKOS_ENABLE_CUDA
template<>
struct ExecutionSpaceFromDevice<Kokkos::Cuda> {
  static const SPARTA_NS::ExecutionSpace space = SPARTA_NS::Device;
};
#elif defined(KOKKOS_ENABLE_HIP)
template<>
struct ExecutionSpaceFromDevice<Kokkos::Experimental::HIP> {
  static const SPARTA_NS::ExecutionSpace space = SPARTA_NS::Device;
};
#elif defined(KOKKOS_ENABLE_SYCL)
template<>
struct ExecutionSpaceFromDevice<Kokkos::Experimental::SYCL> {
  static const SPARTA_NS::ExecutionSpace space = SPARTA_NS::Device;
};
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
template<>
struct ExecutionSpaceFromDevice<Kokkos::Experimental::OpenMPTarget> {
  static const SPARTA_NS::ExecutionSpace space = SPARTA_NS::Device;
};
#endif

// set host pinned space
#if defined(KOKKOS_ENABLE_CUDA)
typedef Kokkos::CudaHostPinnedSpace SPAPinnedHostType;
#elif defined(KOKKOS_ENABLE_HIP)
typedef Kokkos::Experimental::HIPHostPinnedSpace SPAPinnedHostType;
#elif defined(KOKKOS_ENABLE_SYCL)
typedef Kokkos::Experimental::SYCLHostUSMSpace SPAPinnedHostType;
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
typedef Kokkos::Serial SPAPinnedHostType;
#endif

// Determine memory traits for atomic arrays
template<int NEED_ATOMICS>
struct AtomicView {
  enum {value = Kokkos::Unmanaged};
};

template<>
struct AtomicView<1> {
  enum {value = Kokkos::Atomic|Kokkos::Unmanaged};
};

template<>
struct AtomicView<-1> {
  enum {value = Kokkos::Atomic|Kokkos::Unmanaged};
};

// Determine memory traits for array
// Do atomic trait when running with CUDA
template<int NEED_ATOMICS, class DeviceType>
struct AtomicDup {
  using value = Kokkos::Experimental::ScatterNonAtomic;
};

#ifdef KOKKOS_ENABLE_CUDA
template<>
struct AtomicDup<1,Kokkos::Cuda> {
  using value = Kokkos::Experimental::ScatterAtomic;
};

template<>
struct AtomicDup<-1,Kokkos::Cuda> {
  using value = Kokkos::Experimental::ScatterAtomic;
};
#elif defined(KOKKOS_ENABLE_HIP)
template<>
struct AtomicDup<1,Kokkos::Experimental::HIP> {
  using value = Kokkos::Experimental::ScatterAtomic;
};

template<>
struct AtomicDup<-1,Kokkos::Experimental::HIP> {
  using value = Kokkos::Experimental::ScatterAtomic;
};
#elif defined(KOKKOS_ENABLE_SYCL)
template<>
struct AtomicDup<1,Kokkos::Experimental::SYCL> {
  using value = Kokkos::Experimental::ScatterAtomic;
};

template<>
struct AtomicDup<-1,Kokkos::Experimental::SYCL> {
  using value = Kokkos::Experimental::ScatterAtomic;
};
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
template<>
struct AtomicDup<1,Kokkos::Experimental::OpenMPTarget> {
  using value = Kokkos::Experimental::ScatterAtomic;
};

template<>
struct AtomicDup<-1,Kokkos::Experimental::OpenMPTarget> {
  using value = Kokkos::Experimental::ScatterAtomic;
};
#endif

#ifdef SPARTA_KOKKOS_USE_ATOMICS

#ifdef KOKKOS_ENABLE_OPENMP
template<>
struct AtomicDup<1,Kokkos::OpenMP> {
  using value = Kokkos::Experimental::ScatterAtomic;
};

template<>
struct AtomicDup<-1,Kokkos::OpenMP> {
  using value = Kokkos::Experimental::ScatterAtomic;
};
#endif

#ifdef KOKKOS_ENABLE_THREADS
template<>
struct AtomicDup<1,Kokkos::Threads> {
  using value = Kokkos::Experimental::ScatterAtomic;
};

template<>
struct AtomicDup<-1,Kokkos::Threads> {
  using value = Kokkos::Experimental::ScatterAtomic;
};
#endif

#endif


// Determine duplication traits for array
// Use duplication when running threaded and not using atomics
template<int NEED_ATOMICS, class DeviceType>
struct NeedDup {
  using value = Kokkos::Experimental::ScatterNonDuplicated;
};

#ifndef SPARTA_KOKKOS_USE_ATOMICS

#ifdef KOKKOS_ENABLE_OPENMP
template<>
struct NeedDup<1,Kokkos::OpenMP> {
  using value = Kokkos::Experimental::ScatterDuplicated;
};

template<>
struct NeedDup<-1,Kokkos::OpenMP> {
  using value = Kokkos::Experimental::ScatterDuplicated;
};
#endif

#ifdef KOKKOS_ENABLE_THREADS
template<>
struct NeedDup<1,Kokkos::Threads> {
  using value = Kokkos::Experimental::ScatterDuplicated;
};

template<>
struct NeedDup<-1,Kokkos::Threads> {
  using value = Kokkos::Experimental::ScatterDuplicated;
};
#endif

#endif

template<typename value, typename T1, typename T2>
class ScatterViewHelper {};

template<typename T1, typename T2>
class ScatterViewHelper<Kokkos::Experimental::ScatterDuplicated,T1,T2> {
public:
  KOKKOS_INLINE_FUNCTION
  static T1 get(const T1 &dup, const T2 & /*nondup*/) {
    return dup;
  }
};

template<typename T1, typename T2>
class ScatterViewHelper<Kokkos::Experimental::ScatterNonDuplicated,T1,T2> {
public:
  KOKKOS_INLINE_FUNCTION
  static T2 get(const T1 & /*dup*/, const T2 &nondup) {
    return nondup;
  }
};


// define precision
//
// SPARTA_KOKKOS_DOUBLE_DOUBLE: double precision for all calculations (default)
// SPARTA_KOKKOS_SINGLE_DOUBLE: mixed precision; single precision for
//   velocities, energies and per-particle/per-collision arithmetic, double
//   precision for particle positions, geometry and accumulations
// SPARTA_KOKKOS_SINGLE_SINGLE: single precision for all per-particle data
//   and arithmetic, including positions and geometry
//
// KK_FLOAT     = storage and arithmetic precision
// KK_POS_FLOAT = particle positions, remaining timestep, grid cell and
//                surface element coordinates, and the move/geometry kernels
//                that use them
// KK_ACC_FLOAT = per-grid/per-surf tallies and other accumulations, and the
//                per-cell statistics computed from them
//
// KK_ACC_FLOAT is double even in a single precision build: in SI units
//   masses are ~1e-26 kg, so the per-cell moments computed from mass
//   weighted sums, e.g. (sum m*v)^2/sum(m) or (sum m*v)^3/sum(m)^2,
//   underflow single precision
//
// host (legacy) data structures are always double, see TransformView below

#if !defined(SPARTA_KOKKOS_SINGLE_SINGLE) && \
    !defined(SPARTA_KOKKOS_DOUBLE_DOUBLE) && \
    !defined(SPARTA_KOKKOS_SINGLE_DOUBLE)
#define SPARTA_KOKKOS_DOUBLE_DOUBLE
#endif

#if defined(SPARTA_KOKKOS_SINGLE_SINGLE)
typedef float KK_FLOAT;
typedef float KK_POS_FLOAT;
typedef double KK_ACC_FLOAT;
#elif defined(SPARTA_KOKKOS_SINGLE_DOUBLE)
typedef float KK_FLOAT;
typedef double KK_POS_FLOAT;
typedef double KK_ACC_FLOAT;
#else
typedef double KK_FLOAT;
typedef double KK_POS_FLOAT;
typedef double KK_ACC_FLOAT;
#endif

// MPI datatypes of the KK precision types, for MPI calls on Kokkos data
//   (macros, so they expand where mpi.h is included)

#define MPI_KK_FLOAT (std::is_same_v<KK_FLOAT,double> ? MPI_DOUBLE : MPI_FLOAT)
#define MPI_KK_POS_FLOAT (std::is_same_v<KK_POS_FLOAT,double> ? MPI_DOUBLE : MPI_FLOAT)
#define MPI_KK_ACC_FLOAT (std::is_same_v<KK_ACC_FLOAT,double> ? MPI_DOUBLE : MPI_FLOAT)

// true if any Kokkos data is stored in reduced precision

static constexpr bool KK_FP32 = !std::is_same_v<KK_FLOAT,double> ||
  !std::is_same_v<KK_POS_FLOAT,double> || !std::is_same_v<KK_ACC_FLOAT,double>;

// select a tolerance by precision: the double value is the one tuned for
//   the original double precision code, the float value must be large
//   enough to exceed float round-off for the quantity it guards

template<class T>
KOKKOS_INLINE_FUNCTION
constexpr T kk_eps(const double eps_double, const double eps_float)
{
  return std::is_same_v<T,float> ? static_cast<T>(eps_float) :
    static_cast<T>(eps_double);
}

namespace SPARTA_NS {

// convert one element between precisions, specialized for structs in
//   kokkos_structs.h

// dst keeps its value if it already converts exactly to src, so a double
//   host value whose single precision image was not changed on the device
//   survives the round trip host -> device -> host with full precision
//   (e.g. grid cell corners, which host geometry code compares exactly)

template<class DstType, class SrcType>
KOKKOS_INLINE_FUNCTION
std::enable_if_t<std::is_arithmetic_v<DstType>>
kk_convert(DstType &dst, const SrcType &src)
{
  if (static_cast<SrcType>(dst) != src) dst = static_cast<DstType>(src);
}

template<class Type>
KOKKOS_INLINE_FUNCTION
std::enable_if_t<!std::is_arithmetic_v<Type>>
kk_convert(Type &dst, const Type &src) { dst = src; }

}

// KK precision copies of the host structs shared with the device

#include "kokkos_structs.h"

// ------------------------------------------------------------------------
// TransformView: a DualView whose host (legacy) side can have a different
//   value type than its Kokkos side, following the LAMMPS KOKKOS package
//
// three copies of the data can exist:
//   d_view   = device view, KK precision
//   h_viewkk = Kokkos host mirror of d_view, KK precision
//              (same memory as d_view when the device is the host)
//   h_view   = legacy host view, double precision, aliased by the double*
//              pointers of the host (non-Kokkos) classes
//
// when KKType == LegacyType (always the case in a double precision build)
//   h_view is h_viewkk and this class is a thin wrapper of a DualView, so
//   there is no extra memory and no extra copy
//
// view_host()/sync_host()/modify_host() refer to the legacy host view,
//   view_device()/sync_device()/modify_device() to the device view, and
//   view_hostkk()/sync_hostkk()/modify_hostkk() to the Kokkos host mirror
// all type conversion happens host-side, between h_viewkk and h_view;
//   transfers between host and device are always same-type
// ------------------------------------------------------------------------

namespace SPARTA_NS {

// converting host-side copy between two views of different value types

template<bool KEEP, class DstType, class SrcType>
KOKKOS_INLINE_FUNCTION
void convert_one(DstType &dst, const SrcType &src)
{
  if constexpr (KEEP) kk_convert(dst,src);
  else dst = static_cast<DstType>(src);
}

// with KEEP, a dst value that already converts exactly to its src value is
//   kept (see kk_convert()), else every element is overwritten

template<class DstView, class SrcView, bool KEEP = true>
void transform_copy(const DstView &dst, const SrcView &src)
{
  typedef typename DstView::non_const_value_type dst_type;
  typedef Kokkos::RangePolicy<SPAHostType> policy_1d;
  typedef Kokkos::MDRangePolicy<SPAHostType,Kokkos::Rank<2>> policy_2d;
  typedef Kokkos::MDRangePolicy<SPAHostType,Kokkos::Rank<3>> policy_3d;
  if constexpr (std::is_arithmetic_v<dst_type>) {
    // element by element, as kk_convert() keeps unchanged values
    static_assert(DstView::rank == SrcView::rank && DstView::rank <= 3,
                  "TransformView of an arithmetic type must have rank 0 to 3");
    if constexpr (DstView::rank == 0) convert_one<KEEP>(dst(),src());
    else if constexpr (DstView::rank == 1)
      Kokkos::parallel_for(policy_1d(0,dst.extent(0)),
                           [=](const int i) { convert_one<KEEP>(dst(i),src(i)); });
    else if constexpr (DstView::rank == 2)
      Kokkos::parallel_for(policy_2d({0,0},{(int64_t) dst.extent(0),(int64_t) dst.extent(1)}),
                           [=](const int i, const int j) { convert_one<KEEP>(dst(i,j),src(i,j)); });
    else
      Kokkos::parallel_for(policy_3d({0,0,0},{(int64_t) dst.extent(0),(int64_t) dst.extent(1),
                                              (int64_t) dst.extent(2)}),
                           [=](const int i, const int j, const int k) {
                             convert_one<KEEP>(dst(i,j,k),src(i,j,k)); });
    Kokkos::fence();
  } else {
    static_assert(DstView::rank == 1 && SrcView::rank == 1,
                  "TransformView of a struct type must have rank 1");
    const int n = (int) (dst.extent(0) < src.extent(0) ? dst.extent(0) : src.extent(0));
    Kokkos::parallel_for(Kokkos::RangePolicy<SPAHostType>(0,n),
                         [=](const int i) { kk_convert(dst(i),src(i)); });
    Kokkos::fence();
  }
}

// deep copy between views whose value types may differ (e.g. a double host
//   array and a KK_FLOAT device view); the conversion is done on the host

template<class DstView, class SrcView>
void deep_copy_convert(const DstView &dst, const SrcView &src)
{
  if constexpr (std::is_same_v<typename DstView::non_const_value_type,
                typename SrcView::non_const_value_type>) {
    Kokkos::deep_copy(dst,src);
  } else {
    auto h_dst = Kokkos::create_mirror_view(Kokkos::HostSpace(),dst);
    auto h_src = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),src);
    transform_copy<decltype(h_dst),decltype(h_src),false>(h_dst,h_src);
    Kokkos::deep_copy(dst,h_dst);
  }
}

// copy a host double array of length n into a 1d KK view on the device,
//   (re)allocating the view only if it is too short

template<class ViewType>
void copy_host_array_to_view(ViewType &d_view, const double *array, const int n,
                             const char *label)
{
  if ((int) d_view.extent(0) < n)
    d_view = ViewType(Kokkos::view_alloc(label,Kokkos::WithoutInitializing),n);
  Kokkos::View<const double*,Kokkos::LayoutRight,Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_array(array,n);
  deep_copy_convert(Kokkos::subview(d_view,Kokkos::make_pair(0,n)),h_array);
}

template<class KKType, class LegacyType, class KKLayout, class KKSpace = DeviceType>
class TransformView {
 public:
  static constexpr bool NEED_TRANSFORM = !std::is_same_v<KKType,LegacyType>;

  typedef Kokkos::DualView<KKType, KKLayout, KKSpace> kk_view;
  typedef typename kk_view::t_host::memory_space kk_host_memory_space;

  // legacy host view type: same as the Kokkos host mirror when no transform
  //   is needed, else a LayoutRight view of the legacy type in host memory,
  //   so that rows can be aliased by double** pointers

  typedef std::conditional_t<NEED_TRANSFORM,
    Kokkos::View<LegacyType, Kokkos::LayoutRight, kk_host_memory_space>,
    typename kk_view::t_host> legacy_view;

  typedef typename legacy_view::value_type value_type;
  typedef typename legacy_view::array_layout array_layout;
  typedef typename kk_view::t_dev::value_type kk_value_type;

  typedef typename kk_view::t_dev t_dev;
  typedef typename kk_view::t_dev_const t_dev_const;
  typedef typename kk_view::t_dev_um t_dev_um;
  typedef typename kk_view::t_dev_const_um t_dev_const_um;
  typedef typename kk_view::t_dev_const_randomread t_dev_const_randomread;
  typedef typename kk_view::t_host t_hostkk;

  typedef legacy_view t_host;
  typedef Kokkos::View<typename legacy_view::const_data_type,
                       typename legacy_view::array_layout,
                       typename legacy_view::memory_space> t_host_const;
  typedef Kokkos::View<typename legacy_view::data_type,
                       typename legacy_view::array_layout,
                       typename legacy_view::memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>> t_host_um;
  typedef Kokkos::View<typename legacy_view::const_data_type,
                       typename legacy_view::array_layout,
                       typename legacy_view::memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>> t_host_const_um;
  typedef t_host_const t_host_const_randomread;

  // true if device and Kokkos host views share memory

  static constexpr bool SINGLE_DEVICE =
    std::is_same_v<typename kk_view::t_dev::memory_space,kk_host_memory_space>;

 private:
  kk_view k_view;
  legacy_view h_view;

  // modified_legacy = legacy view modified since last conversion to KK
  // modified_kk = device or KK host view modified since last conversion
  //   to legacy

  bool modified_legacy = false;
  bool modified_kk = false;

  template<class... Args>
  void make_legacy_view_args(Args... args) {
    if constexpr (NEED_TRANSFORM) h_view = legacy_view(args...);
    else h_view = k_view.view_host();
  }

 public:
  TransformView() = default;

  template<class... Indices>
  TransformView(const std::string &label, Indices... ns)
    : k_view(label, ns...) {
    make_legacy_view_args(label + "_legacy", ns...);
  }

  template<class... Indices>
  TransformView(const char *label, Indices... ns)
    : TransformView(std::string(label), ns...) {}

  template<class... P, class... Indices>
  TransformView(const Kokkos::Impl::ViewCtorProp<P...> &prop, Indices... ns)
    : k_view(prop, ns...) {
    if constexpr (NEED_TRANSFORM)
      h_view = legacy_view(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                              k_view.view_host().label() + "_legacy"),
                           ns...);
    else h_view = k_view.view_host();
  }

  // accessors

  KOKKOS_INLINE_FUNCTION
  const legacy_view &view_host() const { return h_view; }
  KOKKOS_INLINE_FUNCTION
  const t_hostkk &view_hostkk() const { return k_view.view_host(); }
  KOKKOS_INLINE_FUNCTION
  const t_dev &view_device() const { return k_view.view_device(); }
  kk_view &dual_view() { return k_view; }

  template<class Device>
  auto view() const {
    if constexpr (std::is_same_v<typename Device::memory_space,
                  typename t_dev::memory_space>)
      return k_view.view_device();
    else return k_view.view_host();
  }

  KOKKOS_INLINE_FUNCTION
  size_t extent(const int i) const { return k_view.view_device().extent(i); }
  KOKKOS_INLINE_FUNCTION
  size_t span() const { return k_view.view_device().span(); }

  // modify

  void modify_device() {
    k_view.modify_device();
    if constexpr (NEED_TRANSFORM) modified_kk = true;
  }

  void modify_hostkk() {
    k_view.modify_host();
    if constexpr (NEED_TRANSFORM) modified_kk = true;
  }

  void modify_host() {
    if constexpr (NEED_TRANSFORM) modified_legacy = true;
    else k_view.modify_host();
  }

  template<class Device>
  void modify() {
    if constexpr (std::is_same_v<typename Device::memory_space,
                  typename t_dev::memory_space>) modify_device();
    else modify_hostkk();
  }

  // sync

  void sync_device() {
    if constexpr (NEED_TRANSFORM) {
      if (modified_legacy) {
        legacy_to_kk();
        k_view.clear_sync_state();
        k_view.modify_host();
      }
    }
    k_view.sync_device();
  }

  void sync_hostkk() {
    if constexpr (NEED_TRANSFORM) {
      if (modified_legacy) {
        legacy_to_kk();
        k_view.clear_sync_state();
        k_view.modify_host();
        return;
      }
    }
    k_view.sync_host();
  }

  void sync_host() {
    if constexpr (NEED_TRANSFORM) {
      if (modified_kk) {
        k_view.sync_host();
        transform_copy(h_view,k_view.view_host());
        modified_kk = false;
      }
    } else k_view.sync_host();
  }

  template<class Device>
  void sync() {
    if constexpr (std::is_same_v<typename Device::memory_space,
                  typename t_dev::memory_space>) sync_device();
    else sync_hostkk();
  }

  bool need_sync_host() const {
    if constexpr (NEED_TRANSFORM) return modified_kk;
    else return k_view.need_sync_host();
  }

  bool need_sync_device() const {
    if constexpr (NEED_TRANSFORM)
      return modified_legacy || k_view.need_sync_device();
    else return k_view.need_sync_device();
  }

  void clear_sync_state() {
    k_view.clear_sync_state();
    modified_legacy = modified_kk = false;
  }

  // resize, preserving contents of both copies

  template<class... Indices>
  void resize(Indices... ns) {
    k_view.resize(ns...);
    if constexpr (NEED_TRANSFORM) Kokkos::resize(h_view,ns...);
    else h_view = k_view.view_host();
  }

  template<class... P, class... Indices>
  void resize(const Kokkos::Impl::ViewCtorProp<P...> &prop, Indices... ns) {
    Kokkos::resize(prop,k_view,ns...);
    if constexpr (NEED_TRANSFORM) Kokkos::resize(prop,h_view,ns...);
    else h_view = k_view.view_host();
  }

 private:
  void legacy_to_kk() {
    transform_copy(k_view.view_host(),h_view);
    modified_legacy = false;
    modified_kk = false;
  }
};

}

// Kokkos::resize() of a TransformView, as for a DualView

namespace Kokkos {
template<class KKType, class LegacyType, class KKLayout, class KKSpace, class... Indices>
void resize(SPARTA_NS::TransformView<KKType,LegacyType,KKLayout,KKSpace> &v,
            Indices... ns) { v.resize(ns...); }

template<class... P, class KKType, class LegacyType, class KKLayout, class KKSpace,
         class... Indices>
void resize(const Impl::ViewCtorProp<P...> &prop,
            SPARTA_NS::TransformView<KKType,LegacyType,KKLayout,KKSpace> &v,
            Indices... ns) { v.resize(prop,ns...); }
}

// ------------------------------------------------------------------------

// SPARTA types

namespace SPARTA_NS {

  // the per-particle, per-grid-cell and per-surf structs are stored in
  //   KK precision on the device (see kokkos_structs.h) and converted to
  //   the double precision legacy structs of the host classes on sync

  typedef TransformView<OnePartKK*, Particle::OnePart*, DeviceType::array_layout>
    tdual_particle_1d;
  typedef tdual_particle_1d::t_dev t_particle_1d;
  typedef tdual_particle_1d::t_host t_host_particle_1d;

  typedef Kokkos::
    DualView<OnePartKK**, DeviceType::array_layout, DeviceType> tdual_particle_2d;
  typedef tdual_particle_2d::t_dev t_particle_2d;
  typedef tdual_particle_2d::t_host t_host_particle_2d;

  typedef Kokkos::
    DualView<Particle::Species*, DeviceType::array_layout, DeviceType> tdual_species_1d;
  typedef tdual_species_1d::t_dev t_species_1d;
  typedef tdual_species_1d::t_dev_const t_species_1d_const;
  typedef tdual_species_1d::t_host t_host_species_1d;

  typedef TransformView<ChildCellKK*, Grid::ChildCell*, DeviceType::array_layout>
    tdual_cell_1d;
  typedef tdual_cell_1d::t_dev t_cell_1d;
  typedef tdual_cell_1d::t_host t_host_cell_1d;

  typedef Kokkos::
    DualView<Grid::ChildInfo*, DeviceType::array_layout, DeviceType> tdual_cinfo_1d;
  typedef tdual_cinfo_1d::t_dev t_cinfo_1d;
  typedef tdual_cinfo_1d::t_host t_host_cinfo_1d;

  typedef TransformView<SplitInfoKK*, Grid::SplitInfo*, DeviceType::array_layout>
    tdual_sinfo_1d;
  typedef tdual_sinfo_1d::t_dev t_sinfo_1d;
  typedef tdual_sinfo_1d::t_host t_host_sinfo_1d;

  typedef TransformView<ParentCellKK*, Grid::ParentCell*, DeviceType::array_layout>
    tdual_pcell_1d;
  typedef tdual_pcell_1d::t_dev t_pcell_1d;
  typedef tdual_pcell_1d::t_host t_host_pcell_1d;

  typedef Kokkos::
    DualView<Grid::ParentLevel*, DeviceType::array_layout, DeviceType> tdual_plevel_1d;
  typedef tdual_plevel_1d::t_dev t_plevel_1d;
  typedef tdual_plevel_1d::t_host t_host_plevel_1d;

  typedef TransformView<LineKK*, Surf::Line*, DeviceType::array_layout>
    tdual_line_1d;
  typedef tdual_line_1d::t_dev t_line_1d;
  typedef tdual_line_1d::t_host t_host_line_1d;

  typedef TransformView<TriKK*, Surf::Tri*, DeviceType::array_layout>
    tdual_tri_1d;
  typedef tdual_tri_1d::t_dev t_tri_1d;
  typedef tdual_tri_1d::t_host t_host_tri_1d;

  // device-callable equivalent of Compute::ubuf, used by the KOKKOS tally
  //   computes to pack ints into a double buf slot from within
  //   KOKKOS_INLINE_FUNCTION methods.  Compute::ubuf's constructors are
  //   host-only, which nvcc tolerates but hipcc rejects, so the bit pattern
  //   is reproduced here as a KOKKOS_INLINE_FUNCTION union instead.

  union d_ubuf {
    double d;
    int64_t i;
    KOKKOS_INLINE_FUNCTION d_ubuf(double arg) : d(arg) {}
    KOKKOS_INLINE_FUNCTION d_ubuf(int64_t arg) : i(arg) {}
    KOKKOS_INLINE_FUNCTION d_ubuf(int arg) : i(arg) {}
    KOKKOS_INLINE_FUNCTION d_ubuf(uint32_t arg) : i(arg) {}
    KOKKOS_INLINE_FUNCTION d_ubuf(uint64_t arg) : i(arg) {}
  };
}

// macros to define the typedef families of a DualView or TransformView

#define SPARTA_DEVICE_DUALVIEW(TYPE, LAYOUT, SUFFIX) \
typedef Kokkos::DualView<TYPE, LAYOUT, DeviceType> tdual_##SUFFIX; \
typedef tdual_##SUFFIX::t_dev t_##SUFFIX; \
typedef tdual_##SUFFIX::t_dev_const t_##SUFFIX##_const; \
typedef tdual_##SUFFIX::t_dev_um t_##SUFFIX##_um; \
typedef tdual_##SUFFIX::t_dev_const_um t_##SUFFIX##_const_um; \
typedef tdual_##SUFFIX::t_dev_const_randomread t_##SUFFIX##_randomread;

#define SPARTA_HOST_DUALVIEW(TYPE, LAYOUT, SUFFIX) \
typedef Kokkos::DualView<TYPE, LAYOUT, DeviceType> tdual_##SUFFIX; \
typedef tdual_##SUFFIX::t_host t_##SUFFIX; \
typedef tdual_##SUFFIX::t_host_const t_##SUFFIX##_const; \
typedef tdual_##SUFFIX::t_host_um t_##SUFFIX##_um; \
typedef tdual_##SUFFIX::t_host_const_um t_##SUFFIX##_const_um; \
typedef tdual_##SUFFIX::t_host_const_randomread t_##SUFFIX##_randomread;

#define SPARTA_DEVICE_TRANSFORMVIEW(KKTYPE, LEGACYTYPE, LAYOUT, SUFFIX) \
typedef SPARTA_NS::TransformView<KKTYPE, LEGACYTYPE, LAYOUT> ttransform_##SUFFIX; \
typedef ttransform_##SUFFIX::t_dev t_##SUFFIX; \
typedef ttransform_##SUFFIX::t_dev_const t_##SUFFIX##_const; \
typedef ttransform_##SUFFIX::t_dev_um t_##SUFFIX##_um; \
typedef ttransform_##SUFFIX::t_dev_const_um t_##SUFFIX##_const_um; \
typedef ttransform_##SUFFIX::t_dev_const_randomread t_##SUFFIX##_randomread;

#define SPARTA_HOST_TRANSFORMVIEW(KKTYPE, LEGACYTYPE, LAYOUT, SUFFIX) \
typedef SPARTA_NS::TransformView<KKTYPE, LEGACYTYPE, LAYOUT> ttransform_##SUFFIX; \
typedef ttransform_##SUFFIX::kk_view::t_host t_##SUFFIX; \
typedef ttransform_##SUFFIX::kk_view::t_host_const t_##SUFFIX##_const; \
typedef ttransform_##SUFFIX::kk_view::t_host_um t_##SUFFIX##_um; \
typedef ttransform_##SUFFIX::kk_view::t_host_const_um t_##SUFFIX##_const_um; \
typedef ttransform_##SUFFIX::kk_view::t_host_const_randomread t_##SUFFIX##_randomread;

template <class DeviceType>
struct ArrayTypes;

template <>
struct ArrayTypes<DeviceType> {

// scalar types

typedef Kokkos::
  DualView<int, DeviceType::array_layout, DeviceType> tdual_int_scalar;
typedef tdual_int_scalar::t_dev t_int_scalar;
typedef tdual_int_scalar::t_dev_const t_int_scalar_const;
typedef tdual_int_scalar::t_dev_um t_int_scalar_um;
typedef tdual_int_scalar::t_dev_const_um t_int_scalar_const_um;

typedef Kokkos::
  DualView<SPARTA_NS::bigint, DeviceType::array_layout, DeviceType> tdual_bigint_scalar;
typedef tdual_bigint_scalar::t_dev t_bigint_scalar;
typedef tdual_bigint_scalar::t_dev_const t_bigint_scalar_const;
typedef tdual_bigint_scalar::t_dev_um t_bigint_scalar_um;
typedef tdual_bigint_scalar::t_dev_const_um t_bigint_scalar_const_um;


// generic array types

typedef Kokkos::
  DualView<char*, DeviceType::array_layout, DeviceType> tdual_char_1d;
typedef tdual_char_1d::t_dev t_char_1d;
typedef tdual_char_1d::t_dev_const t_char_1d_const;
typedef tdual_char_1d::t_dev_um t_char_1d_um;
typedef tdual_char_1d::t_dev_const_um t_char_1d_const_um;
typedef tdual_char_1d::t_dev_const_randomread t_char_1d_randomread;

typedef Kokkos::
  DualView<int*, DeviceType::array_layout, DeviceType> tdual_int_1d;
typedef tdual_int_1d::t_dev t_int_1d;
typedef tdual_int_1d::t_dev_const t_int_1d_const;
typedef tdual_int_1d::t_dev_um t_int_1d_um;
typedef tdual_int_1d::t_dev_const_um t_int_1d_const_um;
typedef tdual_int_1d::t_dev_const_randomread t_int_1d_randomread;

typedef Kokkos::
  DualView<SPARTA_NS::bigint*, DeviceType::array_layout, DeviceType> tdual_bigint_1d;
typedef tdual_bigint_1d::t_dev t_bigint_1d;
typedef tdual_bigint_1d::t_dev_const t_bigint_1d_const;
typedef tdual_bigint_1d::t_dev_um t_bigint_1d_um;
typedef tdual_bigint_1d::t_dev_const_um t_bigint_1d_const_um;
typedef tdual_bigint_1d::t_dev_const_randomread t_bigint_1d_randomread;

typedef Kokkos::
  DualView<int*[3], DeviceType::array_layout, DeviceType> tdual_int_1d_3;
typedef tdual_int_1d_3::t_dev t_int_1d_3;
typedef tdual_int_1d_3::t_dev_const t_int_1d_3_const;
typedef tdual_int_1d_3::t_dev_um t_int_1d_3_um;
typedef tdual_int_1d_3::t_dev_const_um t_int_1d_3_const_um;
typedef tdual_int_1d_3::t_dev_const_randomread t_int_1d_3_randomread;

typedef Kokkos::
  DualView<int**, Kokkos::LayoutRight, DeviceType> tdual_int_2d_lr;
typedef tdual_int_2d_lr::t_dev t_int_2d_lr;
typedef tdual_int_2d_lr::t_dev_const t_int_2d_const_lr;
typedef tdual_int_2d_lr::t_dev_um t_int_2d_um_lr;
typedef tdual_int_2d_lr::t_dev_const_um t_int_2d_const_um_lr;
typedef tdual_int_2d_lr::t_dev_const_randomread t_int_2d_randomread_lr;

typedef Kokkos::
  DualView<int**, DeviceType::array_layout, DeviceType> tdual_int_2d;
typedef tdual_int_2d::t_dev t_int_2d;
typedef tdual_int_2d::t_dev_const t_int_2d_const;
typedef tdual_int_2d::t_dev_um t_int_2d_um;
typedef tdual_int_2d::t_dev_const_um t_int_2d_const_um;
typedef tdual_int_2d::t_dev_const_randomread t_int_2d_randomread;

typedef Kokkos::
  DualView<SPARTA_NS::cellint*, DeviceType::array_layout, DeviceType>
  tdual_cellint_1d;
typedef tdual_cellint_1d::t_dev t_cellint_1d;
typedef tdual_cellint_1d::t_dev_const t_cellint_1d_const;
typedef tdual_cellint_1d::t_dev_um t_cellint_1d_um;
typedef tdual_cellint_1d::t_dev_const_um t_cellint_1d_const_um;
typedef tdual_cellint_1d::t_dev_const_randomread t_cellint_1d_randomread;

typedef Kokkos::
  DualView<SPARTA_NS::surfint*, DeviceType::array_layout, DeviceType>
  tdual_surfint_1d;
typedef tdual_surfint_1d::t_dev t_surfint_1d;
typedef tdual_surfint_1d::t_dev_const t_surfint_1d_const;
typedef tdual_surfint_1d::t_dev_um t_surfint_1d_um;
typedef tdual_surfint_1d::t_dev_const_um t_surfint_1d_const_um;
typedef tdual_surfint_1d::t_dev_const_randomread t_surfint_1d_randomread;

// floating point arrays
//   kkfloat = KK_FLOAT, kkpos = KK_POS_FLOAT, kkacc = KK_ACC_FLOAT
//   the ttransform_* types are TransformViews with a double legacy host view
//   the tdual_double_* types are plain DualViews, always double

SPARTA_DEVICE_TRANSFORMVIEW(KK_FLOAT, double, DeviceType::array_layout, kkfloat_scalar)
SPARTA_DEVICE_TRANSFORMVIEW(KK_FLOAT*, double*, DeviceType::array_layout, kkfloat_1d)
SPARTA_DEVICE_TRANSFORMVIEW(KK_FLOAT*[3], double*[3], DeviceType::array_layout, kkfloat_1d_3)
SPARTA_DEVICE_TRANSFORMVIEW(KK_FLOAT**, double**, DeviceType::array_layout, kkfloat_2d)
SPARTA_DEVICE_TRANSFORMVIEW(KK_FLOAT**, double**, Kokkos::LayoutRight, kkfloat_2d_lr)
SPARTA_DEVICE_TRANSFORMVIEW(KK_FLOAT***, double***, DeviceType::array_layout, kkfloat_3d)

SPARTA_DEVICE_TRANSFORMVIEW(KK_POS_FLOAT*, double*, DeviceType::array_layout, kkpos_1d)
SPARTA_DEVICE_TRANSFORMVIEW(KK_POS_FLOAT*[3], double*[3], DeviceType::array_layout, kkpos_1d_3)
SPARTA_DEVICE_TRANSFORMVIEW(KK_POS_FLOAT**, double**, DeviceType::array_layout, kkpos_2d)
SPARTA_DEVICE_TRANSFORMVIEW(KK_POS_FLOAT**, double**, Kokkos::LayoutRight, kkpos_2d_lr)

SPARTA_DEVICE_TRANSFORMVIEW(KK_ACC_FLOAT, double, DeviceType::array_layout, kkacc_scalar)
SPARTA_DEVICE_TRANSFORMVIEW(KK_ACC_FLOAT*, double*, DeviceType::array_layout, kkacc_1d)
SPARTA_DEVICE_TRANSFORMVIEW(KK_ACC_FLOAT*[3], double*[3], DeviceType::array_layout, kkacc_1d_3)
SPARTA_DEVICE_TRANSFORMVIEW(KK_ACC_FLOAT**, double**, DeviceType::array_layout, kkacc_2d)
SPARTA_DEVICE_TRANSFORMVIEW(KK_ACC_FLOAT**, double**, Kokkos::LayoutRight, kkacc_2d_lr)
SPARTA_DEVICE_TRANSFORMVIEW(KK_ACC_FLOAT***, double***, DeviceType::array_layout, kkacc_3d)

SPARTA_DEVICE_DUALVIEW(KK_FLOAT*, Kokkos::LayoutStride, kkfloat_1d_strided)
SPARTA_DEVICE_DUALVIEW(KK_ACC_FLOAT*, Kokkos::LayoutStride, kkacc_1d_strided)

SPARTA_DEVICE_DUALVIEW(double, DeviceType::array_layout, double_scalar)
SPARTA_DEVICE_DUALVIEW(double*, DeviceType::array_layout, double_1d)
SPARTA_DEVICE_DUALVIEW(double**, DeviceType::array_layout, double_2d)
SPARTA_DEVICE_DUALVIEW(double**, Kokkos::LayoutRight, double_2d_lr)
SPARTA_DEVICE_DUALVIEW(double***, DeviceType::array_layout, double_3d)
};

#ifdef SPARTA_KOKKOS_GPU
template <>
struct ArrayTypes<SPAHostType> {

//Scalar Types

typedef Kokkos::DualView<int, DeviceType::array_layout, DeviceType> tdual_int_scalar;
typedef tdual_int_scalar::t_host t_int_scalar;
typedef tdual_int_scalar::t_host_const t_int_scalar_const;
typedef tdual_int_scalar::t_host_um t_int_scalar_um;
typedef tdual_int_scalar::t_host_const_um t_int_scalar_const_um;

typedef Kokkos::DualView<SPARTA_NS::bigint, DeviceType::array_layout, DeviceType> tdual_bigint_scalar;
typedef tdual_bigint_scalar::t_host t_bigint_scalar;
typedef tdual_bigint_scalar::t_host_const t_bigint_scalar_const;
typedef tdual_bigint_scalar::t_host_um t_bigint_scalar_um;
typedef tdual_bigint_scalar::t_host_const_um t_bigint_scalar_const_um;


//Generic ArrayTypes
typedef Kokkos::
  DualView<char*, DeviceType::array_layout, DeviceType> tdual_char_1d;
typedef tdual_char_1d::t_host t_char_1d;
typedef tdual_char_1d::t_host_const t_char_1d_const;
typedef tdual_char_1d::t_host_um t_char_1d_um;
typedef tdual_char_1d::t_host_const_um t_char_1d_const_um;
typedef tdual_char_1d::t_host_const_randomread t_char_1d_randomread;

typedef Kokkos::DualView<int*, DeviceType::array_layout, DeviceType> tdual_int_1d;
typedef tdual_int_1d::t_host t_int_1d;
typedef tdual_int_1d::t_host_const t_int_1d_const;
typedef tdual_int_1d::t_host_um t_int_1d_um;
typedef tdual_int_1d::t_host_const_um t_int_1d_const_um;
typedef tdual_int_1d::t_host_const_randomread t_int_1d_randomread;

typedef Kokkos::DualView<SPARTA_NS::bigint*, DeviceType::array_layout, DeviceType> tdual_bigint_1d;
typedef tdual_bigint_1d::t_host t_bigint_1d;
typedef tdual_bigint_1d::t_host_const t_bigint_1d_const;
typedef tdual_bigint_1d::t_host_um t_bigint_1d_um;
typedef tdual_bigint_1d::t_host_const_um t_bigint_1d_const_um;
typedef tdual_bigint_1d::t_host_const_randomread t_bigint_1d_randomread;

typedef Kokkos::DualView<int*[3], DeviceType::array_layout, DeviceType> tdual_int_1d_3;
typedef tdual_int_1d_3::t_host t_int_1d_3;
typedef tdual_int_1d_3::t_host_const t_int_1d_3_const;
typedef tdual_int_1d_3::t_host_um t_int_1d_3_um;
typedef tdual_int_1d_3::t_host_const_um t_int_1d_3_const_um;
typedef tdual_int_1d_3::t_host_const_randomread t_int_1d_3_randomread;

typedef Kokkos::DualView<int**, Kokkos::LayoutRight, DeviceType> tdual_int_2d_lr;
typedef tdual_int_2d_lr::t_host t_int_2d_lr;
typedef tdual_int_2d_lr::t_host_const t_int_2d_const_lr;
typedef tdual_int_2d_lr::t_host_um t_int_2d_um_lr;
typedef tdual_int_2d_lr::t_host_const_um t_int_2d_const_um_lr;
typedef tdual_int_2d_lr::t_host_const_randomread t_int_2d_randomread_lr;

typedef Kokkos::DualView<int**, DeviceType::array_layout, DeviceType> tdual_int_2d;
typedef tdual_int_2d::t_host t_int_2d;
typedef tdual_int_2d::t_host_const t_int_2d_const;
typedef tdual_int_2d::t_host_um t_int_2d_um;
typedef tdual_int_2d::t_host_const_um t_int_2d_const_um;
typedef tdual_int_2d::t_host_const_randomread t_int_2d_randomread;

typedef Kokkos::DualView<SPARTA_NS::cellint*, DeviceType::array_layout, DeviceType> tdual_cellint_1d;
typedef tdual_cellint_1d::t_host t_cellint_1d;
typedef tdual_cellint_1d::t_host_const t_cellint_1d_const;
typedef tdual_cellint_1d::t_host_um t_cellint_1d_um;
typedef tdual_cellint_1d::t_host_const_um t_cellint_1d_const_um;
typedef tdual_cellint_1d::t_host_const_randomread t_cellint_1d_randomread;

typedef Kokkos::DualView<SPARTA_NS::surfint*, DeviceType::array_layout, DeviceType> tdual_surfint_1d;
typedef tdual_surfint_1d::t_host t_surfint_1d;
typedef tdual_surfint_1d::t_host_const t_surfint_1d_const;
typedef tdual_surfint_1d::t_host_um t_surfint_1d_um;
typedef tdual_surfint_1d::t_host_const_um t_surfint_1d_const_um;
typedef tdual_surfint_1d::t_host_const_randomread t_surfint_1d_randomread;

// floating point arrays
//   kkfloat = KK_FLOAT, kkpos = KK_POS_FLOAT, kkacc = KK_ACC_FLOAT
//   the ttransform_* types are TransformViews with a double legacy host view
//   the tdual_double_* types are plain DualViews, always double

SPARTA_HOST_TRANSFORMVIEW(KK_FLOAT, double, DeviceType::array_layout, kkfloat_scalar)
SPARTA_HOST_TRANSFORMVIEW(KK_FLOAT*, double*, DeviceType::array_layout, kkfloat_1d)
SPARTA_HOST_TRANSFORMVIEW(KK_FLOAT*[3], double*[3], DeviceType::array_layout, kkfloat_1d_3)
SPARTA_HOST_TRANSFORMVIEW(KK_FLOAT**, double**, DeviceType::array_layout, kkfloat_2d)
SPARTA_HOST_TRANSFORMVIEW(KK_FLOAT**, double**, Kokkos::LayoutRight, kkfloat_2d_lr)
SPARTA_HOST_TRANSFORMVIEW(KK_FLOAT***, double***, DeviceType::array_layout, kkfloat_3d)

SPARTA_HOST_TRANSFORMVIEW(KK_POS_FLOAT*, double*, DeviceType::array_layout, kkpos_1d)
SPARTA_HOST_TRANSFORMVIEW(KK_POS_FLOAT*[3], double*[3], DeviceType::array_layout, kkpos_1d_3)
SPARTA_HOST_TRANSFORMVIEW(KK_POS_FLOAT**, double**, DeviceType::array_layout, kkpos_2d)
SPARTA_HOST_TRANSFORMVIEW(KK_POS_FLOAT**, double**, Kokkos::LayoutRight, kkpos_2d_lr)

SPARTA_HOST_TRANSFORMVIEW(KK_ACC_FLOAT, double, DeviceType::array_layout, kkacc_scalar)
SPARTA_HOST_TRANSFORMVIEW(KK_ACC_FLOAT*, double*, DeviceType::array_layout, kkacc_1d)
SPARTA_HOST_TRANSFORMVIEW(KK_ACC_FLOAT*[3], double*[3], DeviceType::array_layout, kkacc_1d_3)
SPARTA_HOST_TRANSFORMVIEW(KK_ACC_FLOAT**, double**, DeviceType::array_layout, kkacc_2d)
SPARTA_HOST_TRANSFORMVIEW(KK_ACC_FLOAT**, double**, Kokkos::LayoutRight, kkacc_2d_lr)
SPARTA_HOST_TRANSFORMVIEW(KK_ACC_FLOAT***, double***, DeviceType::array_layout, kkacc_3d)

SPARTA_HOST_DUALVIEW(KK_FLOAT*, Kokkos::LayoutStride, kkfloat_1d_strided)
SPARTA_HOST_DUALVIEW(KK_ACC_FLOAT*, Kokkos::LayoutStride, kkacc_1d_strided)

SPARTA_HOST_DUALVIEW(double, DeviceType::array_layout, double_scalar)
SPARTA_HOST_DUALVIEW(double*, DeviceType::array_layout, double_1d)
SPARTA_HOST_DUALVIEW(double**, DeviceType::array_layout, double_2d)
SPARTA_HOST_DUALVIEW(double**, Kokkos::LayoutRight, double_2d_lr)
SPARTA_HOST_DUALVIEW(double***, DeviceType::array_layout, double_3d)
};

#endif

template <typename D>
struct Graph {
  using Ints = Kokkos::View<int*, D>;
  Ints offsets;
  Ints at;
  int nedges;
  KOKKOS_INLINE_FUNCTION
  int start(int i) const { return offsets(i); }
  KOKKOS_INLINE_FUNCTION
  int end(int i) const { return offsets(i + 1); }
  KOKKOS_INLINE_FUNCTION
  int count(int i) const { return end(i) - start(i); }
  KOKKOS_INLINE_FUNCTION
  int& get(int i, int j) const { return at(start(i) + j); }
};

// default SPARTA Types
typedef struct ArrayTypes<DeviceType> DAT;
typedef struct ArrayTypes<SPAHostType> HAT;

// custom data types

namespace SPARTA_NS {

  struct struct_tdual_int_1d
  { DAT::tdual_int_1d k_view; };

  struct struct_tdual_float_1d
  { DAT::ttransform_kkfloat_1d k_view; };

  struct struct_tdual_int_2d
  { DAT::tdual_int_2d_lr k_view; };

  struct struct_tdual_float_2d
  { DAT::ttransform_kkfloat_2d_lr k_view; };

  typedef Kokkos::DualView<struct_tdual_int_1d*, DeviceType::array_layout, DeviceType> tdual_struct_tdual_int_1d_1d;
  typedef Kokkos::DualView<struct_tdual_float_1d*, DeviceType::array_layout, DeviceType> tdual_struct_tdual_float_1d_1d;
  typedef Kokkos::DualView<struct_tdual_int_2d*, DeviceType::array_layout, DeviceType> tdual_struct_tdual_int_2d_1d;
  typedef Kokkos::DualView<struct_tdual_float_2d*, DeviceType::array_layout, DeviceType> tdual_struct_tdual_float_2d_1d;
}

#ifndef SPARTA_KOKKOS_FIXED_LISTS

// the per-style device buffers behind the KK_SR_* accessors above, and the two
//   index lists that map a surf react index to a style and to a slot within it.
// blitting a model into a buffer is the same operation, for the same reason, as
//   KKCopy::copy() (kokkos_copy.h:71): on device the model is only read, through
//   KOKKOS_INLINE_FUNCTION members, so its vtable pointer is never used and the
//   Views it carries stay alive in the original that surf->sr holds.  The host
//   lifecycle calls are made on the host half of the same bytes -- a valid
//   object of the class as far as the host is concerned, its vtable pointer
//   copied from a live instance -- and pushed to the device by sr_buf_sync().
// shared here rather than repeated in each of the eight classes that carry
//   these lists, since all eight set them up the same way.

namespace SPARTA_NS {

  template<class T>
  void sr_buf_resize(DAT::tdual_char_1d &k, DAT::t_char_1d &d, int n)
  {
    const size_t need = (size_t) (n > 0 ? n : 1) * sizeof(T);
    if (k.view_device().extent(0) < need) {
      k = DAT::tdual_char_1d("surf_react:models",need);
      d = k.view_device();
    }
  }

  template<class T>
  void sr_buf_blit(DAT::tdual_char_1d &k, int slot, T *obj)
  {
    char *dst = k.view_host().data() + (size_t) slot*sizeof(T);
    memcpy((void*) dst, (const void*) obj, sizeof(T));
    ((T *) dst)->copy = 1;
  }

  inline void sr_buf_sync(DAT::tdual_char_1d &k, DAT::t_char_1d &d)
  {
    if (k.view_device().extent(0) == 0) return;
    k.modify_host();
    k.sync_device();
    d = k.view_device();
  }

  inline void sr_idx_resize(DAT::tdual_int_1d &k, DAT::t_int_1d &d, int n)
  {
    const size_t need = (size_t) (n > 0 ? n : 1);
    if (k.view_device().extent(0) < need) {
      k = DAT::tdual_int_1d("surf_react:index",need);
      d = k.view_device();
    }
  }

  inline void sr_idx_sync(DAT::tdual_int_1d &k, DAT::t_int_1d &d)
  {
    if (k.view_device().extent(0) == 0) return;
    k.modify_host();
    k.sync_device();
    d = k.view_device();
  }
}

#endif

template<class DeviceType, class BufferView, class DualView>
void buffer_view(BufferView &buf, DualView &view,
                 const size_t n0,
                 const size_t n1 = 0,
                 const size_t n2 = 0,
                 const size_t n3 = 0,
                 const size_t n4 = 0,
                 const size_t n5 = 0,
                 const size_t n6 = 0,
                 const size_t n7 = 0) {

  buf = BufferView(
          view.d_view.data(),
          n0,n1,n2,n3,n4,n5,n6,n7);

}

template<class DeviceType>
struct MemsetZeroFunctor {
  typedef DeviceType  execution_space ;
  void* ptr;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    ((int*)ptr)[i] = 0;
  }
};

#define SPARTA_LAMBDA KOKKOS_LAMBDA
#define SPARTA_CLASS_LAMBDA KOKKOS_CLASS_LAMBDA

namespace SPARTA_NS {
template <typename Device>
Kokkos::View<int*, Device> offset_scan(Kokkos::View<int*, Device> a, int& total);
}

#ifdef SPARTA_KOKKOS_GPU
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(__SYCL_DEVICE_ONLY__)
#define SPARTA_KK_DEVICE_COMPILE
#endif
#endif

#endif
