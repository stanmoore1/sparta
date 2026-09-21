// kp_count: a Kokkos Tools library which counts and times every kernel
// launch by name and every host-device copy by the labels of its two views.
// See README in this directory for how to build, load and read it.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <map>
#include <chrono>
#include <vector>
#include <cstring>
#include <execinfo.h>
#include <cxxabi.h>
struct Kokkos_Profiling_SpaceHandle { char name[64]; };
struct Rec { long count = 0; double time = 0; double bytes = 0; };
static std::map<std::string,Rec> kernels, copies;
static std::vector<std::pair<std::string,std::chrono::steady_clock::time_point>> stack;
static std::string prefix;
static void begin(const char *name, uint64_t *kid) {
  static uint64_t next = 0; *kid = next++;
  stack.push_back({name, std::chrono::steady_clock::now()});
}
static void end() {
  auto &e = stack.back();
  double dt = std::chrono::duration<double>(std::chrono::steady_clock::now()-e.second).count();
  Rec &r = kernels[prefix + e.first]; r.count++; r.time += dt;
  stack.pop_back();
}
extern "C" {
void kokkosp_init_library(int, uint64_t, uint32_t, void*) {}
void kokkosp_begin_parallel_for(const char *n, uint32_t, uint64_t *k) { begin(n,k); }
void kokkosp_begin_parallel_scan(const char *n, uint32_t, uint64_t *k) { begin(n,k); }
void kokkosp_begin_parallel_reduce(const char *n, uint32_t, uint64_t *k) { begin(n,k); }
void kokkosp_end_parallel_for(uint64_t) { end(); }
void kokkosp_end_parallel_scan(uint64_t) { end(); }
void kokkosp_end_parallel_reduce(uint64_t) { end(); }
// KP_COUNT_BT=<substring>: every deep copy whose dst or src label contains it
//   is also keyed by the SPARTA frames of its call stack, so the copies of
//   one array are attributed to the routines that cause them
static std::string stack_key() {
  void *frames[24];
  const int n = backtrace(frames,24);
  char **syms = backtrace_symbols(frames,n);
  std::string key;
  int kept = 0;
  for (int i = 0; i < n && kept < 4; i++) {
    const char *open = strchr(syms[i],'('); const char *plus = open ? strchr(open,'+') : nullptr;
    if (!open || !plus) continue;
    std::string mangled(open+1, plus-open-1);
    int status = 0; char *pretty = abi::__cxa_demangle(mangled.c_str(),nullptr,nullptr,&status);
    std::string name = (status == 0 && pretty) ? pretty : mangled;
    if (pretty) free(pretty);
    if (name.find("SPARTA_NS") == std::string::npos) continue;
    if (name.find("DualView") != std::string::npos || name.find("Kokkos") != std::string::npos) continue;
    size_t p = name.find('('); if (p != std::string::npos) name = name.substr(0,p);
    key += (kept ? " < " : "") + name.substr(name.rfind("SPARTA_NS::")+11); kept++;
  }
  free(syms);
  return key;
}
void kokkosp_begin_deep_copy(Kokkos_Profiling_SpaceHandle dh, const char *dn, const void*,
                             Kokkos_Profiling_SpaceHandle sh, const char *sn, const void*,
                             uint64_t size) {
  std::string label = prefix + std::string(dn) + " <- " + sn;
  static const char *bt = getenv("KP_COUNT_BT");
  if (bt && (strstr(dn,bt) || strstr(sn,bt)))
    label += std::string(" [") + dh.name + "<-" + sh.name + "] @ " + stack_key();
  Rec &r = copies[label]; r.count++; r.bytes += (double) size;
}
void kokkosp_push_profile_region(const char *n) { prefix = std::string(n) + " | "; }
void kokkosp_pop_profile_region() { prefix.clear(); }
void kokkosp_finalize_library() {
  const char *out = getenv("KP_COUNT_OUT");
  std::string path = out ? out : "";
  const char *rank = getenv("OMPI_COMM_WORLD_RANK");
  if (!rank) rank = getenv("PMI_RANK");
  if (out && rank) path += std::string(".") + rank;
  FILE *fp = out ? fopen(path.c_str(),"w") : stderr;
  fprintf(fp,"# kernels: count  seconds  name\n");
  for (auto &k : kernels) fprintf(fp,"K %8ld %10.4f  %s\n",k.second.count,k.second.time,k.first.c_str());
  fprintf(fp,"# copies: count  MB  dst <- src\n");
  for (auto &c : copies) fprintf(fp,"C %8ld %10.3f  %s\n",c.second.count,c.second.bytes/1e6,c.first.c_str());
  if (out) fclose(fp);
}
}
