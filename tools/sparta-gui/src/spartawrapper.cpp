// -*- c++ -*- /////////////////////////////////////////////////////////////////////////
// SPARTA-GUI - A Graphical Tool to Learn and Explore the SPARTA DSMC Simulation Software
//
// Copyright (c) 2023, 2024, 2025, 2026  Axel Kohlmeyer
//
// Documentation: https://sparta.github.io/sparta-gui/
// Contact: akohlmey@gmail.com
//
// This software is distributed under the GNU General Public License version 2 or later.
////////////////////////////////////////////////////////////////////////////////////////

#include "spartawrapper.h"

#include "constants.h"
#include "helpers.h"

#if defined(SPARTA_GUI_USE_PLUGIN)
#include "libspartaplugin.h"
#else
#include "library.h"
#endif

#include <cstdio>
#include <cstring>

#include <QFile>

#if defined(SPARTA_GUI_USE_PLUGIN) && defined(Q_OS_LINUX)
#include <elf.h>
#endif

// Dispatch a SPARTA C-library function by its base name.  In plugin mode this
// resolves to the dynamically loaded function table; in linked mode to the
// matching sparta_* symbol.  Usage: SPAFN(version)(sparta_handle).
#if defined(SPARTA_GUI_USE_PLUGIN)
#define SPAFN(fn) (((libspartaplugin_t *)plugin_handle)->fn)
#else
#define SPAFN(fn) (sparta_##fn)
#endif

SpartaWrapper::SpartaWrapper() : sparta_handle(nullptr)
{
#if defined(SPARTA_GUI_USE_PLUGIN)
    plugin_handle = nullptr;
#endif
}

SpartaWrapper::~SpartaWrapper()
{
#if defined(SPARTA_GUI_USE_PLUGIN)
    libspartaplugin_release(static_cast<libspartaplugin_t *>(plugin_handle));
    plugin_handle = nullptr;
#endif
}

void SpartaWrapper::open(int narg, char **args)
{
    // since there may only be one SPARTA instance in SPARTA-GUI we don't open a second one
    if (sparta_handle) return;
    sparta_handle = SPAFN(open_no_mpi)(narg, args, nullptr);
}

int SpartaWrapper::version()
{
    int val = 0;
    if (sparta_handle) {
        val = SPAFN(version)(sparta_handle);
    }
    return val;
}

int SpartaWrapper::extractSetting(const char *keyword)
{
    int val = 0;
    if (sparta_handle) {
        val = SPAFN(extract_setting)(sparta_handle, keyword);
    }
    return val;
}

void *SpartaWrapper::extractGlobal(const char *keyword)
{
    void *val = nullptr;
    if (sparta_handle) {
        val = SPAFN(extract_global)(sparta_handle, keyword);
    }
    return val;
}

// SPARTA has no pair styles, so there is no extract_pair library function.
// Kept as a stub so upstream call sites can be adapted with minimal changes.
void *SpartaWrapper::extractPair(const char *)
{
    return nullptr;
}

// SPARTA has no extract_atom library function (per-particle data is
// accessed via computes and fixes). Kept as a stub, see extractPair().
void *SpartaWrapper::extractAtom(const char *)
{
    return nullptr;
}

void *SpartaWrapper::extractCompute(const QString &id, int style, int type)
{
    int mystyle = -1;
    int mytype  = -1;

    switch (style) {
        case GLOBAL_STYLE:
            mystyle = SPA_STYLE_GLOBAL;
            break;
        case PARTICLE_STYLE:
            mystyle = SPA_STYLE_PARTICLE;
            break;
        case GRID_STYLE:
            mystyle = SPA_STYLE_GRID;
            break;
        case SURF_STYLE:
            mystyle = SPA_STYLE_SURF;
            break;
        case TALLY_STYLE:
            mystyle = SPA_STYLE_TALLY;
            break;
        default:
            mystyle = -1;
            break;
    }
    switch (type) {
        case SCALAR_TYPE:
            mytype = SPA_TYPE_SCALAR;
            break;
        case VECTOR_TYPE:
            mytype = SPA_TYPE_VECTOR;
            break;
        case ARRAY_TYPE:
            mytype = SPA_TYPE_ARRAY;
            break;
        default:
            // NUM_ROWS/NUM_COLS introspection is not available in SPARTA
            mytype = -1;
            break;
    }

    if (sparta_handle && (mystyle >= 0) && (mytype >= 0)) {
        return SPAFN(extract_compute)(sparta_handle, id.toLocal8Bit(), mystyle, mytype);
    }
    return nullptr;
}

void *SpartaWrapper::extractFix(const QString &id, int style, int type, int, int)
{
    // SPARTA's sparta_extract_fix() has no row/column arguments;
    // global data is returned as allocated memory the caller must free

    int mystyle = -1;
    int mytype  = -1;

    switch (style) {
        case GLOBAL_STYLE:
            mystyle = SPA_STYLE_GLOBAL;
            break;
        case PARTICLE_STYLE:
            mystyle = SPA_STYLE_PARTICLE;
            break;
        case GRID_STYLE:
            mystyle = SPA_STYLE_GRID;
            break;
        case SURF_STYLE:
            mystyle = SPA_STYLE_SURF;
            break;
        default:
            mystyle = -1;
            break;
    }
    switch (type) {
        case SCALAR_TYPE:
            mytype = SPA_TYPE_SCALAR;
            break;
        case VECTOR_TYPE:
            mytype = SPA_TYPE_VECTOR;
            break;
        case ARRAY_TYPE:
            mytype = SPA_TYPE_ARRAY;
            break;
        default:
            // NUM_ROWS/NUM_COLS introspection is not available in SPARTA
            mytype = -1;
            break;
    }

    if (sparta_handle && (mystyle >= 0) && (mytype >= 0)) {
        return SPAFN(extract_fix)(sparta_handle, id.toLocal8Bit(), mystyle, mytype);
    }
    return nullptr;
}

int SpartaWrapper::extractVariableDatatype(const QString &keyword)
{
    int type = -1;
    if (sparta_handle) {
        type = SPAFN(extract_variable_datatype)(sparta_handle, keyword.toLocal8Bit());
    }
    switch (type) {
        case SPA_VAR_EQUAL:
        case SPA_VAR_INTERNAL:
            return EQUAL_STYLE;
        case SPA_VAR_PARTICLE:
        case SPA_VAR_GRID:
        case SPA_VAR_SURF:
            // per-particle/grid/surf data, not representable as a scalar
            return ATOM_STYLE;
        case SPA_VAR_STRING:
            return STRING_STYLE;
        default:
            type = -1;
            break;
    }
    return type;
}

// note: equal style and compatible variables only
double SpartaWrapper::extractVariable(const char *keyword)
{
    void *ptr = nullptr;
    if (sparta_handle) {
        ptr = SPAFN(extract_variable)(sparta_handle, keyword);
    }
    double val = (ptr) ? *(static_cast<double *>(ptr)) : 0.0;
    SPAFN(free)(ptr);
    return val;
}

int SpartaWrapper::idCount(const char *idtype)
{
    int val = 0;
    if (sparta_handle) {
        val = SPAFN(id_count)(sparta_handle, idtype);
    }
    return val;
}

int SpartaWrapper::hasId(const char *idtype, const char *id)
{
    int val = 0;
    if (sparta_handle) {
        val = SPAFN(has_id)(sparta_handle, idtype, id);
    }
    return val;
}

int SpartaWrapper::idName(const char *keyword, int idx, char *buf, int len)
{
    int val = 0;
    if (sparta_handle) {
        val = SPAFN(id_name)(sparta_handle, keyword, idx, buf, len);
    }
    return val;
}

QString SpartaWrapper::idName(const char *keyword, int idx)
{
    char buf[Cfg::DEFAULT_BUFLEN];
    if (idName(keyword, idx, buf, Cfg::DEFAULT_BUFLEN)) return QString::fromLocal8Bit(buf);
    return {};
}

int SpartaWrapper::styleCount(const char *keyword)
{
    int val = 0;
    if (sparta_handle) {
        val = SPAFN(style_count)(sparta_handle, keyword);
    }
    return val;
}

int SpartaWrapper::styleName(const char *keyword, int idx, char *buf, int len)
{
    int val = 0;
    if (sparta_handle) {
        val = SPAFN(style_name)(sparta_handle, keyword, idx, buf, len);
    }
    return val;
}

QString SpartaWrapper::styleName(const char *keyword, int idx)
{
    char buf[Cfg::DEFAULT_BUFLEN];
    if (styleName(keyword, idx, buf, Cfg::DEFAULT_BUFLEN)) return QString::fromLocal8Bit(buf);
    return {};
}

int SpartaWrapper::variableInfo(int idx, char *buf, int len)
{
    int val = 0;
    if (sparta_handle) {
        val = SPAFN(variable_info)(sparta_handle, idx, buf, len);
    }
    return val;
}

QString SpartaWrapper::variableInfo(int idx)
{
    char buf[Cfg::DEFAULT_BUFLEN];
    if (variableInfo(idx, buf, Cfg::DEFAULT_BUFLEN)) return QString::fromLocal8Bit(buf);
    return {};
}

double SpartaWrapper::getThermo(const char *keyword)
{
    double val = 0.0;
    if (sparta_handle) {
        val = SPAFN(get_thermo)(sparta_handle, keyword);
    }
    return val;
}

void *SpartaWrapper::lastThermo(const char *keyword, int index)
{
    void *ptr = nullptr;
    if (sparta_handle) {
        ptr = SPAFN(last_thermo)(sparta_handle, keyword, index);
    }
    return ptr;
}

bool SpartaWrapper::isRunning()
{
    int val = 0;
    if (sparta_handle) {
        val = SPAFN(is_running)(sparta_handle);
    }
    return val != 0;
}

void SpartaWrapper::command(const QString &input)
{
    if (sparta_handle) {
        SPAFN(command)(sparta_handle, input.toLocal8Bit());
    }
}

void SpartaWrapper::file(const QString &filename)
{
    if (sparta_handle) {
        SPAFN(file)(sparta_handle, filename.toLocal8Bit());
    }
}

void SpartaWrapper::commandsString(const QString &input)
{
    if (sparta_handle) {
        SPAFN(commands_string)(sparta_handle, input.toLocal8Bit());
    }
}

bool SpartaWrapper::hasError() const
{
    if (!sparta_handle) return false;
    return SPAFN(has_error)(sparta_handle) != 0;
}

int SpartaWrapper::getLastErrorMessage(char *buf, int buflen)
{
    if (!sparta_handle) {
        if (buf && (buflen > 0)) buf[0] = '\0';
        return 0;
    }
    return SPAFN(get_last_error_message)(sparta_handle, buf, buflen);
}

QString SpartaWrapper::lastErrorMessage()
{
    if (!hasError()) return {};
    char buf[Cfg::DEFAULT_BUFLEN];
    getLastErrorMessage(buf, Cfg::DEFAULT_BUFLEN);
    return QString::fromLocal8Bit(buf);
}

void SpartaWrapper::forceTimeout()
{
    if (sparta_handle) SPAFN(force_timeout)(sparta_handle);
}

void SpartaWrapper::close()
{
#if defined(SPARTA_GUI_USE_PLUGIN)
    if (sparta_handle && plugin_handle) ((libspartaplugin_t *)plugin_handle)->close(sparta_handle);
#else
    if (sparta_handle) sparta_close(sparta_handle);
#endif
    sparta_handle = nullptr;
}

void SpartaWrapper::finalize()
{
    // SPARTA has no separate mpi/kokkos finalization in its library
    // interface; closing the instance is all that is needed
    if (sparta_handle) {
        SPAFN(close)(sparta_handle);
        // otherwise isOpen() reports an instance that no longer exists and a
        // later close() would close the stale handle a second time
        sparta_handle = nullptr;
    }
}

// The build-configuration queries below are asked about a library that may not
// be loaded: Preferences builds its accelerator tab from them, and Preferences
// is exactly where a user goes when the library could not be found and the path
// needs setting. In plugin mode SPAFN() dereferences the plugin handle, so
// asking without a library segfaulted the moment that dialog opened. Answering
// "no such feature" is both safe and true of a library that is not there.
bool SpartaWrapper::configHasPackage(const char *package) const
{
    if (!hasLibrary()) return false;
    return SPAFN(config_has_package)(package) != 0;
}

bool SpartaWrapper::configAccelerator(const char *package, const char *category,
                                      const char *setting) const
{
    if (!hasLibrary()) return false;
    return SPAFN(config_accelerator)(package, category, setting) != 0;
}

// SPARTA is not built with libcurl support
bool SpartaWrapper::configHasCurlSupport() const
{
    return false;
}

bool SpartaWrapper::configHasPngSupport() const
{
    if (!hasLibrary()) return false;
    return SPAFN(config_has_png_support)() != 0;
}

bool SpartaWrapper::configHasJpegSupport() const
{
    if (!hasLibrary()) return false;
    return SPAFN(config_has_jpeg_support)() != 0;
}

bool SpartaWrapper::configHasFfmpegSupport() const
{
    if (!hasLibrary()) return false;
    return SPAFN(config_has_ffmpeg_support)() != 0;
}

bool SpartaWrapper::configHasMpiSupport() const
{
    if (!hasLibrary()) return false;
    return SPAFN(config_has_mpi_support)() != 0;
}

bool SpartaWrapper::configHasGzipSupport() const
{
    if (!hasLibrary()) return false;
    return SPAFN(config_has_gzip_support)() != 0;
}

// Provenance strings read from the running instance (empty if no handle or
// the build did not stamp git info).  Used to enrich the archived run record.
QString SpartaWrapper::versionString() const
{
    if (!sparta_handle) return {};
    auto *p = static_cast<const char *>(SPAFN(extract_global)(sparta_handle, "sparta_version"));
    return p ? QString::fromUtf8(p) : QString();
}

QString SpartaWrapper::gitCommit() const
{
    if (!sparta_handle) return {};
    auto *p = static_cast<const char *>(SPAFN(extract_global)(sparta_handle, "git_commit"));
    return p ? QString::fromUtf8(p) : QString();
}

QString SpartaWrapper::gitBranch() const
{
    if (!sparta_handle) return {};
    auto *p = static_cast<const char *>(SPAFN(extract_global)(sparta_handle, "git_branch"));
    return p ? QString::fromUtf8(p) : QString();
}

// GPU support in SPARTA comes via Kokkos; there is no separate
// GPU-device detection in the library interface
bool SpartaWrapper::hasGpuDevice() const
{
    return false;
}

#undef SPAFN

#if defined(SPARTA_GUI_USE_PLUGIN)
bool SpartaWrapper::hasPlugin() const
{
    return true;
}

// Detect an obviously truncated or corrupt ELF shared object before it is
// handed to dlopen().  A partial file -- for example from an interrupted
// download -- makes the dynamic linker dereference relocation data past the end
// of the file and crash the entire process instead of failing cleanly.  The
// check is deliberately fail-safe: it returns true only on positive evidence of
// truncation and otherwise lets the loader proceed.
static bool pluginFileLooksTruncated(const QString &libfile)
{
#if defined(Q_OS_LINUX)
    QFile f(libfile);
    if (!f.open(QIODevice::ReadOnly)) return false;
    const qint64 filesize = f.size();

    Elf64_Ehdr ehdr;
    if (f.read(reinterpret_cast<char *>(&ehdr), sizeof(ehdr)) != sizeof(ehdr)) return false;

    // only validate native 64-bit little-endian ELF objects; anything else the
    // dynamic loader rejects on its own without crashing
    if (memcmp(ehdr.e_ident, ELFMAG, SELFMAG) != 0) return false;
    if ((ehdr.e_ident[EI_CLASS] != ELFCLASS64) || (ehdr.e_ident[EI_DATA] != ELFDATA2LSB))
        return false;

    // the section header table is conventionally at the very end of the file, so
    // a table that runs past EOF is a reliable indication of truncation
    if (ehdr.e_shoff != 0) {
        const qint64 shend = static_cast<qint64>(ehdr.e_shoff) +
                             static_cast<qint64>(ehdr.e_shnum) * ehdr.e_shentsize;
        if (shend > filesize) return true;
    }

    // every loadable segment must lie entirely within the file
    if ((ehdr.e_phoff != 0) && (ehdr.e_phnum > 0)) {
        const qint64 phend = static_cast<qint64>(ehdr.e_phoff) +
                             static_cast<qint64>(ehdr.e_phnum) * ehdr.e_phentsize;
        if (phend > filesize) return true;
        if (!f.seek(ehdr.e_phoff)) return false;
        for (unsigned i = 0; i < ehdr.e_phnum; ++i) {
            Elf64_Phdr phdr;
            if (f.read(reinterpret_cast<char *>(&phdr), sizeof(phdr)) != sizeof(phdr)) return true;
            if (phdr.p_type != PT_LOAD) continue;
            const qint64 segend =
                static_cast<qint64>(phdr.p_offset) + static_cast<qint64>(phdr.p_filesz);
            if (segend > filesize) return true;
        }
    }
    return false;
#else
    Q_UNUSED(libfile);
    return false;
#endif
}

bool SpartaWrapper::loadLib(const QString &libfile)
{
    // reject an obviously truncated or corrupt library up front; handing such a
    // file to dlopen() would crash the process inside the dynamic linker
    if (pluginFileLooksTruncated(libfile)) {
        fprintf(stderr,
                "SPARTA library file %s rejected.\n"
                "The file appears truncated or corrupted (e.g. from an incomplete "
                "download). Please remove it and install the library again.\n",
                libfile.toLocal8Bit().constData());
        return false;
    }

    if (plugin_handle) {
        close();
        libspartaplugin_release(static_cast<libspartaplugin_t *>(plugin_handle));
    }
    plugin_handle = libspartaplugin_load(libfile.toLocal8Bit());
    if (!plugin_handle) return false;
    auto *lmp = static_cast<libspartaplugin_t *>(plugin_handle);

    // check if ABI matches
    if (lmp->abiversion != SPARTAPLUGIN_ABI_VERSION) {
        // cache the ABI version before releasing lmp; the release frees it
        const int abiversion = lmp->abiversion;
        libspartaplugin_release(lmp);
        plugin_handle = nullptr;
        fprintf(stderr, "SPARTA library file %s rejected.\nIncompatible ABI: %d vs %d\n",
                libfile.toLocal8Bit().constData(), abiversion, SPARTAPLUGIN_ABI_VERSION);
        return false;
    }

    // check if all required recently added library functions are present; on
    // failure unload the library again so no unusable handle stays behind
#define CHECKSYM(symbol)                                                          \
    if (lmp->symbol == NULL) {                                                    \
        fprintf(stderr, "SPARTA library file %s is missing sparta_%s function\n", \
                libfile.toLocal8Bit().constData(), #symbol);                      \
        libspartaplugin_release(lmp);                                             \
        plugin_handle = nullptr;                                                  \
        return false;                                                             \
    }

    CHECKSYM(get_thermo);
    CHECKSYM(last_thermo);
    CHECKSYM(commands_string);
    CHECKSYM(style_count);
    CHECKSYM(is_running);

    // check minimum required version
    QString lmpversion;
    auto *ptr = static_cast<const char *>(lmp->extract_global(nullptr, "sparta_version"));
    if (ptr) lmpversion = ptr;

    // found a suitable version
    if (!lmpversion.isEmpty() && (dateCompare(lmpversion, Cfg::MIN_SPARTA_VERSION_STR) >= 0))
        return true;

    // the library loads but is too old: unload it again for consistency with
    // the other failure paths
    libspartaplugin_release(lmp);
    plugin_handle = nullptr;
    return false;
}
#else
bool SpartaWrapper::hasPlugin() const
{
    return false;
}

bool SpartaWrapper::loadLib(const QString &)
{
    return true;
}
#endif

// Local Variables:
// c-basic-offset: 4
// End:
